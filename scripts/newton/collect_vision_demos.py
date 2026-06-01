"""Distill the state RL teacher -> wrist-cam VISION dataset.

Runs the trained state policy (teacher) across N Newton envs, and for every step of
each SUCCESSFUL episode renders the wrist camera (mounted on link_4) and records
(wrist_cam image, proprio state, teacher action) into an HDF5 in the train_act schema:

  data/demo_i/actions            (T, 4)
  data/demo_i/obs/wrist_cam      (T, H, W, 3) uint8
  data/demo_i/obs/eef_pos        (T, 3)
  data/demo_i/obs/eef_quat       (T, 4)
  data/demo_i/obs/gripper        (T, 1)

A vision ACT trained on this imitates the teacher from images -> a real camera->action
policy on the form-closure grip.  (Newton can't batch-render for vision-RL; offline
single-env rendering for distillation is cheap.)

  MUJOCO_GL=osmesa PYTHONPATH=src ~/newton-probe/bin/python scripts/newton/collect_vision_demos.py \
      checkpoints/newton_rl_fc8/actor.pt data/raw/newton_vision/demos.hdf5 --n-demos 80
"""
import os, sys, pathlib, argparse, re
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, torch, mujoco, h5py

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mujoco_urdf_test as M                       # noqa: E402
from arm_act.newton.rl_env import NewtonVecEnv     # noqa: E402
from arm_act.newton.ppo import ActorCritic, RunNorm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="checkpoints/newton_rl_fc8/actor.pt")
ap.add_argument("out", nargs="?", default="data/raw/newton_vision/demos.hdf5")
ap.add_argument("--n-demos", type=int, default=80)
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--ep-len", type=int, default=180)
ap.add_argument("--res", type=int, default=128)
args = ap.parse_args()
W = H = args.res

# wrist cam on link_4 + green plant, for the RENDER model only (physics unchanged)
def render_xml():
    xml = M._build_xml()
    xml = re.sub(r'(<geom type="mesh" mesh="leaf_plant"[^>]*?)rgba="[^"]*"',
                 r'\1rgba="0.15 0.8 0.12 1"', xml)
    xml = re.sub(r'(<asset>)',
                 r'<visual><global offwidth="256" offheight="256"/></visual>\1', xml, count=1)
    cam = '\n      <camera name="wrist_cam" mode="targetbody" target="plant" pos="0.12 0 0.06" fovy="50"/>'
    return re.sub(r'(<body name="link_4"[^>]*>)', r'\1' + cam, xml, count=1)

M.FORM_CLOSURE = True
env = NewtonVecEnv(num_envs=args.num_envs, spawn_jitter=0.006, max_ctrl=args.ep_len,
                   init_mode="grasped", form_closure=True, max_delta=[0.015, 0.015, 0.02])
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
ac = ActorCritic(env.obs_dim, env.act_dim); ac.load_state_dict(ck["actor"]); ac.eval()
rn = RunNorm(env.obs_dim); rn.mean = ck["obs_mean"]; rn.var = ck["obs_var"]
cpw = env.cpw

# rollout: record joint_q, plant xyz, eef, action, success per env per step
o = env.reset()
jq, ppos, eef, acts = [env.s0.joint_q.numpy().copy()], [env._plant().copy()], [env._eef().copy()], []
succ = np.zeros(env.N, bool)
for _ in range(args.ep_len):
    with torch.no_grad():
        mu, _, _ = ac(torch.from_numpy(rn.norm(o)))
    a = mu.numpy()
    o, r, d, info = env.step(a)
    acts.append(a.copy()); jq.append(env.s0.joint_q.numpy().copy())
    ppos.append(env._plant().copy()); eef.append(env._eef().copy())
    succ |= info["success"]
jq = np.stack(jq); ppos = np.stack(ppos); eef = np.stack(eef); acts = np.stack(acts)  # (T(+1),N,..)
good = np.where(succ)[0]
print(f"[vis] {succ.mean()*100:.0f}% success; {len(good)} usable demos (need {args.n_demos})", flush=True)

# render model + wrist cam
m = mujoco.MjModel.from_xml_string(render_xml()); dat = mujoco.MjData(m)
m.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]   # bare MJCF is dark; brighten for usable images
m.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
ren = mujoco.Renderer(m, H, W)
GRIP_CLOSED = 1.0   # auto_grip teacher holds closed all episode

pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
n_written = 0
with h5py.File(args.out, "w") as f:
    g = f.create_group("data")
    for e in good[: args.n_demos]:
        T = args.ep_len
        imgs = np.zeros((T, H, W, 3), np.uint8)
        for t in range(T):
            q = jq[t, e * cpw: e * cpw + cpw][: m.nq].copy()
            q[0:3] = ppos[t, e]; q[3:7] = [1, 0, 0, 0]; q[7:11] = [1, 0, 0, 0]
            dat.qpos[:] = q; mujoco.mj_forward(m, dat)
            ren.update_scene(dat, camera=cid); imgs[t] = ren.render()
        ep = eef[:T, e]                                    # (T,3) eef xyz
        dg = g.create_group(f"demo_{n_written}")
        dg.create_dataset("actions", data=acts[:T, e].astype(np.float32))
        og = dg.create_group("obs")
        og.create_dataset("wrist_cam", data=imgs)
        og.create_dataset("eef_pos", data=ep.astype(np.float32))
        og.create_dataset("eef_quat", data=np.tile([1, 0, 0, 0], (T, 1)).astype(np.float32))
        og.create_dataset("gripper", data=np.full((T, 1), GRIP_CLOSED, np.float32))
        n_written += 1
        if n_written % 10 == 0:
            print(f"[vis] wrote {n_written} demos", flush=True)
print(f"[vis] DONE: {n_written} vision demos -> {args.out}", flush=True)
