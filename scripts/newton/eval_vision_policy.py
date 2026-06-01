"""Closed-loop eval of the VISION ACT policy in Newton (camera -> action).

Runs the trained vision ACT across N Newton envs: every `action_horizon` control
steps it renders the wrist camera for each env (same mount as collect_vision_demos),
feeds (wrist image + proprio) to the ACT, and replays the predicted action chunk.
Reports DETERMINISTIC vial->vial success — the real number for a camera->action
policy on the form-closure grip.

  MUJOCO_GL=osmesa PYTHONPATH=src ~/newton-probe/bin/python scripts/newton/eval_vision_policy.py \
      checkpoints/newton_vision_act --num-envs 48 --action-horizon 10
"""
import os, sys, pathlib, argparse, re
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, torch, mujoco

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mujoco_urdf_test as M                       # noqa: E402
from arm_act.newton.rl_env import NewtonVecEnv     # noqa: E402
from arm_act.training.act_policy import load_policy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="checkpoints/newton_vision_act")
ap.add_argument("--num-envs", type=int, default=48)
ap.add_argument("--ep-len", type=int, default=180)
ap.add_argument("--action-horizon", type=int, default=10)
ap.add_argument("--res", type=int, default=224)
args = ap.parse_args()
W = H = args.res
dev = "cuda" if torch.cuda.is_available() else "cpu"


def render_xml():  # MUST match collect_vision_demos.py exactly (same camera the policy saw)
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
N, cpw = env.N, env.cpw
pol = load_policy(pathlib.Path(args.ckpt), device=dev)
model = pol.model.eval()
s_mean = torch.as_tensor(pol.state_stats.mean, dtype=torch.float32, device=dev)
s_std = torch.as_tensor(pol.state_stats.std, dtype=torch.float32, device=dev)
a_min = torch.as_tensor(pol.action_stats.min, dtype=torch.float32, device=dev)
a_max = torch.as_tensor(pol.action_stats.max, dtype=torch.float32, device=dev)
span = (a_max - a_min).clamp_min(1e-6)
chunk_sz = model.cfg.chunk_size

# render model (single env, wrist cam) reused for every env via qpos swap
m = mujoco.MjModel.from_xml_string(render_xml()); dat = mujoco.MjData(m)
m.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]; m.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
ren = mujoco.Renderer(m, H, W)


def render_all(jq, ppos):
    """Render the wrist cam for every env -> (N,3,H,W) uint8 tensor."""
    out = np.zeros((N, H, W, 3), np.uint8)
    for e in range(N):
        q = jq[e * cpw: e * cpw + cpw][: m.nq].copy()
        q[0:3] = ppos[e]; q[3:7] = [1, 0, 0, 0]; q[7:11] = [1, 0, 0, 0]
        dat.qpos[:] = q; mujoco.mj_forward(m, dat)
        ren.update_scene(dat, camera=cid); out[e] = ren.render()
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous().to(dev)


o = env.reset()
succ = np.zeros(N, bool)
chunk = None; cursor = 0
for t in range(args.ep_len):
    if chunk is None or cursor >= args.action_horizon:
        imgs = render_all(env.s0.joint_q.numpy(), env._plant())          # (N,3,H,W)
        eefp = env._eef()                                                 # (N,3)
        state = np.concatenate([eefp, np.tile([1, 0, 0, 0], (N, 1)),
                                np.ones((N, 1))], axis=1).astype(np.float32)  # (N,8)
        sn = (torch.from_numpy(state).to(dev) - s_mean) / s_std
        with torch.inference_mode():
            pred = model({"wrist_cam": imgs}, sn)                          # (N,chunk,4) normalized
        raw = (pred + 1.0) * 0.5 * span + a_min
        if model.cfg.gripper_classification:
            raw[..., -1] = torch.where(torch.sigmoid(pred[..., -1]) > 0.5, 1.0, -1.0)
        chunk = raw.cpu().numpy(); cursor = 0
    a = chunk[:, cursor]
    o, r, d, info = env.step(a)
    succ |= info["success"]
    cursor += 1

plant = env._plant(); dist = np.linalg.norm(plant[:, :2] - env._dest_xy, axis=1)
print(f"[vis-eval] ckpt={args.ckpt}  N={N}  action_horizon={args.action_horizon}", flush=True)
print(f"[vis-eval] DETERMINISTIC vial->vial success = {succ.mean()*100:.1f}%  ({succ.sum()}/{N})", flush=True)
print(f"[vis-eval] final dist_to_dest: p50={np.percentile(dist,50)*1000:.0f}mm "
      f"lifted_end={(plant[:,2]>0.030).mean()*100:.0f}%", flush=True)
