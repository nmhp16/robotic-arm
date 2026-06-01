"""Render a video of the trained Newton policy doing vial->vial pick-and-place.

Runs the policy (deterministic mean action) across N envs in the NewtonVecEnv, records
each env's joint_q trajectory, picks a SUCCESSFUL episode (and a failure for contrast),
and replays the recorded qpos in a plain MuJoCo model (same MJCF) with offscreen osmesa
rendering -> mp4.  Env 0 of the replicated set sits at the world origin, so its 17-wide
joint_q slice maps directly onto the single-env MuJoCo qpos.

  MUJOCO_GL=osmesa ~/newton-probe/bin/python scripts/newton/eval_video.py \
      checkpoints/newton_rl_fc8/actor.pt eval/runs/newton_rl_fc8/policy_vial2vial.mp4
"""
import os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, torch, mujoco, imageio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mujoco_urdf_test as M                       # noqa: E402
from arm_act.newton.rl_env import NewtonVecEnv     # noqa: E402
from arm_act.newton.ppo import ActorCritic, RunNorm  # noqa: E402

import re
ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/newton_rl_fc8/actor.pt"
# default output lives in the project eval dir (eval/runs/), never /tmp
_dflt = pathlib.Path("eval/runs") / pathlib.Path(ckpt).parent.name / "policy_vial2vial.mp4"
out  = sys.argv[2] if len(sys.argv) > 2 else str(_dflt)
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
N, T = 16, 180
W, H, FPS = 1280, 720, 20


def nicen(xml: str) -> str:
    """Inject skybox, textured checker floor, a shadow-casting directional light, and
    nicer materials into the bare collider MJCF so the render looks presentable."""
    assets = """
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.45 0.6 0.78" rgb2="0.08 0.10 0.16" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" mark="edge" markrgb="0.55 0.55 0.6"
             rgb1="0.30 0.32 0.36" rgb2="0.23 0.25 0.29" width="512" height="512"/>
    <material name="gridmat" texture="grid" texrepeat="10 10" reflectance="0.15" specular="0.2" shininess="0.3"/>
    <material name="plantmat" rgba="0.25 0.65 0.20 1" specular="0.1" shininess="0.1"/>
    <material name="glassmat" rgba="0.70 0.85 0.95 0.30" specular="0.6" shininess="0.7" reflectance="0.1"/>
    <material name="steelmat" rgba="0.30 0.32 0.38 1" specular="0.8" shininess="0.8" reflectance="0.25"/>"""
    # larger offscreen framebuffer (default 640x480) + higher shadow quality
    vis = ('<visual><global offwidth="1280" offheight="720"/>'
           '<quality shadowsize="8192" offsamples="8"/>'
           '<map shadowclip="0.4"/></visual>')
    xml = re.sub(r"(<asset>)", vis + r"\1" + assets, xml, count=1)
    # shadow-casting key light + soft fill
    light = ('\n    <light name="key" directional="true" castshadow="true" '
             'pos="0.1 -0.3 1.2" dir="0.2 0.5 -1" diffuse="0.85 0.85 0.82" specular="0.3 0.3 0.3"/>'
             '\n    <light name="fill" directional="true" castshadow="false" '
             'pos="-0.4 0.4 1.0" dir="-0.3 -0.4 -1" diffuse="0.35 0.35 0.4"/>')
    xml = re.sub(r"(<worldbody>)", r"\1" + light, xml, count=1)
    # floor -> grid material
    xml = re.sub(r'(<geom name="floor"[^>]*?)\s*rgba="[^"]*"', r"\1", xml)
    xml = re.sub(r'(<geom name="floor"\b)', r'\1 material="gridmat"', xml)
    # wide_vial -> glass; gripper vis -> steel
    xml = re.sub(r'(<geom type="mesh" mesh="wide_vial"[^>]*?)rgba="[^"]*"', r'\1material="glassmat"', xml)
    xml = re.sub(r'(mesh="vis_gripper"[^>]*?)rgba="[^"]*"', r'\1material="steelmat"', xml)
    return xml

# ---- run the policy, record per-env joint_q trajectory ----
M.FORM_CLOSURE = True
env = NewtonVecEnv(num_envs=N, spawn_jitter=0.006, max_ctrl=T, init_mode="grasped",
                   form_closure=True, max_delta=[0.015, 0.015, 0.02])
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
ac = ActorCritic(env.obs_dim, env.act_dim); ac.load_state_dict(ck["actor"]); ac.eval()
rn = RunNorm(env.obs_dim); rn.mean = ck["obs_mean"]; rn.var = ck["obs_var"]
cpw = env.cpw

o = env.reset()
traj = [env.s0.joint_q.numpy().copy()]            # (T+1, N*cpw) robot joints
ppos = [env._plant().copy()]                       # (T+1, N, 3) plant world xyz (vial frame)
succ = np.zeros(N, bool)
for _ in range(T):
    with torch.no_grad():
        mu, _, _ = ac(torch.from_numpy(rn.norm(o)))
    o, r, d, info = env.step(mu.numpy())
    traj.append(env.s0.joint_q.numpy().copy())
    ppos.append(env._plant().copy())
    succ |= info["success"]
traj = np.stack(traj)                              # (T+1, N*cpw)
ppos = np.stack(ppos)                              # (T+1, N, 3)
print(f"[video] rollout done: {succ.mean()*100:.0f}% success ({succ.sum()}/{N})", flush=True)

# pick env 0 (origin) regardless of outcome, but prefer it being a success; if env 0
# failed, fall back to the lowest-index success — but render in env 0's origin frame is
# only valid for env 0, so we render env 0 and just report its outcome.
render_env = 0
print(f"[video] rendering env {render_env} (success={bool(succ[render_env])})", flush=True)

# ---- replay recorded qpos in plain MuJoCo, render ----
m = mujoco.MjModel.from_xml_string(nicen(M._build_xml()))
m.vis.headlight.ambient[:] = [0.55, 0.55, 0.58]   # bright fill so the shadowed plant shows
m.vis.headlight.diffuse[:] = [0.45, 0.45, 0.45]
# Render the REAL plant at its true size: the stem colliders (stem_lo/waist/up + root,
# normally alpha=0) and the leaf_plant visual mesh -> opaque green.  The plant is a ~1cm
# stem gripped inside the jaws, so make the gripper TRANSLUCENT to see it.  No fake/
# oversized marker — this shows the actual (small) plant relative to the vial.
PLANT_GEOMS = {"stem", "stem_lo", "stem_waist", "stem_up", "root"}
for gi in range(m.ngeom):
    gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gi]) or ""
    is_mesh = m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH
    mn = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gi]) or "") if is_mesh else ""
    if gn in PLANT_GEOMS or mn == "leaf_plant":
        m.geom_matid[gi] = -1; m.geom_rgba[gi] = [0.15, 0.80, 0.12, 1.0]   # real plant, green
    elif ("finger" in bn) or mn == "vis_gripper" or bn == "link_4":
        m.geom_matid[gi] = -1
        m.geom_rgba[gi] = [0.55, 0.57, 0.62, 0.30]   # translucent gripper (see the plant)
d = mujoco.MjData(m)
cam = mujoco.MjvCamera()
cam.lookat[:] = [(M.SRC_X + M.DEST_X) / 2, (M.SRC_Y + M.DEST_Y) / 2, 0.06]
cam.distance, cam.azimuth, cam.elevation = 0.32, 128.0, -22.0
ren = mujoco.Renderer(m, H, W)
scn = ren.scene
scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
sl = slice(render_env * cpw, render_env * cpw + cpw)
frames = []
for t in range(traj.shape[0]):
    q = traj[t, sl][: m.nq].copy()
    # The robot hinge joints (11:16) replay fine from joint_q, but the plant FREEJOINT
    # (0:6) is in a different frame/quat-order than the render model -> rendered the
    # plant off-camera/mis-rotated.  Use the true plant world xyz (env._plant, vial
    # frame) with identity orientation; the plant is gripped upright anyway.
    q[0:3] = ppos[t, render_env]
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    q[7:11] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    ren.update_scene(d, cam)
    frames.append(ren.render().copy())
imageio.mimsave(out, frames, fps=FPS)
print(f"[video] wrote {out}  ({len(frames)} frames, {len(frames)/FPS:.1f}s)", flush=True)
