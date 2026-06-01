"""Camera rendering for Newton demos via MuJoCo replay.

The Newton sim has no RTX cameras, so to fill the obs camera keys (table_cam,
wrist_cam, wrist_depth) we replay the recorded joint_q frames through MuJoCo's
offscreen renderer (same scene MJCF). Cameras match the YAML poses.

CAVEAT (sim2sim visual gap): these are MuJoCo-rendered images, NOT Isaac RTX.
They are consistent for a Newton/MuJoCo-domain vision pipeline, but are a
DIFFERENT visual domain than the existing Isaac-RTX-trained vision policies — do
not expect them to drop into the RTX vision pipeline. State-based demos have no
such gap and are the primary deliverable.
"""
from __future__ import annotations

import pathlib, sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "newton"))
import mujoco_urdf_test as M  # noqa: E402

CAM_W = CAM_H = 224
# qpos quat blocks (xyzw in Newton joint_q -> wxyz in MuJoCo qpos): free 3:7, ball 7:11
_FREE_Q = slice(3, 7)
_BALL_Q = slice(7, 11)


def _qpos_from_jointq(q):
    qp = q.copy().astype(float)
    qp[3:7] = q[[6, 3, 4, 5]]
    qp[7:11] = q[[10, 7, 8, 9]]
    return qp


def render_cameras(ep):
    """Fill ep.cam['table_cam'|'wrist_cam'|'wrist_depth'] from ep.joint_q_frames."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(M._build_xml())
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [0.85, 0.85, 0.85]
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    link4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_4")

    # table_cam: fixed workcell pose (yaml pos [0.58,0,0.35]) -> az/el/dist
    tcam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(tcam)
    tcam.lookat[:] = [0.31, 0.065, 0.06]; tcam.distance = 0.42
    tcam.azimuth = 345.0; tcam.elevation = -42.0
    # wrist_cam: side/oblique gripper-follow cam aimed at the fingertip grasp
    # zone (~0.19 m below link_4). A top-down view (the old el=-89) saw only the
    # gripper body — the thin stem was occluded; this 3/4 side angle actually
    # shows the plant, the fingers closing on the stem, and the dest vial, and
    # stays framed through approach -> lift -> transport (lookat tracks link_4).
    wcam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(wcam)
    wcam.distance = 0.16; wcam.azimuth = 70.0; wcam.elevation = -35.0

    nq = model.nq
    for q in ep.joint_q_frames:
        qp = _qpos_from_jointq(q)
        data.qpos[:] = qp[:nq]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=tcam)
        ep.cam["table_cam"].append(renderer.render().copy())
        # wrist cam follows the gripper; aim at the fingertip/grasp zone
        # (~0.19 m below link_4) so the plant + closing fingers stay centered
        # through grasp and lift (link_4 rises as j3 retracts, plant rises with it)
        lx, ly, lz = data.xpos[link4]
        wcam.lookat[:] = [lx, ly, max(0.0, lz - 0.19)]
        renderer.update_scene(data, camera=wcam)
        ep.cam["wrist_cam"].append(renderer.render().copy())
        # wrist depth: center-pixel depth
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=wcam)
        depth = renderer.render()
        ep.cam["wrist_depth"].append(np.array([float(depth[CAM_H // 2, CAM_W // 2])], dtype=np.float32))
        renderer.disable_depth_rendering()
    renderer.close()
