"""Closed-loop policy rollout in the standalone Newton (SolverMuJoCo) env.

Runs a trained ACT checkpoint IN the Newton sim (the engine that holds the
round-stem friction grasp) and reports vial->vial SUCCESS RATE — the metric
offline action-error can't give. The policy drives the arm via TCP-delta
actions in the SAME IK-rel convention the demos were recorded with:
    target_tcp = current_tcp + action[:3]   (clipped deltas, as in the oracle)
    j1,j2 = SCARA IK(target xy);  j3 = f(target z);  j4 = wrist-yaw compensation
    gripper = close if action[3] < 0 else open
For a vision policy (camera_keys non-empty) the table+wrist cameras are
rendered LIVE each control step via MuJoCo (same poses as render.py).

HARNESS VALIDATION: `--replay <demos.hdf5>` feeds RECORDED ORACLE actions through
the identical apply path instead of a policy. If that reproduces the grasp, the
IK/apply/success code is trustworthy and policy success rates are fair.

Runs under ~/newton-probe (newton + warp + mujoco + torch). Set MUJOCO_GL=egl.
    ~/newton-probe/bin/python -m arm_act.newton.eval_policy --checkpoint <dir> \
        --episodes 20 --spawn-jitter 0.006 [--cameras]
"""
from __future__ import annotations

import argparse, pathlib, sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "newton"))

import warp as wp                       # noqa: E402
import newton                           # noqa: E402
import newton_urdf_test as NT           # noqa: E402
import mujoco_urdf_test as M            # noqa: E402
from arm_act.newton.collect_demos import (  # noqa: E402
    NewtonOracle, load_task, PLANT_BODY, DESTVIAL_BODY, RECORD_EVERY,
)

CAM_W = CAM_H = 224
# j3 (prismatic descend) operational clamp. Physical URDF limit is [0,0.2], but the
# oracle grasp bottoms at j3~0.046 (tcp_z 0.104); descending past ~0.055 rams the
# prongs into the vial floor and a grip-close there diverges SolverMuJoCo (NaN).
# Cap at 0.06 so an over-deep command fails gracefully (no grasp) instead of crashing.
J3_LO, J3_HI = 0.0, 0.06


# ── state-conditioned reactive expert (DAgger teacher) ─────────────────────
# Given any state, emit clip(phase_goal - eef) in the eef/control frame + a grip
# cmd. Phases inferred from state (not step count) so it labels OFF-policy states
# the rollout visits. eef_z: grasp~0.105, lift~0.149 (tcp_z+j3=0.150 invariant).
EXP_GZ, EXP_LZ = 0.105, 0.149          # grasp / lift eef-z
EXP_MAX = np.array([0.04, 0.04, 0.02], dtype=np.float32)
GRIP_OPEN_A, GRIP_CLOSE_A = 1.0, -1.0


def expert_action(obs):
    eef = obs["eef_pos"]; plant = obs["pickable_pos"]; dest = obs["target_pos"]
    grip = float(obs["gripper_pos"][0])

    def go(tx, ty, tz, g, mxy=0.04, mz=0.02):
        d = np.array([tx - eef[0], ty - eef[1], tz - eef[2]], dtype=np.float32)
        d[:2] = np.clip(d[:2], -mxy, mxy); d[2] = np.clip(d[2], -mz, mz)
        return np.array([d[0], d[1], d[2], g], dtype=np.float32)

    # gripper closed on the 3mm stem saturates at ~0.002-0.004 (stem blocks full
    # 0.010 close), so detect "closed" at >0.0015 (open=0). Don't gate on plant
    # height — that deadlocks (won't lift until grasped, won't grasp until lifted).
    grasped = grip > 0.0015
    if not grasped:
        centered = abs(eef[0] - plant[0]) < 0.006 and abs(eef[1] - plant[1]) < 0.006
        if not centered:                       # center over plant at a safe height, open
            return go(plant[0], plant[1], max(eef[2], EXP_GZ + 0.02), GRIP_OPEN_A)
        if eef[2] > EXP_GZ + 0.004:            # descend to grasp depth, open
            return go(plant[0], plant[1], EXP_GZ, GRIP_OPEN_A)
        return go(plant[0], plant[1], EXP_GZ, GRIP_CLOSE_A)   # centered+deep -> close
    # grasped: GENTLE moves (marginal friction grasp drops under fast lateral yank;
    # oracle transports at ~0.004/step). Lift fully before transporting.
    if eef[2] < EXP_LZ - 0.004:                # lift straight up first (gentle xy hold)
        return go(eef[0], eef[1], EXP_LZ, GRIP_CLOSE_A, mxy=0.004, mz=0.012)
    if not (abs(eef[0] - dest[0]) < 0.008 and abs(eef[1] - dest[1]) < 0.008):
        return go(dest[0], dest[1], EXP_LZ, GRIP_CLOSE_A, mxy=0.008)   # GENTLE transport
    if eef[2] > EXP_GZ + 0.006:                # over dest -> descend into well (gentle)
        return go(dest[0], dest[1], EXP_GZ, GRIP_CLOSE_A, mxy=0.004, mz=0.012)
    return go(dest[0], dest[1], EXP_GZ, GRIP_OPEN_A)          # release


def _jittered_xml(dx, dy):
    mjcf = M._build_xml()
    if dx or dy:
        old = f'pos="{M.SRC_X} {M.SRC_Y} 0"'
        mjcf = mjcf.replace(old, f'pos="{M.SRC_X + dx} {M.SRC_Y + dy} 0"')
    return mjcf


class _LiveCams:
    """MuJoCo offscreen renderer mirroring render.py's table + side-wrist cams."""
    def __init__(self, mjcf):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self.model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
        self.model.vis.headlight.diffuse[:] = [0.85, 0.85, 0.85]
        self.r = mujoco.Renderer(self.model, height=CAM_H, width=CAM_W)
        self.link4 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_4")
        self.nq = self.model.nq

    def shot(self, joint_q):
        qp = joint_q.copy().astype(float)
        qp[3:7] = joint_q[[6, 3, 4, 5]]     # free quat xyzw->wxyz
        qp[7:11] = joint_q[[10, 7, 8, 9]]   # ball quat
        self.data.qpos[:] = qp[: self.nq]
        self.mj.mj_forward(self.model, self.data)
        out = {}
        tcam = self.mj.MjvCamera(); self.mj.mjv_defaultCamera(tcam)
        tcam.lookat[:] = [0.31, 0.065, 0.06]; tcam.distance = 0.42
        tcam.azimuth = 345.0; tcam.elevation = -42.0
        self.r.update_scene(self.data, camera=tcam)
        out["table_cam"] = self.r.render().copy()
        lx, ly, lz = self.data.xpos[self.link4]
        wcam = self.mj.MjvCamera(); self.mj.mjv_defaultCamera(wcam)
        wcam.lookat[:] = [lx, ly, max(0.0, lz - 0.19)]
        wcam.distance = 0.16; wcam.azimuth = 70.0; wcam.elevation = -35.0
        self.r.update_scene(self.data, camera=wcam)
        out["wrist_cam"] = self.r.render().copy()
        return out


def run_episode(oracle, jx, jy, action_fn, cameras, max_ctrl, recorder=None):
    """One closed-loop episode. action_fn(ctrl_idx, state, images)->[dx,dy,dz,grip].
    Returns (success, dist_mm, lifted)."""
    import torch  # noqa
    model = NT.build(src_dxy=(jx, jy))
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    solver = newton.solvers.SolverMuJoCo(model, integrator="implicitfast",
                                         cone="elliptic", iterations=100,
                                         ls_iterations=50, impratio=10.0)
    tgt = control.joint_target_pos if hasattr(control, "joint_target_pos") else control.joint_target
    cams = _LiveCams(_jittered_xml(jx, jy)) if cameras else None
    GO, GC = NT.GRIP_OPEN, NT.GRIP_CLOSED
    I1, I2 = M._ik(M.SRC_X + jx, M.SRC_Y + jy)
    st = {"s0": s0, "s1": s1}

    def set_arm(j1, j2, j3, j4, grip):
        v = tgt.numpy()
        v[..., NT.DOF_J1] = j1; v[..., NT.DOF_J2] = j2
        v[..., NT.DOF_J3] = j3; v[..., NT.DOF_J4] = j4
        v[..., NT.DOF_FL] = grip; v[..., NT.DOF_FR] = grip
        tgt.assign(v)

    def step_n(n):
        for _ in range(n):
            st["s0"].clear_forces()
            solver.step(st["s0"], st["s1"], control, None, NT.DT)
            st["s0"], st["s1"] = st["s1"], st["s0"]

    # settle at hover home (matches demo t=0 init)
    set_arm(I1, I2, M.INIT_J3, 0.0, GO)
    step_n(NT.nsteps("settle"))
    pz_settled = float(st["s0"].body_q.numpy()[PLANT_BODY][2])
    # Calibrate the prismatic z inverse-map from the settle pose: the SCARA has a
    # single z DOF (j3), so tcp_z + j3 == const. Measured invariant ~0.150; deriving
    # it live (tcp_z + the j3 we just set) avoids guessing link/prong offsets.
    CZ = float(oracle._tcp(st["s0"])[2]) + M.INIT_J3

    last_action = np.zeros(4, dtype=np.float32)
    for k in range(max_ctrl):
        s = st["s0"]
        obs = oracle._obs(s, last_action)
        images = cams.shot(s.joint_q.numpy()) if cameras else None
        if recorder is not None:                 # DAgger: label this visited state with the expert
            recorder(obs, expert_action(obs))
        a = action_fn(k, obs, images)
        if a is None:
            break
        last_action = a.astype(np.float32)
        tcp = obs["eef_pos"]
        tx, ty, tz = tcp[0] + a[0], tcp[1] + a[1], tcp[2] + a[2]
        j1, j2 = M._ik(float(tx), float(ty))
        j3 = float(np.clip(CZ - tz, J3_LO, J3_HI))
        j4 = -((j1 - I1) + (j2 - I2))
        grip = GC if a[3] < 0 else GO
        set_arm(j1, j2, j3, j4, grip)
        step_n(RECORD_EVERY)   # one control step = the demo control rate (100 substeps @ dt5e-4 = 20Hz)
        import os
        if os.environ.get("ARM_ACT_RDBG") and (k % 10 == 0 or k < 5):
            pp_dbg = st["s0"].body_q.numpy()[PLANT_BODY][:3]
            gp_dbg = float(obs["gripper_pos"][0])
            print(f"  k={k:3d} act={np.round(a,4)} tgt_tcp=({tx:.3f},{ty:.3f},{tz:.3f}) "
                  f"j3={j3:.3f} grip={gp_dbg:.4f} tcp_z={tcp[2]:.3f} plant=({pp_dbg[0]:.3f},{pp_dbg[1]:.3f},{pp_dbg[2]:.3f})", flush=True)

    pp = st["s0"].body_q.numpy()[PLANT_BODY][:3]
    dist = float(np.hypot(pp[0] - M.DEST_X, pp[1] - M.DEST_Y))
    lifted = pp[2] > pz_settled - 0.005
    return (dist < oracle.xy_thr and lifted), dist * 1000.0, lifted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=str(REPO / "src/arm_act/tasks/pick_plant_out_of_vial_zimmer.yaml"))
    ap.add_argument("--checkpoint", type=pathlib.Path, default=None, help="ACT checkpoint dir")
    ap.add_argument("--replay", type=pathlib.Path, default=None, help="HDF5 of recorded oracle actions (harness validation)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--spawn-jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ctrl", type=int, default=240)
    ap.add_argument("--cameras", action="store_true", help="force live camera render (else inferred from checkpoint)")
    ap.add_argument("--action-horizon", type=int, default=1,
                    help="re-plan cadence: 1 = reactive (re-plan every control step, decelerates at goal); "
                    "chunk_size = open-loop chunk replay (overshoots). Default 1.")
    ap.add_argument("--expert", action="store_true",
                    help="drive with the state-conditioned reactive expert (DAgger teacher) instead of a policy")
    ap.add_argument("--dagger-out", type=pathlib.Path, default=None,
                    help="with --checkpoint: roll out the POLICY but record (obs, EXPERT action) at every "
                    "visited state to this HDF5 (DAgger aggregation set).")
    args = ap.parse_args()

    wp.init()
    cfg = load_task(pathlib.Path(args.task))
    oracle = NewtonOracle(cfg)
    rng = np.random.default_rng(args.seed)

    cameras = args.cameras
    DAGGER_KEYS = ("eef_pos", "eef_quat", "gripper_pos", "pickable_pos", "target_pos", "joint_pos", "joint_vel", "object", "pickable_quat")
    if args.expert:
        print("[eval] EXPERT controller (state-conditioned reactive oracle)", flush=True)
        def action_fn(k, obs, images):
            return expert_action(obs)
    elif args.checkpoint is not None:
        import torch
        from arm_act.training.act_policy import load_policy
        policy = load_policy(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
        policy.action_horizon = max(1, int(args.action_horizon))   # reactive re-planning
        state_keys = tuple(policy.model.cfg.state_keys)
        cam_keys = tuple(policy.model.cfg.camera_keys)
        cameras = cameras or bool(cam_keys)
        print(f"[eval] policy: state_keys={state_keys} cameras={cam_keys or '(none)'}", flush=True)

        def action_fn(k, obs, images):
            state = np.concatenate([np.asarray(obs[kk], dtype=np.float32).reshape(-1) for kk in state_keys])
            imgs = {}
            if cam_keys:
                for c in cam_keys:
                    imgs[c] = torch.from_numpy(images[c]).permute(2, 0, 1).contiguous()
            return policy.select_action(imgs, torch.from_numpy(state).float())
    elif args.replay is not None:
        import h5py
        f = h5py.File(str(args.replay), "r")
        demo_ids = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        print(f"[eval] REPLAY harness-validation from {args.replay} ({len(demo_ids)} demos)", flush=True)

        def make_action_fn(did):
            acts = np.asarray(f["data"][did]["actions"], dtype=np.float32)
            def fn(k, obs, images):
                return acts[k] if k < len(acts) else None
            return fn
    else:
        print("need --checkpoint or --replay"); return 1

    dagger_demos = []   # list of per-episode {key: [frames]} dicts when --dagger-out
    succ = 0
    dists = []
    for ep in range(args.episodes):
        jx, jy = (rng.uniform(-args.spawn_jitter, args.spawn_jitter, size=2) if args.spawn_jitter > 0 else (0.0, 0.0))
        if args.replay is not None:
            fn = make_action_fn(demo_ids[ep % len(demo_ids)]); jx = jy = 0.0
        elif args.expert:
            fn = action_fn
        else:
            policy.reset(); fn = action_fn
        rec = None
        if args.dagger_out is not None:
            ep_rec = {k: [] for k in DAGGER_KEYS}; ep_rec["actions"] = []
            def rec(obs, exp_a, _r=ep_rec):
                for kk in DAGGER_KEYS:
                    _r[kk].append(np.asarray(obs[kk], dtype=np.float32).reshape(-1))
                _r["actions"].append(exp_a.astype(np.float32))
        ok, dist_mm, lifted = run_episode(oracle, float(jx), float(jy), fn, cameras, args.max_ctrl, recorder=rec)
        if args.dagger_out is not None and len(ep_rec["actions"]) > 0:
            dagger_demos.append(ep_rec)
        succ += int(ok)
        dists.append(dist_mm)
        print(f"[eval] ep {ep:2d}: {'SUCCESS' if ok else 'fail   '} dist={dist_mm:6.1f}mm lifted={lifted}", flush=True)

    if args.dagger_out is not None and dagger_demos:
        import h5py
        out = pathlib.Path(args.dagger_out); out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(out), "w") as fo:
            data = fo.create_group("data")
            for i, r in enumerate(dagger_demos):
                g = data.create_group(f"demo_{i}")
                g.create_dataset("actions", data=np.asarray(r["actions"], dtype=np.float32))
                og = g.create_group("obs")
                for kk in DAGGER_KEYS:
                    og.create_dataset(kk, data=np.asarray(r[kk], dtype=np.float32))
            data.attrs["total"] = len(dagger_demos)
        print(f"[eval] DAgger set: {len(dagger_demos)} demos (expert-labeled on-policy states) -> {out}", flush=True)
    print(f"\n>>> {succ}/{args.episodes} = {100.0*succ/args.episodes:.1f}%  "
          f"(median dist {np.median(dists):.1f}mm)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
