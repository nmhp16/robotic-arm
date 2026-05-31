#!/usr/bin/env python3
"""
Collect vial->vial demos with the Newton (SolverMuJoCo, GPU) oracle and write them
in the EXACT Isaac Lab HDF5 schema the existing pipeline (mimic annotate -> LeRobot
-> ACT/SmolVLA) consumes.

Schema replicated (verified against data/raw/.../demos.hdf5):
  /  attr env_args (json)
  /data  attr total
  /data/demo_<n>  attr num_samples
      actions            (T,4) float32   [dx,dy,dz,gripper_cmd]  (TCP deltas m; +1 open / -1 close)
      obs/actions        (T,4)            last action (0 at t=0)
      obs/joint_pos      (T,6)            6 robot joints rel home
      obs/joint_vel      (T,6)
      obs/object         (T,16)           plant_pos(3)+plant_quat(4)+target_pos(3)+(plant-ee)(3)+(target-plant)(3)
      obs/pickable_pos   (T,3)  obs/pickable_quat (T,4)  obs/target_pos (T,3)
      obs/eef_pos        (T,3)  obs/eef_quat (T,4)  obs/gripper_pos (T,1)
      obs/table_cam (T,224,224,3) u8  obs/wrist_cam (T,224,224,3) u8  obs/wrist_depth (T,1)   [--cameras]
      initial_state/articulation/robot/{joint_position,joint_velocity,root_pose,root_velocity}
      initial_state/rigid_object/{pickable,target,vial}/{root_pose,root_velocity}

Records at the CONTROL rate (~20 Hz = every 100 Newton steps at dt=5e-4), matching
the Isaac env (dt 0.01 x decimation 5), so demo length matches the pipeline (~130 frames).

Run (newton-probe venv):
  ~/newton-probe/bin/python -m arm_act.newton.collect_demos --num-demos 50 \
      --out data/raw/pick_plant_out_of_vial_zimmer/demos_newton.hdf5
  (or:  --task <path-to-yaml>  --cameras  --max-attempts 100)
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "newton"))

import warp as wp                       # noqa: E402
import newton                           # noqa: E402
from newton import JointTargetMode      # noqa: E402
import newton_urdf_test as NT           # noqa: E402  (proven build + indices + gains)
import mujoco_urdf_test as M            # noqa: E402  (IK + geometry constants)

import yaml  # noqa: E402

# dof / coord / body indices (from NT, verified against the imported MJCF)
DOF = (NT.DOF_J1, NT.DOF_J2, NT.DOF_J3, NT.DOF_J4, NT.DOF_FL, NT.DOF_FR)
QID = (NT.Q_J1, NT.Q_J2, NT.Q_J3, NT.Q_J4, 15, 16)
PLANT_BODY, SRCVIAL_BODY, DESTVIAL_BODY, LINK4_BODY = 0, 2, 3, 7
FL_FR_BODIES = (8, 9)

CONTROL_HZ = 20.0
RECORD_EVERY = max(1, int(round((1.0 / CONTROL_HZ) / NT.DT)))   # 100 steps @ dt=5e-4


def _wxyz(q_xyzw):
    """warp/Newton stores quats xyzw; Isaac HDF5 expects wxyz."""
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def load_task(task_path: pathlib.Path) -> dict:
    with open(task_path) as f:
        return yaml.safe_load(f)


class Episode:
    """Accumulates control-rate (obs, action) frames for one attempt."""
    def __init__(self):
        self.actions = []
        self.obs = {k: [] for k in
                    ("actions", "joint_pos", "joint_vel", "object", "pickable_pos",
                     "pickable_quat", "target_pos", "eef_pos", "eef_quat", "gripper_pos")}
        self.cam = {"table_cam": [], "wrist_cam": [], "wrist_depth": []}
        self.joint_q_frames = []   # for optional camera replay
        self.initial_state = None


class NewtonOracle:
    def __init__(self, cfg: dict, cameras: bool = False, spawn_jitter: float = 0.0,
                 seed: int = 0):
        self.cfg = cfg
        self.cameras = cameras
        orc = cfg["oracle"]; succ = cfg["success"]
        self.hover = orc["hover_height"]; self.grasp_off = orc["grasp_z_offset"]
        self.lift = orc["lift_height"]; self.place_off = orc["place_z_offset"]
        self.max_dxy = orc["max_dxy"]; self.max_dz = orc["max_dz"]
        self.xy_thr = succ["xy_threshold"]; self.h_thr = succ["height_threshold"]
        self.dest = (M.DEST_X, M.DEST_Y)
        # per-episode pick-side spawn DR (m, uniform +/-jitter on x and y). The
        # oracle is privileged: it targets the jittered SRC, so grasp alignment is
        # preserved (only the absolute pick location varies). 0.0 = fixed spawn.
        self.spawn_jitter = float(spawn_jitter)
        self.rng = np.random.default_rng(seed)
        wp.init()
        # cache the un-jittered model (reused every episode when jitter==0)
        self.model = NT.build()
        self.home6 = np.array([M.INIT_J1, M.INIT_J2, M.INIT_J3, 0.0, 0.0, 0.0], dtype=np.float32)

    # ---- per-step state reads -------------------------------------------------
    def _tcp(self, s):
        q = s.body_q.numpy(); l, r = q[FL_FR_BODIES[0]], q[FL_FR_BODIES[1]]
        return np.array([(l[0]+r[0])/2, (l[1]+r[1])/2, (l[2]+r[2])/2], dtype=np.float32)

    def _obs(self, s, last_action):
        bq = s.body_q.numpy(); jq = s.joint_q.numpy(); jqd = s.joint_qd.numpy()
        tcp = self._tcp(s)
        plant_pos = bq[PLANT_BODY][:3].astype(np.float32)
        plant_quat = _wxyz(bq[PLANT_BODY][3:7])
        target_pos = bq[DESTVIAL_BODY][:3].astype(np.float32)
        eef_quat = _wxyz(bq[LINK4_BODY][3:7])
        joint_pos = jq[list(QID)].astype(np.float32) - self.home6
        joint_vel = jqd[list(DOF)].astype(np.float32)
        gripper_pos = np.array([jq[QID[4]]], dtype=np.float32)
        obj = np.concatenate([plant_pos, plant_quat, target_pos,
                              plant_pos - tcp, target_pos - plant_pos]).astype(np.float32)
        return dict(actions=last_action.astype(np.float32), joint_pos=joint_pos,
                    joint_vel=joint_vel, object=obj, pickable_pos=plant_pos,
                    pickable_quat=plant_quat, target_pos=target_pos, eef_pos=tcp,
                    eef_quat=eef_quat, gripper_pos=gripper_pos)

    def _action(self, tcp, waypoint, grip_open):
        d = np.array(waypoint, dtype=np.float32) - tcp
        dx = float(np.clip(d[0], -self.max_dxy, self.max_dxy))
        dy = float(np.clip(d[1], -self.max_dxy, self.max_dxy))
        dz = float(np.clip(d[2], -self.max_dz, self.max_dz))
        return np.array([dx, dy, dz, 1.0 if grip_open else -1.0], dtype=np.float32)

    # ---- run one attempt, returns (success, Episode) --------------------------
    def run_episode(self):
        # per-episode pick-side spawn jitter (rebuild model with shifted plant+src vial)
        if self.spawn_jitter > 0.0:
            jx, jy = self.rng.uniform(-self.spawn_jitter, self.spawn_jitter, size=2)
            model = NT.build(src_dxy=(float(jx), float(jy)))
            self.src = (M.SRC_X + float(jx), M.SRC_Y + float(jy))
            self._init_j12 = M._ik(*self.src)        # SCARA IK to the jittered pick xy
        else:
            model = self.model
            self.src = (M.SRC_X, M.SRC_Y)
            self._init_j12 = (M.INIT_J1, M.INIT_J2)
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        solver = newton.solvers.SolverMuJoCo(model, integrator="implicitfast",
                                             cone="elliptic", iterations=100,
                                             ls_iterations=50, impratio=10.0)
        tgt = control.joint_target_pos if hasattr(control, "joint_target_pos") else control.joint_target

        ep = Episode()
        last_action = np.zeros(4, dtype=np.float32)
        step_ctr = [0]
        state = {"s0": s0, "s1": s1}

        def set_arm(j1, j2, j3, j4, grip):
            v = tgt.numpy()
            v[..., NT.DOF_J1] = j1; v[..., NT.DOF_J2] = j2
            v[..., NT.DOF_J3] = j3; v[..., NT.DOF_J4] = j4
            v[..., NT.DOF_FL] = grip; v[..., NT.DOF_FR] = grip
            tgt.assign(v)

        def record(waypoint, grip_open):
            nonlocal last_action
            s = state["s0"]
            tcp = self._tcp(s)
            action = self._action(tcp, waypoint, grip_open)
            for k, val in self._obs(s, last_action).items():
                ep.obs[k].append(val)
            ep.actions.append(action)
            if self.cameras:
                ep.joint_q_frames.append(s.joint_q.numpy().copy())
            last_action = action

        def step_n(n, waypoint, grip_open):
            for _ in range(n):
                state["s0"].clear_forces()
                solver.step(state["s0"], state["s1"], control, None, NT.DT)
                state["s0"], state["s1"] = state["s1"], state["s0"]
                if step_ctr[0] % RECORD_EVERY == 0:
                    record(waypoint, grip_open)
                step_ctr[0] += 1

        def ramp(n, f, waypoint, grip_open):
            for i in range(n):
                f((i + 1) / n)
                state["s0"].clear_forces()
                solver.step(state["s0"], state["s1"], control, None, NT.DT)
                state["s0"], state["s1"] = state["s1"], state["s0"]
                if step_ctr[0] % RECORD_EVERY == 0:
                    record(waypoint, grip_open)
                step_ctr[0] += 1

        I1, I2 = self._init_j12; I3 = M.INIT_J3
        D1, D2 = M.DEST_J1, M.DEST_J2
        J3L = M.J3_LIFT
        GO, GC = NT.GRIP_OPEN, NT.GRIP_CLOSED
        ns = NT.nsteps; sm = NT.smooth
        sx, sy = self.src; dx_, dy_ = self.dest

        # waypoint TCP targets per phase (for the recorded TCP-delta action)
        hover_wp = (sx, sy, self.hover)
        grasp_wp = (sx, sy, self.grasp_off)
        lift_wp = (sx, sy, self.lift)
        dest_lift_wp = (dx_, dy_, self.lift)
        place_wp = (dx_, dy_, self.place_off)

        # ---- settle (capture initial_state) -----------------------------------
        set_arm(I1, I2, I3, 0.0, GO)
        step_n(ns("settle"), hover_wp, True)
        ep.initial_state = self._initial_state(state["s0"])
        px, py, pz_settled = state["s0"].body_q.numpy()[PLANT_BODY][:3]
        stem_z = pz_settled + 0.035
        j3_g = M.LINK4_Z_J3_0 - M.PRONG_BELOW_L4 - stem_z
        j3_ins = j3_g

        # ---- descend / close / hold -------------------------------------------
        ramp(ns("descend"), lambda t: set_arm(I1, I2, I3 + t*(j3_g-I3), 0.0, GO), grasp_wp, True)
        set_arm(I1, I2, j3_g, 0.0, GC); step_n(ns("close"), grasp_wp, False)
        step_n(ns("hold"), grasp_wp, False)
        # ---- lift / escape ----------------------------------------------------
        ramp(ns("lift"), lambda t: set_arm(I1, I2, j3_g + t*(J3L-j3_g), 0.0, GC), lift_wp, False)
        set_arm(I1, I2, J3L, 0.0, GC); step_n(ns("escape"), lift_wp, False)
        # ---- transport (eased) ------------------------------------------------
        j4_end = -((D1-I1) + (D2-I2))
        def _trans(t):
            e = sm(t); j1 = I1 + e*(D1-I1); j2 = I2 + e*(D2-I2)
            set_arm(j1, j2, J3L, -((j1-I1)+(j2-I2)), GC)
        ramp(ns("transport"), _trans, dest_lift_wp, False)
        # ---- align (shift dest IK by plant-TCP lag) ---------------------------
        s = state["s0"]; tcp = self._tcp(s); pp = s.body_q.numpy()[PLANT_BODY][:3]
        ox, oy = tcp[0]-pp[0], tcp[1]-pp[1]
        if pp[2] > 0.025 and abs(ox) < 0.05 and abs(oy) < 0.05:
            D1c, D2c = M._ik(dx_ + ox, dy_ + oy)
        else:
            D1c, D2c = D1, D2
        j4c = -((D1c-I1) + (D2c-I2))
        ramp(ns("escape"), lambda t: set_arm(D1+sm(t)*(D1c-D1), D2+sm(t)*(D2c-D2),
                                             J3L, j4_end+sm(t)*(j4c-j4_end), GC), dest_lift_wp, False)
        # ---- insert / release / check -----------------------------------------
        ramp(ns("insert"), lambda t: set_arm(D1c, D2c, J3L + sm(t)*(j3_ins-J3L), j4c, GC), place_wp, False)
        no = ns("release"); half = no // 2
        set_arm(D1c, D2c, j3_ins, j4c, GO); step_n(half, place_wp, True)
        ramp(no - half, lambda t: set_arm(D1c, D2c, j3_ins + t*(J3L-j3_ins), j4c*(1-t), GO), dest_lift_wp, True)
        set_arm(D1c, D2c, J3L, 0.0, GO); step_n(ns("check"), dest_lift_wp, True)

        # ---- success: plant landed in dest well -------------------------------
        px, py, pz = state["s0"].body_q.numpy()[PLANT_BODY][:3]
        dist = float(np.hypot(px - dx_, py - dy_))
        success = dist < self.xy_thr and pz > pz_settled - 0.005
        return success, ep, dist

    def _initial_state(self, s):
        bq = s.body_q.numpy(); jq = s.joint_q.numpy()
        def pose(b):
            p = bq[b]; return np.concatenate([p[:3], _wxyz(p[3:7])]).astype(np.float32)[None, :]
        jp = jq[list(QID)].astype(np.float32)[None, :]
        z6 = np.zeros((1, 6), dtype=np.float32)
        return {
            "articulation/robot/joint_position": jp,
            "articulation/robot/joint_velocity": z6,
            "articulation/robot/root_pose": np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float32),
            "articulation/robot/root_velocity": z6,
            "rigid_object/pickable/root_pose": pose(PLANT_BODY),
            "rigid_object/pickable/root_velocity": z6,
            "rigid_object/target/root_pose": pose(DESTVIAL_BODY),
            "rigid_object/target/root_velocity": z6,
            "rigid_object/vial/root_pose": pose(SRCVIAL_BODY),
            "rigid_object/vial/root_velocity": z6,
        }


def write_demo(grp, ep: Episode, cameras: bool):
    T = len(ep.actions)
    grp.attrs["num_samples"] = T
    grp.create_dataset("actions", data=np.asarray(ep.actions, dtype=np.float32))
    og = grp.create_group("obs")
    for k, frames in ep.obs.items():
        og.create_dataset(k, data=np.asarray(frames, dtype=np.float32))
    if cameras:
        for k in ("table_cam", "wrist_cam"):
            og.create_dataset(k, data=np.asarray(ep.cam[k], dtype=np.uint8))
        og.create_dataset("wrist_depth", data=np.asarray(ep.cam["wrist_depth"], dtype=np.float32))
    isg = grp.create_group("initial_state")
    for path, arr in ep.initial_state.items():
        isg.create_dataset(path, data=arr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=str(REPO / "src/arm_act/tasks/pick_plant_out_of_vial_zimmer.yaml"))
    ap.add_argument("--num-demos", type=int, default=50)
    ap.add_argument("--max-attempts", type=int, default=0, help="0 = num-demos*3")
    ap.add_argument("--out", default=str(REPO / "data/raw/pick_plant_out_of_vial_zimmer/demos_newton.hdf5"))
    ap.add_argument("--cameras", action="store_true", help="render table_cam+wrist_cam via MuJoCo replay")
    ap.add_argument("--spawn-jitter", type=float, default=0.0,
                    help="per-episode pick-side xy spawn DR, +/- meters (0 = fixed spawn)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for spawn jitter")
    args = ap.parse_args()

    cfg = load_task(pathlib.Path(args.task))
    oracle = NewtonOracle(cfg, cameras=args.cameras,
                          spawn_jitter=args.spawn_jitter, seed=args.seed)
    max_attempts = args.max_attempts or args.num_demos * 3

    import h5py
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if args.cameras:
        from arm_act.newton.render import render_cameras

    f = h5py.File(out, "w")
    data = f.create_group("data")
    data.attrs["env_args"] = json.dumps({
        "env_name": cfg["task"]["gym_id"], "type": 2,
        "sim_args": {"dt": NT.DT, "control_hz": CONTROL_HZ, "num_envs": 1, "engine": "newton_mjwarp"},
    })
    kept = 0
    attempts = 0
    while kept < args.num_demos and attempts < max_attempts:
        attempts += 1
        ok, ep, dist = oracle.run_episode()
        tag = "SUCCESS" if ok else "fail   "
        print(f"attempt {attempts:3d}: {tag} dist={dist*1000:6.1f}mm  kept={kept}/{args.num_demos}", flush=True)
        if not ok:
            continue
        if args.cameras:
            render_cameras(ep)   # fills ep.cam from ep.joint_q_frames
        write_demo(data.create_group(f"demo_{kept}"), ep, args.cameras)
        kept += 1
        data.attrs["total"] = kept     # update running count so a partial file is self-describing
        f.flush()                      # persist each demo (DGX Spark power-shutdown safety)
    data.attrs["total"] = kept
    f.close()
    print(f"\n>>> wrote {kept} demos ({attempts} attempts) -> {out}", flush=True)
    return 0 if kept == args.num_demos else 1


if __name__ == "__main__":
    sys.exit(main())
