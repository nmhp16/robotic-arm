"""Batched (N-parallel-world) Newton env for state-RL on the vial->vial pick.

Newton/MuJoCo-Warp runs N independent worlds on the GPU (ModelBuilder.replicate +
SolverMuJoCo(separate_worlds=True)), so PPO can collect from hundreds of envs at
once — the only practical way to RL this. Vision is NOT here (no batched
renderer); the deployable vision path is the green-plant detector -> plant_xy ->
this state policy. Action = IK-rel TCP delta + gripper (same convention as the
oracle/BC). Obs = proprio (eef, gripper) + object state (plant_xy/z, dest) that a
detector supplies at deploy.

Runs under ~/newton-probe (newton + warp + torch). Fixed-length episodes, full
reset each rollout (simple + correct for PPO).
"""
from __future__ import annotations

import pathlib, sys, tempfile
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "newton"))
import warp as wp                       # noqa: E402
import newton                           # noqa: E402
import newton_urdf_test as NT           # noqa: E402
import mujoco_urdf_test as M            # noqa: E402

# Fingertip friction: standalone oracle held the lift at mu 2.0 with a careful gentle
# raise; RL's rougher lift slips, so give the (engineerable, tacky-pad) tip more grip.
MU_TIP, MU_PLANT, MU_OTHER = 4.0, 2.0, 1.0
CZ = 0.150                  # eef_z + j3 invariant (calibrated)
J3_LO, J3_HI = 0.0, 0.06
MAXD = np.array([0.04, 0.04, 0.02], np.float32)    # per-step TCP delta clamp
PLANT_BODY, LINK4_BODY, DESTVIAL_BODY = 0, 7, 3
FL, FR = 8, 9               # finger bodies (world-local)
DOF = dict(j1=9, j2=10, j3=11, j4=12, fl=13, fr=14)
QID = dict(j1=11, j2=12, j3=13, j4=14, fl=15, fr=16)
XY_THR, LIFT_MIN = 0.030, 0.030


def _vec_ik(x, y):
    r2 = x * x + y * y
    c2 = np.clip((r2 - M.ARM_L1**2 - M.ARM_L2**2) / (2 * M.ARM_L1 * M.ARM_L2), -1.0, 1.0)
    j2 = -np.arccos(c2)
    beta = np.arctan2(M.ARM_L2 * np.sin(j2), M.ARM_L1 + M.ARM_L2 * np.cos(j2))
    return np.arctan2(y, x) - beta, j2


class NewtonVecEnv:
    def __init__(self, num_envs=64, spawn_jitter=0.006, max_ctrl=240, seed=0, init_mode="full",
                 form_closure=False, max_delta=None):
        wp.init()
        # Form-closure (waist+ridge): the grip holds the round stem under exploration
        # noise that pure friction slips (sweep: +23pt hold @ ±20mm/step). Set BEFORE
        # building the model so M._build_xml() injects the necked stem + finger ridges.
        M.FORM_CLOSURE = bool(form_closure)
        self.N = int(num_envs)
        self.jitter = float(spawn_jitter)
        self.max_ctrl = int(max_ctrl)
        # per-step TCP delta clamp; smaller => gentler exploration => grip holds better.
        self.maxd = np.array(max_delta if max_delta is not None else MAXD, np.float32)
        self.init_mode = init_mode      # "full" (hover, must grasp) | "grasped" (curriculum: pre-gripped)
        self.auto_grip = (init_mode == "grasped")   # curriculum: env holds gripper closed (policy
        # controls only the arm; avoids the gripper-hold exploration trap of dropping mid-carry)
        self.rng = np.random.default_rng(seed)

        env = self._env_builder()
        main = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(main)
        main.replicate(env, self.N, spacing=(2.0, 2.0, 0.0))
        self.model = main.finalize()
        self.bpw = self.model.body_count // self.N
        self.cpw = len(self.model.joint_q) // self.N
        self.dpw = len(self.model.joint_qd) // self.N
        self._set_friction()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, integrator="implicitfast", cone="elliptic",
            iterations=100, ls_iterations=50, impratio=10.0, separate_worlds=True,
            nconmax=128 * self.N, njmax=160 * self.N)   # ~84/world needed; 128 generous
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()

        e = np.arange(self.N)
        self._b_plant = e * self.bpw + PLANT_BODY
        self._b_fl = e * self.bpw + FL
        self._b_fr = e * self.bpw + FR
        self._b_dest = e * self.bpw + DESTVIAL_BODY
        self._dof = {k: e * self.dpw + v for k, v in DOF.items()}
        self._q = {k: e * self.cpw + v for k, v in QID.items()}
        self._q_plant_xy = (e * self.cpw)[:, None] + np.array([0, 1])   # plant free-joint x,y coords
        self.I1, self.I2 = M.INIT_J1, M.INIT_J2
        self.act_dim = 4
        self.obs_dim = 16
        self._dest_xy = np.array([M.DEST_X, M.DEST_Y], np.float32)
        self._step_ct = 0
        self._origin = None        # per-env xy origin (replicate grid offset), set on first reset
        self._graph = None         # CUDA graph of the 100-substep control step
        self._jq0 = None           # initial joint_q (replicate: arm-home + per-world offsets)

    # ---- construction ----------------------------------------------------------
    def _env_builder(self):
        mjcf = M._build_xml()
        p = pathlib.Path(tempfile.mkstemp(suffix="_t3rl.xml")[1]); p.write_text(mjcf)
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.add_mjcf(str(p)); p.unlink()
        b.joint_q[NT.Q_J1] = M.INIT_J1; b.joint_q[NT.Q_J2] = M.INIT_J2
        b.joint_q[NT.Q_J3] = M.INIT_J3; b.joint_q[NT.Q_J4] = 0.0
        from newton import JointTargetMode
        for d, ke in NT.ARM_KE.items():
            b.joint_target_ke[d] = ke; b.joint_target_kd[d] = NT.ARM_KD[d]
            b.joint_target_mode[d] = int(JointTargetMode.POSITION)
        for d in (NT.DOF_FL, NT.DOF_FR):
            b.joint_target_ke[d] = NT.FINGER_KE; b.joint_target_kd[d] = NT.FINGER_KD
            b.joint_target_mode[d] = int(JointTargetMode.POSITION)
        return b

    def _set_friction(self):
        sb = self.model.shape_body.numpy()
        mu = self.model.shape_material_mu.numpy()
        mt = self.model.shape_material_mu_torsional.numpy()
        mr = self.model.shape_material_mu_rolling.numpy()
        for i in range(len(sb)):
            loc = sb[i] % self.bpw
            m = NT.MU_PLANT if loc == PLANT_BODY else (NT.MU_TIP if loc in (FL, FR) else NT.MU_OTHER)
            mu[i] = m; mt[i] = 0.20 * m; mr[i] = 0.125 * m
        self.model.shape_material_mu.assign(mu)
        self.model.shape_material_mu_torsional.assign(mt)
        self.model.shape_material_mu_rolling.assign(mr)

    # ---- helpers ---------------------------------------------------------------
    def _set_targets(self, j1, j2, j3, j4, grip):
        v = self.control.joint_target_pos.numpy()
        v[self._dof["j1"]] = j1; v[self._dof["j2"]] = j2
        v[self._dof["j3"]] = j3; v[self._dof["j4"]] = j4
        v[self._dof["fl"]] = grip; v[self._dof["fr"]] = grip
        self.control.joint_target_pos.assign(v)

    SUBSTEPS = 100             # one control step (100 x dt5e-4 = 20 Hz control)

    def _substeps_once(self):
        for _ in range(self.SUBSTEPS):
            self.s0.clear_forces()
            self.solver.step(self.s0, self.s1, self.control, None, NT.DT)
            self.s0, self.s1 = self.s1, self.s0

    def _step_solver(self):
        # CUDA-graph the 100-substep control step: kernel-launch overhead otherwise
        # dominates (per-substep Python launches -> ~4 env-steps/s). Targets are
        # written into control.joint_target_pos (same array the graph references)
        # before each replay, so the graph picks up new commands.
        if self._graph is None:
            try:
                with wp.ScopedCapture() as cap:
                    self._substeps_once()
                self._graph = cap.graph
            except Exception as ex:
                self._graph = False
                print(f"[rlenv] graph capture unavailable ({ex}); running eager", flush=True)
        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self._substeps_once()

    def _eef(self):
        q = self.s0.body_q.numpy()
        e = 0.5 * (q[self._b_fl][:, :3] + q[self._b_fr][:, :3])      # (N,3) world
        if self._origin is not None:
            e[:, :2] -= self._origin
        return e

    def _plant(self):
        p = self.s0.body_q.numpy()[self._b_plant][:, :3]             # (N,3) world
        if self._origin is not None:
            p[:, :2] -= self._origin
        return p

    def _obs(self):
        eef = self._eef(); plant = self._plant()
        grip = self.s0.joint_q.numpy()[self._q["fl"]][:, None]
        dest = np.tile(self._dest_xy, (self.N, 1))
        dest3 = np.concatenate([dest, np.zeros((self.N, 1), np.float32)], 1)
        o = np.concatenate([eef, grip, plant, dest3,
                            plant - eef, dest3 - plant], 1).astype(np.float32)
        # a rare env can diverge (marginal grip-close -> solver NaN); sanitize so it
        # can't poison the policy. Such envs recover on the next full reset.
        return np.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- API -------------------------------------------------------------------
    def reset(self):
        if self._jq0 is None:
            self._jq0 = self.model.joint_q.numpy().copy()   # replicate baked in arm-home + world offsets
        jq = self._jq0.copy()
        # per-env plant spawn jitter (free-joint world xy already offset per env -> ADD jitter)
        jit = self.rng.uniform(-self.jitter, self.jitter, size=(self.N, 2)).astype(np.float32)
        jq[self._q_plant_xy] += jit
        plant_xy = np.array([M.SRC_X, M.SRC_Y], np.float32) + jit   # env-local plant xy
        if self.init_mode == "grasped":
            # CURRICULUM: pre-position the arm at the EXACT grasp pose over the plant
            # (centered by construction -> a real grip), gripper closed. Bypasses the
            # precision-grasp wall so the policy learns lift->carry->deliver.
            j1, j2 = _vec_ik(plant_xy[:, 0], plant_xy[:, 1])
            j3g = CZ - 0.105                                        # grasp depth (eef_z~0.105)
            jq[self._q["j1"]] = j1; jq[self._q["j2"]] = j2
            jq[self._q["j3"]] = j3g; jq[self._q["j4"]] = -((j1 - self.I1) + (j2 - self.I2))
            jq[self._q["fl"]] = 0.004; jq[self._q["fr"]] = 0.004    # fingers closed on the 3mm stem
            init_grip = NT.GRIP_CLOSED
            init_j1, init_j2, init_j3, init_j4 = j1, j2, np.full(self.N, j3g), jq[self._q["j4"]]
        else:
            init_grip = NT.GRIP_OPEN
            init_j1 = np.full(self.N, self.I1); init_j2 = np.full(self.N, self.I2)
            init_j3 = np.full(self.N, M.INIT_J3); init_j4 = np.zeros(self.N)
        self.s0.joint_q.assign(jq)
        self.s0.joint_qd.assign(np.zeros(len(self.model.joint_qd), np.float32))
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        if self._origin is None:   # env origin xy = dest-vial(kinematic) world xy - DEST
            self._origin = self.s0.body_q.numpy()[self._b_dest][:, :2].copy() - self._dest_xy
        self._set_targets(init_j1, init_j2, init_j3, init_j4,
                          np.full(self.N, init_grip))
        for _ in range(2):
            self._step_solver()      # settle (~2 control steps; grip establishes if grasped)
        self._step_ct = 0
        p0 = self._plant()
        self._pz0 = p0[:, 2].copy()
        self._prev_dist = np.linalg.norm(p0[:, :2] - self._dest_xy, axis=1)
        self._prev_pz = p0[:, 2].copy()                         # for dense lift potential
        # one-time stage flags (camp-proof reward: each stage pays ONCE)
        z = lambda: np.zeros(self.N, bool)
        self._st = {"aligned": z(), "grasped": z(), "lifted": z(), "neardest": z()}
        return self._obs()

    def step(self, action):
        a = np.asarray(action, np.float32).reshape(self.N, 4)
        eef = self._eef()
        # TANH-squash the raw policy output into the bounded TCP-delta range.  A hard
        # np.clip made the action ~saturated for any Gaussian std >> maxd, so the policy
        # mean had no influence and exploration noise did the carrying (sampled succ
        # ~40% but DETERMINISTIC 0%).  tanh maps the mean smoothly into [-maxd,maxd], so
        # the deterministic mean controls the action AND std~0.6 explores gracefully.
        d = self.maxd * np.tanh(a[:, :3])
        tgt = eef + d
        j1, j2 = _vec_ik(tgt[:, 0], tgt[:, 1])
        j3 = np.clip(CZ - tgt[:, 2], J3_LO, J3_HI)
        j4 = -((j1 - self.I1) + (j2 - self.I2))
        if self.auto_grip:
            grip = np.full(self.N, NT.GRIP_CLOSED, np.float32)   # env holds the grip (carry curriculum)
        else:
            grip = np.where(a[:, 3] < 0, NT.GRIP_CLOSED, NT.GRIP_OPEN).astype(np.float32)
        self._set_targets(j1, j2, j3, j4, grip)
        self._step_solver()
        self._step_ct += 1

        plant = self._plant(); eef2 = self._eef()
        gripc = self.s0.joint_q.numpy()[self._q["fl"]] > 0.0015      # gripper closed (on stem)
        dist_dest = np.linalg.norm(plant[:, :2] - self._dest_xy, axis=1)
        lifted = plant[:, 2] > LIFT_MIN
        # NOTE: 3D eef->plant_origin distance bottoms out ~0.07 at grasp (fingers grip
        # the stem ~35mm above the plant origin), so the grasp criterion is XY-alignment
        # + descended height + gripper-closed, NOT 3D reach (v2 bug: near<0.02 unreachable).
        xy_align = np.linalg.norm(eef2[:, :2] - plant[:, :2], axis=1)
        aligned = xy_align < 0.012                      # approach milestone (loose)
        centered = xy_align < 0.005                     # precise: 3mm stem between the pads
        at_grasp = centered & (eef2[:, 2] < 0.115)
        grasped = at_grasp & gripc                      # REAL grip (precise center + low + closed)
        near_dest = (dist_dest < 0.06) & lifted
        success = (dist_dest < XY_THR) & lifted
        st = self._st
        # v5 reward: CAMP-PROOF. Each stage pays ONCE; only ongoing signals are a small
        # dense alignment shaping (bounded, pre-grasp) + a time/action penalty. Camping
        # anywhere earns nothing, so the policy must keep advancing stages to deliver.
        # v8: SMALL, carry-DOMINANT rewards.  Diagnosis (diagnose_carry.py): with the
        # big bonuses (success+50, potentials 20/30) returns hit ~150 and v_loss
        # exploded ~10^4 -> advantages were noise -> PPO never learned the carry, the
        # deterministic policy just lifts-and-holds (75% stall ~14cm from dest, 0% det
        # success).  Shrink everything so returns are O(10) and the dense carry
        # potential is the LARGEST term -> the value fn fits, advantages are clean, and
        # reducing distance-to-dest is the dominant learnable signal.
        rew = (-0.3 * np.minimum(xy_align, 0.2)                  # mild pre-grasp align shaping
               + 0.5 * (aligned & ~st["aligned"])               # stage: aligned over plant
               + 1.0 * (grasped & ~st["grasped"])               # stage: grasped
               + 1.0 * (lifted & ~st["lifted"])                 # stage: lifted out
               + 2.0 * (near_dest & ~st["neardest"])            # stage: carried to dest
               + 10.0 * success                                 # delivered
               - 0.01                                           # time penalty (finish fast)
               - 0.01 * (a[:, :3] ** 2).sum(1)                  # action penalty (no thrashing)
               ).astype(np.float32)
        # v9 (option 2): carry pays ONLY while the plant is clearly LIFTED (z>0.035), and
        # a SMOOTH height-maintenance penalty makes dropping actively costly.  This fixes
        # v8's carry-rush-and-DROP (det-eval: 72% dropped, gripper closed but stem sheared
        # out over the lateral move): the policy now maximizes reward by carrying WHILE
        # keeping the plant seated, not by lunging at the dest.  Interlock geometry is
        # unchanged (r2/1mm waist+ridge — best shear hold; deeper r1.5 was WORSE).
        # Returns stay O(10) so the value fn keeps fitting (v_loss bounded).
        held_high = gripc & (plant[:, 2] > 0.035)
        rew += 3.0 * (plant[:, 2] - self._prev_pz) * gripc           # height potential (symmetric)
        rew += 20.0 * (self._prev_dist - dist_dest) * held_high      # carry pays only when lifted HIGH
        rew -= 2.0 * np.clip(0.035 - plant[:, 2], 0.0, 0.035) * st["lifted"]  # smooth drop penalty
        self._prev_pz = plant[:, 2]; self._prev_dist = dist_dest
        st["aligned"] |= aligned; st["grasped"] |= grasped
        st["lifted"] |= lifted; st["neardest"] |= near_dest
        rew = np.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        done = np.full(self.N, self._step_ct >= self.max_ctrl)
        info = {"success": success, "dist_dest": dist_dest, "lifted": lifted,
                "grasped": grasped, "near_dest": near_dest}     # stage flags for monitoring
        return self._obs(), rew, done, info


if __name__ == "__main__":
    import time
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    env = NewtonVecEnv(num_envs=N, spawn_jitter=0.006, max_ctrl=40)
    print(f"[rlenv] N={env.N} bpw={env.bpw} cpw={env.cpw} dpw={env.dpw} obs_dim={env.obs_dim}", flush=True)
    obs = env.reset()
    print(f"[rlenv] reset obs {obs.shape}; eef[0]={obs[0,:3].round(3)} plant_xy spread x={env._plant()[:,0].std()*1000:.1f}mm", flush=True)
    t0 = time.time(); steps = 30
    for k in range(steps):
        a = np.zeros((env.N, 4), np.float32); a[:, 3] = 1.0   # hold open
        # script a crude descend to sanity-check per-env motion
        a[:, 2] = -0.02
        obs, rew, done, info = env.step(a)
    dt = time.time() - t0
    print(f"[rlenv] {steps} ctrl-steps x {env.N} envs in {dt:.1f}s = {steps*env.N/dt:.0f} env-steps/s", flush=True)
    print(f"[rlenv] after descend: eef_z mean={env._eef()[:,2].mean():.3f} (per-env indep: eef_z std across envs={env._eef()[:,2].std()*1000:.2f}mm)", flush=True)
    print("[rlenv] OK", flush=True)
