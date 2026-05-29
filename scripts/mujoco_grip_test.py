#!/usr/bin/env python3
"""
Minimal MuJoCo grip test: Zimmer V-groove fingers vs. plant stem.

Tests whether force-controlled V-groove fingers pick and hold the plant
collision proxy (6 mm grasp cylinder + 8 mm jam nodes) without Isaac Lab.
Runs on macOS/Windows/Linux with only the `mujoco` pip package.

Usage:
    pip install mujoco
    python scripts/mujoco_grip_test.py             # headless, prints result
    python scripts/mujoco_grip_test.py --render    # opens interactive viewer
    python scripts/mujoco_grip_test.py --sweep     # sweep grip force 50..200 N

Actuator model:
    Position-servo with kp=25000 N/m, kv=10 N·s/m.
    ctrl=0.0   → open  (fingers fully retracted)
    ctrl=0.010 → close (try to reach 10 mm travel; stem stops at ~2 mm)
    Stall force at stem contact = kp × (0.010 − 0.002) = kp × 0.008.
    Default kp → 200 N.  --force N overrides kp = N / 0.008.

Geometry note:
    Plant body spawns at world z=0 but its lowest geom is at local z=+0.019 m,
    so the body settles to z≈−0.019 m on the floor.  H_GRASP is computed
    dynamically after the settle phase so the V-prongs land at stem centre
    regardless of where the body comes to rest.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT      = pathlib.Path(__file__).resolve().parent.parent
SCENE_XML      = REPO_ROOT / "assets" / "mujoco_grip_test" / "scene.xml"
SCENE_VIAL_XML = REPO_ROOT / "assets" / "mujoco_grip_test" / "scene_vial.xml"
MEDIA_DIR      = REPO_ROOT / "media"

# ── Recording ─────────────────────────────────────────────────────────────
RECORD_WIDTH  = 640
RECORD_HEIGHT = 480
RECORD_FPS    = 60
# Render one frame every N simulation steps (dt=0.2ms → 60fps = every 83 steps)
RECORD_EVERY  = max(1, int(round(1.0 / (RECORD_FPS * 0.0002))))

# ── Phase durations (steps at dt=0.0002 s) ────────────────────────────────
SETTLE_STEPS  = 2500   # 0.5 s — let plant fall onto floor
DESCEND_STEPS = 4000   # 0.8 s — descend from H_START to H_GRASP
CLOSE_STEPS   = 3000   # 0.6 s — drive fingers to close target
HOLD_STEPS    = 2000   # 0.4 s — settle with grip engaged
LIFT_STEPS    = 4000   # 0.8 s — lift H_GRASP → H_LIFT
CHECK_STEPS   = 2000   # 0.4 s — hold at lift height, measure

# ── Fixed geometry offsets (world frame, Z-up) ────────────────────────────
PLANT_X, PLANT_Y  = 0.31, -0.01
H_START           = 0.370  # initial gripper height (clear of everything)

# H_GRASP uses the same formula as the oracle state machine:
#   TCP_target = plant_body_z + GRASP_Z_OFFSET
#   H_GRASP    = TCP_target + TCP_Z_OFFSET   (gripper_base is TCP_Z_OFFSET above TCP)
# With plant_body_z ≈ −0.019 (settled on floor), H_GRASP ≈ 0.171 m.
# This puts the V-prong bottom at z ≈ +0.001 m — just clears the floor.
GRASP_Z_OFFSET    = 0.030   # oracle.grasp_z_offset from pick_plant_out_of_vial_zimmer.yaml
TCP_Z_OFFSET      = 0.160   # robot.tcp_z_offset (0.16 m below gripper_base = TCP)

# Lift height matching oracle lift_height:
LIFT_HEIGHT       = 0.095
# Plant must rise this much above its settled position to count as lifted:
LIFT_ABOVE_SETTLED = 0.050
# Stall force = kp × STEM_STROKE at stem contact (each jaw travels ~2 mm):
STEM_STROKE       = 0.008
DEFAULT_FORCE_N   = 200.0
KP_MAX            = 30000.0   # N/m — safe at dt=0.2 ms, m=8 g

# ── Vial-to-vial geometry (matches pick_plant_out_of_vial_zimmer.yaml) ────
DEST_X, DEST_Y    = 0.31, 0.14   # destination vial centre
PLACE_Z_OFFSET    = 0.040        # oracle place_z_offset
VIAL_HEIGHT       = 0.045        # vial rim z
DEST_XY_TOL       = 0.030        # success: plant within 30 mm of dest centre

# ── Vial-to-vial phase durations (steps at dt=0.0002 s) ──────────────────
TRANSPORT_STEPS   = 5000   # 1.0 s lateral move at lift height
INSERT_STEPS      = 3000   # 0.6 s descend into dest vial
RELEASE_STEPS     = 2000   # 0.4 s open fingers + rise clear


# ── Core test ─────────────────────────────────────────────────────────────

def run_test(
    close_force_n: float = DEFAULT_FORCE_N,
    render: bool = False,
    record: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run one descend-close-lift cycle.

    Returns:
        success          bool
        plant_z_settled  float  m — body z after settling on floor
        plant_z_at_lift  float  m — body z after lift-and-hold phase
        fl_closure_mm    float  mm — left  finger joint position at end
        fr_closure_mm    float  mm — right finger joint position at end
        jaw_gap_mm       float  mm — gap between inner faces at stem
    """
    try:
        import mujoco
    except ImportError:
        sys.exit("mujoco not installed — run: pip install mujoco")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)

    # kp = force / STEM_STROKE, capped at KP_MAX for numerical stability.
    # At dt=1 ms, m=8 g: kp_max ≈ m/dt² = 8000 N/m → max stall ≈ 64 N.
    # The test verifies V-groove GEOMETRY (holds 2 g plant), not exact force.
    kp = min(close_force_n / STEM_STROKE, KP_MAX)
    fl_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_left_act")
    fr_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_right_act")
    model.actuator_gainprm[fl_act_id, 0] = kp
    model.actuator_gainprm[fr_act_id, 0] = kp
    # kv (velocity gain) is in biasprm[2] for position actuators in MuJoCo
    # Leave kv at the XML default (100 N·s/m) regardless of force level.

    mujoco.mj_resetData(model, data)

    # ── ID lookups ────────────────────────────────────────────────────────
    tcp_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "tcp")
    plant_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "plant")
    fl_jnt    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    "finger_left_joint")
    fr_jnt    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    "finger_right_joint")
    gb_jnt    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    "gripper_freejoint")
    mocap_idx = model.body_mocapid[tcp_id]

    def set_tcp(z: float) -> None:
        data.mocap_pos[mocap_idx]  = [PLANT_X, PLANT_Y, z]
        data.mocap_quat[mocap_idx] = [1.0, 0.0, 0.0, 0.0]

    # Initialise gripper freejoint to match mocap so the weld starts settled
    qi = model.jnt_qposadr[gb_jnt]
    data.qpos[qi : qi + 3]    = [PLANT_X, PLANT_Y, H_START]
    data.qpos[qi + 3 : qi + 7] = [1.0, 0.0, 0.0, 0.0]
    set_tcp(H_START)
    mujoco.mj_forward(model, data)

    # ── Optional viewer ───────────────────────────────────────────────────
    viewer = None
    if render:
        try:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(model, data)
        except Exception as exc:
            print(f"[warn] viewer unavailable: {exc}", file=sys.stderr)

    # ── Optional offscreen recorder ───────────────────────────────────────
    frames: list | None = None
    renderer = None
    cam = None
    if record:
        renderer = mujoco.Renderer(model, height=RECORD_HEIGHT, width=RECORD_WIDTH)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [PLANT_X, PLANT_Y, 0.12]  # centre of full settle→lift range
        cam.distance  = 0.45
        cam.azimuth   = 135.0   # front-left diagonal shows V-groove closing
        cam.elevation = -15.0
        frames = []

    def sync():
        if viewer is not None:
            viewer.sync()

    def _capture() -> None:
        if frames is not None:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    # ctrl semantics: 0.0 = open, 0.010 = fully closed target
    def step_n(n: int, tcp_z=None, finger_ctrl: float = 0.0) -> None:
        for i in range(n):
            if tcp_z is not None:
                set_tcp(tcp_z(i, n) if callable(tcp_z) else tcp_z)
            data.ctrl[fl_act_id] = finger_ctrl
            data.ctrl[fr_act_id] = finger_ctrl
            mujoco.mj_step(model, data)
            if i % RECORD_EVERY == 0:
                _capture()
            sync()

    def plant_z() -> float:
        return float(data.xpos[plant_id, 2])

    def finger_pos() -> tuple[float, float]:
        fl = float(data.qpos[model.jnt_qposadr[fl_jnt]])
        fr = float(data.qpos[model.jnt_qposadr[fr_jnt]])
        return fl, fr

    # ── Phase 0: settle plant on floor (fingers open, gripper parked high) ─
    step_n(SETTLE_STEPS, tcp_z=H_START, finger_ctrl=0.0)

    plant_z_settled = plant_z()
    # Oracle formula: TCP_target = plant_body_z + grasp_z_offset; H_GRASP = TCP_target + tcp_z_offset
    h_grasp = plant_z_settled + GRASP_Z_OFFSET + TCP_Z_OFFSET
    h_lift  = h_grasp + LIFT_HEIGHT
    success_z     = plant_z_settled + LIFT_ABOVE_SETTLED

    if verbose:
        print(f"  plant settled at z={plant_z_settled*1000:.1f} mm  "
              f"H_GRASP={h_grasp*1000:.1f} mm")

    # ── Phase 1: descend ──────────────────────────────────────────────────
    step_n(DESCEND_STEPS,
           tcp_z=lambda i, n: H_START + (i / n) * (h_grasp - H_START),
           finger_ctrl=0.0)

    if verbose:
        fl, fr = finger_pos()
        print(f"  [after descent]  plant_z={plant_z()*1000:.1f} mm  "
              f"fl={fl*1000:.2f} mm  fr={fr*1000:.2f} mm")

    # ── Phase 2: close fingers ────────────────────────────────────────────
    step_n(CLOSE_STEPS, tcp_z=h_grasp, finger_ctrl=0.010)

    if verbose:
        fl, fr = finger_pos()
        gap = (0.006 - fl - fr) * 1000   # 6 mm = stem diameter
        print(f"  [after close]    plant_z={plant_z()*1000:.1f} mm  "
              f"fl={fl*1000:.2f} mm  fr={fr*1000:.2f} mm  gap={gap:.2f} mm")

    # ── Phase 3: hold (let contact forces settle) ─────────────────────────
    step_n(HOLD_STEPS, tcp_z=h_grasp, finger_ctrl=0.010)

    # ── Phase 4: lift ─────────────────────────────────────────────────────
    step_n(LIFT_STEPS,
           tcp_z=lambda i, n: h_grasp + (i / n) * (h_lift - h_grasp),
           finger_ctrl=0.010)

    # ── Phase 5: hold at lift height, measure ─────────────────────────────
    step_n(CHECK_STEPS, tcp_z=h_lift, finger_ctrl=0.010)

    if viewer is not None:
        viewer.close()

    # ── Write video ───────────────────────────────────────────────────────
    if record and frames:
        try:
            import imageio.v2 as iio
        except ImportError:
            sys.exit("imageio not installed — run: pip install 'imageio[ffmpeg]'")
        out = pathlib.Path(record)
        out.parent.mkdir(parents=True, exist_ok=True)
        iio.mimwrite(str(out), frames, fps=RECORD_FPS, quality=8, macro_block_size=None)
        if verbose:
            print(f"  video saved → {out}  ({len(frames)} frames @ {RECORD_FPS} fps)")
    if renderer is not None:
        renderer.close()

    fl, fr     = finger_pos()
    pz         = plant_z()
    jaw_gap_mm = (0.006 - fl - fr) * 1000   # 6 mm = stem diameter

    return {
        "success":         pz > success_z,
        "plant_z_settled": plant_z_settled,
        "plant_z_at_lift": pz,
        "fl_closure_mm":   fl * 1000,
        "fr_closure_mm":   fr * 1000,
        "jaw_gap_mm":      jaw_gap_mm,
    }


# ── Vial-to-vial test ─────────────────────────────────────────────────────

def run_vial_test(
    close_force_n: float = DEFAULT_FORCE_N,
    render: bool = False,
    record: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Full vial-to-vial pick-and-place mirroring pick_plant_out_of_vial_zimmer.

    Phases: settle → descend → close → hold → lift → transport → insert →
            release → check.

    Returns:
        success          bool   — plant within DEST_XY_TOL of dest vial
        plant_z_settled  float  m
        plant_z_at_lift  float  m — after extraction from source vial
        plant_z_final    float  m — after release and settle in dest vial
        dist_to_dest_mm  float  mm — XY distance from dest vial centre
        jaw_gap_mm       float  mm — grip gap at release
    """
    try:
        import mujoco
    except ImportError:
        sys.exit("mujoco not installed — run: pip install mujoco")

    model = mujoco.MjModel.from_xml_path(str(SCENE_VIAL_XML))
    data  = mujoco.MjData(model)

    kp = min(close_force_n / STEM_STROKE, KP_MAX)
    fl_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_left_act")
    fr_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_right_act")
    model.actuator_gainprm[fl_act_id, 0] = kp
    model.actuator_gainprm[fr_act_id, 0] = kp

    mujoco.mj_resetData(model, data)

    tcp_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "tcp")
    plant_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "plant")
    fl_jnt    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_left_joint")
    fr_jnt    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_right_joint")
    gb_jnt    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_freejoint")
    mocap_idx = model.body_mocapid[tcp_id]

    qi = model.jnt_qposadr[gb_jnt]
    data.qpos[qi : qi + 3]    = [PLANT_X, PLANT_Y, H_START]
    data.qpos[qi + 3 : qi + 7] = [1.0, 0.0, 0.0, 0.0]
    data.mocap_pos[mocap_idx]  = [PLANT_X, PLANT_Y, H_START]
    data.mocap_quat[mocap_idx] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    viewer = None
    if render:
        try:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(model, data)
        except Exception as exc:
            print(f"[warn] viewer unavailable: {exc}", file=sys.stderr)

    frames: list | None = None
    renderer = None
    cam = None
    if record:
        renderer = mujoco.Renderer(model, height=RECORD_HEIGHT, width=RECORD_WIDTH)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [PLANT_X, (PLANT_Y + DEST_Y) / 2, 0.10]
        cam.distance  = 0.75
        cam.azimuth   = 50.0
        cam.elevation = -15.0
        frames = []

    def sync():
        if viewer is not None:
            viewer.sync()

    def _capture():
        if frames is not None:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    def step_n(n: int, tcp_xyz=None, finger_ctrl: float = 0.0) -> None:
        for i in range(n):
            if tcp_xyz is not None:
                xyz = tcp_xyz(i, n) if callable(tcp_xyz) else tcp_xyz
                data.mocap_pos[mocap_idx]  = [xyz[0], xyz[1], xyz[2]]
                data.mocap_quat[mocap_idx] = [1.0, 0.0, 0.0, 0.0]
            data.ctrl[fl_act_id] = finger_ctrl
            data.ctrl[fr_act_id] = finger_ctrl
            mujoco.mj_step(model, data)
            if i % RECORD_EVERY == 0:
                _capture()
            sync()

    def plant_z() -> float:
        return float(data.xpos[plant_id, 2])

    def plant_xy() -> tuple[float, float]:
        return float(data.xpos[plant_id, 0]), float(data.xpos[plant_id, 1])

    def finger_pos() -> tuple[float, float]:
        fl = float(data.qpos[model.jnt_qposadr[fl_jnt]])
        fr = float(data.qpos[model.jnt_qposadr[fr_jnt]])
        return fl, fr

    # ── Phase 0: settle plant inside source vial ──────────────────────────
    step_n(SETTLE_STEPS, tcp_xyz=[PLANT_X, PLANT_Y, H_START], finger_ctrl=0.0)

    plant_z_settled = plant_z()
    h_grasp = plant_z_settled + GRASP_Z_OFFSET + TCP_Z_OFFSET
    h_lift  = h_grasp + LIFT_HEIGHT
    h_place = plant_z_settled + PLACE_Z_OFFSET + TCP_Z_OFFSET

    if verbose:
        print(f"  plant settled at z={plant_z_settled*1000:.1f} mm  "
              f"H_GRASP={h_grasp*1000:.1f} mm  H_LIFT={h_lift*1000:.1f} mm")

    # ── Phase 1: descend into source vial ────────────────────────────────
    step_n(DESCEND_STEPS,
           tcp_xyz=lambda i, n: [PLANT_X, PLANT_Y,
                                  H_START + (i / n) * (h_grasp - H_START)],
           finger_ctrl=0.0)

    # ── Phase 2: close fingers ───────────────────────────────────────────
    step_n(CLOSE_STEPS, tcp_xyz=[PLANT_X, PLANT_Y, h_grasp], finger_ctrl=0.010)

    if verbose:
        fl, fr = finger_pos()
        print(f"  [after close]    fl={fl*1000:.2f} mm  fr={fr*1000:.2f} mm  "
              f"plant_z={plant_z()*1000:.1f} mm")

    # ── Phase 3: hold ────────────────────────────────────────────────────
    step_n(HOLD_STEPS, tcp_xyz=[PLANT_X, PLANT_Y, h_grasp], finger_ctrl=0.010)

    # ── Phase 4: lift — extract from source vial ─────────────────────────
    step_n(LIFT_STEPS,
           tcp_xyz=lambda i, n: [PLANT_X, PLANT_Y,
                                  h_grasp + (i / n) * (h_lift - h_grasp)],
           finger_ctrl=0.010)

    plant_z_at_lift = plant_z()
    if verbose:
        cleared = "YES ✓" if plant_z_at_lift > VIAL_HEIGHT else "NO ✗"
        print(f"  [after lift]     plant_z={plant_z_at_lift*1000:.1f} mm  "
              f"cleared {VIAL_HEIGHT*1000:.0f} mm rim: {cleared}")

    # ── Phase 5: transport laterally to dest vial ─────────────────────────
    step_n(TRANSPORT_STEPS,
           tcp_xyz=lambda i, n: [PLANT_X + (i / n) * (DEST_X - PLANT_X),
                                  PLANT_Y + (i / n) * (DEST_Y - PLANT_Y),
                                  h_lift],
           finger_ctrl=0.010)

    if verbose:
        px, py = plant_xy()
        print(f"  [after transport] plant_xy=({px*1000:.1f}, {py*1000:.1f}) mm  "
              f"(dest: {DEST_X*1000:.0f}, {DEST_Y*1000:.0f})")

    # ── Phase 6: insert into dest vial ───────────────────────────────────
    step_n(INSERT_STEPS,
           tcp_xyz=lambda i, n: [DEST_X, DEST_Y,
                                  h_lift + (i / n) * (h_place - h_lift)],
           finger_ctrl=0.010)

    # ── Phase 7: release — open fingers and rise clear ───────────────────
    step_n(RELEASE_STEPS,
           tcp_xyz=lambda i, n: [DEST_X, DEST_Y,
                                  h_place + (i / n) * 0.10],
           finger_ctrl=0.0)

    # ── Phase 8: settle check ────────────────────────────────────────────
    step_n(CHECK_STEPS, tcp_xyz=[DEST_X, DEST_Y, h_place + 0.10],
           finger_ctrl=0.0)

    if viewer is not None:
        viewer.close()

    if record and frames:
        try:
            import imageio.v2 as iio
        except ImportError:
            sys.exit("imageio not installed — run: pip install 'imageio[ffmpeg]'")
        out = pathlib.Path(record)
        out.parent.mkdir(parents=True, exist_ok=True)
        iio.mimwrite(str(out), frames, fps=RECORD_FPS, quality=8, macro_block_size=None)
        if verbose:
            print(f"  video saved → {out}  ({len(frames)} frames @ {RECORD_FPS} fps)")
    if renderer is not None:
        renderer.close()

    fl, fr = finger_pos()
    pz     = plant_z()
    px, py = plant_xy()
    jaw_gap_mm      = (0.006 - fl - fr) * 1000
    dist_to_dest_mm = ((px - DEST_X)**2 + (py - DEST_Y)**2)**0.5 * 1000
    success         = dist_to_dest_mm < DEST_XY_TOL * 1000 and pz > -0.030

    return {
        "success":         success,
        "plant_z_settled": plant_z_settled,
        "plant_z_at_lift": plant_z_at_lift,
        "plant_z_final":   pz,
        "dist_to_dest_mm": dist_to_dest_mm,
        "jaw_gap_mm":      jaw_gap_mm,
    }


# ── Force sweep ───────────────────────────────────────────────────────────

def run_sweep() -> None:
    forces = [50.0, 100.0, 150.0, 200.0]
    print("\n── Grip force sweep ──────────────────────────────────────────")
    print(f"  {'Force':>8}  {'plant Δz':>9}  {'gap mm':>7}  result")
    print(f"  {'(N)':>8}  {'(mm)':>9}  {'':>7}")
    for f in forces:
        r = run_test(close_force_n=f, render=False, verbose=False)
        dz  = (r["plant_z_at_lift"] - r["plant_z_settled"]) * 1000
        tag = "SUCCESS ✓" if r["success"] else "FAIL ✗"
        print(f"  {f:>8.0f}  {dz:>9.1f}  {r['jaw_gap_mm']:>7.2f}  {tag}")
    print("──────────────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", action="store_true",
                    help="Open interactive MuJoCo viewer")
    ap.add_argument("--record", metavar="PATH", nargs="?", const="",
                    help="Record video (default path auto-selected based on --vial flag)")
    ap.add_argument("--force",  type=float, default=DEFAULT_FORCE_N,
                    help=f"Grip force per finger in N (default {DEFAULT_FORCE_N:.0f})")
    ap.add_argument("--sweep",  action="store_true",
                    help="Sweep grip force 50..200 N and print table")
    ap.add_argument("--vial",   action="store_true",
                    help="Run full vial-to-vial pick-and-place test")
    args = ap.parse_args()

    # Resolve default record path
    record = args.record
    if record == "":
        record = str(MEDIA_DIR / ("mujoco_vial_test.mp4" if args.vial
                                  else "mujoco_grip_test.mp4"))

    if args.sweep:
        run_sweep()
        return

    if args.vial:
        print(f"\nRunning vial-to-vial test  force={args.force:.0f} N  render={args.render}"
              + (f"  record={record}" if record else ""))
        r = run_vial_test(close_force_n=args.force, render=args.render,
                          record=record, verbose=True)

        ok = "SUCCESS ✓" if r["success"] else "FAIL ✗"
        print(f"\n── Result ────────────────────────────────────────────────────")
        print(f"  {ok}")
        print(f"  plant z settled    : {r['plant_z_settled']*1000:>6.1f} mm")
        print(f"  plant z at lift    : {r['plant_z_at_lift']*1000:>6.1f} mm  "
              f"(vial rim {VIAL_HEIGHT*1000:.0f} mm)")
        print(f"  plant z final      : {r['plant_z_final']*1000:>6.1f} mm")
        print(f"  dist to dest       : {r['dist_to_dest_mm']:>6.1f} mm  "
              f"(threshold {DEST_XY_TOL*1000:.0f} mm)")
        print(f"  jaw gap at release : {r['jaw_gap_mm']:>6.2f} mm")
        print(f"──────────────────────────────────────────────────────────────\n")
        sys.exit(0 if r["success"] else 1)

    print(f"\nRunning grip test  force={args.force:.0f} N  render={args.render}"
          + (f"  record={record}" if record else ""))
    r = run_test(close_force_n=args.force, render=args.render,
                 record=record, verbose=True)

    dz  = (r["plant_z_at_lift"] - r["plant_z_settled"]) * 1000
    ok  = "SUCCESS ✓" if r["success"] else "FAIL ✗"
    print(f"\n── Result ────────────────────────────────────────────────────")
    print(f"  {ok}")
    print(f"  plant z settled    : {r['plant_z_settled']*1000:>6.1f} mm")
    print(f"  plant z after lift : {r['plant_z_at_lift']*1000:>6.1f} mm  (Δ {dz:.1f} mm)")
    print(f"  lift threshold     : settled + {LIFT_ABOVE_SETTLED*1000:.0f} mm")
    print(f"  left  finger close : {r['fl_closure_mm']:>6.2f} mm")
    print(f"  right finger close : {r['fr_closure_mm']:>6.2f} mm")
    print(f"  jaw gap at stem    : {r['jaw_gap_mm']:>6.2f} mm  (stem dia 6 mm)")
    print(f"──────────────────────────────────────────────────────────────\n")

    sys.exit(0 if r["success"] else 1)


if __name__ == "__main__":
    main()
