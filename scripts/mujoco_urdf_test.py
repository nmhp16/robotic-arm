#!/usr/bin/env python3
"""
Full T3-401+Zimmer URDF vial-to-vial test in MuJoCo.

Loads the real t3_401_zimmer.urdf, injects plant + vials + actuators via
MuJoCo's <mujoco> URDF extension, then runs the oracle phases from
pick_plant_out_of_vial_zimmer.yaml:

  settle → descend → close → hold → lift → transport → insert → release → check

Arm driven by SCARA position actuators (gains from robot_cfg.py).
Gripper driven by position actuator with kp=25000 (same as scene.xml).

Usage:
    python scripts/mujoco_urdf_test.py             # headless
    python scripts/mujoco_urdf_test.py --record    # save media/mujoco_urdf_test.mp4
    python scripts/mujoco_urdf_test.py --render    # interactive viewer
"""
from __future__ import annotations

import argparse, math, pathlib, re, sys
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "assets" / "t3_401_zimmer" / "t3_401_zimmer.urdf"
MESH_DIR  = REPO_ROOT / "assets" / "t3_401_zimmer" / "meshes"
VIAL_STL  = REPO_ROOT / "assets" / "wide_vial" / "meshes" / "wide_vial.stl"
MEDIA_DIR = REPO_ROOT / "media"

# ── SCARA geometry (verified by FK sweep) ─────────────────────────────────
ARM_L1, ARM_L2  = 0.225, 0.175   # URDF joint origins
LINK4_Z_J3_0    = 0.270           # link4 world-z when j3=0 (fully retracted)
# prong_centre_z = LINK4_Z_J3_0 - j3 - 0.190
# 0.190 = 0.040 (link4→tool0 offset) + 0.080 (gripper body) + 0.070 (prong offset)
PRONG_BELOW_L4  = 0.190

# ── Scene constants matching YAML ─────────────────────────────────────────
SRC_X, SRC_Y   = 0.31, -0.01
DEST_X, DEST_Y = 0.31,  0.14
VIAL_HEIGHT    = 0.045
DEST_XY_TOL    = 0.030

# ── Recording ─────────────────────────────────────────────────────────────
DT           = 0.0002
RECORD_FPS   = 60
RECORD_EVERY = max(1, int(round(1.0 / (RECORD_FPS * DT))))
RECORD_W, RECORD_H = 640, 480

# ── Phase durations (steps at dt=0.0002 s) ────────────────────────────────
SETTLE_STEPS    = 2500   # 0.5 s
DESCEND_STEPS   = 5000   # 1.0 s
CLOSE_STEPS     = 3000   # 0.6 s
HOLD_STEPS      = 2000   # 0.4 s
LIFT_STEPS      = 3000   # 0.6 s
TRANSPORT_STEPS = 6000   # 1.2 s
INSERT_STEPS    = 4000   # 0.8 s
RELEASE_STEPS   = 2000   # 0.4 s
CHECK_STEPS     = 3000   # 0.6 s

# ── Initial joint positions (from YAML init_joint_pos) ────────────────────
INIT_J1, INIT_J2, INIT_J3 = 0.554, -1.381, 0.02
J3_LIFT = 0.002   # near j3-min (arm at max height) for clearance over vial rim


# ── SCARA IK ──────────────────────────────────────────────────────────────

def _ik(x: float, y: float) -> tuple[float, float]:
    """Elbow-down SCARA IK for the T3-401S. Returns (j1, j2) in radians."""
    r2 = x * x + y * y
    c2 = (r2 - ARM_L1**2 - ARM_L2**2) / (2.0 * ARM_L1 * ARM_L2)
    j2 = -math.acos(max(-1.0, min(1.0, c2)))
    beta = math.atan2(ARM_L2 * math.sin(j2), ARM_L1 + ARM_L2 * math.cos(j2))
    j1 = math.atan2(y, x) - beta
    return j1, j2


DEST_J1, DEST_J2 = _ik(DEST_X, DEST_Y)   # ≈ (0.905, -1.118)


# ── XML builder ───────────────────────────────────────────────────────────

def _vial_walls(indent: str = "        ") -> str:
    """8 thin box wall segments forming a 27 mm-ID cylinder, 45 mm tall."""
    r_i, t   = 0.0135, 0.0015
    r_c      = r_i + t / 2
    half_arc = r_i * math.sin(math.pi / 8)
    lines = []
    for i in range(8):
        a = i * math.pi / 4
        lines.append(
            f'{indent}<geom type="box"'
            f' size="{t/2:.5f} {half_arc:.5f} 0.0225"'
            f' pos="{r_c*math.cos(a):.5f} {r_c*math.sin(a):.5f} 0.0225"'
            f' euler="0 0 {a:.4f}" rgba="0.5 0.7 0.9 0.06"/>'
        )
    return "\n".join(lines)


def _build_xml() -> str:
    """Load URDF → save as MJCF → inject scene elements into the MJCF."""
    import mujoco as mj, os, tempfile

    # Step 1: load URDF with only balanceinertia (the only supported compiler tag)
    urdf = URDF_PATH.read_text()
    compiler_only = (
        f'\n  <mujoco><compiler balanceinertia="true" meshdir="{MESH_DIR}"/>'
        f'</mujoco>\n'
    )
    urdf2 = re.sub(r'(<robot[^>]*>)', lambda m: m.group(1) + compiler_only, urdf, count=1)
    model_tmp = mj.MjModel.from_xml_string(urdf2)

    # Step 2: save as MJCF — gives a clean MJCF with absolute meshdir
    fd, tmp_path = tempfile.mkstemp(suffix='.xml')
    os.close(fd)
    mj.mj_saveLastXML(tmp_path, model_tmp)
    mjcf = pathlib.Path(tmp_path).read_text()
    os.unlink(tmp_path)

    walls = _vial_walls()

    # Step 3: inject option/size/asset/default after <compiler .../>
    # Visual mesh assets included here; MuJoCo's URDF loader ignores <visual> mesh
    # elements and only loads <collision> geometry, so we inject them manually.
    # gripper_base maps to link_4: all intermediate fixed joints have zero origin.
    _VIS_MESHES = [
        ('vis_base',    'base_link.stl'),
        ('vis_link1',   'link_1.stl'),
        ('vis_link2',   'link_2.stl'),
        ('vis_link3',   'link_3.stl'),
        ('vis_gripper', 'gripper_base.stl'),
    ]
    vis_mesh_assets = ''.join(
        f'\n    <mesh name="{mn}" file="{MESH_DIR}/{sf}" scale="0.001 0.001 0.001"/>'
        for mn, sf in _VIS_MESHES
    )
    inject_after_compiler = f"""
  <option gravity="0 0 -9.81" timestep="{DT}" integrator="implicitfast"
          iterations="100" noslip_iterations="10" tolerance="1e-10">
    <flag contact="enable"/>
  </option>
  <size njmax="500" nconmax="500"/>
  <asset>
    <mesh name="wide_vial" file="{VIAL_STL}" scale="0.001 0.001 0.000643"/>{vis_mesh_assets}
  </asset>
  <default>
    <geom condim="6" friction="3.0 0.01 0.001"
          solimp="0 0.999 0.0001 0.5 2" solref="0.0005 1"/>
  </default>"""
    mjcf = re.sub(
        r'(<compiler\b[^>]*/\s*>)',
        lambda m: m.group(1) + inject_after_compiler,
        mjcf
    )

    # Step 3b: disable collision on the base column (visual geom in worldbody).
    # MuJoCo 3.x filterparent uses weld-group membership, not kinematic adjacency,
    # so world↔link_1 contacts (50mm penetration, ~15 MN force) are NOT auto-excluded.
    # Also make it invisible — the base_link.stl visual mesh will provide the render.
    mjcf = re.sub(
        r'(<geom\b[^>]*size="0\.05 0\.25"[^>]*>)',
        lambda m: m.group(1).replace('/>', ' contype="0" conaffinity="0"/>'),
        mjcf
    )
    # Hide collision primitives for arm links that have STL visual meshes.
    # rgba values are unique per link group in the MJCF saved by mj_saveLastXML:
    #   "0.7 0.7 0.7 1"   → base column cylinder
    #   "0.95 0.95 0.95 1" → link_1, link_2 boxes
    #   "0.2 0.2 0.2 1"   → link_3 shaft, link_4 wrist cylinder
    # Finger geoms use "0.85 0.85 0.88 1" and are intentionally kept visible.
    mjcf = mjcf.replace('rgba="0.7 0.7 0.7 1"',   'rgba="0.7 0.7 0.7 0"')
    mjcf = mjcf.replace('rgba="0.95 0.95 0.95 1"', 'rgba="0.95 0.95 0.95 0"')
    mjcf = mjcf.replace('rgba="0.2 0.2 0.2 1"',   'rgba="0.2 0.2 0.2 0"')

    # Step 3c: inject visual-only STL mesh geoms into worldbody and each arm body.
    # contype=0/conaffinity=0 → no physics contact, render only.
    def _vg(mn, pos, eu, rgba):
        return (f'<geom type="mesh" mesh="{mn}" pos="{pos}" euler="{eu}" '
                f'rgba="{rgba}" contype="0" conaffinity="0"/>')

    mjcf = re.sub(
        r'(<worldbody>)',
        lambda m: m.group(1) + '\n    ' + _vg('vis_base', '-0.189 0 0.17', '1.5707963 0 0', '0.70 0.70 0.70 1'),
        mjcf, count=1,
    )
    for _bname, _mname, _pos, _euler, _rgba in [
        ('link_1', 'vis_link1',   '-0.189 0 -0.182',     '1.5707963 0 0',   '0.95 0.95 0.95 1'),
        ('link_2', 'vis_link2',   '-0.414 0 -0.204',     '1.5707963 0 0',   '0.95 0.95 0.95 1'),
        ('link_3', 'vis_link3',   '-0.543 0.023 -0.261', '1.5707963 0 0',   '0.20 0.20 0.20 1'),
        ('link_4', 'vis_gripper', '0 0 0.002',           '-1.5707963 0 0',  '0.20 0.22 0.25 1'),
    ]:
        mjcf = re.sub(
            rf'(<body\s+name="{re.escape(_bname)}"[^>]*>)',
            lambda m, g=_vg(_mname, _pos, _euler, _rgba): m.group(1) + '\n      ' + g,
            mjcf, count=1,
        )

    # Step 4: inject floor, plant, vials at the start of <worldbody>
    inject_into_worldbody = f"""
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.55 0.50 0.40 1"
          friction="0.5 0.005 0.0001"/>
    <body name="plant" pos="{SRC_X} {SRC_Y} 0">
      <freejoint name="plant_freejoint"/>
      <inertial pos="0 0 0.025" mass="0.002" diaginertia="8e-8 8e-8 1e-8"/>
      <geom name="stem"  type="cylinder" pos="0 0 0.029" size="0.003 0.010"
            friction="3.0 0.01 0.001" rgba="0.35 0.60 0.20 1"/>
      <geom name="node1" type="cylinder" pos="0 0 0.025" size="0.004 0.00125"
            friction="3.0 0.01 0.001" rgba="0.25 0.50 0.15 1"/>
      <geom name="node2" type="cylinder" pos="0 0 0.031" size="0.004 0.00125"
            friction="3.0 0.01 0.001" rgba="0.25 0.50 0.15 1"/>
      <geom name="node3" type="cylinder" pos="0 0 0.037" size="0.004 0.00125"
            friction="3.0 0.01 0.001" rgba="0.25 0.50 0.15 1"/>
      <geom type="sphere" pos="0 0 0.040" size="0.011"
            rgba="0.15 0.70 0.10 0.8" contype="0" conaffinity="0"/>
    </body>
    <body name="source_vial" pos="{SRC_X} {SRC_Y} 0">
      <geom type="mesh" mesh="wide_vial" rgba="0.85 0.90 0.95 0.35"
            contype="0" conaffinity="0"/>
{walls}
    </body>
    <body name="dest_vial" pos="{DEST_X} {DEST_Y} 0">
      <geom type="mesh" mesh="wide_vial" rgba="0.85 0.90 0.95 0.35"
            contype="0" conaffinity="0"/>
{walls}
    </body>"""
    mjcf = re.sub(
        r'(<worldbody>)',
        lambda m: m.group(1) + inject_into_worldbody,
        mjcf
    )

    # Step 5: inject contact exclusions and actuators before </mujoco>
    inject_before_end = """
  <contact>
    <exclude body1="link_1"       body2="source_vial"/>
    <exclude body1="link_2"       body2="source_vial"/>
    <exclude body1="link_3"       body2="source_vial"/>
    <exclude body1="link_4"       body2="source_vial"/>
    <exclude body1="link_1"       body2="dest_vial"/>
    <exclude body1="link_2"       body2="dest_vial"/>
    <exclude body1="link_3"       body2="dest_vial"/>
    <exclude body1="link_4"       body2="dest_vial"/>
    <exclude body1="finger_left"  body2="source_vial"/>
    <exclude body1="finger_right" body2="source_vial"/>
    <exclude body1="finger_left"  body2="dest_vial"/>
    <exclude body1="finger_right" body2="dest_vial"/>
    <exclude body1="link_1"       body2="plant"/>
    <exclude body1="link_2"       body2="plant"/>
    <exclude body1="link_3"       body2="plant"/>
    <exclude body1="link_4"       body2="plant"/>
  </contact>
  <actuator>
    <position name="j1_act" joint="joint_1" kp="1200" kv="70"/>
    <position name="j2_act" joint="joint_2" kp="1200" kv="70"/>
    <position name="j3_act" joint="joint_3" kp="4000" kv="80"/>
    <position name="j4_act" joint="joint_4" kp="200"  kv="20"/>
    <position name="fl_act" joint="finger_left_joint"  kp="25000" kv="10" ctrlrange="0 0.010"/>
    <position name="fr_act" joint="finger_right_joint" kp="25000" kv="10" ctrlrange="0 0.010"/>
  </actuator>
  <equality>
    <weld name="plant_attach" body1="link_4" body2="plant"
          active="false"
          solref="0.001 1" solimp="0 0.9999 0.001 0.5 2"/>
  </equality>"""
    mjcf = re.sub(
        r'(</mujoco>)',
        lambda m: inject_before_end + '\n' + m.group(1),
        mjcf
    )

    return mjcf


# ── Main test ─────────────────────────────────────────────────────────────

def run_urdf_test(
    render: bool = False,
    record: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run full T3+Zimmer vial-to-vial test with the real URDF.

    Returns:
        success          bool
        plant_z_settled  float m
        plant_z_at_lift  float m
        plant_z_final    float m
        dist_to_dest_mm  float mm
    """
    try:
        import mujoco
    except ImportError:
        sys.exit("mujoco not installed — run: pip install mujoco")

    xml   = _build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    B = mujoco.mjtObj.mjOBJ_BODY
    A = mujoco.mjtObj.mjOBJ_ACTUATOR
    J = mujoco.mjtObj.mjOBJ_JOINT

    def _aid(n): return mujoco.mj_name2id(model, A, n)
    def _bid(n): return mujoco.mj_name2id(model, B, n)
    def _jid(n): return mujoco.mj_name2id(model, J, n)

    plant_id = _bid("plant")
    j1a, j2a, j3a, j4a = _aid("j1_act"), _aid("j2_act"), _aid("j3_act"), _aid("j4_act")
    fla, fra = _aid("fl_act"), _aid("fr_act")
    fl_jnt   = _jid("finger_left_joint")
    j1_jnt, j2_jnt, j3_jnt, j4_jnt = _jid("joint_1"), _jid("joint_2"), _jid("joint_3"), _jid("joint_4")

    E = mujoco.mjtObj.mjOBJ_EQUALITY
    def _eid(n): return mujoco.mj_name2id(model, E, n)
    link4_id = _bid("link_4")
    weld_id  = _eid("plant_attach")
    data.eq_active[weld_id] = 0  # start inactive; activated before transport

    # Init arm joints by address — plant freejoint occupies qpos[0:7]
    data.qpos[model.jnt_qposadr[j1_jnt]] = INIT_J1
    data.qpos[model.jnt_qposadr[j2_jnt]] = INIT_J2
    data.qpos[model.jnt_qposadr[j3_jnt]] = INIT_J3
    data.qpos[model.jnt_qposadr[j4_jnt]] = 0.0
    data.ctrl[[j1a, j2a, j3a, j4a, fla, fra]] = [INIT_J1, INIT_J2, INIT_J3, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    viewer = None
    if render:
        try:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(model, data)
        except Exception as e:
            print(f"[warn] viewer: {e}", file=sys.stderr)

    frames, renderer, cam = None, None, None
    if record:
        renderer = mujoco.Renderer(model, height=RECORD_H, width=RECORD_W)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [0.18, 0.07, 0.18]
        cam.distance  = 1.35
        cam.azimuth   = 210.0
        cam.elevation = -18.0
        frames = []

    def sync():
        if viewer: viewer.sync()

    def _cap():
        if frames is not None:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    def _arm(j1, j2, j3, j4=0.0):
        data.ctrl[j1a] = j1; data.ctrl[j2a] = j2
        data.ctrl[j3a] = j3; data.ctrl[j4a] = j4

    def _grip(c):  # 0.0 open, 0.010 closed target
        data.ctrl[fla] = c; data.ctrl[fra] = c

    def _step(n: int):
        for i in range(n):
            mujoco.mj_step(model, data)
            if i % RECORD_EVERY == 0: _cap()
            sync()

    def _ppos():
        p = data.xpos[plant_id]
        return float(p[0]), float(p[1]), float(p[2])

    # ── Phase 0: settle ───────────────────────────────────────────────────
    _arm(INIT_J1, INIT_J2, INIT_J3); _grip(0.0)
    _step(SETTLE_STEPS)

    _, _, pz_settled = _ppos()
    stem_z  = pz_settled + 0.029  # plant stem centre world-z
    j3_g    = LINK4_Z_J3_0 - PRONG_BELOW_L4 - stem_z   # j3 for prong at stem
    j3_ins  = j3_g   # same depth in dest vial (same floor height)

    if verbose:
        print(f"  plant settled z={pz_settled*1000:.1f} mm  "
              f"j3_grasp={j3_g*1000:.1f} mm  j3_lift={J3_LIFT*1000:.0f} mm")

    # ── Phase 1: descend ──────────────────────────────────────────────────
    for i in range(DESCEND_STEPS):
        _arm(INIT_J1, INIT_J2, INIT_J3 + (i / DESCEND_STEPS) * (j3_g - INIT_J3))
        mujoco.mj_step(model, data)
        if i % RECORD_EVERY == 0: _cap()
        sync()

    # ── Phase 2: close ────────────────────────────────────────────────────
    _arm(INIT_J1, INIT_J2, j3_g); _grip(0.010)
    _step(CLOSE_STEPS)

    if verbose:
        fl_q = float(data.qpos[model.jnt_qposadr[fl_jnt]])
        _, _, pz_now = _ppos()
        print(f"  [after close]    fl={fl_q*1000:.2f} mm  plant_z={pz_now*1000:.1f} mm")

    # ── Phase 3: hold ─────────────────────────────────────────────────────
    _step(HOLD_STEPS)

    # ── Phase 4: lift ─────────────────────────────────────────────────────
    for i in range(LIFT_STEPS):
        _arm(INIT_J1, INIT_J2, j3_g + (i / LIFT_STEPS) * (J3_LIFT - j3_g))
        mujoco.mj_step(model, data)
        if i % RECORD_EVERY == 0: _cap()
        sync()

    _, _, pz_lift = _ppos()
    if verbose:
        stem_bottom = pz_lift + 0.019   # stem geom bottom in world z (pos=0.029 − half=0.010)
        cleared = "YES ✓" if stem_bottom > VIAL_HEIGHT else "NO ✗"
        print(f"  [after lift]     plant_z={pz_lift*1000:.1f} mm  "
              f"stem_bottom={stem_bottom*1000:.1f} mm  "
              f"cleared {VIAL_HEIGHT*1000:.0f} mm rim: {cleared}")

    # ── Phase 5: transport ────────────────────────────────────────────────
    # Activate kinematic_attach: weld plant to link_4 for the duration of transport
    # and insert. Mirrors use_kinematic_attach=True in pick_plant_out_of_vial_zimmer.yaml.
    # Without this, MuJoCo's no-slip contact solver provides exactly -frc_act on j1,
    # acting as a rigid weld via friction and preventing j1 from rotating at all.
    mujoco.mj_forward(model, data)
    R1 = data.xmat[link4_id].reshape(3, 3)
    rel_pos = R1.T @ (data.xpos[plant_id] - data.xpos[link4_id])
    q1_conj = data.xquat[link4_id].copy(); q1_conj[1:] *= -1
    rel_quat = np.zeros(4)
    mujoco.mju_mulQuat(rel_quat, q1_conj, data.xquat[plant_id])
    model.eq_data[weld_id, 3:6]  = rel_pos   # relpose pos  at [3:6]
    model.eq_data[weld_id, 6:10] = rel_quat  # relpose quat at [6:10]
    data.eq_active[weld_id] = 1

    # j4 compensates for j1+j2 rotation so gripper orientation stays fixed
    j4_end = -((DEST_J1 - INIT_J1) + (DEST_J2 - INIT_J2))
    for i in range(TRANSPORT_STEPS):
        t = i / TRANSPORT_STEPS
        j1_t = INIT_J1 + t * (DEST_J1 - INIT_J1)
        j2_t = INIT_J2 + t * (DEST_J2 - INIT_J2)
        j4_t = -((j1_t - INIT_J1) + (j2_t - INIT_J2))
        _arm(j1_t, j2_t, J3_LIFT, j4_t)
        mujoco.mj_step(model, data)
        if i % RECORD_EVERY == 0: _cap()
        sync()

    if verbose:
        px, py, pz = _ppos()
        lx = data.xpos[link4_id][0]; ly = data.xpos[link4_id][1]
        print(f"  [after transport] plant=({px*1000:.1f},{py*1000:.1f},{pz*1000:.1f})mm  "
              f"arm=({lx*1000:.1f},{ly*1000:.1f})mm")

    # ── Phase 6: insert ───────────────────────────────────────────────────
    # Deactivate weld before insert: j3 is vertical, finger friction doesn't oppose it,
    # and keeping the weld active would conflict with the dest-vial floor.
    data.eq_active[weld_id] = 0
    for i in range(INSERT_STEPS):
        _arm(DEST_J1, DEST_J2, J3_LIFT + (i / INSERT_STEPS) * (j3_ins - J3_LIFT), j4_end)
        mujoco.mj_step(model, data)
        if i % RECORD_EVERY == 0: _cap()
        sync()

    if verbose:
        px, py, pz = _ppos()
        print(f"  [after insert]    plant=({px*1000:.1f},{py*1000:.1f},{pz*1000:.1f})mm")

    # ── Phase 7: release ──────────────────────────────────────────────────
    # Stage A: hold arm at insert depth, open fingers (plant settles in dest vial).
    # Stage B: retract arm only after fingers are fully open so V-groove prongs
    # don't drag the plant stem upward.
    OPEN_STEPS    = RELEASE_STEPS // 2
    RETRACT_STEPS = RELEASE_STEPS - OPEN_STEPS
    _grip(0.0)
    _step(OPEN_STEPS)
    for i in range(RETRACT_STEPS):
        t = i / RETRACT_STEPS
        _arm(DEST_J1, DEST_J2, j3_ins + t * (J3_LIFT - j3_ins), j4_end * (1 - t))
        mujoco.mj_step(model, data)
        if i % RECORD_EVERY == 0: _cap()
        sync()

    if verbose:
        px, py, pz = _ppos()
        print(f"  [after release]   plant=({px*1000:.1f},{py*1000:.1f},{pz*1000:.1f})mm")

    # ── Phase 8: check ────────────────────────────────────────────────────
    _arm(DEST_J1, DEST_J2, J3_LIFT); _grip(0.0)
    _step(CHECK_STEPS)

    if viewer: viewer.close()

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
    if renderer: renderer.close()

    px, py, pz = _ppos()
    dist = math.hypot(px - DEST_X, py - DEST_Y) * 1000
    success = dist < DEST_XY_TOL * 1000 and pz > -0.030

    return {
        "success":         success,
        "plant_z_settled": pz_settled,
        "plant_z_at_lift": pz_lift,
        "plant_z_final":   pz,
        "dist_to_dest_mm": dist,
    }


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", action="store_true",
                    help="Open interactive MuJoCo viewer")
    ap.add_argument("--record", metavar="PATH", nargs="?", const="",
                    help="Record video (default: media/mujoco_urdf_test.mp4)")
    args = ap.parse_args()

    record = args.record
    if record == "":
        record = str(MEDIA_DIR / "mujoco_urdf_test.mp4")

    print(f"\nRunning URDF vial-to-vial test  render={args.render}"
          + (f"  record={record}" if record else ""))

    r = run_urdf_test(render=args.render, record=record)

    ok = "SUCCESS ✓" if r["success"] else "FAIL ✗"
    print(f"\n── Result ────────────────────────────────────────────────────")
    print(f"  {ok}")
    print(f"  plant z settled  : {r['plant_z_settled']*1000:>6.1f} mm")
    print(f"  plant z at lift  : {r['plant_z_at_lift']*1000:>6.1f} mm  "
          f"(vial rim {VIAL_HEIGHT*1000:.0f} mm)")
    print(f"  plant z final    : {r['plant_z_final']*1000:>6.1f} mm")
    print(f"  dist to dest     : {r['dist_to_dest_mm']:>6.1f} mm  "
          f"(threshold {DEST_XY_TOL*1000:.0f} mm)")
    print(f"──────────────────────────────────────────────────────────────\n")
    sys.exit(0 if r["success"] else 1)


if __name__ == "__main__":
    main()
