"""Generate + convert the multi-segment articulated leaf plant.

Replaces the FEM approach (which hit an Isaac Lab DeformableObject + IK
cache-mismatch bug). Instead, the plant is modeled as a chain of small
rigid segments connected by spring revolute joints — physically equivalent
to a real flexing stem, but running on the well-tested rigid-body +
articulation path that Isaac Lab handles cleanly.

URDF shape:
    base (heavier root, rests on table) -[joint_0]-> seg_1 -[joint_1]-> seg_2
    ... -> seg_N -> leaves (fixed at tip, carries the leaf_plant.stl visual)

Each joint is revolute with a soft spring (stiffness via ImplicitActuator in
env_cfg, NOT in URDF dynamics). The springs return the plant to upright
under its own weight; under jaw pressure mid-stem, the local segments bend
(local-dent analogue) while the rest stays upright — what a real plant does.

Run with:
    env -u VIRTUAL_ENV -u CONDA_PREFIX ~/IsaacLab/isaaclab.sh -p \\
        scripts/convert_leaf_plant_segmented.py
"""

from __future__ import annotations

import logging
import os

from isaaclab.app import AppLauncher

_app_launcher = AppLauncher(headless=True)
_simulation_app = _app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSET_DIR = os.path.join(REPO_ROOT, "assets", "leaf_plant_segmented")
URDF_PATH = os.path.join(ASSET_DIR, "leaf_plant_segmented.urdf")
USD_NAME = "leaf_plant_segmented.usd"

# ----------------- Geometry & physics constants -----------------
N_SEGMENTS = 2                # MINIMAL chain — final test before declaring articulation dead-end
SEGMENT_H = 0.005             # m — each segment 5 mm tall (30 mm total stem)
SEGMENT_R = 0.003             # m — 6 mm dia stem (matches existing rigid)
BASE_H = 0.005                # m — base disc, 5 mm tall
BASE_R = 0.003                # m — 6 mm dia (was 10 mm). 10 mm overlapped the
                              # SCARA gripper finger inner faces at TCP init
                              # (X=±0.005 from gripper centre), driving init
                              # depenetration impulses that exploded the
                              # articulation. 6 mm = 1.5 mm clearance per side.
# Mass + inertia inflated 10× over real values for PhysX numerical stability.
# At tiny inertias (~1e-9 kg·m²) any micro-contact impulse produces huge
# accelerations and the articulation explodes (verified, v1: pos diverged
# to 10^11 m by step 240). 10× inflation puts inertia in PhysX's reliable
# range while keeping the plant feeling soft (springs are tuned in env_cfg
# independently, so spring response is unaffected).
BASE_MASS = 0.050             # kg — 50 g base (was 20 g) for anchoring
SEGMENT_MASS = 0.015           # kg — 15 g per segment (was 1.5 g)
JOINT_DAMPING = 0.005         # moderate URDF damping (back from 0.05 which was over-stiff)
JOINT_ARMATURE = 0.0001       # extra rotor inertia for solver stability
LEAF_MESH_REL = "../leaf_plant/meshes/leaf_plant.stl"  # reuse the existing leaf visual mesh
LEAF_SCALE_XY = 0.001         # leaf_plant.stl is mm → m
LEAF_SCALE_Z = 0.0006         # squashed for the segmented stem's tip
COLOR = (0.30, 0.55, 0.25, 1.0)
FRICTION_MU = (3.0, 2.5, 0.0)  # static, dynamic, restitution


def build_urdf() -> str:
    """Generate the URDF text for the chain."""
    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<robot name="leaf_plant_segmented">')

    # Base link — heavier disc that sits on the vial floor / table.
    lines.append('  <link name="base">')
    lines.append(f'    <visual><origin xyz="0 0 {BASE_H/2}" rpy="0 0 0"/>')
    lines.append(f'      <geometry><cylinder radius="{BASE_R}" length="{BASE_H}"/></geometry>')
    lines.append(f'      <material name="stem"><color rgba="{COLOR[0]} {COLOR[1]} {COLOR[2]} {COLOR[3]}"/></material></visual>')
    lines.append(f'    <collision><origin xyz="0 0 {BASE_H/2}" rpy="0 0 0"/>')
    lines.append(f'      <geometry><cylinder radius="{BASE_R}" length="{BASE_H}"/></geometry></collision>')
    # Inertia for a cylinder: ixx = iyy = (1/12) * m * (3r^2 + h^2), izz = (1/2) * m * r^2
    i_xy = (1/12) * BASE_MASS * (3 * BASE_R**2 + BASE_H**2)
    i_z = 0.5 * BASE_MASS * BASE_R**2
    lines.append(f'    <inertial><origin xyz="0 0 {BASE_H/2}"/><mass value="{BASE_MASS}"/>')
    lines.append(f'      <inertia ixx="{i_xy:.3e}" ixy="0" ixz="0" iyy="{i_xy:.3e}" iyz="0" izz="{i_z:.3e}"/></inertial>')
    lines.append('  </link>')

    # Stem segments.
    seg_i_xy = (1/12) * SEGMENT_MASS * (3 * SEGMENT_R**2 + SEGMENT_H**2)
    seg_i_z = 0.5 * SEGMENT_MASS * SEGMENT_R**2
    for i in range(N_SEGMENTS):
        name = f"segment_{i}"
        lines.append(f'  <link name="{name}">')
        lines.append(f'    <visual><origin xyz="0 0 {SEGMENT_H/2}" rpy="0 0 0"/>')
        lines.append(f'      <geometry><cylinder radius="{SEGMENT_R}" length="{SEGMENT_H}"/></geometry>')
        lines.append(f'      <material name="stem"><color rgba="{COLOR[0]} {COLOR[1]} {COLOR[2]} {COLOR[3]}"/></material></visual>')
        lines.append(f'    <collision><origin xyz="0 0 {SEGMENT_H/2}" rpy="0 0 0"/>')
        lines.append(f'      <geometry><cylinder radius="{SEGMENT_R}" length="{SEGMENT_H}"/></geometry></collision>')
        lines.append(f'    <inertial><origin xyz="0 0 {SEGMENT_H/2}"/><mass value="{SEGMENT_MASS}"/>')
        lines.append(f'      <inertia ixx="{seg_i_xy:.3e}" ixy="0" ixz="0" iyy="{seg_i_xy:.3e}" iyz="0" izz="{seg_i_z:.3e}"/></inertial>')
        lines.append('  </link>')

        # Joint connecting THIS segment to the previous link.
        parent = "base" if i == 0 else f"segment_{i-1}"
        parent_h = BASE_H if i == 0 else SEGMENT_H
        joint_name = f"seg_joint_{i}"
        # Axis alternates X/Y per segment so the plant can bend in any direction
        # (1-DOF per joint, but axes alternated → effectively 2D bending across
        # the chain). Simpler than full ball joints.
        axis = "1 0 0" if i % 2 == 0 else "0 1 0"
        lines.append(f'  <joint name="{joint_name}" type="revolute">')
        lines.append(f'    <parent link="{parent}"/>')
        lines.append(f'    <child link="{name}"/>')
        lines.append(f'    <origin xyz="0 0 {parent_h}" rpy="0 0 0"/>')
        lines.append(f'    <axis xyz="{axis}"/>')
        # Joint limits: ±60° max bend, plenty for natural flex without inversion.
        lines.append('    <limit lower="-1.0" upper="1.0" effort="2.0" velocity="10.0"/>')
        # Small URDF-level damping for solver stability (spring stiffness is
        # set in env_cfg via ImplicitActuator, on top of this baseline).
        lines.append(f'    <dynamics damping="{JOINT_DAMPING}" friction="0.0"/>')
        lines.append('  </joint>')

    # Leaves at the tip — fixed visual mesh, no collider (lets the jaws pass
    # through to reach the stem segments).
    last_seg = f"segment_{N_SEGMENTS-1}"
    lines.append('  <link name="leaves">')
    lines.append(f'    <visual><origin xyz="0 0 -0.040" rpy="0 0 0"/>')
    lines.append(f'      <geometry><mesh filename="{LEAF_MESH_REL}" scale="{LEAF_SCALE_XY} {LEAF_SCALE_XY} {LEAF_SCALE_Z}"/></geometry>')
    lines.append(f'      <material name="leaf"><color rgba="{COLOR[0]} {COLOR[1]} {COLOR[2]} {COLOR[3]}"/></material></visual>')
    lines.append('    <inertial><origin xyz="0 0 0.005"/><mass value="0.0003"/>')
    lines.append('      <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/></inertial>')
    lines.append('  </link>')
    lines.append(f'  <joint name="leaf_attach" type="fixed">')
    lines.append(f'    <parent link="{last_seg}"/>')
    lines.append(f'    <child link="leaves"/>')
    lines.append(f'    <origin xyz="0 0 {SEGMENT_H}" rpy="0 0 0"/>')
    lines.append('  </joint>')

    # Gazebo-style friction (read by Isaac's UrdfConverter as PhysxMaterial).
    # Apply to the base + every segment for consistent stem-on-jaw friction.
    for link in ["base"] + [f"segment_{i}" for i in range(N_SEGMENTS)]:
        lines.append(f'  <gazebo reference="{link}">')
        lines.append(f'    <mu1>{FRICTION_MU[0]}</mu1>')
        lines.append(f'    <mu2>{FRICTION_MU[1]}</mu2>')
        lines.append('  </gazebo>')

    lines.append('</robot>')
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    os.makedirs(ASSET_DIR, exist_ok=True)
    urdf_text = build_urdf()
    with open(URDF_PATH, "w") as fh:
        fh.write(urdf_text)
    print(f"Wrote URDF: {URDF_PATH} ({len(urdf_text)} bytes, {N_SEGMENTS} segments + base + leaves)", flush=True)

    cfg = UrdfConverterCfg(
        asset_path=URDF_PATH,
        usd_dir=ASSET_DIR,
        usd_file_name=USD_NAME,
        fix_base=False,             # plant rests on table/vial floor; gravity holds the base
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"Wrote USD: {converter.usd_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        _simulation_app.close()
