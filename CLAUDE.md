# CLAUDE.md — robotic-arm (Isaac Lab sim-to-real)

Rules for working in this repo. Dimensional and visual accuracy matter for
sim-to-real; "the pipeline runs end-to-end" is necessary but not sufficient.

## Robot-sim-to-real (general)

1. **Kinematic dimensions must match the real robot's spec.** No extended joint
   strokes, raised bases, or fudged tool offsets for "sim convenience".
   Adjust the scene instead (table height, object spawn z, oracle waypoints).

2. **Asset swaps are never drop-in.** Before swapping any CAD/USD, list every
   downstream task and budget time to re-tune each one's
   hover/lift/insert/grasp heights, success thresholds, and collisions.

3. **Making a URDF fully accurate to its source CAD is a multi-day project.**
   It requires actuator retuning, inertial frame math (axis rotations,
   ixx↔iyy swaps), scene-layout adjustment, oracle waypoint retuning, and
   re-validation of every task using the asset. If you start, scope it
   explicitly. After the first cascading bug you're at the beginning of the
   debug, not the end.

4. **Distinguish kinematic correctness, physical realism, and visual realism.**
   Defined by URDF joint origins / mass+inertia+actuator gains / `<visual>`
   placement respectively. Fixing one can break another.

5. **Document every physics shortcut.** kinematic_attach payload welding,
   collision-disabled sub-prims, fixed bases, high actuator stiffness, zero
   motor friction — list per task, with why it's there and what breaks if
   removed. Audit before real-robot deployment.

6. **The recorded demo's last obs is not the success state.** Verify success
   semantics by reading the obs trajectory, not just the SUCCESS flag.

7. **CAD assembly pose ≠ home pose.** Mesh bounds in CAD don't directly tell
   you joint positions in the operational pose; the assembly may be a
   shipping/storage pose.

8. **Off-the-shelf table/floor USDs are rarely flat at z=0.** Validate the
   settled z across the workspace before relying on it.

9. **Iterate with render → look → simulate → commit.** Standalone matplotlib
   render is seconds; full simulator launch is minutes. Don't skip the look
   step.

## URDF / CAD-mesh handling

10. **Never trust mesh bounding boxes for joint positions.** Find the actual
    joint axis line on each link. Cylindrical parts have their joint axis on
    the cylinder centerline, which may be off-center from the bounding box —
    e.g., the T3-401S shaft has center at CAD (X=543, Z=23), not (X=543, Z=0).

11. **Verify CAD coordinate convention before applying URDF rpy.** Which CAD
    axis is vertical, and which is the arm-extension direction? T3-B401S CAD
    had arm in CAD −X (rpy used π yaw); T3-401S CAD has arm in CAD +X (no π
    yaw). When swapping CAD source files, redo this check.

12. **Joint kinematic origin must match visual mesh joint position.** If you
    change one, change the other, or the mesh appears floating / misaligned.

13. **STEP part-name conventions vary between CAD assemblies.** Inspect
    discovered parts after swapping the STEP file; update the `PART_TO_LINK`
    map in `scripts/convert/convert_step_to_meshes.py`.

14. **Always inspect per-axis bounds AND centerlines after a mesh swap.**

    ```bash
    python3 -c "import struct, numpy as np; ..."  # min/max/center per axis
    ```

15. **Use standalone trimesh + matplotlib renders for mesh-transform
    iteration**, not the full Isaac Sim pipeline.

## Sim-to-real fidelity

16. **`collision: false` in YAML does not always disable a baked SDF collider.**
    The env_cfg builder sets `CollisionPropertiesCfg(collision_enabled=False)`
    at the USD root only; sub-prim physics layers still fire. To truly
    disable, edit the source URDF's `<collision>` and re-run the converter.

17. **`SeattleLabTable` top surface varies by xy.**
    - (0.30, -0.01): z ≈ 0.039
    - (0.27, 0.14): z ≈ -0.003

    Spawn objects in the central flat region or use a kinematic landing pad.

18. **`kinematic_attach` is on by default for replayable demos.** Real
    friction grasping (`use_kinematic_attach: false`) requires payload
    geometry compatible with the gripper, gentle actuator tuning, and tight
    oracle alignment. If real grip is <20% reliable on an asset, document
    the failure mode and stay on `kinematic_attach`.

## Workflow

19. **After URDF edits, re-convert every dependent USD.** Multiple end-effector
    variants share arm meshes; re-run every converter:

    ```bash
    env -u VIRTUAL_ENV -u CONDA_PREFIX ~/IsaacLab/isaaclab.sh -p scripts/convert/convert_t3_401_tweezer.py
    env -u VIRTUAL_ENV -u CONDA_PREFIX ~/IsaacLab/isaaclab.sh -p scripts/convert/convert_t3_401_simple_gripper.py
    ```

    Unset the host venv (`env -u VIRTUAL_ENV -u CONDA_PREFIX`) or Isaac Sim's
    python loads the wrong site-packages.

20. **Oracle bash output buffers when piped to `tail`.** Use `tee` for live
    monitoring:

    ```bash
    stdbuf -oL -eL ./scripts/oracle.sh ... 2>&1 | tee /tmp/log >/dev/null
    grep -E "episode |wrote |SUCCESS|FAIL" /tmp/log
    ```

21. **Validate by reading the recorded HDF5**, not just the SUCCESS count.
    `data/raw/<task>/demos.hdf5` has the actual recorded `pickable_pos` /
    `eef_pos` trajectories.
