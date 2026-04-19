# Teleop

`scripts/teleop.sh` wraps Isaac Lab's `scripts/tools/record_demos.py` with
task-specific defaults. Output is HDF5 in the format consumed by
`isaaclab_mimic` and by `arm_vla.data.rlds_convert`.

## HDF5 schema

```
data/
  demo_<i>/
    actions            (T, 7)
    obs/
      policy/
        joint_pos      (T, 6)
        joint_vel      (T, 6)
        eef_pos        (T, 3)
        eef_quat       (T, 4)
        gripper_pos    (T, 1)
        object         (T, 16)
        cube_pos       (T, 3)
        cube_quat      (T, 4)
        target_pos     (T, 3)
        table_cam      (T, 224, 224, 3)
        wrist_cam      (T, 224, 224, 3)
      subtask_terms/
        grasp          (T,)
        place          (T,)
    initial_state      (struct)
    seed               scalar
```
