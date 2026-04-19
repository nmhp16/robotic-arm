# Teleop data collection

Pure wrapper around Isaac Lab's `scripts/tools/record_demos.py`. Nothing novel
lives in this directory — it's a marker for the pipeline stage. See
`scripts/teleop.sh` for the entry point.

## Output schema

The HDF5 produced by `record_demos.py` has the structure that `isaaclab_mimic`
expects:

```
data/
  demo_0/
    actions            (T, action_dim)
    obs/
      policy/
        joint_pos      (T, 6)
        joint_vel      (T, 6)
        eef_pos        (T, 3)
        eef_quat       (T, 4)
        gripper_pos    (T, 1)
        object         (T, 16)    # cube pos/quat + target + relative
        cube_pos       (T, 3)
        cube_quat      (T, 4)
        target_pos     (T, 3)
        table_cam      (T, 224, 224, 3)
        wrist_cam      (T, 224, 224, 3)
      subtask_terms/
        grasp          (T,)       # bool, cube in gripper
        place          (T,)       # bool, cube on target
      ...
    initial_state      (struct)
    seed               scalar
```

This schema drops into mimic's dataset generator with no conversion. The RLDS
converter (phase 5) reads this same file and emits OpenVLA's expected format.

## Goal for phase 3

15 successful demos, varied cube/target positions. Mimic will augment these
into ~500 in phase 4.
