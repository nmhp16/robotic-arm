# HDF5 demo data format

`scripts/teleop.sh`, `scripts/oracle.sh`, and `scripts/mimic.sh` all
write demos to HDF5 in the schema below — the format Isaac Lab's
`isaaclab_mimic` tools produce and that
`arm_act.training.dataset.HDF5DemoDataset` consumes.

## Schema

```
data/
  demo_<i>/
    actions             (T, 7)              # delta_pose(6) + binary gripper(1)
    obs/
      policy/
        joint_pos       (T, 6)
        joint_vel       (T, 6)
        eef_pos         (T, 3)
        eef_quat        (T, 4)
        gripper_pos     (T, 1)              # driver-finger position
        object          (T, 16)             # concatenated object obs
        pickable_pos    (T, 3)              # the object being manipulated
        pickable_quat   (T, 4)
        target_pos      (T, 3)              # the place target
        table_cam       (T, 224, 224, 3)    # uint8 RGB
        wrist_cam       (T, 224, 224, 3)    # uint8 RGB
      subtask_terms/
        grasp           (T,)                # bool — gripper closed on object
        place           (T,)                # bool — object on target with gripper open
    initial_state       (struct)
    seed                scalar
```

## Trainer-relevant fields

`HDF5DemoDataset` reads only the proprio state + cameras + actions:

- `obs/eef_pos`, `obs/eef_quat`, `obs/gripper_pos[:1]` → 8-D state vector
- `obs/<camera>` for each camera in `data.camera_keys` (default: `table_cam`, `wrist_cam`)
- `actions` → action chunks of length `policy.chunk_size`

The `pickable_*`, `target_pos`, `subtask_terms/*` fields are recorded for
completeness (the oracle and mimic use them) but the ACT trainer ignores
them — its policy input is image + proprio only.
