# Scripts

Shell wrappers around the Python entry points. Each script selects the
Python environment it needs.

| Script        | Env        | Purpose                                       |
|---------------|------------|-----------------------------------------------|
| `smoke.sh`    | Isaac Lab  | Headless sanity check of the env              |
| `teleop.sh`   | Isaac Lab  | Record keyboard demos to HDF5                 |
| `mimic.sh`    | Isaac Lab  | Augment demos via curobo                      |
| `convert.sh`  | training   | HDF5 → RLDS TFDS                              |
| `train.sh`    | training   | OpenVLA LoRA fine-tune                        |
| `eval.sh`     | Isaac Lab  | Sim rollouts of a fine-tuned checkpoint       |

The Isaac Lab entry is at `$ISAACLAB/isaaclab.sh -p` (defaults to
`~/IsaacLab`). The training env is at `./.venv` (see `README.md`).
