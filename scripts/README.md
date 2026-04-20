# Scripts

Shell wrappers around the Python entry points. Each script selects the
Python environment it needs.

| Script              | Env        | Purpose                                             |
|---------------------|------------|-----------------------------------------------------|
| `smoke.sh`          | Isaac Lab  | Scene preview for the pick-place env                |
| `smoke_stack.sh`    | Isaac Lab  | Scene preview for the stack env                     |
| `teleop.sh`         | Isaac Lab  | Record keyboard demos to HDF5                       |
| `mimic.sh`          | Isaac Lab  | Augment demos via curobo                            |
| `convert.sh`        | training   | HDF5 → RLDS TFDS                                    |
| `train.sh`          | training   | OpenVLA LoRA fine-tune                              |
| `eval.sh`           | Isaac Lab  | Sim rollouts of a fine-tuned checkpoint (mp4 out)   |
| `zeroshot.sh`       | Isaac Lab  | Pretrained-OpenVLA baseline, no fine-tune (mp4 out) |

Isaac Lab entry: `$ISAACLAB/isaaclab.sh -p` (defaults to `~/IsaacLab`).
Training venv: `./.venv` (see top-level `README.md`).
