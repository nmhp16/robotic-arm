# Scripts

Shell wrappers around the Python entry points. Each script selects the
Python environment it needs.

| Script              | Env        | Purpose                                             |
|---------------------|------------|-----------------------------------------------------|
| `smoke.sh`          | Isaac Lab  | Scene preview for the pick-place env                |
| `smoke_stack.sh`    | Isaac Lab  | Scene preview for the stack env                     |
| `teleop.sh`         | Isaac Lab  | Record keyboard demos to HDF5                       |
| `annotate.sh`       | Isaac Lab  | Annotate raw demos with mimic datagen_info          |
| `mimic.sh`          | Isaac Lab  | Augment demos via curobo                            |
| `oracle.sh`         | Isaac Lab  | Scripted oracle demo collection                     |
| `train.sh`          | training   | Train ACT on the HDF5 demo file                     |
| `eval.sh`           | Isaac Lab  | Sim rollouts of a trained checkpoint (mp4 out)      |

Isaac Lab entry: `$ISAACLAB/isaaclab.sh -p` (defaults to `~/IsaacLab`).
Training venv: see top-level `README.md` (default `/home/ketchup-core/arm-vla-venv`).
