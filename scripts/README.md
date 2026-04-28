# Scripts

Thin shell wrappers around the Python entry points. Each script picks
the right Python environment (Isaac Lab's bundled python for sim work,
the local training venv for ACT training) and forwards all flags through.

| Script              | Env        | Wraps                              | Purpose                                    |
|---------------------|------------|------------------------------------|--------------------------------------------|
| `setup.sh`          | both       | —                                  | Bootstrap: training venv, install, URDF→USD |
| `smoke.sh`          | Isaac Lab  | `arm_vla.cli.smoke`                | Scene preview / sanity check               |
| `teleop.sh`         | Isaac Lab  | `arm_vla.cli.teleop`               | Record keyboard demos                      |
| `oracle.sh`         | Isaac Lab  | `arm_vla.cli.oracle`               | Scripted oracle demo collection            |
| `annotate.sh`       | Isaac Lab  | `arm_vla.cli.annotate`             | Add datagen_info to raw demos              |
| `mimic.sh`          | Isaac Lab  | `arm_vla.cli.mimic`                | Curobo-based augmentation                  |
| `train.sh`          | training   | `arm_vla.training.train_act`       | Train ACT                                  |
| `eval.sh`           | Isaac Lab  | `arm_vla.eval.rollout`             | Sim rollouts of a trained checkpoint       |

Every script accepts `--task <name>` (default: `pick_place`). Other
defaults come from:

* `src/arm_vla/training/defaults.yaml` (shared hyperparams)
* `src/arm_vla/tasks/<task>/task.yaml` (task-specific overrides)

CLI flags override both YAMLs.

Isaac Lab entry: `$ISAACLAB/isaaclab.sh -p` (defaults to `~/IsaacLab`).
Training venv: `$ARM_VLA_VENV` (defaults to `~/arm-vla-venv`).
