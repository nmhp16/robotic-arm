# Scripts

One-line shell wrappers around the Python entry points. Each script
selects the right Python environment (Isaac Lab's bundled python for sim
work, the local training venv for ACT training) and forwards every flag
through to the underlying module.

| Script         | Environment | Wraps                          | Purpose                                      |
|----------------|-------------|--------------------------------|----------------------------------------------|
| `setup.sh`     | both        | —                              | Bootstrap: training venv, install, URDF→USD  |
| `smoke.sh`     | Isaac Lab   | `arm_act.cli.smoke`            | Scene preview / sanity check                 |
| `teleop.sh`    | Isaac Lab   | `arm_act.cli.teleop`           | Record keyboard demos                        |
| `oracle.sh`    | Isaac Lab   | `arm_act.cli.oracle`           | Scripted oracle demo collection              |
| `annotate.sh`  | Isaac Lab   | `arm_act.cli.annotate`         | Add ``datagen_info`` for mimic               |
| `mimic.sh`     | Isaac Lab   | `arm_act.cli.mimic`            | Curobo-based augmentation                    |
| `train.sh`     | training    | `arm_act.training.train_act`   | Train ACT                                    |
| `eval.sh`      | Isaac Lab   | `arm_act.eval.rollout`         | Sim rollouts of a trained checkpoint         |

## Conventions

Every script except `setup.sh` accepts `--task <name>` (default:
`pick_plant_out`). Other defaults come from:

- `src/arm_act/training/defaults.yaml` — shared hyperparameters
- `src/arm_act/tasks/<task>.yaml` — task-specific overrides

CLI flags override both YAMLs. Run any script with `--help` for its
exact flag list.

## Environment variables

| Variable        | Default           | Effect                                |
|-----------------|-------------------|---------------------------------------|
| `ISAACLAB`      | `~/IsaacLab`      | Isaac Lab install root                |
| `ARM_ACT_VENV`  | `~/arm-act-venv`  | Training venv location                |
| `PYTHON`        | `python3.12`      | System python used by `setup.sh`      |
| `SKIP_TORCH`    | (unset)           | `setup.sh`: skip torch wheels install |
| `SKIP_ISAAC`    | (unset)           | `setup.sh`: skip Isaac Lab pip install|
| `SKIP_USD`      | (unset)           | `setup.sh`: skip URDF→USD conversion  |

## Helper scripts

- `convert_t3_401_simple_gripper.py` — one-shot URDF→USD converter for
  the bundled Epson T3-401 SCARA + parallel-jaw gripper. Run via
  `setup.sh` (idempotent); only needed manually if you edit the URDF.
