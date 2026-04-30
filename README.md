# arm-act

Action Chunking Transformer (ACT) imitation pipeline for Isaac Lab.
Epson T3-401 SCARA + parallel-jaw, sim-only, one task per checkpoint.
The bundled tasks teach the arm to pick a plant out of a jar and put it
back. New task variants are a single YAML file — no Python required for
variants of the bundled archetype.

```text
keyboard teleop ─┐
                 ├─→ isaaclab_mimic ──→ ACT training ──→ sim rollout eval
scripted oracle ─┘    augmentation                       (mp4 + summary.json)
```

## Features

- **YAML-driven tasks.** A task is one file:
  `src/arm_act/tasks/<name>.yaml`. The runtime reads it and builds the
  Isaac Lab env, oracle, and Mimic config dynamically.
- **Auto-registered gym ids.** Drop in a new YAML and the next process
  import sees it; no code edits required.
- **Two-layer config.** Shared hyperparameters in `defaults.yaml`,
  per-task overrides in `<task>.yaml`, deep-merged at load time. CLI
  flags override both.
- **Vendored ACT** (~30 M params) — trains in 30–60 min on a single GPU
  and runs inference in milliseconds. No 7 B VLA fine-tune required.
- **No RLDS / TFDS step.** The trainer reads Mimic-augmented HDF5
  directly; action chunks are sampled lazily.

## Requirements

- NVIDIA GPU with CUDA 12.8+ (tested on GB10 / DGX Spark, aarch64)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 2.3.2 at `~/IsaacLab` (override via `ISAACLAB=...`)
- Python 3.12

## Installation

```bash
./scripts/setup.sh
```

The bootstrap script is re-runnable. It creates the training venv,
installs `torch + torchvision` from NVIDIA's CUDA index, installs the
project into both the training venv and Isaac Lab's bundled python, and
converts the T3-401 + simple-gripper URDF to USD.

Manual equivalents are documented at the top of `scripts/setup.sh`.

### Why two Python environments?

Isaac Lab ships a `torch` build that does not coexist cleanly with the
training-side `torchvision` + ImageNet weights stack. Separating them is
cheaper than debugging clashes:

| Environment      | Default location  | Used for                                       |
|------------------|-------------------|------------------------------------------------|
| Isaac Lab python | `$ISAACLAB`       | teleop, oracle, mimic, annotate, eval, smoke   |
| Training venv    | `$ARM_ACT_VENV`   | ACT training only                              |

`$ARM_ACT_VENV` defaults to `~/arm-act-venv`. Keep the venv on local
NVMe — FUSE mounts have produced intermittent `EPERM` under the heavy
concurrent imports `torch + torchvision` does at startup.

## Quickstart

Default task is `pick_plant_out`. Pass `--task <name>` to retarget any
step.

```bash
./scripts/smoke.sh                            # headless scene preview
./scripts/oracle.sh   --num-demos 15          # collect 15 scripted demos
./scripts/annotate.sh                         # add datagen_info for mimic
./scripts/mimic.sh    --num-demos 500         # curobo augmentation
./scripts/train.sh                            # train ACT (~30–60 min)
./scripts/eval.sh                             # rollout eval, writes mp4s
```

Substitute `oracle.sh` with `teleop.sh` for keyboard collection. See
[`scripts/README.md`](scripts/README.md) for the full script reference
and [`docs/data_format.md`](docs/data_format.md) for the HDF5 schema.

### The bundled tasks

| Task              | Template       | Goal                                              |
|-------------------|----------------|---------------------------------------------------|
| `pick_plant_out`  | `_runtime`     | Pick the plant out of its starting pose, place on tray |
| `put_plant_back`  | `_runtime_jar` | Pick the plant from the tray, insert into the jar     |

`pick_plant_out` uses the default state machine. `put_plant_back` uses
the orientation-aware `_runtime_jar` template that adds an `INSERT`
phase (descend below jar top into the cavity) and a `RETRACT` phase
(lift the gripper out of the jar before the episode ends).

## Configuration

Two YAML layers, deep-merged at load time:

```text
src/arm_act/training/defaults.yaml          shared hyperparams (rarely touched)
src/arm_act/tasks/<task>.yaml               full task spec (one file per task)
```

`defaults.yaml` carries the policy architecture, training schedule, eval
defaults, and data-collection step rates. `<task>.yaml` carries
everything task-specific: gym ids, instruction, robot type, objects
(color, size, mass, friction, spawn ranges), cameras, success
thresholds, oracle waypoint heights, mimic subtask declarations, and
data + checkpoint paths.

CLI flags on `train.sh` / `eval.sh` override individual fields:

```bash
./scripts/train.sh --max-steps 1000 --batch-size 32
./scripts/eval.sh  --checkpoint checkpoints/pick_plant_out/step_010000 --num-episodes 5
```

## Adding a new task

### Variants of the bundled archetype — YAML only

The bundled runtime supports any task that fits "pick *one* object,
place on *one* target, hover→grasp→lift→place→release" or its
jar-insertion sibling. For variants, copy and edit the YAML:

```bash
cp src/arm_act/tasks/pick_plant_out.yaml src/arm_act/tasks/red_to_blue.yaml
$EDITOR  src/arm_act/tasks/red_to_blue.yaml
# Edit:
#   task.name           → red_to_blue
#   task.gym_id         → unique gym id (must not collide with others)
#   task.mimic_gym_id   → unique mimic gym id
#   task.instruction    → describe the task in plain English
#   objects.cube.color  → e.g. [0.9, 0.1, 0.1]
#   objects.target.*    → new spawn range / USD if you want a different pad
#   data.*              → per-task data paths
#   training.output_dir → e.g. checkpoints/red_to_blue
#   eval.checkpoint     → e.g. checkpoints/red_to_blue/final

./scripts/oracle.sh   --task red_to_blue --num-demos 15
./scripts/annotate.sh --task red_to_blue
./scripts/mimic.sh    --task red_to_blue
./scripts/train.sh    --task red_to_blue
./scripts/eval.sh     --task red_to_blue
```

### Where YAML stops being enough

The `_runtime` template assumes one *pickable* object, one *target*
object, and the state machine
`hover → descend → grasp → lift → move → place → release`. The
`_runtime_jar` sibling adds `align → insert → retract` phases for
jar-insertion-style tasks. Tasks that break that structure need a new
template under `tasks/_runtime_<archetype>/`:

| Task                                          | YAML-only? | Reason                                           |
|-----------------------------------------------|:----------:|--------------------------------------------------|
| Pick plant out of jar (shipped)               | yes        | matches `_runtime` archetype                     |
| Put plant back into jar (shipped)             | yes        | matches `_runtime_jar` archetype                 |
| Same task, different color / size / mass      | yes        | edit `objects.*`                                 |
| Different success thresholds                  | yes        | edit `success.*`                                 |
| Stack cube A on cube B                        | yes        | target is a cube; tighten `height_threshold`     |
| Pick a USD object (mug, cylinder)             | yes\*      | `objects.<name>.type: usd` — needs the USD file  |
| Push without grasping                         | no         | no grasp/release phases                          |
| Pour from a held cup                          | no         | needs orientation control beyond yaw             |
| Multi-step (stack three cubes)                | no         | one pickable / one target assumed                |
| Multi-object scene (3+ objects)               | no         | role system assumes one of each                  |

For the "no" rows, add one new template under `tasks/_runtime_<archetype>/`
and select it via the YAML `template:` key. Variants of *that* template
are again YAML-only.

### Robot envelope notes

The Epson T3-401 SCARA has a 400 mm planar reach (J1 + J2 = 225 + 175 mm)
and a 150 mm Z stroke (the URDF stretches this slightly to 200 mm for
sim convenience). When designing spawn ranges in a new task YAML, keep
each object centroid within radius 0.05–0.40 m of the base origin (the
inner singularity is at full elbow fold). The vertical workspace is
roughly z=0.08 to z=0.28 above the table at home pose.

## Project layout

```text
src/arm_act/
  config.py                   defaults + task overlay loader
  cli/                        thin Python entry points used by scripts/
    teleop.py / mimic.py / annotate.py / oracle.py / smoke.py
    __init__.py               isaaclab_script() + register_tasks() helpers
  training/
    defaults.yaml             shared hyperparams
    train_act.py              training entry point
    act_policy.py             ACT model + ACTPolicy inference wrapper
    dataset.py                HDF5-backed action-chunking dataset
  eval/
    rollout.py                sim rollouts of a trained checkpoint
    common.py                 video / summary writers + logging setup
  tasks/
    __init__.py               scans tasks/*.yaml, builds + registers all gym ids
    _runtime/                 default parametric template
      env_cfg.py / base_env_cfg.py / robot_cfg.py
      mdp.py / events.py
      mimic_env.py / mimic_env_cfg.py
      oracle.py / smoke.py
    _runtime_jar/             jar-insertion sibling template
      __init__.py / env_cfg.py / mimic_env.py / mimic_env_cfg.py
      oracle.py               extends the state machine with align/insert/retract
    pick_plant_out.yaml       phase 1: pick plant out of jar, place on tray
    put_plant_back.yaml       phase 2: pick plant from tray, insert into jar

assets/t3_401_simple_gripper/
  t3_401_simple_gripper.urdf  hand-written T3-401 + parallel-jaw URDF
  t3_401_simple_gripper.usd   layered USD entry point (after setup.sh)
  configuration/              physics + visual sublayers from the URDF importer

scripts/                      one-line shell wrappers (see scripts/README.md)
docs/
  data_format.md              HDF5 schema reference
tests/
  test_config_loader.py
  test_training_config.py
  test_eval_helpers.py
```

## Design notes

- **Per-camera ResNet18 backbones** (ImageNet-init), 7×7 spatial features
  → 1×1 conv → `hidden_dim`. State is one extra token via a `Linear`
  projection. Standard PyTorch transformer encoder + decoder; learned
  action queries.
- **No CVAE.** Faithful to the original ACT architecture *minus* the
  latent-z encoder — empirically optional for short-horizon single-task
  behavior cloning, and removing it cut ~150 lines.
- **Checkpoint format.** `model.pt` (state dict) + `config.json`
  (`ACTConfig` dataclass) + `norm_stats.json` (action/state min/max +
  mean/std). `load_policy(ckpt_dir)` reconstructs everything.
- **Auto-registered gym ids.** `tasks/__init__.py` walks every
  `tasks/*.yaml` at import time, builds the env/mimic classes from each
  spec via the named template, and calls `gym.register` for both the env
  and its mimic variant.

## Scope

Out of scope for this repo:

- Real-robot transfer
- Training from scratch (the trainer assumes ACT for behavior cloning)
- Hyperparameter sweeps
- Multi-task generalization in a single checkpoint

## Development

```bash
source ~/arm-act-venv/bin/activate
pip install -e ".[dev]"
pytest                                        # config + eval-helper tests
ruff check src tests scripts
ruff format src tests
```

## Further reading

- [`scripts/README.md`](scripts/README.md) — script reference table
- [`docs/data_format.md`](docs/data_format.md) — HDF5 demo schema
- ACT paper: Zhao et al., *Learning Fine-Grained Bimanual Manipulation
  with Low-Cost Hardware*, RSS 2023
