# arm-vla

Imitation-learning pipeline for **UR5 + parallel-jaw** manipulation in
Isaac Lab. Sim-only, ACT (Action Chunking Transformer), one task per
checkpoint, multi-task by design — adding a new pick-and-place variant
is **edit one YAML file**.

```
keyboard teleop  →  isaaclab_mimic augmentation  →  ACT training  →  sim rollout eval
   (or oracle)                                                         (mp4 + summary.json)
```

Ships with one task — `pick_place` (pick the blue cube, place on the
green target). New tasks live under `src/arm_vla/tasks/<your_task>/` as a
single `task.yaml`; the runtime in `tasks/_runtime/` reads it and builds
the env, oracle, and mimic config dynamically.

## Why ACT?

A small task-specific Action Chunking Transformer (~30 M params) trains
in 30–60 min on a single GPU and runs inference in milliseconds. For a
single sim task with a few hundred demos that's the right tool — no need
to fine-tune a 7 B VLA per task.

## Requirements

- NVIDIA GPU with CUDA 12.8+ (tested on GB10 / DGX Spark, aarch64)
- Isaac Lab 2.3.2 (expected at `~/IsaacLab`; override via `ISAACLAB=...`)
- Python 3.12

## Environments

Two Python environments, separated to keep Isaac Sim's bundled torch from
clashing with the training stack.

| Env              | Location                                   | Used for                                       |
|------------------|--------------------------------------------|------------------------------------------------|
| Isaac Lab python | `~/IsaacLab/isaaclab.sh -p`                | teleop, oracle, mimic, annotate, eval, smoke   |
| training venv    | `$ARM_VLA_VENV` (default `~/arm-vla-venv`) | ACT training                                   |

The training venv lives on local NVMe — putting it on the FUSE-mounted
project directory has caused intermittent `EPERM` under the heavy
concurrent imports a torch+vision stack does at startup.

### Bootstrap

```bash
./scripts/setup.sh
```

Re-runnable; existing artifacts are reused.

## Usage

Every script defaults to `--task pick_place`. Pass `--task <name>` to
target a different one.

```bash
# Scene preview
./scripts/smoke.sh
./scripts/smoke.sh --visible --random --steps 200

# Full pipeline
./scripts/teleop.sh   --num-demos 15      # keyboard teleop
# or
./scripts/oracle.sh   --num-demos 15      # scripted oracle
./scripts/annotate.sh                     # add datagen_info for mimic
./scripts/mimic.sh    --num-demos 500     # curobo-based augmentation
./scripts/train.sh                        # train ACT (~30–60 min)
./scripts/eval.sh                         # rollout eval (writes mp4s)
```

## Configuration

Two YAML layers, deep-merged at load time:

```
src/arm_vla/training/defaults.yaml          ← shared hyperparams (rarely touched)
src/arm_vla/tasks/<task>/task.yaml          ← full task spec
```

`defaults.yaml` carries policy hyperparameters, the training schedule,
and eval defaults — things that apply to every task.

`task.yaml` carries everything task-specific: the gym id, instruction,
robot type, objects (color, size, mass, friction, spawn ranges),
cameras, success thresholds, the scripted-oracle waypoint heights, the
mimic subtask declarations, and the data + checkpoint paths.

CLI flags on `train.sh` / `eval.sh` override individual fields without
editing either YAML:

```bash
./scripts/train.sh --max-steps 1000 --batch-size 32
./scripts/eval.sh  --checkpoint checkpoints/pick_place/step_010000 --num-episodes 5
```

## Adding a new task — YAML only

For variants of the bundled pick-and-place archetype (different objects,
colors, sizes, target positions, instructions, success thresholds) you
**don't write any Python**:

```bash
cp -r src/arm_vla/tasks/pick_place src/arm_vla/tasks/pick_red_to_blue
$EDITOR src/arm_vla/tasks/pick_red_to_blue/task.yaml
# Edit:
#   task.name           → pick_red_to_blue
#   task.gym_id         → unique gym id
#   task.mimic_gym_id   → unique mimic gym id
#   task.instruction    → describe the new task in plain English
#   objects.cube.color  → e.g. [0.9, 0.1, 0.1]
#   objects.target.*    → new spawn range / USD if you want a different pad
#   data.*              → per-task data paths
#   training.output_dir → e.g. checkpoints/pick_red_to_blue
#   eval.checkpoint     → e.g. checkpoints/pick_red_to_blue/final

./scripts/oracle.sh   --task pick_red_to_blue --num-demos 15
./scripts/annotate.sh --task pick_red_to_blue
./scripts/mimic.sh    --task pick_red_to_blue
./scripts/train.sh    --task pick_red_to_blue
./scripts/eval.sh     --task pick_red_to_blue
```

The runtime in `tasks/_runtime/` reads your `task.yaml`, generates the
configclasses, registers the gym ids, and the same scripts work
unchanged.

For a genuinely new action archetype (stack two cubes, pour, push) you'd
add a sibling template under `tasks/_runtime/` — but variants of *that*
template would again be YAML-only.

## Layout

```
src/arm_vla/
  config.py                   defaults + task overlay loader
  cli/                        thin Python entry points used by scripts/
    teleop.py / mimic.py / annotate.py / oracle.py / smoke.py
  training/
    defaults.yaml             shared hyperparams
    train_act.py              training entry point
    act_policy.py             ACT model + ACTPolicy wrapper
    dataset.py                HDF5-backed action-chunking dataset
  eval/
    rollout.py                sim rollouts of a trained checkpoint
    common.py                 video/summary writers + logging setup
  tasks/
    __init__.py               scans tasks/*/task.yaml, builds + registers all gym ids
    _runtime/                 the parametric template (one Python set, all tasks share it)
      env_cfg.py / base_env_cfg.py / robot_cfg.py
      mdp.py / events.py
      mimic_env.py / mimic_env_cfg.py
      oracle.py / smoke.py
    pick_place/
      task.yaml               ← only file in here; full task spec
scripts/                      one-line shell wrappers (see scripts/README.md)
```

## Design notes

- **YAML-driven tasks.** Every parameter (object positions, colors, sizes,
  cameras, success thresholds, oracle heights, mimic subtasks) lives in
  `task.yaml`. The runtime reads it and builds Isaac Lab configclasses
  at import time; no per-task Python required for variants.
- **Two-layer config.** Shared hyperparams in `defaults.yaml`, per-task
  overrides in `tasks/<task>/task.yaml`. Deep-merged at load time. CLI
  flags win over both.
- **Auto-registered gym ids.** `tasks/__init__.py` walks every
  `tasks/*/task.yaml`, builds the env/mimic/cfg classes from the spec,
  and calls `gym.register` for each — drop in a new `task.yaml` and the
  next process import sees it.
- **No RLDS / TFDS conversion step.** The trainer reads the Mimic-augmented
  HDF5 directly. Action chunks are sampled lazily; memory stays bounded
  even for thousand-demo runs.
- **Per-camera ResNet18 backbones** (ImageNet-init), 7×7 spatial features
  → 1×1 → hidden_dim. State is one extra token via a Linear projection.
  Standard PyTorch transformer encoder + decoder; learned action queries.
- **No CVAE.** Faithful to the original ACT architecture *minus* the
  latent-z encoder — empirically optional for short-horizon single-task
  BC, and removing it cut ~150 lines.
- **Checkpoint format.** `model.pt` (state dict) + `config.json`
  (`ACTConfig` dataclass) + `norm_stats.json` (action/state min/max +
  mean/std). `load_policy(ckpt_dir)` reconstructs everything.

## Scope

Out of scope: real-robot transfer, training from scratch, hyperparameter
sweeps, multi-task generalization in a single checkpoint.
