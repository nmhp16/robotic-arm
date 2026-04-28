# arm-vla

Single-task imitation-learning pipeline for UR5 + parallel-jaw pick-and-place
in Isaac Lab. Sim-only.

Pipeline: **keyboard teleop → Isaac Lab Mimic augmentation → ACT fine-tune
→ sim rollout eval (mp4 per episode)**.

Two tasks are shipped:
- `pick_place` — pick the blue cube, place on the green target pad.
- `stack` — pick the blue cube, stack on top of the red cube. *(UR10 variant)*

## Why ACT?

A small task-specific Action Chunking Transformer (~27 M params) trains
in 30–60 min on a single GPU and runs inference in milliseconds — vs. an
OpenVLA-style 7 B model that takes hours to fine-tune and seconds per
inference. For a single sim task with a few hundred demos, ACT is the
right tool. The repo previously targeted OpenVLA; that path is gone.

## Requirements

- NVIDIA GPU with CUDA 12.8+ (tested on GB10 / DGX Spark, aarch64)
- Isaac Lab 2.3.2 (expected at `~/IsaacLab`; override via `ISAACLAB=...`)
- Python 3.12

## Environments

Two Python environments, separated to keep Isaac Sim's bundled torch from
clashing with the training stack.

| Env              | Location                          | Used for                                      |
|------------------|-----------------------------------|-----------------------------------------------|
| Isaac Lab python | `~/IsaacLab/isaaclab.sh -p`       | teleop, Mimic augmentation, eval, smoke       |
| training venv    | `$ARM_VLA_VENV` (default `/home/ketchup-core/arm-vla-venv`) | ACT training |

The training venv lives on local NVMe — putting it on the FUSE-mounted
project directory has caused intermittent `EPERM` under the heavy
concurrent imports a torch+vision stack does at startup.

### Training venv

```bash
python3.12 -m venv /home/ketchup-core/arm-vla-venv
source /home/ketchup-core/arm-vla-venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[ml]"
```

### Isaac Lab env

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -e ".[sim]"
```

## Usage

```bash
# Scene preview (visible or headless-with-video)
./scripts/smoke.sh                                # headless sanity check
./scripts/smoke.sh --visible --random --steps 200 # GUI, random motion

# Full pipeline
./scripts/teleop.sh --num-demos 15            # keyboard teleop, record HDF5
./scripts/annotate.sh                         # add datagen_info for mimic
./scripts/mimic.sh --num-demos 500            # curobo-based augmentation
./scripts/train.sh                            # train ACT (~30-60 min)
./scripts/eval.sh --checkpoint checkpoints/act-ur5-pickplace/final
```

Eval writes per-episode mp4 videos + `summary.json` into
`eval/runs/<timestamp>/`.

## Configuration

Almost everything you'd tune lives in one file:
**`src/arm_vla/training/config.yaml`**.

```yaml
data:
  hdf5_path: data/raw/demos.hdf5     # or data/augmented/demos.hdf5 after mimic
  camera_keys: [table_cam, wrist_cam]
policy:
  chunk_size: 50         # actions per forward pass (open-loop chunk replay at eval)
  hidden_dim: 256
  ...
training:
  max_steps: 50000
  batch_size: 64
  learning_rate: 1.0e-4
```

CLI overrides for ad-hoc runs (don't have to edit the YAML):

```bash
./scripts/train.sh --max-steps 1000 --batch-size 32
```

## Layout

```
src/arm_vla/
  tasks/ur5_pick_place/       UR5 + simple-gripper pick-and-place env
  tasks/ur10_pick_place/      UR10 + suction pick-and-place env (legacy)
  tasks/ur10_stack/           UR10 stack env (legacy)
  datagen/                    Mimic env cfg + runtime + scripted oracle
  teleop/                     Keyboard recording helpers
  training/
    config.yaml               One config to rule them all
    dataset.py                HDF5-backed action-chunking dataset
    act_policy.py             ACT model + ACTPolicy wrapper for inference
    train_act.py              Training entry point
  eval/
    rollout.py                Sim rollouts of a trained checkpoint
    common.py                 Task registry, video/summary writers
scripts/                      CLI wrappers (see scripts/README.md)
```

## Design notes

- **No RLDS / TFDS conversion step.** The trainer reads the Mimic-augmented
  HDF5 directly. Action chunks are sampled lazily, so memory stays
  bounded even for thousand-demo runs.
- **Per-camera ResNet18 backbones** (ImageNet-init), 7×7 spatial features
  → 1×1 → hidden_dim. State is one extra token via a Linear projection.
  Standard PyTorch transformer encoder + decoder; learned action queries.
- **No CVAE.** Faithful to the original ACT architecture *minus* the
  latent-z encoder — empirically optional for short-horizon single-task
  BC, and removing it cut ~150 lines of code.
- **Checkpoint format.** `model.pt` (state dict) + `config.json`
  (`ACTConfig` dataclass) + `norm_stats.json` (action/state min/max +
  mean/std). `load_policy(ckpt_dir)` reconstructs everything; no separate
  base-model download needed at eval time.
- **Isaac Lab rather than raw Isaac Sim.** Mimic, teleop device dispatch,
  and the UR assets are already plumbed through manager-based envs.
- **Two Python environments.** Isaac Lab ships a torch build that does
  not mix well with our training-side requirements; separating them is
  cheaper than debugging clashes.

## Scope

Explicitly out of scope: real-robot transfer, training from scratch,
hyperparameter sweeps, multi-task generalization.
