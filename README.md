# arm-vla

OpenVLA fine-tuning pipeline for UR10 + long-suction pick-and-place tasks
in Isaac Lab. Sim-only.

Pipeline: keyboard teleop → Isaac Lab Mimic augmentation → RLDS/TFDS →
OpenVLA LoRA fine-tune → sim rollout eval (mp4 per episode).

Two tasks are shipped:
- `pick_place` — pick the blue cube, place on the green target pad.
- `stack` — pick the blue cube, stack on top of the red cube.

## Requirements

- NVIDIA GPU with CUDA 12.8+ (tested on GB10 / DGX Spark, aarch64)
- Isaac Lab 2.3.2 (expected at `~/IsaacLab`; override via `ISAACLAB=...`)
- Python 3.10+

## Environments

Two Python environments, separated to keep Isaac Sim's bundled torch from
clashing with OpenVLA's.

| Env              | Location                          | Used for                                      |
|------------------|-----------------------------------|-----------------------------------------------|
| Isaac Lab python | `~/IsaacLab/isaaclab.sh -p`       | teleop, Mimic augmentation, eval, zero-shot   |
| training venv    | `./.venv`                         | RLDS conversion, OpenVLA LoRA fine-tune       |

### Training venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[ml,dev]"
```

`flash-attn` is intentionally omitted — it does not build on aarch64 as
of this writing. Training uses `attn_implementation="sdpa"`.

### Isaac Lab env

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -e ".[sim]"
```

## Usage

```bash
# scene preview (visible or headless-with-video)
./scripts/smoke.sh                                # headless sanity check
./scripts/smoke.sh --visible --random --steps 200 # GUI, random motion
./scripts/smoke_stack.sh --random --video-out media/stack.mp4  # stack task

# pretrained baseline (no fine-tune, should fail on our robot)
./scripts/zeroshot.sh --task pick_place --episodes 3
./scripts/zeroshot.sh --task stack --episodes 3

# full pipeline
./scripts/teleop.sh --num-demos 15            # keyboard teleop, record HDF5
./scripts/mimic.sh --num-demos 500            # curobo-based augmentation
./scripts/convert.sh                          # HDF5 → RLDS TFDS
./scripts/train.sh                            # OpenVLA LoRA fine-tune
./scripts/eval.sh --checkpoint checkpoints/openvla-ur10-pickplace-lora/final --task pick_place
./scripts/eval.sh --checkpoint <ckpt> --task stack
```

Eval writes per-episode mp4 videos + `summary.json` into
`eval/runs/<timestamp>/`.

## Layout

```
src/arm_vla/
  tasks/ur10_pick_place/     pick-and-place env (gym id Isaac-PickPlace-UR10-IK-Rel-v0)
  tasks/ur10_stack/          stack env (gym id Isaac-Stack-UR10-IK-Rel-v0)
  datagen/                   Mimic env cfg + runtime
  data/rlds_convert.py       HDF5 → RLDS TFDS builder
  training/                  OpenVLA LoRA fine-tune
  eval/rollout.py            sim rollouts of a fine-tuned checkpoint
  eval/zeroshot.py           pretrained-baseline rollouts (no LoRA)
scripts/                     CLI wrappers
```

## Design notes

- **Isaac Lab rather than raw Isaac Sim.** Mimic, teleop device dispatch,
  and the UR assets are already plumbed through manager-based envs.
- **UR10 + long suction.** Isaac Lab ships `UR10_LONG_SUCTION_CFG` with a
  pre-authored `SurfaceGripper` schema — the shortest working path. The
  UR5e USD does not have the schema and its Robotiq 2F-85 variant spawns
  a second articulation under the robot prim that Isaac Lab rejects. The
  pipeline itself is robot-agnostic; swap `UR10_LONG_SUCTION_CFG` for a
  different arm when one's available.
- **Two Python environments.** Isaac Lab ships a torch build that does
  not mix well with OpenVLA's pinned requirements; separating them is
  cheaper than debugging clashes.
- **Runtime registration with `OXE_DATASET_CONFIGS`** rather than a fork
  of upstream `openvla`. Schema drift surfaces as a `KeyError` at
  data-loader construction.
- **Suction gripper forces CPU physics** (`self.device = "cpu"`) as of
  Isaac Lab 2.3.2. Acceptable for Mimic augmentation (few envs) and
  single-env eval.

## Scope

Explicitly out of scope: real-robot transfer, training from scratch,
hyperparameter sweeps.
