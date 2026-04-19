# arm-vla

OpenVLA fine-tuning pipeline for a UR5e + Robotiq 2F-85 pick-and-place task
in Isaac Lab. Sim-only.

Pipeline: keyboard teleop → Isaac Lab Mimic augmentation → RLDS/TFDS →
OpenVLA LoRA fine-tune → sim rollout eval.

## Requirements

- NVIDIA GPU with CUDA 12.8+ (tested on GB10 / DGX Spark, aarch64)
- Isaac Lab 2.3.2 (expected at `~/IsaacLab`; override via `ISAACLAB=...`)
- Python 3.10+

## Environments

Two Python environments, separated to keep Isaac Sim's bundled torch from
clashing with OpenVLA's.

| Env              | Location                          | Used for                                 |
|------------------|-----------------------------------|------------------------------------------|
| Isaac Lab python | `~/IsaacLab/isaaclab.sh -p`       | teleop, Mimic augmentation, eval rollout |
| training venv    | `./.venv`                         | RLDS conversion, OpenVLA LoRA fine-tune  |

### Training venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[ml,dev]"
```

`flash-attn` is intentionally omitted — it does not build on aarch64 as of
this writing. The training script uses `attn_implementation="sdpa"`.

### Isaac Lab env

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -e ".[sim]"
```

## Usage

```bash
./scripts/smoke.sh                         # sanity-check the env loads
./scripts/teleop.sh --num-demos 15         # keyboard teleop, record demos
./scripts/mimic.sh --num-demos 500         # augment via curobo
./scripts/convert.sh                       # HDF5 → RLDS TFDS
./scripts/train.sh                         # OpenVLA LoRA fine-tune
./scripts/eval.sh --checkpoint checkpoints/openvla-ur5-pickplace-lora/final
```

See [`PLAN.md`](./PLAN.md) for the full design rationale.

## Layout

```
src/arm_vla/
  assets/ur5e_cfg.py         UR5e + Robotiq 2F-85 ArticulationCfg
  tasks/ur5_pick_place/      Isaac Lab env (gym id Isaac-PickPlace-UR5-IK-Rel-v0)
  datagen/                   Mimic env cfg + runtime
  data/rlds_convert.py       HDF5 → RLDS TFDS builder
  training/                  OpenVLA LoRA fine-tune
  eval/rollout.py            sim rollouts of a fine-tuned checkpoint
scripts/                     CLI wrappers
```

## UR5e USD fallback

`assets/ur5e_cfg.py` points at NVIDIA's Nucleus server for the UR5e USD. If
that path is unreachable, convert the URDF bundled with Isaac Sim:

```bash
mkdir -p assets/ur5e
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/tools/convert_urdf.py \
  ~/isaac/env_isaacsim/lib/python3.12/site-packages/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur5e/ur5e.urdf \
  assets/ur5e/ur5e.usd
```

Then point `_NUCLEUS_UR5E_USD` in `src/arm_vla/assets/ur5e_cfg.py` at the
local path.
