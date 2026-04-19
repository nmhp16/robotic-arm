# arm-vla

OpenVLA fine-tuning pipeline on a UR5e (Robotiq 2F-85) pick-and-place task, sim-only via Isaac Lab.

Full design in [`PLAN.md`](./PLAN.md).

## Status

All phases **built**, none yet **validated**. Every script parses but nothing
has been executed end-to-end (Isaac Sim boot requires the user to drive it).

Suggested validation order (see `PLAN.md` for what each phase does):

1. `./scripts/smoke.sh`               — env loads + steps cleanly, obs shapes print
2. `./scripts/teleop.sh --num-demos 2`  — record a couple of demos, confirm HDF5 shape
3. `./scripts/teleop.sh --num-demos 15` — real data collection
4. `./scripts/mimic.sh --num-demos 50`  — try augmentation at small scale first
5. `./scripts/mimic.sh --num-demos 500` — full augmentation
6. `./scripts/convert.sh`             — produces TFDS under `data/rlds/`
7. `./scripts/train.sh`               — expect bumps on first run (aarch64, OXE registry)
8. `./scripts/eval.sh --checkpoint checkpoints/.../final`

## Platform

- NVIDIA DGX Spark (GB10, aarch64, CUDA 13)
- Isaac Lab 2.3.2 at `~/IsaacLab`
- Python 3.10+

## Two Python environments

This project splits across two envs on purpose:

| Env | Where | Used for |
|---|---|---|
| `isaaclab` (bundled) | `~/IsaacLab/_isaac_sim/python.sh` | Teleop data collection, mimic augmentation, eval rollouts |
| `training` (fresh venv) | `./.venv` | RLDS conversion, OpenVLA LoRA fine-tune |

Mixing torch versions between Isaac Sim's bundled one and OpenVLA's requirements is the main foot-gun — keeping them separate avoids it.

### Install (training env)

```bash
# aarch64 + CUDA 13 torch (DGX Spark)
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[ml,dev]"
```

`flash-attn` intentionally omitted — it fails to build on aarch64 as of Apr 2026. The training script falls back to `attn_implementation="sdpa"`.

### Install (sim env)

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -e ".[sim]"
```

## Commands (not yet implemented — see PLAN.md)

```bash
./scripts/teleop.sh                     # keyboard teleop, collect 15 demos
./scripts/mimic.sh --num-demos 500      # augment
./scripts/convert.sh                    # HDF5 → RLDS
./scripts/train.sh                      # OpenVLA LoRA fine-tune
./scripts/eval.sh                       # rollouts in sim, log success rate
```

## Repo layout

```
src/arm_vla/
  assets/ur5e_cfg.py       UR5e + Robotiq 2F-85 ArticulationCfg (not in Isaac Lab)
  tasks/ur5_pick_place/    Isaac Lab env
  teleop/                  keyboard → HDF5
  datagen/                 mimic config + augmentation
  data/                    HDF5 → RLDS
  training/                LoRA fine-tune
  eval/                    sim rollouts
scripts/                   thin CLI wrappers
```

### UR5e USD fallback

`assets/ur5e_cfg.py` points at NVIDIA's Nucleus server for the UR5e USD
(`{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd`). If that path
404s on your network, convert the URDF bundled with Isaac Sim instead:

```bash
mkdir -p assets/ur5e
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/tools/convert_urdf.py \
  ~/isaac/env_isaacsim/lib/python3.12/site-packages/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur5e/ur5e.urdf \
  assets/ur5e/ur5e.usd
```

Then edit `_NUCLEUS_UR5E_USD` in `src/arm_vla/assets/ur5e_cfg.py` to a local path.
