# arm-vla

OpenVLA fine-tuning pipeline for a UR10 + long-suction pick-and-place task
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
./scripts/smoke.sh --visible --random --steps 200
                                           # open GUI, drive the arm
./scripts/teleop.sh --num-demos 15         # keyboard teleop, record demos
./scripts/mimic.sh --num-demos 500         # augment via curobo
./scripts/convert.sh                       # HDF5 → RLDS TFDS
./scripts/train.sh                         # OpenVLA LoRA fine-tune
./scripts/eval.sh --checkpoint checkpoints/openvla-ur10-pickplace-lora/final
```

### Eval failure analysis

`eval.sh` classifies every failed episode into one of
`collision / grasp_slip / drop / misplacement / drift / other` and writes
`failure_analysis.json` + a histogram into the run directory. The classifier
shells out to the `claude` CLI, so it uses your existing Claude Code auth
(`claude login`) — no `ANTHROPIC_API_KEY` required. If `claude` is not on
your `PATH`, failures are labeled `other` and classification is skipped.
Disable entirely with `--no-classify`.

## Layout

```
src/arm_vla/
  tasks/ur10_pick_place/     Isaac Lab env (gym id Isaac-PickPlace-UR10-IK-Rel-v0)
  datagen/                   Mimic env cfg + runtime
  data/rlds_convert.py       HDF5 → RLDS TFDS builder
  training/                  OpenVLA LoRA fine-tune
  eval/rollout.py            sim rollouts of a fine-tuned checkpoint
scripts/                     CLI wrappers
```

## Design notes

- **Isaac Lab rather than raw Isaac Sim.** Mimic, teleop device dispatch,
  and the UR assets are already plumbed through manager-based envs.
- **UR10 + long suction gripper.** Isaac Lab ships this as a pre-built
  config (`UR10_LONG_SUCTION_CFG`) whose USD has a pre-authored
  `SurfaceGripper` schema. The UR5e USD on Nucleus does not, and the
  Robotiq 2F-85 variant spawns a second articulation under the robot
  prim that Isaac Lab rejects. UR10 is the shortest working path; the
  pipeline itself is robot-agnostic and swappable.
- **Two Python environments.** Isaac Lab ships a torch build that does
  not mix well with OpenVLA's requirements; separating them is cheaper
  than debugging clashes.
- **Runtime registration with `OXE_DATASET_CONFIGS`** rather than a fork
  of upstream `openvla`. Schema drift surfaces as a `KeyError` at
  data-loader construction.
- **IK-relative Δpose on `ee_link`, body offset at the suction TCP.**
  Keeps the action interpretable and matches OpenVLA's training distribution.
- **Suction gripper forces CPU physics** (`self.device = "cpu"`) as of
  Isaac Lab 2.3.2. Acceptable for Mimic augmentation and eval; single-
  env throughput is the bottleneck, not headcount.

## Scope

Explicitly out of scope: real-robot transfer, multi-task training,
training from scratch, hyperparameter sweeps.
