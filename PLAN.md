# OpenVLA Fine-Tuning Pipeline — Plan

**Goal:** Learn the end-to-end fine-tuning pipeline for a Vision-Language-Action model using a sim-only pick-and-place task. Outcome is a working pipeline, not a production policy.

**Platform:** DGX Spark (NVIDIA GB10, aarch64, 128 GB unified memory) + Isaac Lab 2.3.2 (already installed at `~/IsaacLab`).

## Design decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Simulator | Isaac Lab (Isaac Sim under the hood) | Installed; has teleop + mimic + UR assets built in |
| Robot | UR10 w/ Long Suction gripper | Isaac Lab's `ur10_gripper` stack config uses this — shortest path |
| Task | Pick single cube → drop in target zone | Simplest pick-and-place; reuses stack env plumbing |
| Action space | IK-relative 6-DoF pose + binary suction | Matches OpenVLA's 7-DoF action distribution |
| Cameras | Wrist (224×224 RGB) + third-person (224×224 RGB) | Matches OpenVLA's expected input |
| Teleop | Keyboard (`Se3KeyboardCfg` — built in) | Only hardware user has |
| Demo augmentation | `isaaclab_mimic` (curobo-based) | 10–20 teleop demos → 500–1000 augmented; teleop alone isn't enough for VLA fine-tuning |
| Dataset format | RLDS / TFDS | What OpenVLA's finetune script expects |
| Model | `openvla/openvla-7b` | Per user choice; GB10 can handle full fine-tune |
| Fine-tune method | LoRA (rank 32) | ~15 GB VRAM, trains in hours not days |
| Eval | Rollouts in same Isaac Lab env, log success rate + video | Closes the loop |
| ROS 2 | **Dropped** | Not needed for sim-only training |

## Known risks / caveats

- **aarch64 wheels.** PyTorch + transformers have ARM CUDA wheels on DGX Spark. `flash-attn` may need building from source; fallback is `attn_implementation="sdpa"` in the HF config (slower but works).
- **Suction + CPU sim.** The `UR10_LONG_SUCTION_CFG` currently forces `device="cpu"` in Isaac Lab's stack env. Mimic augmentation will be slower than GPU sim. Acceptable for a few hundred demos.
- **Action distribution drift.** OpenVLA was pretrained on a mix of Franka/UR/xArm data with parallel-jaw grippers. Our suction gripper is slightly off-distribution. Fine-tuning should bridge the gap but be aware.
- **Not real-robot.** Everything here is sim. No sim-to-real step in scope.

## Repository layout after cleanup

```
robotic-arm/
├── PLAN.md                           # this file
├── README.md                         # replaces SETUP.md + SIMULATION_GUIDE.md
├── pyproject.toml                    # replaces requirements.txt; split deps into groups
├── .gitignore                        # includes venv/, data/, checkpoints/
│
├── src/arm_vla/
│   ├── tasks/
│   │   └── ur10_pick_place/          # Isaac Lab env (cloned from ur10_gripper stack)
│   │       ├── __init__.py           # gym.register("Isaac-PickPlace-UR10-IK-Rel-v0")
│   │       ├── pick_place_env_cfg.py # base env cfg with IK-rel + cameras + 1 cube + target
│   │       └── mdp.py                # success = cube in target zone
│   │
│   ├── teleop/
│   │   └── collect.py                # keyboard teleop → HDF5 (isaaclab_mimic-compatible)
│   │
│   ├── datagen/
│   │   ├── mimic_env_cfg.py          # subclasses pick_place with subtask terms
│   │   └── generate.py               # wraps isaaclab.mimic.generate_dataset
│   │
│   ├── data/
│   │   ├── rlds_convert.py           # HDF5 → RLDS TFDS builder
│   │   └── schema.py                 # OpenVLA-compatible feature spec
│   │
│   ├── training/
│   │   ├── finetune_lora.py          # adapted from openvla/vla-scripts/finetune.py
│   │   └── config.yaml               # hyperparams
│   │
│   └── eval/
│       └── rollout.py                # load LoRA checkpoint → run N eps → log success
│
├── scripts/                          # thin CLI wrappers
│   ├── teleop.sh
│   ├── mimic.sh
│   ├── convert.sh
│   ├── train.sh
│   └── eval.sh
│
└── checkpoints/  (.gitignored)
    data/         (.gitignored)
```

## Execution phases

### Phase 1 — Cleanup (no new functionality)
Remove anything not on the training path so the repo tells a coherent story.

**Delete:**
- `ros_ws/` (all ROS 2 / MoveIt / URScript TCP / Gazebo launch files)
- `tests/control.py` (real-robot TCP control)
- `venv/` (committed 3.12 venv pointing to macOS `/Users/hp/.pyenv`)
- `graphify-out/` (auto-generated cache)
- `SETUP.md`, `SIMULATION_GUIDE.md` (ROS-era docs with hardcoded `/home/hp/` paths)
- `requirements.txt` (replaced by `pyproject.toml`)

**Keep:**
- `.git/`, `.gitignore` (extended)
- Nothing else from the old tree

**Add:**
- Updated `.gitignore` (venv, __pycache__, data/, checkpoints/, wandb/, .vscode/, .idea/)
- `pyproject.toml` with dependency groups: `core`, `ml`, `dev`
- Minimal `README.md` describing the pipeline and pointing at `PLAN.md`

### Phase 2 — Env scaffolding
Clone `~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/ur10_gripper/` into `src/arm_vla/tasks/ur10_pick_place/`. Modify:
- Scene: 1 cube (not 3) + a visual target zone (colored decal on the table).
- Observation: add `wrist_cam` and `table_cam` `TiledCameraCfg` at 224×224 RGB, feeding `RGBCameraPolicyCfg`.
- Termination: `success = cube_in_target_zone(threshold=0.05m)`.
- Action: keep existing IK-relative 6-DoF + `SurfaceGripperBinaryActionCfg`. That's 7-D total, matching OpenVLA.
- Events: randomize cube start pose + target zone pose per-episode.
- Register `Isaac-PickPlace-UR10-IK-Rel-v0` in `__init__.py`.

**Validation:** `python -m arm_vla.tasks.ur10_pick_place.smoke` — spawns env, steps with zero actions for 1 s, no errors.

### Phase 3 — Teleop + recording
Wrap `isaaclab.devices.keyboard.Se3Keyboard` into `arm_vla.teleop.collect`. Keyboard maps:
- `W/S/A/D/Q/E`: ±x, ±y, ±z (translation)
- `Z/X/C/V/R/F`: ±roll/pitch/yaw
- `Space`: toggle suction
- `N`: mark episode success + advance; `B`: discard + retry; `Esc`: quit

Record to `data/raw/demo_XXX.hdf5` with Isaac Lab's native schema so it drops into mimic without reformatting. Target: 15 successful demos.

### Phase 4 — Mimic augmentation
Create `src/arm_vla/datagen/mimic_env_cfg.py` — subclasses `PickPlaceEnvCfg` and adds the subtask annotation terms (object_grasped, object_in_target). Then invoke Isaac Lab's `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py` with our env id, input HDF5, target `num_demos=500`. This uses curobo motion planning to transplant segments across randomized scenes.

**Output:** `data/augmented/dataset.hdf5` with ~500 successful episodes.

### Phase 5 — RLDS conversion
Write `src/arm_vla/data/rlds_convert.py` matching OpenVLA's feature schema:
```python
{
    "observation": {
        "image": (224, 224, 3, uint8),        # third-person
        "wrist_image": (224, 224, 3, uint8),
        "state": (7,) float32,                 # eef xyz + rpy + gripper
    },
    "action": (7,) float32,                    # Δxyz + Δrpy + gripper
    "language_instruction": str,               # "put the blue block in the green zone"
    "is_first" / "is_last" / "is_terminal": bool,
}
```
Emit as a TFDS `DatasetBuilder`, register in the Open X-Embodiment-style TFDS registry locally. Split 90/10 train/val.

### Phase 6 — LoRA fine-tune
Adapt `openvla/vla-scripts/finetune.py`:
- Point `data_root_dir` at our TFDS dataset.
- `vla_path="openvla/openvla-7b"`, `lora_rank=32`, `batch_size=16`, `grad_accum=1`, `learning_rate=5e-4`, `max_steps=50_000` (check wall-clock, may reduce).
- On GB10 aarch64: if `flash-attn` install fails, set `attn_implementation="sdpa"` in model kwargs.
- Save action normalization stats alongside checkpoint (`dataset_statistics.json`) — required for inference unnormalization.
- Log to wandb (offline mode; user can sync later).

**Output:** `checkpoints/openvla-ur10-pickplace-lora/`.

### Phase 7 — Eval rollout
`src/arm_vla/eval/rollout.py`:
1. Load base model + LoRA adapters.
2. Spawn Isaac Lab env (same cfg as collection).
3. For each of N=50 episodes: reset env, get image + state + instruction, model forward, unnormalize action, step env, repeat until success/timeout.
4. Log: success rate, average time-to-success, per-episode video under `eval/runs/<ts>/`.

**Success bar:** >30% success rate demonstrates the pipeline works. Production quality isn't the goal.

## Commands (once built)

```bash
# one-time install
uv sync --all-groups

# collect 15 demos
./scripts/teleop.sh

# augment to 500
./scripts/mimic.sh --num-demos 500

# convert to RLDS
./scripts/convert.sh

# fine-tune (expect 6-12h on GB10 for 50k steps)
./scripts/train.sh

# evaluate (50 rollouts)
./scripts/eval.sh
```

## What's explicitly out of scope

- Real UR robot (only sim)
- ROS 2 integration (dropped)
- Training from scratch (always starts from `openvla/openvla-7b`)
- Multi-task training (single pick-and-place only)
- Sim-to-real transfer
- Hyperparameter sweeps
