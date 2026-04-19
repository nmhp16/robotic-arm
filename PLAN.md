# Design

## Goal

End-to-end OpenVLA fine-tuning pipeline on a single pick-and-place task,
simulated in Isaac Lab. The repo targets pipeline correctness over policy
quality.

## Stack

| Component         | Choice                                                  |
|-------------------|---------------------------------------------------------|
| Simulator         | Isaac Lab 2.3.2 (Isaac Sim)                             |
| Robot             | UR5e + Robotiq 2F-85                                    |
| Task              | Single cube → target pad                                |
| Action space      | 6-DoF IK-relative Δpose + 1-D binary gripper (7-D)      |
| Observations      | wrist RGB + third-person RGB (224×224) + proprio state  |
| Teleop            | Isaac Lab `Se3Keyboard`                                 |
| Demo augmentation | `isaaclab_mimic` (curobo-based segment replay)          |
| Dataset format    | RLDS (TFDS), Open X-Embodiment feature schema           |
| Model             | `openvla/openvla-7b`                                    |
| Fine-tune method  | LoRA (rank 32) via `peft`                               |
| Eval              | Sim rollouts in the collection env                      |

## Pipeline

1. **Teleop** (`scripts/teleop.sh`). Isaac Lab's `record_demos.py` drives
   the env with keyboard SE(3) input and writes episodes to HDF5 in a
   format compatible with Mimic. Success detection via the env's own
   `cube_on_target` termination.

2. **Augmentation** (`scripts/mimic.sh`). Mimic segments each demo at the
   `grasp` / `place` boundary and replays subtask segments across
   randomized scenes using curobo motion planning. 10–20 hand-recorded
   demos expand to several hundred.

3. **RLDS conversion** (`scripts/convert.sh`). HDF5 episodes are read and
   emitted as a TFDS dataset matching Open X-Embodiment's feature schema:

   ```
   observation.image          (224, 224, 3) uint8    # third-person
   observation.wrist_image    (224, 224, 3) uint8    # wrist
   observation.state          (8,) float32           # eef_pos + eef_quat + gripper
   action                     (7,) float32           # Δxyz + Δrpy + gripper
   language_instruction       string
   is_first / is_last / is_terminal
   ```

4. **LoRA fine-tune** (`scripts/train.sh`). Action quantiles (q01/q99 per
   dim) are computed from the training split and stored alongside the
   checkpoint. The OXE config entry for the dataset is registered at
   runtime to avoid forking upstream `openvla`. `attn_implementation` is
   `sdpa` — `flash-attn` does not build on aarch64.

5. **Eval** (`scripts/eval.sh`). Loads base model + LoRA, opens the same
   env, runs N episodes, logs success rate and per-episode video.
   `predict_action` handles token→continuous decode and unnormalization
   using the stats file written during training.

## Design decisions

- **Isaac Lab rather than raw Isaac Sim.** Mimic, teleop device dispatch,
  and the UR assets are already plumbed through manager-based envs.

- **Parallel-jaw gripper rather than suction.** OpenVLA's pretraining data
  is dominated by parallel jaws, and the suction gripper path in Isaac Lab
  currently requires CPU physics.

- **Isaac Lab does not ship a UR5/UR5e `ArticulationCfg`**, only UR10/UR10e.
  `src/arm_vla/assets/ur5e_cfg.py` mirrors `UR10e_ROBOTIQ_2F_85_CFG` and
  points at NVIDIA's Nucleus UR5e USD with the `Robotiq_2f_85` gripper
  variant. A local-URDF fallback is documented in `README.md`.

- **Two Python environments.** Isaac Lab ships a torch build that does not
  mix well with OpenVLA's requirements; keeping them separate is cheaper
  than debugging clashes.

- **Runtime registration of the dataset with `OXE_DATASET_CONFIGS`** rather
  than forking upstream `openvla`. Upstream schema drift will surface as
  a `KeyError` at data-loader construction, which is easy to catch.

- **IK-relative Δpose on `tool0`, with a body offset placing the IK target
  at the fingertip center.** Keeps the action interpretable and matches
  OpenVLA's training distribution.

## Repository layout

```
robotic-arm/
├── PLAN.md
├── README.md
├── pyproject.toml
├── src/arm_vla/
│   ├── assets/ur5e_cfg.py                UR5e + 2F-85 ArticulationCfg
│   ├── tasks/ur5_pick_place/
│   │   ├── pick_place_env_cfg.py         scene, obs, terminations
│   │   ├── pick_place_ur5_env_cfg.py     robot, actions, cameras, events
│   │   ├── mdp.py                        task-specific obs + subtask terms
│   │   └── smoke.py                      headless sanity check
│   ├── datagen/                          Mimic env cfg + runtime
│   ├── data/rlds_convert.py              HDF5 → RLDS TFDS
│   ├── training/
│   │   ├── config.yaml                   hyperparameters
│   │   └── finetune_lora.py              LoRA fine-tune
│   └── eval/rollout.py                   sim rollouts + success rate
└── scripts/                              CLI wrappers
```

## Scope

Explicitly out of scope: real-robot transfer, multi-task training,
training from scratch, hyperparameter sweeps.
