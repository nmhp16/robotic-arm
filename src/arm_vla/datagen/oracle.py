"""Scripted oracle demo collection for UR5 + 2F-85 pick-and-place.

Replaces keyboard teleop with a deterministic state machine that reads
ground-truth cube/target positions from the env and drives the arm through
fixed waypoints. Writes the same HDF5 format as Isaac Lab's
``record_demos.py`` (via ``ActionStateRecorderManagerCfg``), so everything
downstream (mimic augmentation, RLDS convert, OpenVLA fine-tune) consumes
the output identically.

State machine per episode:
  HOVER    → TCP at cube.xy, cube.z + HOVER_HEIGHT, gripper open
  DESCEND  → TCP at cube.xy, cube.z + GRASP_Z_OFFSET
  GRASP    → hold N steps with gripper=close
  LIFT     → TCP at cube.xy, LIFT_HEIGHT
  MOVE     → TCP at target.xy, LIFT_HEIGHT
  PLACE    → TCP at target.xy, target.z + PLACE_Z_OFFSET
  RELEASE  → gripper=open, hold until success termination

    ./scripts/oracle.sh --num-demos 15
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import traceback
from enum import Enum

from isaaclab.app import AppLauncher


# Waypoint geometry (meters). Blue block is 4 cm cube with centroid ~2 cm
# above the table. TCP is between fingertips — targeting cube centroid
# puts the fingers straddling top-to-bottom.
HOVER_HEIGHT: float = 0.15       # m above cube centroid before descent
GRASP_Z_OFFSET: float = 0.0      # m relative to cube centroid
LIFT_HEIGHT: float = 0.25        # m above table during transport
PLACE_Z_OFFSET: float = 0.035    # m above target pad centroid at release

# Per-step clamp matches the IK-rel action scale. Oracle advances at max
# speed; Mimic augmentation preserves timing.
MAX_DXY: float = 0.05
MAX_DZ: float = 0.05

# Default tolerances (loose); DESCEND/PLACE override with tight z.
XY_REACH_TOL: float = 0.020      # m — "waypoint reached" in xy
Z_REACH_TOL: float = 0.015       # m — default z tolerance
Z_REACH_TOL_TIGHT: float = 0.008 # m — used for DESCEND and PLACE so the
                                  # arm doesn't leave the gripper hovering
                                  # above its target
GRASP_HOLD_STEPS: int = 15       # gripper=close + motion pause
RELEASE_HOLD_STEPS: int = 5      # gripper=open, then wait for env success

# Safety caps.
MAX_STEPS_PER_EPISODE: int = 400
NUM_SUCCESS_STEPS: int = 10       # env's cube_on_target must hold this many
MAX_EPISODE_ATTEMPTS_PER_DEMO: int = 5  # cap retries when grasps miss


class Phase(Enum):
    HOVER = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    MOVE = 5
    PLACE = 6
    RELEASE = 7


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-demos", type=int, default=15)
    p.add_argument(
        "--dataset-file",
        type=pathlib.Path,
        default=pathlib.Path("data/raw/demos.hdf5"),
    )
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import gymnasium as gym
        import numpy as np
        import torch

        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.managers.recorder_manager import DatasetExportMode

        import arm_vla.tasks.ur5_pick_place  # noqa: F401  registers gym id
        from arm_vla.tasks.ur5_pick_place.pick_place_ur5_env_cfg import UR5PickPlaceEnvCfg

        output_dir = str(args.dataset_file.parent.resolve())
        output_file_name = args.dataset_file.stem
        os.makedirs(output_dir, exist_ok=True)

        cfg = UR5PickPlaceEnvCfg()
        cfg.scene.num_envs = 1
        # Run until the env's own success termination fires, not the time cap.
        cfg.terminations.time_out = None
        cfg.observations.policy.concatenate_terms = False
        cfg.recorders = ActionStateRecorderManagerCfg()
        cfg.recorders.dataset_export_dir_path = output_dir
        cfg.recorders.dataset_filename = output_file_name
        cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

        env = gym.make("Isaac-PickPlace-UR5-IK-Rel-v0", cfg=cfg).unwrapped
        device = env.device

        success_term = cfg.terminations.success
        exported = 0
        episode_idx = 0

        max_attempts = args.num_demos * MAX_EPISODE_ATTEMPTS_PER_DEMO
        while exported < args.num_demos and episode_idx < max_attempts:
            episode_idx += 1
            obs, _ = env.reset()

            cube_pos = obs["policy"]["cube_pos"][0].cpu().numpy()
            target_pos = obs["policy"]["target_pos"][0].cpu().numpy()
            waypoints = _plan(cube_pos, target_pos)

            phase = Phase.HOVER
            hold_counter = 0
            success_step_count = 0
            succeeded = False

            prev_phase = None
            lift_xy_ref = None      # TCP xy frozen at LIFT entry
            grasp_cube_pos = None   # cube pose frozen at GRASP entry, so
                                    # asymmetric finger closure can't pull
                                    # the waypoint around mid-grasp
            for step in range(args.max_steps):
                tcp = obs["policy"]["eef_pos"][0].cpu().numpy()
                cube_now = obs["policy"]["cube_pos"][0].cpu().numpy()
                target_now = obs["policy"]["target_pos"][0].cpu().numpy()
                gripper_rad = float(obs["policy"]["gripper_pos"][0, 0].cpu())

                if phase is Phase.GRASP and grasp_cube_pos is None:
                    grasp_cube_pos = cube_now.copy()
                if phase is Phase.LIFT and lift_xy_ref is None:
                    lift_xy_ref = tcp[:2].copy()

                # Pre-GRASP phases track the live cube so small nudges
                # during descent don't strand the approach. Once we hit
                # GRASP, freeze the cube reference and let the jaws close
                # on whatever's actually there.
                cube_for_plan = grasp_cube_pos if grasp_cube_pos is not None else cube_now
                waypoints = _plan_dynamic(
                    cube_for_plan, target_now, tcp, lift_xy_ref, phase
                )

                wp_pos, gripper_open = waypoints[phase]
                reached = _reached(tcp, wp_pos, phase)
                action = _compute_action(tcp, wp_pos, gripper_open)

                if phase is not prev_phase or step % 5 == 0:
                    print(
                        f"  ep{episode_idx:>2d} t{step:>3d} {phase.name:<7s} "
                        f"tcp=[{tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}] "
                        f"cube=[{cube_now[0]:+.3f},{cube_now[1]:+.3f},{cube_now[2]:+.3f}] "
                        f"act_grip={action[6]:+.1f} obs_grip={gripper_rad:.2f} "
                        f"reached={reached}",
                        flush=True,
                    )
                prev_phase = phase

                if phase in (Phase.GRASP, Phase.RELEASE):
                    hold_counter += 1
                    if phase is Phase.GRASP and hold_counter >= GRASP_HOLD_STEPS:
                        phase = Phase.LIFT
                        hold_counter = 0
                    # For RELEASE we keep sending gripper=open until the env
                    # success termination fires below.
                elif reached:
                    phase = _next_phase(phase)
                    hold_counter = 0

                action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                obs, _, terminated, truncated, info = env.step(action_t)

                # The env's `success` termination fires on a single step of
                # `cube_on_target` with gripper open. That's the completion
                # signal we want — no need to wait for NUM_SUCCESS_STEPS of
                # accumulated success. When it fires, export the episode
                # before the env auto-resets.
                tm = env.termination_manager
                if bool(tm.get_term("success")[0]):
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])
                    succeeded = True
                    exported += 1
                    break

                if bool(terminated[0]) or bool(truncated[0]):
                    flags = {
                        name: bool(tm.get_term(name)[0])
                        for name in tm.active_terms
                    }
                    print(
                        f"  ep{episode_idx:>2d} ENDED at t={step} phase={phase.name} "
                        f"flags={flags}",
                        flush=True,
                    )
                    break

            tag = "SUCCESS" if succeeded else "FAIL"
            print(
                f"episode {episode_idx:>3d}: {tag:<7s} "
                f"({exported}/{args.num_demos} exported, final phase={phase.name})",
                flush=True,
            )

        print(f"\nwrote {exported} successful episodes to {args.dataset_file}", flush=True)
        try:
            env.close()
        except Exception as close_exc:
            print(f"env.close() raised (ignored): {close_exc}", flush=True)

    except Exception:
        traceback.print_exc()
        return 1
    finally:
        app.close()

    return 0


def _plan(cube_pos, target_pos) -> dict:
    """Initial waypoint table computed at reset. Superseded each step by
    ``_plan_dynamic``; kept for the first iteration before the state machine
    has run."""
    return {
        Phase.HOVER:   (_xyz(cube_pos[0],   cube_pos[1],   cube_pos[2] + HOVER_HEIGHT), True),
        Phase.DESCEND: (_xyz(cube_pos[0],   cube_pos[1],   cube_pos[2] + GRASP_Z_OFFSET), True),
        Phase.GRASP:   (_xyz(cube_pos[0],   cube_pos[1],   cube_pos[2] + GRASP_Z_OFFSET), False),
        Phase.LIFT:    (_xyz(cube_pos[0],   cube_pos[1],   LIFT_HEIGHT),                   False),
        Phase.MOVE:    (_xyz(target_pos[0], target_pos[1], LIFT_HEIGHT),                   False),
        Phase.PLACE:   (_xyz(target_pos[0], target_pos[1], target_pos[2] + PLACE_Z_OFFSET), False),
        Phase.RELEASE: (_xyz(target_pos[0], target_pos[1], target_pos[2] + PLACE_Z_OFFSET), True),
    }


def _plan_dynamic(cube_pos, target_pos, tcp, lift_xy_ref, phase: Phase) -> dict:
    """Per-step waypoint table with lateral-drift prevention.

    Pre-grasp phases track the live cube so the approach corrects for any
    small cube nudges from the descent. Post-grasp phases freeze xy to
    either the TCP-at-LIFT-entry (LIFT) or the target pose (MOVE/PLACE/
    RELEASE) — the cube rides the arm, so re-reading cube_pos here creates
    a feedback loop that drags it out of the jaws."""
    # Fallback to current TCP xy if LIFT reference hasn't been captured yet.
    lift_xy = lift_xy_ref if lift_xy_ref is not None else tcp[:2]
    return {
        Phase.HOVER:   (_xyz(cube_pos[0],   cube_pos[1],   cube_pos[2] + HOVER_HEIGHT), True),
        Phase.DESCEND: (_xyz(cube_pos[0],   cube_pos[1],   cube_pos[2] + GRASP_Z_OFFSET), True),
        Phase.GRASP:   (_xyz(cube_pos[0],   cube_pos[1],   cube_pos[2] + GRASP_Z_OFFSET), False),
        Phase.LIFT:    (_xyz(float(lift_xy[0]), float(lift_xy[1]), LIFT_HEIGHT),          False),
        Phase.MOVE:    (_xyz(target_pos[0], target_pos[1], LIFT_HEIGHT),                   False),
        Phase.PLACE:   (_xyz(target_pos[0], target_pos[1], target_pos[2] + PLACE_Z_OFFSET), False),
        Phase.RELEASE: (_xyz(target_pos[0], target_pos[1], target_pos[2] + PLACE_Z_OFFSET), True),
    }


def _xyz(x: float, y: float, z: float):
    import numpy as np
    return np.array([x, y, z], dtype=np.float32)


def _next_phase(phase: Phase) -> Phase:
    order = [
        Phase.HOVER, Phase.DESCEND, Phase.GRASP,
        Phase.LIFT, Phase.MOVE, Phase.PLACE, Phase.RELEASE,
    ]
    i = order.index(phase)
    return order[min(i + 1, len(order) - 1)]


def _reached(tcp, wp_pos, phase: Phase) -> bool:
    import numpy as np
    dxy = float(np.linalg.norm(tcp[:2] - wp_pos[:2]))
    dz = abs(float(tcp[2] - wp_pos[2]))
    # Tighten z at descent-to-contact phases so the gripper doesn't close
    # on empty air 1 cm above the cube.
    z_tol = Z_REACH_TOL_TIGHT if phase in (Phase.DESCEND, Phase.PLACE) else Z_REACH_TOL
    return dxy < XY_REACH_TOL and dz < z_tol


def _compute_action(tcp, wp_pos, gripper_open: bool):
    import numpy as np

    dx = float(np.clip(wp_pos[0] - tcp[0], -MAX_DXY, MAX_DXY))
    dy = float(np.clip(wp_pos[1] - tcp[1], -MAX_DXY, MAX_DXY))
    dz = float(np.clip(wp_pos[2] - tcp[2], -MAX_DZ, MAX_DZ))
    # Isaac Lab's BinaryJointAction convention (see binary_joint_actions.py):
    # positive action = OPEN command, negative = CLOSE. The inverted
    # convention from the earlier opus_agent.py / rollout.py was a bug.
    gripper_cmd = 1.0 if gripper_open else -1.0

    return np.array(
        [dx, dy, dz, 0.0, 0.0, 0.0, gripper_cmd],
        dtype=np.float32,
    )


if __name__ == "__main__":
    sys.exit(main())
