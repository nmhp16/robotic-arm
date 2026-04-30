"""Scripted oracle for jar-insertion tasks (the `_runtime_jar` template).

Extends the default state machine with three new phases tailored for
descending into a confined cavity:

  HOVER -> DESCEND -> GRASP -> LIFT -> MOVE -> ALIGN -> INSERT
        -> RELEASE -> RETRACT

Differences from `_runtime/oracle.py`:

* ALIGN  -- after MOVE, hold xy at the target with a tighter tolerance
            (`align_xy_tol`) before starting the descent. Insertion only
            begins once the gripper is precisely centred over the
            opening; sloppy alignment wedges the stem against the rim.
* INSERT -- replaces PLACE. tcp z target is `target.z + insert_z_offset`
            where `insert_z_offset` is NEGATIVE (descend BELOW target).
* RETRACT -- after RELEASE, drive tcp back up to `retract_height` so the
            episode ends with the arm clear of the cavity. This matters
            for mimic: without retract, mimic's rollouts terminate with
            the arm wedged inside and learn trajectories that never exit.

Reuses everything else (env loading, recording, IK action shape) from
the default oracle.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)


class Phase(Enum):
    HOVER = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    MOVE = 5
    ALIGN = 6
    INSERT = 7
    RELEASE = 8
    RETRACT = 9


@dataclass(frozen=True)
class _OracleParams:
    hover_height: float
    grasp_z_offset: float
    lift_height: float
    insert_z_offset: float
    retract_height: float
    align_xy_tol: float
    max_dxy: float
    max_dz: float
    xy_reach_tol: float
    z_reach_tol: float
    z_reach_tol_tight: float
    grasp_hold_steps: int
    release_hold_steps: int
    max_steps_per_episode: int
    max_attempts_per_demo: int

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> _OracleParams:
        o = spec["oracle"]
        return cls(
            hover_height=float(o["hover_height"]),
            grasp_z_offset=float(o["grasp_z_offset"]),
            lift_height=float(o["lift_height"]),
            insert_z_offset=float(o["insert_z_offset"]),
            retract_height=float(o.get("retract_height", o["lift_height"])),
            align_xy_tol=float(o.get("align_xy_tol", o["xy_reach_tol"] * 0.5)),
            max_dxy=float(o["max_dxy"]),
            max_dz=float(o["max_dz"]),
            xy_reach_tol=float(o["xy_reach_tol"]),
            z_reach_tol=float(o["z_reach_tol"]),
            z_reach_tol_tight=float(o["z_reach_tol_tight"]),
            grasp_hold_steps=int(o["grasp_hold_steps"]),
            release_hold_steps=int(o["release_hold_steps"]),
            max_steps_per_episode=int(o["max_steps_per_episode"]),
            max_attempts_per_demo=int(o["max_episode_attempts_per_demo"]),
        )


def _parse_args(default_dataset: str, default_max_steps: int) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-demos", type=int, default=15)
    p.add_argument("--dataset-file", type=pathlib.Path, default=pathlib.Path(default_dataset))
    p.add_argument("--max-steps", type=int, default=default_max_steps)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video-out", type=pathlib.Path, default=None,
                   help="if set, write mp4(s) of each episode's table_cam + wrist_cam streams")
    p.add_argument("--video-fps", type=int, default=15)
    return p.parse_args()


def main(spec: dict[str, Any]) -> int:
    """Entry point invoked by ``arm_act.cli.oracle`` when the task spec
    sets ``template: _runtime_jar``."""
    # AppLauncher's carb handler swallows our INFO logs. Force a stream
    # handler on the root logger so phase progress reaches stderr.
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(h)
    root.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    params = _OracleParams.from_spec(spec)
    args = _parse_args(
        default_dataset=spec["data"]["raw_path"],
        default_max_steps=params.max_steps_per_episode,
    )

    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import gymnasium as gym
        import torch
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.managers.recorder_manager import DatasetExportMode

        import arm_act.tasks
        arm_act.tasks.register()

        gym_id = spec["task"]["gym_id"]
        env_cfg_spec = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
        env_cfg = _instantiate_env_cfg(env_cfg_spec)
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.observations.policy.concatenate_terms = False

        output_dir = str(args.dataset_file.parent.resolve())
        os.makedirs(output_dir, exist_ok=True)
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = args.dataset_file.stem
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

        env = gym.make(gym_id, cfg=env_cfg).unwrapped
        device = env.device

        exported = 0
        episode_idx = 0
        max_attempts = args.num_demos * params.max_attempts_per_demo

        record_video = args.video_out is not None
        video_table: list = []
        video_wrist: list = []

        while exported < args.num_demos and episode_idx < max_attempts:
            episode_idx += 1
            obs, _ = env.reset()

            phase = Phase.HOVER
            hold_counter = 0
            succeeded = False
            prev_phase = None
            lift_xy_ref = None
            grasp_pickable_pos = None

            for step in range(args.max_steps):
                tcp = obs["policy"]["eef_pos"][0].cpu().numpy()
                pickable_now = obs["policy"]["pickable_pos"][0].cpu().numpy()
                target_now = obs["policy"]["target_pos"][0].cpu().numpy()
                gripper_rad = float(obs["policy"]["gripper_pos"][0, 0].cpu())

                if phase is Phase.GRASP and grasp_pickable_pos is None:
                    grasp_pickable_pos = pickable_now.copy()
                if phase is Phase.LIFT and lift_xy_ref is None:
                    lift_xy_ref = tcp[:2].copy()

                pickable_for_plan = grasp_pickable_pos if grasp_pickable_pos is not None else pickable_now
                waypoints = _plan(pickable_for_plan, target_now, lift_xy_ref, params)

                wp_pos, gripper_open = waypoints[phase]
                reached = _reached(tcp, wp_pos, phase, params)
                action = _compute_action(tcp, wp_pos, gripper_open, params)

                if phase is not prev_phase or step % 20 == 0:
                    print(
                        f"  ep{episode_idx:2d} t{step:3d} {phase.name:<7s} "
                        f"tcp=[{tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}] "
                        f"obj=[{pickable_now[0]:+.3f},{pickable_now[1]:+.3f},{pickable_now[2]:+.3f}] "
                        f"tgt=[{target_now[0]:+.3f},{target_now[1]:+.3f},{target_now[2]:+.3f}] "
                        f"act_grip={action[-1]:+.1f} obs_grip={gripper_rad:.2f} reached={reached}",
                        flush=True,
                    )
                prev_phase = phase

                if phase in (Phase.GRASP, Phase.RELEASE):
                    hold_counter += 1
                    if phase is Phase.GRASP and hold_counter >= params.grasp_hold_steps:
                        phase = Phase.LIFT
                        hold_counter = 0
                    elif phase is Phase.RELEASE and hold_counter >= params.release_hold_steps:
                        phase = Phase.RETRACT
                        hold_counter = 0
                elif reached:
                    phase = _next_phase(phase)
                    hold_counter = 0

                action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                obs, _, terminated, truncated, info = env.step(action_t)

                if record_video:
                    pol = obs.get("policy", {}) if isinstance(obs, dict) else {}
                    t_cam = pol.get("table_cam")
                    w_cam = pol.get("wrist_cam")
                    if t_cam is not None:
                        video_table.append(t_cam[0].cpu().numpy().astype("uint8").copy())
                    if w_cam is not None:
                        video_wrist.append(w_cam[0].cpu().numpy().astype("uint8").copy())

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
                    flags = {name: bool(tm.get_term(name)[0]) for name in tm.active_terms}
                    print(f"  ep{episode_idx:2d} ENDED at t={step} phase={phase.name} flags={flags}", flush=True)
                    break

            tag = "SUCCESS" if succeeded else "FAIL"
            print(
                f"episode {episode_idx:3d}: {tag:<7s} ({exported}/{args.num_demos} exported, final phase={phase.name})",
                flush=True,
            )

        print(f"wrote {exported} successful episodes to {args.dataset_file}", flush=True)

        if record_video:
            from arm_act.tasks._runtime.oracle import _save_oracle_videos
            _save_oracle_videos(args.video_out, video_table, video_wrist, args.video_fps)

        try:
            env.close()
        except (RuntimeError, AttributeError, AssertionError):
            logger.warning("env.close() raised during teardown (ignored)", exc_info=True)

    except Exception:
        logger.exception("oracle demo collection failed")
        return 1
    finally:
        app.close()

    return 0


def _instantiate_env_cfg(env_cfg_spec: str):
    import importlib

    mod_name, cls_name = env_cfg_spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()


def _plan(pickable_pos, target_pos, lift_xy_ref, p: _OracleParams) -> dict:
    """Waypoint geometry for every phase.

    Note that ALIGN sits at the lift height directly above the target;
    INSERT goes BELOW target.z by `|insert_z_offset|`; RETRACT goes back
    up to `retract_height` while keeping xy at the target so the gripper
    exits the cavity vertically.
    """
    lift_xy = lift_xy_ref if lift_xy_ref is not None else pickable_pos[:2]
    return {
        Phase.HOVER:   (_xyz(pickable_pos[0], pickable_pos[1], pickable_pos[2] + p.hover_height), True),
        Phase.DESCEND: (_xyz(pickable_pos[0], pickable_pos[1], pickable_pos[2] + p.grasp_z_offset), True),
        Phase.GRASP:   (_xyz(pickable_pos[0], pickable_pos[1], pickable_pos[2] + p.grasp_z_offset), False),
        Phase.LIFT:    (_xyz(float(lift_xy[0]), float(lift_xy[1]), p.lift_height), False),
        Phase.MOVE:    (_xyz(target_pos[0], target_pos[1], p.lift_height), False),
        Phase.ALIGN:   (_xyz(target_pos[0], target_pos[1], p.lift_height), False),
        Phase.INSERT:  (_xyz(target_pos[0], target_pos[1], target_pos[2] + p.insert_z_offset), False),
        Phase.RELEASE: (_xyz(target_pos[0], target_pos[1], target_pos[2] + p.insert_z_offset), True),
        Phase.RETRACT: (_xyz(target_pos[0], target_pos[1], p.retract_height), True),
    }


def _xyz(x: float, y: float, z: float):
    import numpy as np
    return np.array([x, y, z], dtype=np.float32)


def _next_phase(phase: Phase) -> Phase:
    order = [
        Phase.HOVER, Phase.DESCEND, Phase.GRASP, Phase.LIFT,
        Phase.MOVE, Phase.ALIGN, Phase.INSERT, Phase.RELEASE, Phase.RETRACT,
    ]
    i = order.index(phase)
    return order[min(i + 1, len(order) - 1)]


def _reached(tcp, wp_pos, phase: Phase, p: _OracleParams) -> bool:
    """Phase-specific tolerance gates.

    DESCEND and INSERT use the tight z tolerance (precision matters when
    entering or exiting a confined volume). ALIGN uses a tighter xy
    tolerance than the default so we don't begin descent until we are
    centred over the opening.
    """
    import numpy as np
    dxy = float(np.linalg.norm(tcp[:2] - wp_pos[:2]))
    dz = abs(float(tcp[2] - wp_pos[2]))
    z_tol = p.z_reach_tol_tight if phase in (Phase.DESCEND, Phase.INSERT) else p.z_reach_tol
    xy_tol = p.align_xy_tol if phase is Phase.ALIGN else p.xy_reach_tol
    return dxy < xy_tol and dz < z_tol


def _compute_action(tcp, wp_pos, gripper_open: bool, p: _OracleParams):
    import numpy as np

    dx = float(np.clip(wp_pos[0] - tcp[0], -p.max_dxy, p.max_dxy))
    dy = float(np.clip(wp_pos[1] - tcp[1], -p.max_dxy, p.max_dxy))
    dz = float(np.clip(wp_pos[2] - tcp[2], -p.max_dz, p.max_dz))
    gripper_cmd = 1.0 if gripper_open else -1.0
    # 4-DOF action: 3 position deltas + 1 gripper.
    return np.array([dx, dy, dz, gripper_cmd], dtype=np.float32)


if __name__ == "__main__":
    from arm_act.config import load
    sys.exit(main(load("put_plant_back")))
