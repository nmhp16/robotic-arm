"""Parallel scripted oracle for parametric pick-and-place tasks.

Each Isaac Sim env_id runs an independent episode. The Isaac Lab env
batches the physics + rendering across all envs on the GPU; this oracle
keeps a Python-side state machine per env and feeds a batched action
tensor to ``env.step()`` each frame.

CLI: ``--num-envs N`` selects parallelism. Default 1 (matches the
pre-parallelization behavior). 8 is the recommended cap on the DGX
Spark: N=16 has tripped abrupt power-off shutdowns under peak GPU load
(see CLAUDE.md / hardware notes), and throughput already scales
sublinearly past ~8 as the GPU saturates. Demos export incrementally,
so a power-off mid-run loses at most the in-flight batch.

State machine per episode (unchanged from the serial version):
  PRE_HOVER        → TCP at start.xy, hover_height
  HOVER            → TCP at pickable.xy, pickable.z + hover_height
  DESCEND_LATERAL  → TCP at pickable.xy + jitter_offset, hover_z
  DESCEND          → TCP at pickable.xy + jitter_offset, pickable.z + grasp_z_offset
  GRASP            → hold N steps with gripper=close
  LIFT             → TCP at pickable.xy, lift_height
  MOVE             → TCP at target.xy, lift_height
  PLACE            → TCP at target.xy, target.z + place_z_offset
  RELEASE          → gripper=open, hold until env's success termination fires

The phase splits exist to separate xy and z motion (see file backup
``oracle_serial.py.bak`` for the rationale comments).
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
    PRE_HOVER = 1
    HOVER = 2
    DESCEND_LATERAL = 3
    DESCEND = 4
    GRASP = 5
    LIFT = 6
    MOVE = 7
    PLACE = 8
    RELEASE = 9


@dataclass
class _EpisodeJitter:
    """Per-episode variability so demos aren't N near-identical trajectories."""
    hover_height: float
    grasp_z_offset: float
    lift_height: float
    place_z_offset: float
    descend_xy_offset_x: float
    descend_xy_offset_y: float
    move_xy_offset_x: float
    move_xy_offset_y: float
    max_dxy: float
    max_dz: float
    extra_hover_pause: int
    extra_lift_pause: int
    grasp_hold_steps: int


@dataclass
class _EnvState:
    """Per-env oracle state. Lives in Python; cheap to update."""
    phase: Phase
    hold_counter: int
    hover_pause_counter: int
    lift_pause_counter: int
    pre_hover_xy_ref: Any  # np.ndarray (2,) or None
    grasp_pickable_pos: Any  # np.ndarray (3,) or None
    lift_xy_ref: Any  # np.ndarray (2,) or None
    jitter: _EpisodeJitter
    episode_step: int      # steps elapsed in current episode
    episode_idx: int       # global episode counter (for logging)
    succeeded: bool        # set True once this env's success export fires


@dataclass(frozen=True)
class _OracleParams:
    hover_height: float
    grasp_z_offset: float
    grasp_xy_offset_x: float
    grasp_xy_offset_y: float
    lift_height: float
    place_z_offset: float
    max_dxy: float
    max_dz: float
    xy_reach_tol: float
    z_reach_tol: float
    z_reach_tol_tight: float
    grasp_hold_steps: int
    release_hold_steps: int
    max_steps_per_episode: int
    max_attempts_per_demo: int
    terminate_after_lift: bool
    ik_command_type: str

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> _OracleParams:
        o = spec["oracle"]
        return cls(
            hover_height=float(o["hover_height"]),
            grasp_z_offset=float(o["grasp_z_offset"]),
            grasp_xy_offset_x=float(o.get("grasp_xy_offset_x", 0.0)),
            grasp_xy_offset_y=float(o.get("grasp_xy_offset_y", 0.0)),
            lift_height=float(o["lift_height"]),
            place_z_offset=float(o.get("place_z_offset", 0.0)),
            max_dxy=float(o["max_dxy"]),
            max_dz=float(o["max_dz"]),
            xy_reach_tol=float(o["xy_reach_tol"]),
            z_reach_tol=float(o["z_reach_tol"]),
            z_reach_tol_tight=float(o["z_reach_tol_tight"]),
            grasp_hold_steps=int(o["grasp_hold_steps"]),
            release_hold_steps=int(o.get("release_hold_steps", 0)),
            max_steps_per_episode=int(o["max_steps_per_episode"]),
            max_attempts_per_demo=int(o["max_episode_attempts_per_demo"]),
            terminate_after_lift=bool(o.get("terminate_after_lift", False)),
            ik_command_type=str(spec.get("robot", {}).get("ik_command_type", "position")),
        )


def _parse_args(default_dataset: str, default_max_steps: int) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-demos", type=int, default=15)
    p.add_argument("--num-envs", type=int, default=1,
                   help="parallel envs (1 = serial behavior). Try 8 for ~4-5x speedup. "
                        "Avoid >8 on DGX Spark: peak GPU load at N=16 has tripped power-off.")
    p.add_argument("--dataset-file", type=pathlib.Path, default=pathlib.Path(default_dataset))
    p.add_argument("--max-steps", type=int, default=default_max_steps)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video-out", type=pathlib.Path, default=None,
                   help="(unsupported in parallel mode; ignored when --num-envs > 1)")
    p.add_argument("--video-fps", type=int, default=15)
    return p.parse_args()


def main(spec: dict[str, Any]) -> int:
    """Entry point. Supports both serial (num_envs=1) and parallel (num_envs > 1)."""
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
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
        import numpy as np
        import gymnasium as gym
        import torch
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.managers.recorder_manager import DatasetExportMode

        import arm_act.tasks
        arm_act.tasks.register()

        gym_id = spec["task"]["gym_id"]
        env_cfg_spec = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
        env_cfg = _instantiate_env_cfg(env_cfg_spec)
        N = max(1, int(args.num_envs))
        if N > 8:
            logger.warning(
                "--num-envs=%d exceeds the DGX Spark safe cap of 8: peak GPU load "
                "at N=16 has tripped abrupt power-off shutdowns. Demos export "
                "incrementally so they survive, but consider --num-envs 8.", N,
            )
        env_cfg.scene.num_envs = N
        env_cfg.terminations.time_out = None  # oracle handles timeouts manually
        env_cfg.observations.policy.concatenate_terms = False

        output_dir = str(args.dataset_file.parent.resolve())
        os.makedirs(output_dir, exist_ok=True)
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = args.dataset_file.stem
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

        env = gym.make(gym_id, cfg=env_cfg).unwrapped
        device = env.device

        record_video = args.video_out is not None and N == 1
        video_table: list = []
        video_wrist: list = []
        if args.video_out is not None and N > 1:
            logger.warning("--video-out not supported with --num-envs > 1; ignored")

        # Per-env RNGs so each env's jitter is independent + reproducible.
        rng_root = np.random.default_rng(seed=args.seed)
        per_env_rngs = [np.random.default_rng(rng_root.integers(2**31)) for _ in range(N)]

        # Initialize all N envs with their first episode state.
        next_episode_idx = 0
        states: list[_EnvState] = []
        for env_id in range(N):
            next_episode_idx += 1
            states.append(_EnvState(
                phase=Phase.PRE_HOVER,
                hold_counter=0,
                hover_pause_counter=0,
                lift_pause_counter=0,
                pre_hover_xy_ref=None,
                grasp_pickable_pos=None,
                lift_xy_ref=None,
                jitter=_sample_jitter(params, per_env_rngs[env_id]),
                episode_step=0,
                episode_idx=next_episode_idx,
                succeeded=False,
            ))

        obs, _ = env.reset()
        exported = 0
        attempts = N
        max_attempts = args.num_demos * params.max_attempts_per_demo

        # action_dim depends on ik_command_type
        action_dim = 7 if params.ik_command_type == "pose" else 4

        while exported < args.num_demos and attempts < max_attempts + N:
            # Read batched obs.
            tcp_b = obs["policy"]["eef_pos"].cpu().numpy()           # (N, 3)
            pickable_b = obs["policy"]["pickable_pos"].cpu().numpy()  # (N, 3)
            target_b = obs["policy"]["target_pos"].cpu().numpy()      # (N, 3)
            gripper_b = obs["policy"]["gripper_pos"][:, 0].cpu().numpy()  # (N,)

            # Per-env: update refs, compute waypoint + action.
            actions_np = np.zeros((N, action_dim), dtype=np.float32)
            for env_id in range(N):
                s = states[env_id]
                tcp = tcp_b[env_id]
                pickable_now = pickable_b[env_id]
                target_now = target_b[env_id]

                # Set frozen references on phase entry.
                if s.phase is Phase.PRE_HOVER and s.pre_hover_xy_ref is None:
                    s.pre_hover_xy_ref = tcp[:2].copy()
                if s.phase is Phase.DESCEND_LATERAL and s.grasp_pickable_pos is None:
                    s.grasp_pickable_pos = pickable_now.copy()
                if s.phase is Phase.LIFT and s.lift_xy_ref is None:
                    s.lift_xy_ref = (tcp[:2] * 0.75).copy()

                pickable_for_plan = s.grasp_pickable_pos if s.grasp_pickable_pos is not None else pickable_now
                waypoints = _plan(
                    pickable_for_plan, target_now,
                    s.lift_xy_ref, s.pre_hover_xy_ref,
                    params, s.jitter,
                )
                wp_pos, gripper_open = waypoints[s.phase]
                actions_np[env_id] = _compute_action(tcp, wp_pos, gripper_open, params, s.jitter, action_dim)

            actions = torch.from_numpy(actions_np).to(device)

            # Step physics on all N envs.
            obs, _, terminated, truncated, info = env.step(actions)
            terminated_np = terminated.cpu().numpy()
            truncated_np = truncated.cpu().numpy()

            # Post-step: detect successes, timeouts, env-auto-resets.
            success_env_ids: list[int] = []
            reset_oracle_env_ids: list[int] = []

            for env_id in range(N):
                s = states[env_id]
                s.episode_step += 1

                # Re-read post-step obs for this env's phase update.
                tcp = obs["policy"]["eef_pos"][env_id].cpu().numpy()
                pickable_now = obs["policy"]["pickable_pos"][env_id].cpu().numpy()
                target_now = obs["policy"]["target_pos"][env_id].cpu().numpy()
                pickable_for_plan = s.grasp_pickable_pos if s.grasp_pickable_pos is not None else pickable_now
                waypoints = _plan(
                    pickable_for_plan, target_now,
                    s.lift_xy_ref, s.pre_hover_xy_ref,
                    params, s.jitter,
                )
                wp_pos, _ = waypoints[s.phase]
                reached = _reached(tcp, wp_pos, s.phase, params)

                # Grasp-geometry trace (env-var gated; DBG_GRASP=1). Shows
                # whether the plant rises WITH the TCP (form-closure caught:
                # tcp-plant gap stays ~constant) or stays flat while the TCP
                # climbs (slip: gap grows). Ground truth for tuning the lip.
                if env_id == 0 and os.environ.get("DBG_GRASP") and (
                    s.episode_step % 10 == 0
                ):
                    print(
                        f"[GRASP-DBG ep{s.episode_idx} t{s.episode_step}] "
                        f"phase={s.phase.name} tcp_z={float(tcp[2]):.4f} "
                        f"plant_z={float(pickable_now[2]):.4f} "
                        f"gap={float(tcp[2]-pickable_now[2]):.4f}",
                        flush=True,
                    )

                # Phase transitions (mirrors serial logic).
                if s.phase in (Phase.GRASP, Phase.RELEASE):
                    s.hold_counter += 1
                    if s.phase is Phase.GRASP and s.hold_counter >= s.jitter.grasp_hold_steps:
                        s.phase = Phase.LIFT
                        s.hold_counter = 0
                elif reached and s.phase is Phase.HOVER and s.hover_pause_counter < s.jitter.extra_hover_pause:
                    s.hover_pause_counter += 1
                elif reached and s.phase is Phase.LIFT and s.lift_pause_counter < s.jitter.extra_lift_pause and not params.terminate_after_lift:
                    s.lift_pause_counter += 1
                elif reached:
                    # Early success on LIFT in terminate_after_lift mode —
                    # but ONLY if the PLANT actually came up. The flat-finger
                    # friction grip can raise the TCP to lift_height while the
                    # stem slips out, which used to be exported as a false
                    # SUCCESS (verified: demos where plant z fell to the table
                    # were marked success). Gate on the live pickable z.
                    if params.terminate_after_lift and s.phase is Phase.LIFT:
                        # In-vial "lifted out": plant root above ~0.02 (clears the
                        # vial rim). Real lifts hit ~0.038; misses stay <0. 0.6*
                        # lift_height was 0.042 — too high, under-counted. Use 0.02.
                        if float(pickable_now[2]) >= 0.020:
                            success_env_ids.append(env_id)
                            s.succeeded = True
                            continue
                        # TCP at height but plant didn't lift — not a success.
                        # Stay in LIFT; it'll time out as an honest FAIL.
                    else:
                        s.phase = _next_phase(s.phase)
                        s.hold_counter = 0

                # Env-detected success (placement-mode tasks).
                if bool(env.termination_manager.get_term("success")[env_id]):
                    success_env_ids.append(env_id)
                    s.succeeded = True
                    continue

                # Env auto-reset already happened (e.g., pickable_dropping).
                if bool(terminated_np[env_id]) or bool(truncated_np[env_id]):
                    reset_oracle_env_ids.append(env_id)
                    continue

                # Manual timeout: episode_step exceeded oracle's max.
                if s.episode_step >= params.max_steps_per_episode:
                    reset_oracle_env_ids.append(env_id)

            # Export successes (one batched call per group).
            if success_env_ids:
                ids_tensor = success_env_ids
                env.recorder_manager.record_pre_reset(ids_tensor, force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    ids_tensor,
                    torch.ones((len(ids_tensor), 1), dtype=torch.bool, device=device),
                )
                env.recorder_manager.export_episodes(ids_tensor)
                _ids_t = torch.tensor(ids_tensor, device=device, dtype=torch.long)
                env._reset_idx(_ids_t)
                # record_post_reset captures the NEW episode's /initial_state.
                # Without it, the next episode for these envs has no
                # initial_state group and isaaclab_mimic's annotate step
                # fails with KeyError('initial_state').
                env.recorder_manager.record_post_reset(_ids_t)
                exported += len(success_env_ids)
                for env_id in success_env_ids:
                    s = states[env_id]
                    print(
                        f"episode {s.episode_idx:3d}: SUCCESS ({exported}/{args.num_demos} exported, env={env_id}, t={s.episode_step})",
                        flush=True,
                    )

            # Reset oracle-side state for failed/timed-out envs.
            for env_id in reset_oracle_env_ids:
                s = states[env_id]
                print(
                    f"episode {s.episode_idx:3d}: FAIL    ({exported}/{args.num_demos} exported, env={env_id}, t={s.episode_step}, phase={s.phase.name})",
                    flush=True,
                )
                # Manually reset the env for non-success cases too — env.step
                # only auto-resets on terminations from termination_manager,
                # not on our oracle-internal timeout. record_pre_reset flushes
                # (and, in EXPORT_SUCCEEDED_ONLY mode, discards) the failed
                # episode buffer; record_post_reset captures the next
                # episode's /initial_state.
                if not (terminated_np[env_id] or truncated_np[env_id]):
                    _id_t = torch.tensor([env_id], device=device, dtype=torch.long)
                    env.recorder_manager.record_pre_reset(_id_t, force_export_or_skip=False)
                    env._reset_idx(_id_t)
                    env.recorder_manager.record_post_reset(_id_t)

            # Re-arm any env that just finished an episode (success OR fail).
            finished_ids = set(success_env_ids) | set(reset_oracle_env_ids) | {
                i for i in range(N) if (terminated_np[i] or truncated_np[i]) and i not in success_env_ids and i not in reset_oracle_env_ids
            }
            for env_id in finished_ids:
                attempts += 1
                next_episode_idx += 1
                states[env_id] = _EnvState(
                    phase=Phase.PRE_HOVER,
                    hold_counter=0,
                    hover_pause_counter=0,
                    lift_pause_counter=0,
                    pre_hover_xy_ref=None,
                    grasp_pickable_pos=None,
                    lift_xy_ref=None,
                    jitter=_sample_jitter(params, per_env_rngs[env_id]),
                    episode_step=0,
                    episode_idx=next_episode_idx,
                    succeeded=False,
                )

            # Optional video capture (single-env mode only).
            if record_video and N == 1:
                pol = obs.get("policy", {}) if isinstance(obs, dict) else {}
                t_cam = pol.get("table_cam")
                w_cam = pol.get("wrist_cam")
                if t_cam is not None:
                    video_table.append(t_cam[0].cpu().numpy().astype("uint8").copy())
                if w_cam is not None:
                    video_wrist.append(w_cam[0].cpu().numpy().astype("uint8").copy())

        print(f"wrote {exported} successful episodes to {args.dataset_file}", flush=True)

        if record_video:
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


def _save_oracle_videos(out_path, table_frames, wrist_frames, fps: int) -> None:
    try:
        import imageio
    except ImportError:
        print("imageio not installed; skipping oracle video dump", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    composite = []
    for t, w in zip(table_frames, wrist_frames, strict=False):
        if t is None and w is None:
            continue
        if w is None:
            composite.append(t)
            continue
        if t is None:
            composite.append(w)
            continue
        composite.append(np.concatenate([t, w], axis=1))

    imageio.mimsave(str(out_path), composite, fps=fps)
    print(f"wrote {out_path} ({len(composite)} frames @ {fps} fps)", flush=True)

    stem = out_path.with_suffix("")
    for frames, name in ((table_frames, "table_cam"), (wrist_frames, "wrist_cam")):
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        path = pathlib.Path(f"{stem}_{name}.mp4")
        imageio.mimsave(str(path), frames, fps=fps)
        print(f"wrote {path}", flush=True)


def _plan(pickable_pos, target_pos, lift_xy_ref, pre_hover_xy_ref, p: _OracleParams, j: _EpisodeJitter | None = None) -> dict:
    lift_xy = lift_xy_ref if lift_xy_ref is not None else pickable_pos[:2]
    pre_hover_xy = pre_hover_xy_ref if pre_hover_xy_ref is not None else pickable_pos[:2]
    g_off_x = p.grasp_xy_offset_x
    g_off_y = p.grasp_xy_offset_y
    if j is None:
        hover_h = p.hover_height
        grasp_z = p.grasp_z_offset
        lift_h = p.lift_height
        place_z = p.place_z_offset
        d_off_x = d_off_y = 0.0
        m_off_x = m_off_y = 0.0
    else:
        hover_h = j.hover_height
        grasp_z = j.grasp_z_offset
        lift_h = j.lift_height
        place_z = j.place_z_offset
        d_off_x, d_off_y = j.descend_xy_offset_x, j.descend_xy_offset_y
        m_off_x, m_off_y = j.move_xy_offset_x, j.move_xy_offset_y
    return {
        Phase.PRE_HOVER:       (_xyz(float(pre_hover_xy[0]), float(pre_hover_xy[1]), pickable_pos[2] + hover_h), True),
        Phase.HOVER:           (_xyz(pickable_pos[0] + g_off_x, pickable_pos[1] + g_off_y, pickable_pos[2] + hover_h), True),
        Phase.DESCEND_LATERAL: (_xyz(pickable_pos[0] + g_off_x + d_off_x, pickable_pos[1] + g_off_y + d_off_y, pickable_pos[2] + hover_h), True),
        Phase.DESCEND:         (_xyz(pickable_pos[0] + g_off_x + d_off_x, pickable_pos[1] + g_off_y + d_off_y, pickable_pos[2] + grasp_z), True),
        Phase.GRASP:           (_xyz(pickable_pos[0] + g_off_x + d_off_x, pickable_pos[1] + g_off_y + d_off_y, pickable_pos[2] + grasp_z), False),
        Phase.LIFT:            (_xyz(float(lift_xy[0]), float(lift_xy[1]), lift_h), False),
        Phase.MOVE:            (_xyz(target_pos[0] + m_off_x, target_pos[1] + m_off_y, lift_h), False),
        Phase.PLACE:           (_xyz(target_pos[0] + m_off_x, target_pos[1] + m_off_y, target_pos[2] + place_z), False),
        Phase.RELEASE:         (_xyz(target_pos[0] + m_off_x, target_pos[1] + m_off_y, target_pos[2] + place_z), True),
    }


def _xyz(x: float, y: float, z: float):
    import numpy as np
    return np.array([x, y, z], dtype=np.float32)


def _reached(tcp, wp_pos, phase: Phase, p: _OracleParams) -> bool:
    import numpy as np
    dxy = float(np.linalg.norm(tcp[:2] - wp_pos[:2]))
    dz = abs(float(tcp[2] - wp_pos[2]))
    if phase in (Phase.MOVE, Phase.PLACE):
        return dxy < p.xy_reach_tol and tcp[2] < wp_pos[2] + p.z_reach_tol
    z_tol = p.z_reach_tol_tight if phase in (Phase.DESCEND, Phase.PLACE) else p.z_reach_tol
    return dxy < p.xy_reach_tol and dz < z_tol


def _compute_action(tcp, wp_pos, gripper_open: bool, p: _OracleParams, j: _EpisodeJitter | None = None, action_dim: int = 4):
    import numpy as np

    max_dxy = j.max_dxy if j is not None else p.max_dxy
    max_dz = j.max_dz if j is not None else p.max_dz
    dx = float(np.clip(wp_pos[0] - tcp[0], -max_dxy, max_dxy))
    dy = float(np.clip(wp_pos[1] - tcp[1], -max_dxy, max_dxy))
    dz = float(np.clip(wp_pos[2] - tcp[2], -max_dz, max_dz))
    gripper_cmd = 1.0 if gripper_open else -1.0
    if action_dim == 7:
        return np.array([dx, dy, dz, 0.0, 0.0, 0.0, gripper_cmd], dtype=np.float32)
    return np.array([dx, dy, dz, gripper_cmd], dtype=np.float32)


def _sample_jitter(p: _OracleParams, rng) -> _EpisodeJitter:
    return _EpisodeJitter(
        hover_height=float(p.hover_height + rng.uniform(-0.05, 0.05)),
        grasp_z_offset=float(p.grasp_z_offset + rng.uniform(-0.01, 0.01)),
        lift_height=float(p.lift_height + rng.uniform(-0.02, 0.02)),
        place_z_offset=float(p.place_z_offset + rng.uniform(-0.01, 0.01)),
        descend_xy_offset_x=float(rng.uniform(-0.01, 0.01)),
        descend_xy_offset_y=float(rng.uniform(-0.01, 0.01)),
        move_xy_offset_x=float(rng.uniform(-0.01, 0.01)),
        move_xy_offset_y=float(rng.uniform(-0.01, 0.01)),
        max_dxy=float(rng.uniform(0.02, 0.06)),
        max_dz=float(rng.uniform(0.02, 0.06)),
        extra_hover_pause=int(rng.integers(0, 21)),
        extra_lift_pause=int(rng.integers(0, 16)),
        grasp_hold_steps=int(rng.integers(10, 26)),
    )


def _next_phase(phase: Phase) -> Phase:
    order = [
        Phase.PRE_HOVER,
        Phase.HOVER,
        Phase.DESCEND_LATERAL,
        Phase.DESCEND,
        Phase.GRASP,
        Phase.LIFT,
        Phase.MOVE,
        Phase.PLACE,
        Phase.RELEASE,
    ]
    idx = order.index(phase)
    return order[min(idx + 1, len(order) - 1)]


if __name__ == "__main__":
    from arm_act.config import DEFAULT_TASK, load
    sys.exit(main(load(DEFAULT_TASK)))
