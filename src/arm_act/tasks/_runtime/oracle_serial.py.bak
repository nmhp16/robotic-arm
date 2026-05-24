"""Generic scripted oracle for parametric pick-and-place tasks.

Reads its waypoint geometry + state-machine timing from the task spec
(``oracle:`` section in task.yaml). Exposes ``main(spec, cli_args)``;
the cli wrapper passes the loaded spec.

State machine per episode:
  PRE_HOVER        → TCP at start.xy, hover_height (rise straight up at start xy)
  HOVER            → TCP at pickable.xy, pickable.z + hover_height, gripper open
  DESCEND_LATERAL  → TCP at pickable.xy + jitter_offset, hover_z (xy realign at hover height)
  DESCEND          → TCP at pickable.xy + jitter_offset, pickable.z + grasp_z_offset (pure vertical)
  GRASP            → hold N steps with gripper=close
  LIFT             → TCP at pickable.xy, lift_height
  MOVE             → TCP at target.xy, lift_height
  PLACE            → TCP at target.xy, target.z + place_z_offset
  RELEASE          → gripper=open, hold until env's success termination fires

The phase splits exist to keep xy and z motion separated, because the
UR5 + 6-DOF DLS IK does NOT produce clean linear paths between
waypoints: the joint-space solution can swing wide, sweeping links
through table-level space before settling at the commanded TCP. Two
separations matter:

  1. PRE_HOVER → HOVER (start of episode). The arm's home pose places
     TCP very low (z ≈ 0.02 — tcp essentially at the table). Going
     diagonally to (vial.xy, vial.z + hover_height) sends the gripper
     body through a band of low-z xy positions that intersect the vial,
     knocking it off the table during the very first action. PRE_HOVER
     does pure vertical rise at the start xy first, so by the time
     HOVER's xy realign fires, the gripper is 30 cm above the vial
     and lateral motion is harmless.

  2. HOVER → DESCEND_LATERAL → DESCEND. Same idea at the descent end:
     diagonal motion plus the descend_xy jitter offset (±1 cm) drove the
     finger through the vial's side, shifting the vial 5-8 cm laterally
     so the kinematic-attach captured a bad offset. DESCEND_LATERAL
     realigns xy at hover height; DESCEND drops pure vertical.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)


class Phase(Enum):
    PRE_HOVER = 1        # rise straight up at start xy to hover height
    HOVER = 2
    DESCEND_LATERAL = 3  # xy-align at hover height before vertical descent
    DESCEND = 4
    GRASP = 5
    LIFT = 6
    MOVE = 7
    PLACE = 8
    RELEASE = 9


@dataclass(frozen=True)
class _EpisodeJitter:
    """Per-episode random variability injected into the scripted oracle so
    the recorded dataset doesn't have N near-identical trajectories. ACT
    (and any imitation method) mode-collapses onto the dataset mean when
    every demo is the same straight-line path — the model's lowest-loss
    solution becomes "output the average action" without conditioning on
    state. This struct samples per-episode offsets/pauses that are still
    geometrically valid (waypoints stay reachable) but make every demo
    visibly different in the visual stream and the joint trajectory.

    Sample once at the start of each episode; the values are fixed for
    the whole run so a single demo is internally consistent.
    """
    hover_height: float        # ±5 cm around params.hover_height
    grasp_z_offset: float      # ±1 cm
    lift_height: float         # ±2 cm
    place_z_offset: float      # ±1 cm
    descend_xy_offset_x: float  # ±1 cm offset from plant xy at DESCEND
    descend_xy_offset_y: float
    move_xy_offset_x: float    # ±1 cm offset from tray xy at MOVE/PLACE
    move_xy_offset_y: float
    max_dxy: float             # in [0.02, 0.06] — varies demo speed
    max_dz: float              # in [0.02, 0.06]
    extra_hover_pause: int     # 0–20 extra steps held at HOVER
    extra_lift_pause: int      # 0–15 extra steps held at LIFT
    grasp_hold_steps: int      # 10–25 (vs fixed 15 before)


@dataclass(frozen=True)
class _OracleParams:
    hover_height: float
    grasp_z_offset: float
    # Base XY offset applied to the descend / grasp positions on top of the
    # pickable's centroid. Useful when the grip point is NOT the pickable's
    # centroid — e.g., gripping the front-edge rim of a holder rack rather
    # than its geometric center. Default 0,0 (grip directly above centroid).
    # Distinct from the per-episode jitter ±1cm in OracleJitter.
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
    # If True, the episode ends as soon as LIFT reaches lift_height — skips
    # MOVE/PLACE/RELEASE entirely. Use for "pick and lift up" tasks where
    # there is no placement target, just a graspable object that needs to
    # be removed from its starting pose.
    terminate_after_lift: bool
    # Mirrors robot.ik_command_type in task.yaml. "position" => 4-DOF action
    # (3 translation deltas + gripper). "pose" => 7-DOF action (3 translation
    # + 3 axis-angle rotation deltas + gripper); rotation deltas are zero
    # since the SCARA can't pitch/roll. Pose mode is required when the env
    # is annotated through Isaac Lab's mimic stack, which subclasses the
    # 7-D Franka IK env.
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
    p.add_argument("--dataset-file", type=pathlib.Path, default=pathlib.Path(default_dataset))
    p.add_argument("--max-steps", type=int, default=default_max_steps)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video-out", type=pathlib.Path, default=None,
                   help="if set, write mp4(s) of each episode's table_cam + wrist_cam streams")
    p.add_argument("--video-fps", type=int, default=15)
    return p.parse_args()


def main(spec: dict[str, Any]) -> int:
    """Entry point invoked by ``arm_act.cli.oracle``."""
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
        # Run until the env's own success termination fires, not the time cap.
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

        # Per-episode RNG so each demo's jitter is reproducible from the
        # CLI seed but different across demos.
        import numpy as _np
        jitter_rng_root = _np.random.default_rng(seed=args.seed)

        while exported < args.num_demos and episode_idx < max_attempts:
            episode_idx += 1
            obs, _ = env.reset()
            jitter = _sample_jitter(params, jitter_rng_root.spawn(1)[0])

            phase = Phase.PRE_HOVER
            hold_counter = 0
            hover_pause_counter = 0
            lift_pause_counter = 0
            succeeded = False
            prev_phase = None
            lift_xy_ref = None          # TCP xy frozen at LIFT entry
            pre_hover_xy_ref = None     # TCP xy frozen at PRE_HOVER entry (start of episode)
            grasp_pickable_pos = None   # pickable pose frozen at GRASP entry

            for step in range(args.max_steps):
                tcp = obs["policy"]["eef_pos"][0].cpu().numpy()
                pickable_now = obs["policy"]["pickable_pos"][0].cpu().numpy()
                target_now = obs["policy"]["target_pos"][0].cpu().numpy()
                gripper_rad = float(obs["policy"]["gripper_pos"][0, 0].cpu())

                if phase is Phase.PRE_HOVER and pre_hover_xy_ref is None:
                    pre_hover_xy_ref = tcp[:2].copy()
                if phase is Phase.DESCEND_LATERAL and grasp_pickable_pos is None:
                    # Freeze the pickable position at the start of descent
                    # (HOVER reach, vial still at spawn). DESCEND/GRASP plan
                    # against this frozen pos instead of the live vial — so
                    # if a finger micro-nudges the vial mid-descent, the TCP
                    # keeps descending to the original xy instead of chasing
                    # the displaced vial. Without this freeze, a small bump
                    # spirals into a 10 cm chase as TCP and vial co-move
                    # diagonally toward the table edge.
                    grasp_pickable_pos = pickable_now.copy()
                if phase is Phase.LIFT and lift_xy_ref is None:
                    # Pull the lift waypoint xy 25% toward the robot base
                    # (origin). Pure-vertical lift at the grasp xy hits a
                    # near-singular UR5 config when the grasp is at extended
                    # reach (~55 cm radius); the redundant 6-DOF DLS IK then
                    # oscillates 100+ steps trying to hold xy while raising
                    # z. Biasing toward the base shortens the radius from
                    # ~0.55 m to ~0.41 m, well inside UR5's comfortable
                    # envelope, and the subsequent MOVE phase does the
                    # lateral traversal from there.
                    lift_xy_ref = (tcp[:2] * 0.75).copy()

                # Pre-GRASP: track the live pickable so descent corrects for nudges.
                # Post-GRASP: freeze xy so reading the pickable's pose (which now
                # rides the arm) doesn't form a feedback loop dragging it out of the jaws.
                pickable_for_plan = grasp_pickable_pos if grasp_pickable_pos is not None else pickable_now
                waypoints = _plan(pickable_for_plan, target_now, lift_xy_ref, pre_hover_xy_ref, params, jitter)

                wp_pos, gripper_open = waypoints[phase]
                reached = _reached(tcp, wp_pos, phase, params)
                action = _compute_action(tcp, wp_pos, gripper_open, params, jitter)

                if phase is not prev_phase or step % 20 == 0:
                    print(
                        f"  ep{episode_idx:2d} t{step:3d} {phase.name:<15s} "
                        f"tcp=[{tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}] "
                        f"obj=[{pickable_now[0]:+.3f},{pickable_now[1]:+.3f},{pickable_now[2]:+.3f}] "
                        f"tgt=[{target_now[0]:+.3f},{target_now[1]:+.3f},{target_now[2]:+.3f}] "
                        f"act_grip={action[-1]:+.1f} obs_grip={gripper_rad:.4f} reached={reached}",
                        flush=True,
                    )
                prev_phase = phase

                if phase in (Phase.GRASP, Phase.RELEASE):
                    hold_counter += 1
                    if phase is Phase.GRASP and hold_counter >= jitter.grasp_hold_steps:
                        phase = Phase.LIFT
                        hold_counter = 0
                elif reached and phase is Phase.HOVER and hover_pause_counter < jitter.extra_hover_pause:
                    # Hold the gripper hovering above the plant for a few extra
                    # steps before descending. Mimics the natural human pause
                    # to "look before you leap".
                    hover_pause_counter += 1
                elif reached and phase is Phase.LIFT and lift_pause_counter < jitter.extra_lift_pause and not params.terminate_after_lift:
                    # Same idea at LIFT — pause briefly with the cuboid raised
                    # before starting the lateral move. Skipped in
                    # ``terminate_after_lift`` mode (early-success tasks).
                    lift_pause_counter += 1
                elif reached:
                    # Pick-and-lift-only mode: declare success on LIFT
                    # completion; never advance to MOVE.
                    if params.terminate_after_lift and phase is Phase.LIFT:
                        print(
                            f"  ep{episode_idx:2d} t{step:3d} LIFT-DONE "
                            f"tcp=[{tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}] "
                            f"obj=[{pickable_now[0]:+.3f},{pickable_now[1]:+.3f},{pickable_now[2]:+.3f}] "
                            f"-> early success",
                            flush=True,
                        )
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        succeeded = True
                        exported += 1
                        break
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


def _plan(
    pickable_pos,
    target_pos,
    lift_xy_ref,
    pre_hover_xy_ref,
    p: _OracleParams,
    j: _EpisodeJitter | None = None,
) -> dict:
    lift_xy = lift_xy_ref if lift_xy_ref is not None else pickable_pos[:2]
    pre_hover_xy = pre_hover_xy_ref if pre_hover_xy_ref is not None else pickable_pos[:2]
    # Base XY offset shifts the grip target away from the pickable's centroid
    # — useful for objects gripped at an edge/feature rather than at center
    # (e.g., the front-rim grip on a holder rack).
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


def _sample_jitter(p: _OracleParams, rng) -> _EpisodeJitter:
    """Draw per-episode variability."""
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
    i = order.index(phase)
    return order[min(i + 1, len(order) - 1)]


def _reached(tcp, wp_pos, phase: Phase, p: _OracleParams) -> bool:
    import numpy as np
    dxy = float(np.linalg.norm(tcp[:2] - wp_pos[:2]))
    dz = abs(float(tcp[2] - wp_pos[2]))
    if phase in (Phase.MOVE, Phase.PLACE):
        # MOVE/PLACE only require xy alignment. UR5 6-DOF DLS IK z-sinks
        # at extended reach with the payload offset, so TCP typically
        # arrives at the tray xy at z well below the commanded waypoint
        # z (lift_height for MOVE, place height for PLACE) — demanding
        # strict z would leave the phase permanently unreached. The
        # kinematic_attach offset (~5 cm) puts the vial slightly below
        # the TCP, so when MOVE/PLACE settles at low z above the tray
        # the vial is already approximately above the tray surface;
        # RELEASE then drops the vial a short distance onto the tray.
        # Cap on the high side so the phase doesn't transition while
        # still rising toward the target z early in the phase.
        return dxy < p.xy_reach_tol and tcp[2] < wp_pos[2] + p.z_reach_tol
    z_tol = p.z_reach_tol_tight if phase in (Phase.DESCEND, Phase.PLACE) else p.z_reach_tol
    return dxy < p.xy_reach_tol and dz < z_tol


def _compute_action(tcp, wp_pos, gripper_open: bool, p: _OracleParams, j: _EpisodeJitter | None = None):
    import numpy as np

    max_dxy = j.max_dxy if j is not None else p.max_dxy
    max_dz = j.max_dz if j is not None else p.max_dz
    dx = float(np.clip(wp_pos[0] - tcp[0], -max_dxy, max_dxy))
    dy = float(np.clip(wp_pos[1] - tcp[1], -max_dxy, max_dxy))
    dz = float(np.clip(wp_pos[2] - tcp[2], -max_dz, max_dz))
    # Isaac Lab BinaryJointAction: positive = OPEN, negative = CLOSE.
    gripper_cmd = 1.0 if gripper_open else -1.0
    if p.ik_command_type == "pose":
        # 7-DOF: 3 translation deltas + 3 axis-angle rotation deltas + gripper.
        # Rotation deltas are zero — SCARA can't pitch/roll, and we keep yaw
        # at home. Required for mimic-env compatibility (it subclasses the
        # 7-D Franka IK env that slices action[:, 3:6] for delta_rotation).
        return np.array([dx, dy, dz, 0.0, 0.0, 0.0, gripper_cmd], dtype=np.float32)
    return np.array([dx, dy, dz, gripper_cmd], dtype=np.float32)


class EnvStateSnapshot(NamedTuple):
    """Env-local positions + gripper driver-joint reading for one env.

    Field names match the kwargs of :func:`oracle_action_at_state` so the
    common pattern ``snapshot_env_state(...)`` → ``oracle_action_at_state(...)``
    flows naturally.
    """
    tcp: "np.ndarray"
    pickable: "np.ndarray"
    target: "np.ndarray"
    gripper_pos: float


def snapshot_env_state(env, gripper_driver_joint: str) -> "EnvStateSnapshot":
    """Sample the env state the oracle needs, in env-local coordinates.

    Translates TCP / pickable / target positions by
    ``scene.env_origins[0]`` so the values match what the oracle and the
    task YAML's waypoints expect (task-frame, not world-frame). Reads
    scene keys ``"ee_frame"``, ``"pickable"``, ``"target"``, ``"robot"`` —
    the parametric runtime guarantees all four exist.

    Assumes ``num_envs == 1`` (oracle queries are single-env).

    Args:
        env: The ``gym.make()``-wrapped Isaac Lab env.
        gripper_driver_joint: Joint name from ``cfg["robot"]["gripper_driver_joint"]``;
            its commanded position drives the open/closed check downstream.

    Raises:
        ValueError: If ``gripper_driver_joint`` isn't in the robot's joint list.
    """
    import numpy as np  # noqa: F401  — kept lazy to match module convention

    scene = env.unwrapped.scene
    env_origin = scene.env_origins[0].cpu().numpy()
    tcp = scene["ee_frame"].data.target_pos_w[0, 0, :].cpu().numpy() - env_origin
    pickable = scene["pickable"].data.root_pos_w[0].cpu().numpy() - env_origin
    target = scene["target"].data.root_pos_w[0].cpu().numpy() - env_origin
    robot = scene["robot"]
    grip_idx = robot.data.joint_names.index(gripper_driver_joint)
    gripper_pos = float(robot.data.joint_pos[0, grip_idx])
    return EnvStateSnapshot(tcp=tcp, pickable=pickable, target=target, gripper_pos=gripper_pos)


def oracle_action_at_state(
    *,
    tcp: "np.ndarray",
    pickable_pos: "np.ndarray",
    target_pos: "np.ndarray",
    gripper_drive_pos: float,
    gripper_closed_threshold: float,
    params: _OracleParams,
) -> "np.ndarray":
    """Stateless oracle: return the action the scripted policy would emit for
    the given env snapshot.

    The original ``main()`` loop tracks an explicit ``Phase`` plus several
    counters (``hold_counter``, ``grasp_pickable_pos``, ``lift_xy_ref``).
    For DAgger we need to query the oracle from arbitrary policy-visited
    states, so we re-derive "what to do next" from observable signals only:

      1. Is the gripper currently closed (driver joint > closed threshold)?
      2. Is the pickable lifted off the table (z above a small clearance)?
      3. Is the TCP above the pickable / above the target?
      4. Has the TCP descended to grasp / place height?

    No hidden counters means we can't reproduce the original "hold for N
    steps after closing the jaws" behavior, but with a stiff gripper the
    finger reaches its commanded position in ~5 steps and the closed-loop
    policy will keep sending the close command on its own.

    Returns: 4-D ``[dx, dy, dz, gripper]`` for ``ik_command_type=position``,
    7-D ``[dx, dy, dz, drx, dry, drz, gripper]`` for ``ik_command_type=pose``.
    Position deltas are clipped to ``params.max_dxy`` / ``params.max_dz``
    so a single oracle call can't command an over-aggressive jump.
    """
    import numpy as np

    p = params
    gripper_is_closed = gripper_drive_pos > gripper_closed_threshold

    # Heuristic: pickable is "lifted" when its centroid is appreciably above
    # the table. The cuboid is 0.15 m tall with centroid at z=0.075 when
    # resting on the table; we consider z > 0.10 as "in the air".
    plant_lifted = pickable_pos[2] > 0.10

    # xy alignment checks reuse the same tolerances as ``_reached``.
    def _xy_close(a, b):
        return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2])) < p.xy_reach_tol

    # Detect "gripper closed but holding nothing": the policy may have
    # squeezed the jaws on empty air, then the cuboid stayed on the table.
    # Treat this as "open the gripper and try the pick again" rather than
    # advancing into LIFT/MOVE on a phantom payload.
    if gripper_is_closed:
        import numpy as _np
        ee_to_plant = float(_np.linalg.norm(_np.asarray(tcp) - _np.asarray(pickable_pos)))
        if ee_to_plant > 0.10:
            # Reopen and head back above the plant.
            wp = _xyz(pickable_pos[0], pickable_pos[1], pickable_pos[2] + p.hover_height)
            return _compute_action(tcp, wp, gripper_open=True, p=p)

    if not gripper_is_closed:
        # ---- Pre-grasp branch: HOVER → DESCEND → close gripper -----------
        if not _xy_close(tcp, pickable_pos):
            wp = _xyz(pickable_pos[0], pickable_pos[1], pickable_pos[2] + p.hover_height)
            return _compute_action(tcp, wp, gripper_open=True, p=p)

        descend_z = pickable_pos[2] + p.grasp_z_offset
        if tcp[2] > descend_z + p.z_reach_tol_tight:
            wp = _xyz(pickable_pos[0], pickable_pos[1], descend_z)
            return _compute_action(tcp, wp, gripper_open=True, p=p)

        # In place to grasp: hold position, command close.
        wp = _xyz(pickable_pos[0], pickable_pos[1], descend_z)
        return _compute_action(tcp, wp, gripper_open=False, p=p)

    # ---- Post-grasp branch: LIFT → MOVE → PLACE → RELEASE ---------------
    if not plant_lifted:
        # Gripper just closed; rise straight up before moving laterally.
        wp = _xyz(tcp[0], tcp[1], p.lift_height)
        return _compute_action(tcp, wp, gripper_open=False, p=p)

    if not _xy_close(tcp, target_pos):
        wp = _xyz(target_pos[0], target_pos[1], p.lift_height)
        return _compute_action(tcp, wp, gripper_open=False, p=p)

    place_z = target_pos[2] + p.place_z_offset
    if tcp[2] > place_z + p.z_reach_tol_tight:
        wp = _xyz(target_pos[0], target_pos[1], place_z)
        return _compute_action(tcp, wp, gripper_open=False, p=p)

    # In place to release: hold position, command open.
    wp = _xyz(target_pos[0], target_pos[1], place_z)
    return _compute_action(tcp, wp, gripper_open=True, p=p)


if __name__ == "__main__":
    # Standalone: load default task config, then run.
    from arm_act.config import DEFAULT_TASK, load
    sys.exit(main(load(DEFAULT_TASK)))
