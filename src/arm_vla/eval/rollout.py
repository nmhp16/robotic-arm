"""Evaluate a trained ACT checkpoint via Isaac Lab sim rollouts.

Loads the policy + normalization stats, opens the gym env, runs N
episodes, writes per-episode mp4 + a summary.json into
``eval/runs/<timestamp>/``.

    ./scripts/eval.sh --checkpoint checkpoints/act-ur5-pickplace/final
    ./scripts/eval.sh --checkpoint <ckpt> --task pick_place_ur10
"""

from __future__ import annotations

import argparse
import importlib
import logging
import pathlib
import sys
import time
from dataclasses import dataclass

from isaaclab.app import AppLauncher

from arm_vla.eval.common import TASK_REGISTRY, save_summary, save_video, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class EvalArgs:
    checkpoint: pathlib.Path
    task: str = "pick_place"
    num_episodes: int = 20
    max_steps_per_episode: int = 300
    output_dir: pathlib.Path = pathlib.Path("eval/runs")
    record_video: bool = True
    action_horizon: int | None = None


def _parse_args() -> EvalArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    p.add_argument("--task", choices=list(TASK_REGISTRY), default="pick_place")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps-per-episode", type=int, default=300)
    p.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("eval/runs"))
    p.add_argument("--no-video", action="store_true")
    p.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="How many actions of each predicted chunk to execute before re-planning. Default: chunk_size (full open-loop replay).",
    )
    a = p.parse_args()
    return EvalArgs(
        checkpoint=a.checkpoint,
        task=a.task,
        num_episodes=a.num_episodes,
        max_steps_per_episode=a.max_steps_per_episode,
        output_dir=a.output_dir,
        record_video=not a.no_video,
        action_horizon=a.action_horizon,
    )


def _check_success(env) -> bool:
    try:
        term = env.unwrapped.cfg.terminations.success
        return bool(term.func(env.unwrapped, **term.params)[0])
    except (AttributeError, KeyError, IndexError, TypeError):
        return False


def main() -> int:
    setup_logging()
    args = _parse_args()
    spec = TASK_REGISTRY[args.task]

    run_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import gymnasium as gym
        import numpy as np
        import torch

        from arm_vla.training.act_policy import load_policy

        importlib.import_module(spec["module"])  # registers gym id

        policy = load_policy(args.checkpoint, device="cuda")
        if args.action_horizon is not None:
            policy.action_horizon = args.action_horizon
        cam_keys = policy.model.camera_keys
        logger.info("loaded ACT policy: chunk=%d, action_horizon=%d, cams=%s",
                    policy.model.cfg.chunk_size, policy.action_horizon, cam_keys)

        cfg_mod_path, cfg_cls = spec["cfg_path"].split(":")
        cfg_mod = importlib.import_module(cfg_mod_path)
        env_cfg = getattr(cfg_mod, cfg_cls)()
        env_cfg.scene.num_envs = 1
        env = gym.make(spec["gym_id"], cfg=env_cfg)
        device = env.unwrapped.device

        successes = 0
        results: list[dict] = []
        episode_frames: list[list] = []

        for ep in range(args.num_episodes):
            obs, _ = env.reset()
            policy.reset()
            frames: list = []
            success = False
            steps_to_success: int | None = None

            for t in range(args.max_steps_per_episode):
                # Build the policy input from the env observation.
                cam_imgs: dict[str, torch.Tensor] = {}
                for k in cam_keys:
                    arr = obs["policy"][k][0].cpu().numpy().astype("uint8")  # (H, W, 3)
                    cam_imgs[k] = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

                # Reconstruct the 8-D state from the env: eef_pos(3) + eef_quat(4) + gripper(1).
                # The Mimic-augmented HDF5 stored these under obs/{eef_pos, eef_quat, gripper_pos};
                # at runtime they're available the same way via obs["policy"].
                eef_pos = obs["policy"]["eef_pos"][0].cpu().numpy().astype(np.float32).reshape(-1)
                eef_quat = obs["policy"]["eef_quat"][0].cpu().numpy().astype(np.float32).reshape(-1)
                grip = obs["policy"]["gripper_pos"][0].cpu().numpy().astype(np.float32).reshape(-1)[:1]
                state = torch.from_numpy(np.concatenate([eef_pos, eef_quat, grip], axis=0))

                # Record video as a side-by-side panel (table | wrist) for inspection.
                table = obs["policy"]["table_cam"][0].cpu().numpy().astype("uint8")
                wrist = obs["policy"]["wrist_cam"][0].cpu().numpy().astype("uint8")
                frames.append(np.concatenate([table, wrist], axis=1))

                action = policy.select_action(cam_imgs, state)  # (action_dim,)
                action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                obs, _, terminated, truncated, info = env.step(action_t)

                if bool(terminated[0]) or bool(truncated[0]):
                    term_info = info.get("termination", {}) if isinstance(info, dict) else {}
                    if bool(term_info.get("success", False)) or _check_success(env):
                        success = True
                        steps_to_success = t + 1
                    break

            successes += int(success)
            entry = {
                "episode": ep,
                "success": success,
                "steps": steps_to_success or args.max_steps_per_episode,
            }
            results.append(entry)
            logger.info(
                "ep %3d: %-7s  (%d steps)",
                ep,
                "SUCCESS" if success else "FAIL",
                entry["steps"],
            )

            if args.record_video and frames:
                save_video(run_dir / f"ep_{ep:03d}.mp4", frames, fps=15)
            episode_frames.append(frames)

        # Write summary + reel BEFORE env.close() — Isaac Lab's
        # SurfaceGripper cleanup can SIGABRT on shutdown.
        rate = successes / max(1, args.num_episodes)
        summary = {
            "task": args.task,
            "num_episodes": args.num_episodes,
            "successes": successes,
            "success_rate": rate,
            "checkpoint": str(args.checkpoint),
            "action_horizon": policy.action_horizon,
            "episodes": results,
        }
        save_summary(run_dir / "summary.json", summary)

        if args.record_video and episode_frames:
            all_frames: list = []
            for i, ep_frames in enumerate(episode_frames):
                all_frames.extend(ep_frames)
                if i < len(episode_frames) - 1 and ep_frames:
                    all_frames.extend([np.zeros_like(ep_frames[0])] * 8)
            save_video(run_dir / "reel.mp4", all_frames, fps=15)

        logger.info("success rate: %.1f%% (%d/%d)", rate * 100, successes, args.num_episodes)
        logger.info("results: %s", run_dir)

        try:
            env.close()
        except (RuntimeError, AttributeError, AssertionError):
            logger.warning("env.close() raised during teardown (ignored)", exc_info=True)

    except Exception:
        logger.exception("rollout failed")
        return 1
    finally:
        app.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
