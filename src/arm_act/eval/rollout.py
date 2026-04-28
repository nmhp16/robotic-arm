"""Evaluate a trained ACT checkpoint via Isaac Lab sim rollouts.

Reads ``defaults.yaml`` overlaid with ``tasks/<task>/task.yaml`` for the
task's gym id and eval defaults. Runs N episodes, writes per-episode mp4 +
``summary.json`` to ``<eval.output_dir>/<timestamp>/``.

    ./scripts/eval.sh                                              # uses --task pick_place
    ./scripts/eval.sh --task pick_place --num-episodes 5
    ./scripts/eval.sh --checkpoint checkpoints/pick_place/step_010000
"""

from __future__ import annotations

import argparse
import importlib
import logging
import pathlib
import sys
import time

from isaaclab.app import AppLauncher

from arm_vla.config import DEFAULT_TASK, load as load_config
from arm_vla.eval.common import save_summary, save_video, setup_logging

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULT_TASK, help="task name under src/arm_vla/tasks/")
    # All overrides are optional; default to the value in the merged config.
    p.add_argument("--checkpoint", type=pathlib.Path, default=None)
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument("--max-steps-per-episode", type=int, default=None)
    p.add_argument("--output-dir", type=pathlib.Path, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="How many actions of each predicted chunk to execute before re-planning. Default: chunk_size (full open-loop replay).",
    )
    return p.parse_args()


def _check_success(env) -> bool:
    try:
        term = env.unwrapped.cfg.terminations.success
        return bool(term.func(env.unwrapped, **term.params)[0])
    except (AttributeError, KeyError, IndexError, TypeError):
        return False


def main() -> int:
    setup_logging()
    args = _parse_args()
    cfg = load_config(args.task)
    task_cfg = cfg["task"]
    eval_cfg = cfg["eval"]

    checkpoint = pathlib.Path(args.checkpoint or eval_cfg["checkpoint"])
    num_episodes = args.num_episodes or eval_cfg["num_episodes"]
    max_steps_per_ep = args.max_steps_per_episode or eval_cfg["max_steps_per_episode"]
    output_dir = pathlib.Path(args.output_dir or eval_cfg["output_dir"])
    record_video = (not args.no_video) and bool(eval_cfg.get("record_video", True))
    action_horizon_override = (
        args.action_horizon if args.action_horizon is not None else eval_cfg.get("action_horizon")
    )

    run_dir = output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import gymnasium as gym
        import numpy as np
        import torch

        from arm_vla.training.act_policy import load_policy

        importlib.import_module("arm_vla.tasks")  # triggers task auto-registration

        policy = load_policy(checkpoint, device="cuda")
        if action_horizon_override is not None:
            policy.action_horizon = action_horizon_override
        cam_keys = policy.model.camera_keys
        logger.info(
            "task=%s  ckpt=%s  chunk=%d  action_horizon=%d  cams=%s",
            task_cfg["name"], checkpoint, policy.model.cfg.chunk_size,
            policy.action_horizon, cam_keys,
        )

        gym_id = task_cfg["gym_id"]
        env_cfg_spec = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
        cfg_mod_path, cfg_cls = env_cfg_spec.split(":")
        cfg_mod = importlib.import_module(cfg_mod_path)
        env_cfg = getattr(cfg_mod, cfg_cls)()
        env_cfg.scene.num_envs = 1
        env = gym.make(gym_id, cfg=env_cfg)
        device = env.unwrapped.device

        successes = 0
        results: list[dict] = []
        episode_frames: list[list] = []

        for ep in range(num_episodes):
            obs, _ = env.reset()
            policy.reset()
            frames: list = []
            success = False
            steps_to_success: int | None = None

            for t in range(max_steps_per_ep):
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
                "steps": steps_to_success or max_steps_per_ep,
            }
            results.append(entry)
            logger.info(
                "ep %3d: %-7s  (%d steps)",
                ep,
                "SUCCESS" if success else "FAIL",
                entry["steps"],
            )

            if record_video and frames:
                save_video(run_dir / f"ep_{ep:03d}.mp4", frames, fps=15)
            episode_frames.append(frames)

        # Write summary + reel BEFORE env.close() — Isaac Lab's
        # SurfaceGripper cleanup can SIGABRT on shutdown.
        rate = successes / max(1, num_episodes)
        summary = {
            "task": task_cfg["name"],
            "num_episodes": num_episodes,
            "successes": successes,
            "success_rate": rate,
            "checkpoint": str(checkpoint),
            "action_horizon": policy.action_horizon,
            "episodes": results,
        }
        save_summary(run_dir / "summary.json", summary)

        if record_video and episode_frames:
            all_frames: list = []
            for i, ep_frames in enumerate(episode_frames):
                all_frames.extend(ep_frames)
                if i < len(episode_frames) - 1 and ep_frames:
                    all_frames.extend([np.zeros_like(ep_frames[0])] * 8)
            save_video(run_dir / "reel.mp4", all_frames, fps=15)

        logger.info("success rate: %.1f%% (%d/%d)", rate * 100, successes, num_episodes)
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
