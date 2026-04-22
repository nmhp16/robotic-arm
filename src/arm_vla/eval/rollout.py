"""Evaluate a fine-tuned OpenVLA checkpoint via sim rollouts.

Loads the base model + LoRA adapters, opens the Isaac Lab task env, runs
N episodes, and writes a summary plus per-episode mp4 videos into
``eval/runs/<timestamp>/``.

    ./scripts/eval.sh --checkpoint checkpoints/openvla-ur5-pickplace-lora/final
    ./scripts/eval.sh --checkpoint <ckpt> --task pick_place_ur10
    ./scripts/eval.sh --checkpoint <ckpt> --task stack
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
    instruction: str | None = None
    output_dir: pathlib.Path = pathlib.Path("eval/runs")
    unnorm_key: str | None = None
    record_video: bool = True


def _parse_args() -> EvalArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    p.add_argument("--task", choices=list(TASK_REGISTRY), default="pick_place")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps-per-episode", type=int, default=300)
    p.add_argument("--instruction", type=str, default=None)
    p.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("eval/runs"))
    p.add_argument(
        "--unnorm-key",
        type=str,
        default=None,
        help="Override the per-task default (which matches the RLDS dataset name)",
    )
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    return EvalArgs(
        checkpoint=args.checkpoint,
        task=args.task,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        instruction=args.instruction,
        output_dir=args.output_dir,
        unnorm_key=args.unnorm_key,
        record_video=not args.no_video,
    )


def main() -> int:
    setup_logging()
    args = _parse_args()
    spec = TASK_REGISTRY[args.task]
    instruction = args.instruction or spec["instruction"]
    unnorm_key = args.unnorm_key or spec["unnorm_key"]

    run_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from peft import PeftModel
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor

        importlib.import_module(spec["module"])  # registers gym id

        # Patch OpenVLA class for transformers 4.57+ compatibility.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        _vla_cls = get_class_from_dynamic_module(
            "modeling_prismatic.OpenVLAForActionPrediction", "openvla/openvla-7b"
        )
        _vla_cls._supports_sdpa = True
        _vla_cls._supports_flash_attn_2 = False

        processor = AutoProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=True)
        base = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).to("cuda")
        vla = PeftModel.from_pretrained(base, str(args.checkpoint))
        vla.eval()

        cfg_mod_path, cfg_cls = spec["cfg_path"].split(":")
        cfg_mod = importlib.import_module(cfg_mod_path)
        cfg = getattr(cfg_mod, cfg_cls)()
        cfg.scene.num_envs = 1
        env = gym.make(spec["gym_id"], cfg=cfg)
        device = env.unwrapped.device

        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        logger.info("task: %s  instruction: %s", args.task, instruction)

        successes = 0
        results: list[dict] = []
        episode_frames: list[list] = []

        for ep in range(args.num_episodes):
            obs, _ = env.reset()
            frames: list = []
            success = False
            steps_to_success: int | None = None

            for t in range(args.max_steps_per_episode):
                table = obs["policy"]["table_cam"][0].cpu().numpy().astype("uint8")
                wrist = obs["policy"]["wrist_cam"][0].cpu().numpy().astype("uint8")
                frames.append(np.concatenate([table, wrist], axis=1))

                inputs = processor(prompt, Image.fromarray(table)).to("cuda", dtype=torch.bfloat16)
                with torch.inference_mode():
                    action = vla.predict_action(
                        **inputs, unnorm_key=unnorm_key, do_sample=False
                    )
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

        # Write summary + optional combined reel BEFORE env.close() —
        # Isaac Lab's SurfaceGripper cleanup can SIGABRT on shutdown.
        rate = successes / args.num_episodes
        summary = {
            "task": args.task,
            "instruction": instruction,
            "unnorm_key": unnorm_key,
            "num_episodes": args.num_episodes,
            "successes": successes,
            "success_rate": rate,
            "checkpoint": str(args.checkpoint),
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

        logger.info(
            "success rate: %.1f%% (%d/%d)",
            rate * 100,
            successes,
            args.num_episodes,
        )
        logger.info("results: %s", run_dir)

        try:
            env.close()
        except (RuntimeError, AttributeError, AssertionError):
            # Isaac Lab SurfaceGripper + PhysX teardown ordering can raise
            # here; the simulation is already shutting down so it's safe
            # to swallow. SIGABRT can still happen and is uncatchable.
            logger.warning("env.close() raised during teardown (ignored)", exc_info=True)

    except Exception:
        logger.exception("rollout failed")
        return 1
    finally:
        app.close()

    return 0


def _check_success(env) -> bool:
    try:
        term = env.unwrapped.cfg.terminations.success
        return bool(term.func(env.unwrapped, **term.params)[0])
    except (AttributeError, KeyError, IndexError, TypeError):
        return False


if __name__ == "__main__":
    sys.exit(main())
