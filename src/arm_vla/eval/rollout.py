"""Evaluate a fine-tuned OpenVLA checkpoint via sim rollouts.

    ./scripts/eval.sh --checkpoint checkpoints/openvla-ur5-pickplace-lora/final

Loads the base model + LoRA adapters, opens the same Isaac Lab env used for
data collection, runs N episodes, and writes a summary plus per-episode
video to ``eval/runs/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import dataclass

from isaaclab.app import AppLauncher


@dataclass
class EvalArgs:
    checkpoint: pathlib.Path
    num_episodes: int = 50
    max_steps_per_episode: int = 400
    instruction: str = "put the blue cube on the green target"
    output_dir: pathlib.Path = pathlib.Path("eval/runs")
    unnorm_key: str = "ur5_pick_place"
    record_video: bool = True


def _parse_args() -> EvalArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--max-steps-per-episode", type=int, default=400)
    p.add_argument("--instruction", type=str, default="put the blue cube on the green target")
    p.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("eval/runs"))
    p.add_argument("--unnorm-key", type=str, default="ur5_pick_place")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    return EvalArgs(
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        instruction=args.instruction,
        output_dir=args.output_dir,
        unnorm_key=args.unnorm_key,
        record_video=not args.no_video,
    )


def main() -> None:
    args = _parse_args()
    run_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Boot Isaac Sim before importing any isaaclab module.
    app = AppLauncher(headless=not args.record_video, enable_cameras=True)
    sim_app = app.app

    try:
        import gymnasium as gym
        import torch
        from peft import PeftModel
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor

        import arm_vla.tasks.ur5_pick_place  # noqa: F401  registers gym id

        processor = AutoProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=True)
        base = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).to("cuda")
        vla = PeftModel.from_pretrained(base, str(args.checkpoint))
        vla.eval()

        prompt = f"In: What action should the robot take to {args.instruction}?\nOut:"

        env = gym.make("Isaac-PickPlace-UR5-IK-Rel-v0")
        successes = 0
        results = []

        for ep in range(args.num_episodes):
            obs, _ = env.reset()
            frames: list | None = [] if args.record_video else None
            success = False
            steps_to_success: int | None = None

            for t in range(args.max_steps_per_episode):
                img = obs["policy"]["table_cam"][0].cpu().numpy().astype("uint8")
                inputs = processor(prompt, Image.fromarray(img)).to("cuda", dtype=torch.bfloat16)

                with torch.inference_mode():
                    action = vla.predict_action(
                        **inputs, unnorm_key=args.unnorm_key, do_sample=False
                    )
                action_t = torch.as_tensor(
                    action, dtype=torch.float32, device=env.unwrapped.device
                ).unsqueeze(0)

                obs, _, terminated, truncated, info = env.step(action_t)

                if frames is not None and t % 4 == 0:
                    frames.append(img.copy())

                if bool(terminated[0]) or bool(truncated[0]):
                    term_info = info.get("termination", {}) if isinstance(info, dict) else {}
                    if bool(term_info.get("success", False)) or _check_success(env):
                        success = True
                        steps_to_success = t + 1
                    break

            successes += int(success)
            results.append({
                "episode": ep,
                "success": success,
                "steps": steps_to_success or args.max_steps_per_episode,
            })
            print(f"ep {ep:>3d}: {'SUCCESS' if success else 'FAIL':<7}  ({results[-1]['steps']} steps)")

            if frames is not None and frames:
                _save_video(run_dir / f"ep_{ep:03d}.mp4", frames)

        env.close()

        rate = successes / args.num_episodes
        summary = {
            "num_episodes": args.num_episodes,
            "successes": successes,
            "success_rate": rate,
            "episodes": results,
            "checkpoint": str(args.checkpoint),
            "instruction": args.instruction,
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nsuccess rate: {rate:.1%} ({successes}/{args.num_episodes})")
        print(f"results: {run_dir}")

    finally:
        sim_app.close()


def _check_success(env) -> bool:
    try:
        term = env.unwrapped.cfg.terminations.success
        return bool(term.func(env.unwrapped, **term.params)[0])
    except Exception:
        return False


def _save_video(path: pathlib.Path, frames: list) -> None:
    try:
        import imageio
        imageio.mimsave(str(path), frames, fps=8)
    except ImportError:
        from PIL import Image
        png_dir = path.with_suffix("")
        png_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(png_dir / f"{i:04d}.png")


if __name__ == "__main__":
    main()
