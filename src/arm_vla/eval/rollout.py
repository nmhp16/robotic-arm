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
import json
import pathlib
import sys
import time
import traceback
from dataclasses import dataclass

from isaaclab.app import AppLauncher


TASK_REGISTRY = {
    "pick_place": {
        "gym_id": "Isaac-PickPlace-UR5-IK-Rel-v0",
        "module": "arm_vla.tasks.ur5_pick_place",
        "cfg_path": "arm_vla.tasks.ur5_pick_place.pick_place_ur5_env_cfg:UR5PickPlaceEnvCfg",
        "instruction": "pick up the blue block and place it on the green pad",
        "unnorm_key": "ur5_pick_place",
    },
    "pick_place_ur10": {
        "gym_id": "Isaac-PickPlace-UR10-IK-Rel-v0",
        "module": "arm_vla.tasks.ur10_pick_place",
        "cfg_path": "arm_vla.tasks.ur10_pick_place.pick_place_ur10_env_cfg:UR10PickPlaceEnvCfg",
        "instruction": "pick up the blue block and place it on the green pad",
        "unnorm_key": "ur10_pick_place",
    },
    "stack": {
        "gym_id": "Isaac-Stack-UR10-IK-Rel-v0",
        "module": "arm_vla.tasks.ur10_stack",
        "cfg_path": "arm_vla.tasks.ur10_stack.stack_ur10_env_cfg:UR10StackEnvCfg",
        "instruction": "stack the blue block on top of the red block",
        "unnorm_key": "ur10_pick_place",
    },
}


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
    p.add_argument("--unnorm-key", type=str, default=None,
                   help="Override the per-task default (which matches the RLDS dataset name)")
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
    args = _parse_args()
    spec = TASK_REGISTRY[args.task]
    instruction = args.instruction or spec["instruction"]
    unnorm_key = args.unnorm_key or spec["unnorm_key"]

    run_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import gymnasium as gym
        import numpy as np  # noqa: F401
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
        print(f"task: {args.task}  instruction: {instruction}", flush=True)

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
                import numpy as np
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
            print(f"ep {ep:>3d}: {'SUCCESS' if success else 'FAIL':<7}  ({entry['steps']} steps)", flush=True)

            if args.record_video and frames:
                _save_video(run_dir / f"ep_{ep:03d}.mp4", frames, fps=15)
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
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if args.record_video and episode_frames:
            all_frames: list = []
            for i, ep_frames in enumerate(episode_frames):
                all_frames.extend(ep_frames)
                if i < len(episode_frames) - 1 and ep_frames:
                    import numpy as np
                    all_frames.extend([np.zeros_like(ep_frames[0])] * 8)
            _save_video(run_dir / "reel.mp4", all_frames, fps=15)

        print(f"\nsuccess rate: {rate:.1%} ({successes}/{args.num_episodes})", flush=True)
        print(f"results: {run_dir}", flush=True)

        try:
            env.close()
        except Exception as e:
            print(f"env.close() raised (ignored): {e}", flush=True)

    except Exception:
        traceback.print_exc()
        return 1
    finally:
        app.close()

    return 0


def _check_success(env) -> bool:
    try:
        term = env.unwrapped.cfg.terminations.success
        return bool(term.func(env.unwrapped, **term.params)[0])
    except Exception:
        return False


def _save_video(path: pathlib.Path, frames: list, fps: int = 15) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        imageio.mimsave(str(path), frames, fps=fps)
        print(f"wrote {path}", flush=True)
    except ImportError:
        print("imageio missing, skipping video", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
