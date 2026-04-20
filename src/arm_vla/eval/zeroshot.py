"""Zero-shot OpenVLA eval on our Isaac Lab tasks.

Loads the pretrained ``openvla/openvla-7b`` checkpoint without any
fine-tuning and runs it on our env. Uses ``bridge_orig`` as the
unnorm_key (the WidowX BridgeData V2 action distribution shipped with the
model) since our 7-D delta-pose + gripper action space matches its shape.

Don't expect success: zero-shot on a new robot is typically sub-20%.
The point is to characterize the baseline before fine-tuning.

    ./scripts/zeroshot.sh --task pick_place --episodes 3
    ./scripts/zeroshot.sh --task stack --episodes 3
"""

from __future__ import annotations

import argparse
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
    },
    "pick_place_ur10": {
        "gym_id": "Isaac-PickPlace-UR10-IK-Rel-v0",
        "module": "arm_vla.tasks.ur10_pick_place",
        "cfg_path": "arm_vla.tasks.ur10_pick_place.pick_place_ur10_env_cfg:UR10PickPlaceEnvCfg",
        "instruction": "pick up the blue block and place it on the green pad",
    },
    "stack": {
        "gym_id": "Isaac-Stack-UR10-IK-Rel-v0",
        "module": "arm_vla.tasks.ur10_stack",
        "cfg_path": "arm_vla.tasks.ur10_stack.stack_ur10_env_cfg:UR10StackEnvCfg",
        "instruction": "stack the blue block on top of the red block",
    },
}


@dataclass
class Args:
    task: str
    episodes: int
    max_steps: int
    unnorm_key: str
    out: pathlib.Path
    fps: int


def _parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASK_REGISTRY), required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--unnorm-key", type=str, default="bridge_orig")
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()
    out = args.out or pathlib.Path(f"media/openvla_zeroshot_{args.task}.mp4")
    return Args(
        task=args.task,
        episodes=args.episodes,
        max_steps=args.max_steps,
        unnorm_key=args.unnorm_key,
        out=out,
        fps=args.fps,
    )


def main() -> int:
    args = _parse_args()
    spec = TASK_REGISTRY[args.task]

    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import gymnasium as gym
        import importlib
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor

        importlib.import_module(spec["module"])  # registers gym id

        print(f"loading openvla/openvla-7b (first run downloads ~14 GB)", flush=True)
        processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

        # transformers 4.57+ expects every HF model class to declare
        # _supports_sdpa / _supports_flash_attn_2, but OpenVLA (pinned to
        # transformers 4.40) doesn't. Patch the dynamically-loaded class
        # before from_pretrained instantiates it.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        _vla_cls = get_class_from_dynamic_module(
            "modeling_prismatic.OpenVLAForActionPrediction",
            "openvla/openvla-7b",
        )
        _vla_cls._supports_sdpa = True
        _vla_cls._supports_flash_attn_2 = False

        vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).to("cuda")
        vla.eval()

        prompt = f"In: What action should the robot take to {spec['instruction']}?\nOut:"
        print(f"task: {args.task}  instruction: {spec['instruction']}", flush=True)

        cfg_mod_path, cfg_cls = spec["cfg_path"].split(":")
        cfg_mod = importlib.import_module(cfg_mod_path)
        cfg = getattr(cfg_mod, cfg_cls)()
        cfg.scene.num_envs = 1
        env = gym.make(spec["gym_id"], cfg=cfg)
        device = env.unwrapped.device

        all_frames: list = []
        summary: list[dict] = []

        for ep in range(args.episodes):
            obs, _ = env.reset()
            ep_frames: list = []
            success = False
            steps = 0

            for t in range(args.max_steps):
                table = obs["policy"]["table_cam"][0].cpu().numpy().astype("uint8")
                wrist = obs["policy"]["wrist_cam"][0].cpu().numpy().astype("uint8")
                ep_frames.append(np.concatenate([table, wrist], axis=1))

                pil = Image.fromarray(table)
                inputs = processor(prompt, pil).to("cuda", dtype=torch.bfloat16)

                with torch.inference_mode():
                    action = vla.predict_action(
                        **inputs, unnorm_key=args.unnorm_key, do_sample=False
                    )
                action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

                if t % 10 == 0:
                    a = [float(v) for v in action]
                    eef = obs["policy"]["eef_pos"][0].cpu().numpy()
                    print(
                        f"    t={t:>3d}  action=[{a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f} | "
                        f"{a[3]:+.2f},{a[4]:+.2f},{a[5]:+.2f} | grip={a[6]:+.1f}]  "
                        f"tcp=[{eef[0]:.3f},{eef[1]:.3f},{eef[2]:.3f}]",
                        flush=True,
                    )

                obs, _, terminated, truncated, info = env.step(action_t)
                steps = t + 1

                if bool(terminated[0]) or bool(truncated[0]):
                    term_info = info.get("termination", {}) if isinstance(info, dict) else {}
                    if bool(term_info.get("success", False)):
                        success = True
                    break

            print(f"  episode {ep}: {'SUCCESS' if success else 'FAIL':<7}  ({steps} steps)", flush=True)
            summary.append({"episode": ep, "success": success, "steps": steps})
            all_frames.extend(ep_frames)
            # Separator frame between episodes (black)
            if ep < args.episodes - 1 and ep_frames:
                blank = np.zeros_like(ep_frames[0])
                all_frames.extend([blank] * 5)

        # Save video + JSON BEFORE env.close() — the close teardown can
        # SIGABRT on Isaac Lab's SurfaceGripper cleanup and lose everything.
        _save_video(args.out, all_frames, args.fps)
        with open(args.out.with_suffix(".json"), "w") as f:
            json.dump(
                {
                    "task": args.task,
                    "instruction": spec["instruction"],
                    "unnorm_key": args.unnorm_key,
                    "episodes": summary,
                    "success_rate": sum(e["success"] for e in summary) / max(1, len(summary)),
                },
                f,
                indent=2,
            )
        rate = sum(e["success"] for e in summary) / max(1, len(summary))
        print(f"\nzero-shot success rate on {args.task}: {rate:.0%} ({sum(e['success'] for e in summary)}/{len(summary)})", flush=True)

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


def _save_video(path: pathlib.Path, frames: list, fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        imageio.mimsave(str(path), frames, fps=fps)
        print(f"wrote {path} ({len(frames)} frames @ {fps} fps)")
    except ImportError:
        print("imageio missing, skipping video", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
