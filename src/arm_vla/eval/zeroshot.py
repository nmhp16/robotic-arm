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
import importlib
import logging
import pathlib
import sys
from dataclasses import dataclass

from isaaclab.app import AppLauncher

from arm_vla.eval.common import TASK_REGISTRY, save_summary, save_video, setup_logging

logger = logging.getLogger(__name__)


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
    setup_logging()
    args = _parse_args()
    spec = TASK_REGISTRY[args.task]

    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor

        importlib.import_module(spec["module"])  # registers gym id

        logger.info("loading openvla/openvla-7b (first run downloads ~14 GB)")
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
        logger.info("task: %s  instruction: %s", args.task, spec["instruction"])

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
                    logger.info(
                        "    t=%3d  action=[%+.3f,%+.3f,%+.3f | %+.2f,%+.2f,%+.2f | grip=%+.1f]"
                        "  tcp=[%.3f,%.3f,%.3f]",
                        t, a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                        eef[0], eef[1], eef[2],
                    )

                obs, _, terminated, truncated, info = env.step(action_t)
                steps = t + 1

                if bool(terminated[0]) or bool(truncated[0]):
                    term_info = info.get("termination", {}) if isinstance(info, dict) else {}
                    if bool(term_info.get("success", False)):
                        success = True
                    break

            logger.info(
                "  episode %d: %-7s  (%d steps)",
                ep,
                "SUCCESS" if success else "FAIL",
                steps,
            )
            summary.append({"episode": ep, "success": success, "steps": steps})
            all_frames.extend(ep_frames)
            # Separator frame between episodes (black)
            if ep < args.episodes - 1 and ep_frames:
                blank = np.zeros_like(ep_frames[0])
                all_frames.extend([blank] * 5)

        # Save video + JSON BEFORE env.close() — the close teardown can
        # SIGABRT on Isaac Lab's SurfaceGripper cleanup and lose everything.
        save_video(args.out, all_frames, args.fps)
        rate = sum(e["success"] for e in summary) / max(1, len(summary))
        save_summary(
            args.out.with_suffix(".json"),
            {
                "task": args.task,
                "instruction": spec["instruction"],
                "unnorm_key": args.unnorm_key,
                "episodes": summary,
                "success_rate": rate,
            },
        )
        logger.info(
            "zero-shot success rate on %s: %.0f%% (%d/%d)",
            args.task,
            rate * 100,
            sum(e["success"] for e in summary),
            len(summary),
        )

        try:
            env.close()
        except (RuntimeError, AttributeError, AssertionError):
            # Isaac Lab SurfaceGripper + PhysX teardown ordering can raise
            # here; simulation is already shutting down so it's safe to
            # swallow. SIGABRT can still happen and is uncatchable.
            logger.warning("env.close() raised during teardown (ignored)", exc_info=True)

    except Exception:
        logger.exception("zeroshot eval failed")
        return 1
    finally:
        app.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
