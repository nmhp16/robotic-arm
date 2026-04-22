"""Sanity check for the UR10 pick-and-place env.

    ~/IsaacLab/isaaclab.sh -p -m arm_vla.tasks.ur10_pick_place.smoke

Flags:
    --visible         open the Isaac Sim GUI instead of running headless
    --random          drive the arm with small random Delta-pose actions so
                      motion is visible (otherwise arm stays at home)
    --steps N         number of sim steps (default: 20)
    --dump-cams DIR   write the final frame of each RGB camera to DIR
    --video-out PATH  write per-step frames to an mp4 at PATH
    --video-fps N     fps for the video (default: 15)
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--visible", action="store_true")
    p.add_argument("--random", action="store_true")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--dump-cams", type=pathlib.Path, default=None)
    p.add_argument("--video-out", type=pathlib.Path, default=None)
    p.add_argument("--video-fps", type=int, default=15)
    return p.parse_args()


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = AppLauncher(headless=not args.visible, enable_cameras=True)
    simulation_app = app.app

    try:
        import gymnasium as gym  # noqa: F401
        import numpy as np
        import torch

        import arm_vla.tasks.ur10_pick_place  # noqa: F401  registers gym id
        from arm_vla.tasks.ur10_pick_place.pick_place_ur10_env_cfg import UR10PickPlaceEnvCfg

        cfg = UR10PickPlaceEnvCfg()
        cfg.scene.num_envs = 1

        env = gym.make("Isaac-PickPlace-UR10-IK-Rel-v0", cfg=cfg)
        try:
            obs, _ = env.reset()
            logger.info("observation shapes:")
            for group, terms in obs.items():
                if isinstance(terms, dict):
                    for name, val in terms.items():
                        shape = tuple(val.shape) if hasattr(val, "shape") else type(val).__name__
                        logger.info("  %s.%s: %s", group, name, shape)
                else:
                    shape = tuple(terms.shape) if hasattr(terms, "shape") else type(terms).__name__
                    logger.info("  %s: %s", group, shape)

            device = env.unwrapped.device
            rng = np.random.default_rng(0)

            record_video = args.video_out is not None
            table_frames: list = []
            wrist_frames: list = []

            for _ in range(args.steps):
                if args.random:
                    a = rng.uniform(-1.0, 1.0, size=7).astype("float32")
                    a[:3] *= 0.02
                    a[3:6] *= 0.05
                    a[6] = 0.0
                    action = torch.from_numpy(a).to(device).unsqueeze(0)
                else:
                    action = torch.zeros((1, 7), device=device)
                obs, *_ = env.step(action)

                if record_video:
                    table_frames.append(_grab(obs, "table_cam"))
                    wrist_frames.append(_grab(obs, "wrist_cam"))

            logger.info("stepped %d times, env alive", args.steps)

            if args.dump_cams is not None:
                _dump_camera_frames(obs, args.dump_cams)

            if record_video and table_frames:
                _save_videos(args.video_out, table_frames, wrist_frames, args.video_fps)

        finally:
            env.close()

    except Exception:
        logger.exception("smoke test failed")
        return 1
    finally:
        simulation_app.close()

    return 0


def _grab(obs, name: str):
    policy = obs.get("policy", {}) if isinstance(obs, dict) else {}
    tensor = policy.get(name)
    if tensor is None:
        return None
    return tensor[0].cpu().numpy().astype("uint8").copy()


def _dump_camera_frames(obs, out_dir: pathlib.Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not installed; skipping camera dump")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    policy = obs.get("policy", {}) if isinstance(obs, dict) else {}
    for name in ("table_cam", "wrist_cam"):
        tensor = policy.get(name)
        if tensor is None:
            continue
        frame = tensor[0].cpu().numpy().astype("uint8")
        path = out_dir / f"{name}.png"
        Image.fromarray(frame).save(path)
        logger.info("wrote %s", path)


def _save_videos(out_path: pathlib.Path, table_frames, wrist_frames, fps: int) -> None:
    try:
        import imageio
    except ImportError:
        logger.warning("imageio not installed; skipping video dump")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Side-by-side composite: table_cam on the left, wrist_cam on the right.
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
    logger.info("wrote %s (%d frames @ %d fps)", out_path, len(composite), fps)

    # Also write per-camera mp4s.
    stem = out_path.with_suffix("")
    for frames, name in ((table_frames, "table_cam"), (wrist_frames, "wrist_cam")):
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        path = pathlib.Path(f"{stem}_{name}.mp4")
        imageio.mimsave(str(path), frames, fps=fps)
        logger.info("wrote %s", path)


if __name__ == "__main__":
    sys.exit(_run(_parse_args()))
