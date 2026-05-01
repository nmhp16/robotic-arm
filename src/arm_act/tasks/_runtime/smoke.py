"""Generic scene preview / sanity check for any registered task.

Invoked via ``./scripts/smoke.sh --task <name>``. The cli wrapper passes
the loaded spec into ``main(spec)``.

Flags (forwarded after --task):
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


def _run(args: argparse.Namespace, spec: dict) -> int:
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
    app = AppLauncher(headless=not args.visible, enable_cameras=True)
    simulation_app = app.app

    try:
        import gymnasium as gym
        import numpy as np
        import torch

        import arm_act.tasks
        arm_act.tasks.register()

        gym_id = spec["task"]["gym_id"]
        env_cfg_spec = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
        env_cfg = _instantiate_env_cfg(env_cfg_spec)
        env_cfg.scene.num_envs = 1

        env = gym.make(gym_id, cfg=env_cfg)
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
            action_dim = int(env.unwrapped.action_manager.total_action_dim)
            # Walk the EE around home with a small, smooth, drift-bounded
            # signal. Pure-random deltas of 2 cm/tick drive the SCARA into
            # joint-3 limits and J1/J2 singularities; that looks like shake
            # in the video but tells you nothing about the robot. A 5 mm
            # bounded random walk stays inside the reach envelope.
            walk_scale = 0.005      # 5 mm per axis per step
            walk_clip = 0.04        # clamp cumulative drift to +/- 4 cm
            walk_state = np.zeros(action_dim, dtype="float32")

            record_video = args.video_out is not None
            table_frames: list = []
            wrist_frames: list = []

            for _ in range(args.steps):
                if args.random:
                    walk_state += rng.uniform(-1.0, 1.0, size=action_dim).astype("float32") * walk_scale
                    np.clip(walk_state, -walk_clip, walk_clip, out=walk_state)
                    a = walk_state.copy()
                    if action_dim > 0:
                        a[-1] = 0.0  # gripper stays at neutral
                    action = torch.from_numpy(a).to(device).unsqueeze(0)
                else:
                    action = torch.zeros((1, action_dim), device=device)
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

    stem = out_path.with_suffix("")
    for frames, name in ((table_frames, "table_cam"), (wrist_frames, "wrist_cam")):
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        path = pathlib.Path(f"{stem}_{name}.mp4")
        imageio.mimsave(str(path), frames, fps=fps)
        logger.info("wrote %s", path)


def _instantiate_env_cfg(env_cfg_spec: str):
    import importlib

    mod_name, cls_name = env_cfg_spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()


def main(spec: dict) -> int:
    return _run(_parse_args(), spec)


if __name__ == "__main__":
    from arm_act.config import DEFAULT_TASK, load
    sys.exit(main(load(DEFAULT_TASK)))
