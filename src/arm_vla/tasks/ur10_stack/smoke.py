"""Scene-preview smoke test for the UR10 stacking env.

Same pattern as ur10_pick_place.smoke, but spawns the stacking env.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback

from isaaclab.app import AppLauncher


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
    app = AppLauncher(headless=not args.visible, enable_cameras=True)
    simulation_app = app.app

    try:
        import gymnasium as gym  # noqa: F401
        import numpy as np
        import torch

        import arm_vla.tasks.ur10_stack  # noqa: F401  registers gym id
        from arm_vla.tasks.ur10_stack.stack_ur10_env_cfg import UR10StackEnvCfg

        cfg = UR10StackEnvCfg()
        cfg.scene.num_envs = 1

        env = gym.make("Isaac-Stack-UR10-IK-Rel-v0", cfg=cfg)
        try:
            obs, _ = env.reset()
            print("observation shapes:", flush=True)
            for group, terms in obs.items():
                if isinstance(terms, dict):
                    for name, val in terms.items():
                        shape = tuple(val.shape) if hasattr(val, "shape") else type(val).__name__
                        print(f"  {group}.{name}: {shape}", flush=True)
                else:
                    shape = tuple(terms.shape) if hasattr(terms, "shape") else type(terms).__name__
                    print(f"  {group}: {shape}", flush=True)

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

            print(f"stepped {args.steps} times, env alive", flush=True)

            if args.dump_cams is not None:
                _dump_camera_frames(obs, args.dump_cams)

            if record_video and table_frames:
                _save_videos(args.video_out, table_frames, wrist_frames, args.video_fps)

        finally:
            env.close()

    except Exception:
        traceback.print_exc()
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
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = obs.get("policy", {}) if isinstance(obs, dict) else {}
    for name in ("table_cam", "wrist_cam"):
        tensor = policy.get(name)
        if tensor is None:
            continue
        frame = tensor[0].cpu().numpy().astype("uint8")
        path = out_dir / f"{name}.png"
        Image.fromarray(frame).save(path)
        print(f"wrote {path}", flush=True)


def _save_videos(out_path: pathlib.Path, table_frames, wrist_frames, fps: int) -> None:
    import imageio
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite = []
    for t, w in zip(table_frames, wrist_frames):
        if t is None and w is None:
            continue
        if w is None:
            composite.append(t); continue
        if t is None:
            composite.append(w); continue
        composite.append(np.concatenate([t, w], axis=1))

    imageio.mimsave(str(out_path), composite, fps=fps)
    print(f"wrote {out_path} ({len(composite)} frames @ {fps} fps)", flush=True)


if __name__ == "__main__":
    sys.exit(_run(_parse_args()))
