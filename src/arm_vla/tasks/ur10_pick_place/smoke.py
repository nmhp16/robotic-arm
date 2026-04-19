"""Sanity check for the UR10 pick-and-place env.

    ~/IsaacLab/isaaclab.sh -p -m arm_vla.tasks.ur10_pick_place.smoke

Flags:
    --visible        open the Isaac Sim GUI instead of running headless
    --random         drive the arm with small random Delta-pose actions so
                     motion is visible (otherwise arm stays at home)
    --steps N        number of sim steps (default: 20)
    --dump-cams DIR  write one frame from each RGB camera to DIR

Prints observation shapes, then steps the env.
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
    return p.parse_args()


def _run(args: argparse.Namespace) -> int:
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
            print("observation shapes:")
            for group, terms in obs.items():
                if isinstance(terms, dict):
                    for name, val in terms.items():
                        shape = tuple(val.shape) if hasattr(val, "shape") else type(val).__name__
                        print(f"  {group}.{name}: {shape}")
                else:
                    shape = tuple(terms.shape) if hasattr(terms, "shape") else type(terms).__name__
                    print(f"  {group}: {shape}")

            device = env.unwrapped.device
            rng = np.random.default_rng(0)
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

            print(f"stepped {args.steps} times, env alive")

            if args.dump_cams is not None:
                _dump_camera_frames(obs, args.dump_cams)

        finally:
            env.close()

    except Exception:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()

    return 0


def _dump_camera_frames(obs, out_dir: pathlib.Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed; skipping camera dump", file=sys.stderr)
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
        print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(_run(_parse_args()))
