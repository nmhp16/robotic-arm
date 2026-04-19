"""Headless sanity check for the UR5e pick-and-place env.

    ~/IsaacLab/isaaclab.sh -p -m arm_vla.tasks.ur5_pick_place.smoke

Spawns the env, steps with zero actions for ~1 s, prints observation
shapes, exits.
"""

from __future__ import annotations

import sys
import traceback

from isaaclab.app import AppLauncher


def _run() -> int:
    app = AppLauncher(headless=True)
    simulation_app = app.app

    try:
        import gymnasium as gym  # noqa: F401
        import torch

        import arm_vla.tasks.ur5_pick_place  # noqa: F401  registers gym id
        from arm_vla.tasks.ur5_pick_place.pick_place_ur5_env_cfg import UR5PickPlaceEnvCfg

        cfg = UR5PickPlaceEnvCfg()
        cfg.scene.num_envs = 1

        env = gym.make("Isaac-PickPlace-UR5-IK-Rel-v0", cfg=cfg)
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

            action = torch.zeros((1, 7), device=env.unwrapped.device)
            for _ in range(20):
                env.step(action)
            print("stepped 20 times, env alive")
        finally:
            env.close()

    except Exception:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()

    return 0


if __name__ == "__main__":
    sys.exit(_run())
