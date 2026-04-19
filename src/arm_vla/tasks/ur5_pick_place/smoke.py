"""Smoke test for the UR5e pick-and-place env.

Boots headless, spawns the env, steps with zero actions for ~1 s, prints obs
keys + shapes, exits. No policy, no teleop — just confirms wiring.

    ~/IsaacLab/isaaclab.sh -p -m arm_vla.tasks.ur5_pick_place.smoke
"""

from __future__ import annotations

import sys
import traceback

from isaaclab.app import AppLauncher


def _run():
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
            obs, info = env.reset()
            print("=== observation keys / shapes ===")
            for group, terms in obs.items():
                if isinstance(terms, dict):
                    for name, val in terms.items():
                        shape = tuple(val.shape) if hasattr(val, "shape") else type(val).__name__
                        print(f"  {group}.{name}: {shape}")
                else:
                    shape = tuple(terms.shape) if hasattr(terms, "shape") else type(terms).__name__
                    print(f"  {group}: {shape}")

            action = torch.zeros((1, 7), device=env.unwrapped.device)
            for i in range(20):  # 20 * decimation(5) * dt(0.01) = 1 s
                env.step(action)
            print(f"=== stepped {i + 1} times, env alive ===")
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
