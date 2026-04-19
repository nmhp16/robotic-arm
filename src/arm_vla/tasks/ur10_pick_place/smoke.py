"""Smoke test for the UR10 pick-and-place env.

Spawns the env headless, steps with zero actions for ~1 second of sim time,
and prints observation keys + shapes. No policy, no teleop — just confirms
the env cfg wires up, all assets load, and observations flow.

Run inside Isaac Lab's bundled python:

    ~/IsaacLab/isaaclab.sh -p -m arm_vla.tasks.ur10_pick_place.smoke

Exits 0 on success, non-zero on any exception.
"""

from __future__ import annotations

import sys
import traceback

from isaaclab.app import AppLauncher


def _run():
    # AppLauncher must be constructed before any isaaclab imports that touch
    # Omniverse — it boots the SimulationApp.
    app = AppLauncher(headless=True)
    simulation_app = app.app

    try:
        import gymnasium as gym  # noqa: F401
        import torch

        # Register the gym id (side effect of importing the package).
        import arm_vla.tasks.ur10_pick_place  # noqa: F401
        from arm_vla.tasks.ur10_pick_place.pick_place_ur10_env_cfg import UR10PickPlaceEnvCfg

        cfg = UR10PickPlaceEnvCfg()
        cfg.scene.num_envs = 1

        env = gym.make("Isaac-PickPlace-UR10-IK-Rel-v0", cfg=cfg)
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

            # Zero action — 6-D IK-rel pose + 1-D suction = 7
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
