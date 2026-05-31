"""Headless joint trace for diagnosing arm shake.

Steps the env with zero actions for N ticks and prints joint positions and
velocities so we can see whether the arm is steady or oscillating.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True, enable_cameras=True)
_sim = _app.app


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pick_plant_out")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--random", action="store_true")
    p.add_argument("--scale", type=float, default=0.005, help="random delta scale (matches smoke.py walk)")
    p.add_argument("--clip", type=float, default=0.04, help="cumulative drift clamp")
    args = p.parse_args()

    import gymnasium as gym
    import numpy as np
    import torch

    import arm_act.tasks
    from arm_act.config import load

    arm_act.tasks.register()
    spec = load(args.task)
    gym_id = spec["task"]["gym_id"]
    env_cfg_spec = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    mod_name, cls_name = env_cfg_spec.split(":")
    env_cfg = getattr(importlib.import_module(mod_name), cls_name)()
    env_cfg.scene.num_envs = 1

    env = gym.make(gym_id, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        device = env.unwrapped.device
        action_dim = int(env.unwrapped.action_manager.total_action_dim)
        zero = torch.zeros((1, action_dim), device=device)
        rng = np.random.default_rng(0)

        robot = env.unwrapped.scene["robot"]
        names = list(robot.data.joint_names)
        # Header
        sys.stdout.write("step  " + "  ".join(f"{n[:8]:>8s}" for n in names) +
                         "  | " + "  ".join(f"{('v_'+n[:5]):>7s}" for n in names) + "\n")
        sys.stdout.flush()
        walk_state = np.zeros(action_dim, dtype="float32")
        for s in range(args.steps):
            if args.random:
                walk_state += rng.uniform(-1.0, 1.0, size=action_dim).astype("float32") * args.scale
                np.clip(walk_state, -args.clip, args.clip, out=walk_state)
                a = walk_state.copy()
                if action_dim > 0:
                    a[-1] = 0.0
                action = torch.from_numpy(a).to(device).unsqueeze(0)
            else:
                action = zero
            obs, *_ = env.step(action)
            q = robot.data.joint_pos[0].cpu().numpy()
            v = robot.data.joint_vel[0].cpu().numpy()
            sys.stdout.write(f"{s:>4d}  " + "  ".join(f"{x:+8.5f}" for x in q) +
                             "  | " + "  ".join(f"{x:+7.4f}" for x in v) + "\n")
            sys.stdout.flush()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    rc = 0
    try:
        rc = main()
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        _sim.close()
    sys.exit(rc)
