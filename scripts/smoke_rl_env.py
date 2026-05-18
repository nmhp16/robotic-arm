"""Smoke test: register the RL env variant for a task, gym.make it, run a
few random steps, log the per-step reward. Verifies the RewardsCfg + the
reward functions in ``mdp.py`` work end-to-end before launching a multi-
hour PPO run.

Works for any task that has a YAML in ``src/arm_act/tasks/`` — the RL gym
id is derived from the task's ``gym_id`` by replacing the trailing ``-v0``
with ``-RL-v0`` (handled by ``arm_act.tasks.register()``).

Usage:
    env -u VIRTUAL_ENV -u CONDA_PREFIX \\
        PYTHONPATH=src $ISAAC_LAB/isaaclab.sh -p scripts/smoke_rl_env.py \\
        --task put_plant_back --num-envs 4 --steps 10
"""
from __future__ import annotations
import argparse
import logging
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--task", default="put_plant_back",
               help="task name under src/arm_act/tasks/ (uses its gym_id with -RL- suffix)")
p.add_argument("--num-envs", type=int, default=4,
               help="parallel envs to spawn — keep small for smoke testing")
p.add_argument("--steps", type=int, default=10,
               help="random-action steps per env")
args = p.parse_args()

logging.basicConfig(level=logging.INFO, format="[rl-smoke] %(message)s")
log = logging.getLogger("rl-smoke")

app = AppLauncher(headless=True, enable_cameras=True).app

try:
    import importlib
    import torch
    import gymnasium as gym

    import arm_act.tasks
    arm_act.tasks.register()

    from arm_act.config import load

    il_gym_id = load(args.task)["task"]["gym_id"]
    if not il_gym_id.endswith("-v0"):
        raise RuntimeError(
            f"task gym_id {il_gym_id!r} doesn't end with -v0; the RL variant "
            "naming convention assumes it does")
    gym_id = il_gym_id[: -len("-v0")] + "-RL-v0"
    log.info("registering %s", gym_id)
    env_cfg_spec = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    cfg_mod_path, cfg_cls = env_cfg_spec.split(":")
    env_cfg = getattr(importlib.import_module(cfg_mod_path), cfg_cls)()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(gym_id, cfg=env_cfg)
    device = env.unwrapped.device
    log.info("env created — num_envs=%d device=%s", env_cfg.scene.num_envs, device)

    obs, _ = env.reset()
    action_dim = env.action_space.shape[-1]
    log.info("action_dim=%d obs keys=%s", action_dim, list(obs.keys()))

    for t in range(args.steps):
        # Small random action — just exercise the env + reward path.
        a = (torch.rand((env_cfg.scene.num_envs, action_dim), device=device) - 0.5) * 0.1
        obs, reward, term, trunc, info = env.step(a)
        log.info(
            "t=%d reward=%s term=%s trunc=%s",
            t,
            reward.cpu().numpy().tolist(),
            term.cpu().numpy().tolist(),
            trunc.cpu().numpy().tolist(),
        )

    env.close()
    log.info("smoke test PASSED")
except Exception:
    log.exception("smoke test FAILED")
    sys.exit(1)
finally:
    app.close()
