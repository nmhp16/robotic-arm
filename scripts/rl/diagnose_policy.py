"""Side-by-side: what does the policy predict vs what the oracle would do at the same state.

Reset the env, run for N steps under the POLICY, log policy_action vs oracle_action
at each step. If they diverge from the very first step, the policy has a state/action
mapping bug. If they agree at step 0 but diverge later, it's compounding error.
"""
from __future__ import annotations
import argparse
import logging
import pathlib
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--task", default="put_plant_back",
               help="task name under src/arm_act/tasks/")
p.add_argument("--checkpoint", required=True,
               help="path to an ACT checkpoint directory (must contain model.pt + config.json)")
p.add_argument("--num-steps", type=int, default=30,
               help="steps per episode to roll out (kept short to focus on the early divergence)")
p.add_argument("--num-episodes", type=int, default=3,
               help="number of env resets to diagnose")
args = p.parse_args()

logging.basicConfig(level=logging.INFO, format="[diag] %(message)s", force=True)
log = logging.getLogger("diag")

app = AppLauncher(headless=True, enable_cameras=True).app

try:
    import importlib
    import numpy as np
    import torch
    import gymnasium as gym

    import arm_act.tasks
    arm_act.tasks.register()

    from arm_act.config import load
    from arm_act.training.act_policy import load_policy
    from arm_act.tasks._runtime.oracle import (
        _OracleParams,
        oracle_action_at_state,
        snapshot_env_state,
    )

    cfg = load(args.task)
    task_cfg = cfg["task"]
    gym_id = task_cfg["gym_id"]

    env_cfg_spec = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    cfg_mod_path, cfg_cls = env_cfg_spec.split(":")
    env_cfg = getattr(importlib.import_module(cfg_mod_path), cfg_cls)()
    env_cfg.scene.num_envs = 1
    env = gym.make(gym_id, cfg=env_cfg)
    device = env.unwrapped.device

    policy = load_policy(pathlib.Path(args.checkpoint), device="cuda")
    policy.action_horizon = 1  # closed-loop: re-predict every step
    cam_keys = policy.model.camera_keys

    params = _OracleParams.from_spec(cfg)
    grip_close_thr = float(cfg["robot"]["gripper_closed_threshold"])
    driver_joint = cfg["robot"]["gripper_driver_joint"]

    log.info("=== diagnostic start: %d episodes x %d steps ===", args.num_episodes, args.num_steps)
    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        policy.reset()
        log.info("--- ep %d ---", ep)
        for t in range(args.num_steps):
            snap = snapshot_env_state(env, driver_joint)

            # Oracle action at this state
            oracle_a = oracle_action_at_state(
                tcp=snap.tcp, pickable_pos=snap.pickable, target_pos=snap.target,
                gripper_drive_pos=snap.gripper_pos, gripper_closed_threshold=grip_close_thr,
                params=params,
            )

            # Policy action at this state
            eef_pos_obs = obs["policy"]["eef_pos"][0].cpu().numpy().astype(np.float32).reshape(-1)
            eef_quat_obs = obs["policy"]["eef_quat"][0].cpu().numpy().astype(np.float32).reshape(-1)
            grip_obs = obs["policy"]["gripper_pos"][0].cpu().numpy().astype(np.float32).reshape(-1)[:1]
            state_t = torch.from_numpy(np.concatenate([eef_pos_obs, eef_quat_obs, grip_obs], axis=0))
            cam_imgs = {}
            for k in cam_keys:
                arr = obs["policy"][k][0].cpu().numpy().astype("uint8")
                cam_imgs[k] = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            pol_a = policy.select_action(cam_imgs, state_t)

            # Log: state + both actions
            log.info(
                "ep%d t%02d  tcp_loc=[%+.3f %+.3f %+.3f]  pick_loc=[%+.3f %+.3f %+.3f]  tgt_loc=[%+.3f %+.3f %+.3f]  grip=%.3f",
                ep, t, *snap.tcp, *snap.pickable, *snap.target, snap.gripper_pos,
            )
            log.info(
                "       oracle=[%+.3f %+.3f %+.3f %+.2f]   policy=[%+.3f %+.3f %+.3f %+.2f]   diff=[%+.3f %+.3f %+.3f %+.2f]",
                *oracle_a, *pol_a,
                pol_a[0] - oracle_a[0], pol_a[1] - oracle_a[1],
                pol_a[2] - oracle_a[2], pol_a[3] - oracle_a[3],
            )

            # Step env with POLICY action (so we see the policy's actual rollout)
            a_t = torch.as_tensor(pol_a, dtype=torch.float32, device=device).unsqueeze(0)
            obs, _, term, trunc, _ = env.step(a_t)
            if bool(term[0]) or bool(trunc[0]):
                log.info("ep %d terminated/truncated at t=%d", ep, t)
                break

    env.close()
except Exception:
    log.exception("diagnose failed")
    sys.exit(1)
finally:
    app.close()
