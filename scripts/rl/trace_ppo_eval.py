"""Diagnostic: one env, dump plant_z / tcp_z / gripper / kin_attach over time.

Run after a PPO training to verify the policy is actually lifting the
plant rather than the success metric firing on a kinematic_attach
shortcut (or some other geometry trick).

    env -u VIRTUAL_ENV -u CONDA_PREFIX PYTHONPATH=src \
        ~/IsaacLab/isaaclab.sh -p scripts/trace_ppo_eval.py \
        --task Isaac-PutPlantBack-T3-IK-Rel-RL-v0 \
        --checkpoint checkpoints/arm_act_rl/<run>/model_199.pt
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--max_steps", type=int, default=80)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)  # adds --checkpoint, --resume, --load_run, ...
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import arm_act.tasks  # noqa: E402
arm_act.tasks.register()
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
import importlib.metadata as _md
_installed = _md.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _installed)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    term_mgr = env.unwrapped.termination_manager
    scene = env.unwrapped.scene
    plant = scene["pickable"]
    ee = scene["ee_frame"]
    robot = scene["robot"]
    gripper_joint = "finger_left_joint"
    j_idx = robot.joint_names.index(gripper_joint)
    unwrapped = env.unwrapped

    # The actual lift threshold used by the termination depends on the
    # cached _pickable_initial_z (set lazily on first eval call). Print
    # its value once so we can sanity-check whether plant.z would ever
    # exceed it during a rollout.
    print(f"[INFO] success term params: {term_mgr.get_term_cfg('success').params}")

    for ep in range(args_cli.num_episodes):
        plant_z0 = float(plant.data.root_pos_w[0, 2].item())
        cached_z0 = getattr(unwrapped, "_initial_pickable_z_pickable", None)
        cz = float(cached_z0[0].item()) if cached_z0 is not None else float("nan")
        print(f"\n=== episode {ep}  plant_z0_now={plant_z0:.4f}  cached_z0={cz:.4f} ===")
        print(f"{'step':>4} {'plant_xyz':>26} {'tcp_xyz':>26} {'grip':>6} {'attach':>6} {'plant_z-cz':>10} {'thresh':>8} {'lifted':>6}")
        for step in range(args_cli.max_steps):
            with torch.inference_mode():
                action = policy(obs)
                obs, _, dones, _ = env.step(action)
            plant_p = plant.data.root_pos_w[0].cpu().numpy()
            tcp_p = ee.data.target_pos_w[0, 0].cpu().numpy()
            grip = float(robot.data.joint_pos[0, j_idx].item())
            attach = bool(getattr(unwrapped, "_kinematic_attach_active", torch.zeros(1, dtype=torch.bool))[0].item())
            lifted = bool(term_mgr.get_term("success")[0].item()) if "success" in term_mgr.active_terms else False
            cached_z0 = getattr(unwrapped, "_initial_pickable_z_pickable", None)
            cz = float(cached_z0[0].item()) if cached_z0 is not None else float("nan")
            lift_h = term_mgr.get_term_cfg("success").params.get("lift_height", 0.10)
            thresh = cz + lift_h
            print(
                f"{step:>4} "
                f"({plant_p[0]:6.3f},{plant_p[1]:6.3f},{plant_p[2]:6.3f}) "
                f"({tcp_p[0]:6.3f},{tcp_p[1]:6.3f},{tcp_p[2]:6.3f}) "
                f"{grip:>6.4f} {str(attach):>6} {plant_p[2]-cz:>+10.4f} {thresh:>8.4f} {str(lifted):>6}"
            )
            if dones[0].item():
                print(f"  -> episode ended at step {step}, terminated={lifted}")
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
