"""DAgger collector: run the current ACT policy in the env, query the
stateless oracle at each visited state, write (obs, oracle_action) pairs as
a new HDF5 demo file in the same shape the training dataset reader expects.

Usage:
  ./scripts/dagger.sh --task <task>            # 50 episodes, default ckpt
  ./scripts/dagger.sh --task <task> --num-episodes 100 --output-suffix iter1

The output file goes next to the augmented dataset:
  data/augmented/<task>/demos_dagger_<suffix>.hdf5

The follow-up retrain step concatenates this file with the original
augmented demos (or replaces it) and trains ACT again. This script doesn't
trigger the retrain itself — it just produces the new data so a human can
inspect it before committing to a multi-hour training run.

Why DAgger here: the recorded mimic demos pass the simulator's success
check during recording but those exact action sequences DO NOT replay
deterministically (GPU PhysX FP noise drops the cuboid mid-trajectory).
The trained policy ends up in states the demos never covered and has
nothing to imitate. DAgger fixes the coverage problem by relabeling the
states the policy actually visits with what the oracle would do there.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import pathlib
import sys
from datetime import datetime

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    from arm_act.config import DEFAULT_TASK
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--checkpoint", type=pathlib.Path, default=None,
                   help="default: <training.output_dir>/final from task.yaml")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--max-steps-per-episode", type=int, default=None,
                   help="default: eval.max_steps_per_episode from task.yaml")
    p.add_argument("--action-horizon", type=int, default=1,
                   help="how many policy-chunk steps to execute open-loop "
                        "between re-plans. 1 = fully closed-loop (recommended "
                        "for DAgger so the policy's own state distribution is "
                        "what gets relabeled)")
    p.add_argument("--output-suffix", default=None,
                   help="suffix for the output filename. Default: timestamp.")
    p.add_argument(
        "--policy-type",
        choices=("act", "smolvla"),
        default="act",
        help="act: load checkpoint directly (in-process). "
        "smolvla: spawn a subprocess in arm-act-venv that hosts the lerobot policy.",
    )
    p.add_argument(
        "--server-python",
        type=pathlib.Path,
        default=None,
        help="Python interpreter used for --policy-type=smolvla. "
        "Defaults to $ARM_ACT_VENV/bin/python or ~/arm-act-venv/bin/python.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args = _parse_args()

    from arm_act.config import load
    cfg = load(args.task)
    task_cfg = cfg["task"]
    eval_cfg = cfg["eval"]
    max_steps = args.max_steps_per_episode or int(eval_cfg["max_steps_per_episode"])
    checkpoint = args.checkpoint or pathlib.Path(eval_cfg["checkpoint"])

    suffix = args.output_suffix or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = pathlib.Path(cfg["data"]["augmented_path"]).parent / f"demos_dagger_{suffix}.hdf5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # AppLauncher must come before any isaaclab imports that need pxr.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app

    try:
        import h5py
        import numpy as np
        import torch
        import gymnasium as gym

        # Late import: registration needs AppLauncher init.
        import arm_act.tasks
        arm_act.tasks.register()

        from arm_act.tasks._runtime.oracle import (
            _OracleParams,
            oracle_action_at_state,
            snapshot_env_state,
        )

        params = _OracleParams.from_spec(cfg)
        gripper_closed_threshold = float(cfg["robot"]["gripper_closed_threshold"])
        driver_joint = cfg["robot"]["gripper_driver_joint"]

        if args.policy_type == "smolvla":
            import os
            from arm_act.eval.remote_policy import RemotePolicy

            server_python = (
                args.server_python
                or pathlib.Path(
                    os.environ.get("ARM_ACT_VENV", str(pathlib.Path.home() / "arm-act-venv"))
                ) / "bin" / "python"
            )
            if not server_python.exists():
                raise FileNotFoundError(f"server python not found: {server_python}")
            policy = RemotePolicy(
                checkpoint=checkpoint,
                server_python=server_python,
                task_instruction=task_cfg["instruction"],
                # Same camera order as smolvla_convert + rollout.py
                camera_keys=["table_cam", "wrist_cam"],
            )
        else:
            from arm_act.training.act_policy import load_policy
            policy = load_policy(checkpoint, device="cuda")
        policy.action_horizon = args.action_horizon

        gym_id = task_cfg["gym_id"]
        env_cfg_spec = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
        cfg_mod_path, cfg_cls = env_cfg_spec.split(":")
        env_cfg = getattr(importlib.import_module(cfg_mod_path), cfg_cls)()
        env_cfg.scene.num_envs = 1
        env = gym.make(gym_id, cfg=env_cfg)
        device = env.unwrapped.device
        cam_keys = policy.model.camera_keys
        logger.info("task=%s ckpt=%s episodes=%d max_steps=%d out=%s",
                    task_cfg["name"], checkpoint, args.num_episodes, max_steps, out_path)

        # ---- HDF5 writer in the same shape as the augmented dataset --------
        out_h5 = h5py.File(out_path, "w")
        out_grp = out_h5.create_group("data")
        out_grp.attrs["env_args"] = (
            f'{{"env_name": "{gym_id}", "type": 2, "source": "dagger", '
            f'"checkpoint": "{checkpoint}"}}'
        )

        success_term = env.unwrapped.cfg.terminations.success
        # Fail fast at startup so snapshot_env_state doesn't blow up mid-episode.
        if driver_joint not in env.unwrapped.scene["robot"].data.joint_names:
            raise RuntimeError(f"driver joint {driver_joint!r} not found on robot")

        total_steps = 0
        successes = 0
        for ep in range(args.num_episodes):
            obs, _ = env.reset()
            policy.reset()

            ep_actions: list[np.ndarray] = []
            ep_eef_pos: list[np.ndarray] = []
            ep_eef_quat: list[np.ndarray] = []
            ep_grip: list[np.ndarray] = []
            ep_table: list[np.ndarray] = []
            ep_wrist: list[np.ndarray] = []

            success = False
            for t in range(max_steps):
                # ---- Snapshot env state for the oracle query ------------
                snap = snapshot_env_state(env, driver_joint)
                oracle_a = oracle_action_at_state(
                    tcp=snap.tcp,
                    pickable_pos=snap.pickable,
                    target_pos=snap.target,
                    gripper_drive_pos=snap.gripper_pos,
                    gripper_closed_threshold=gripper_closed_threshold,
                    params=params,
                )

                # ---- Capture obs the dataset reader will expect ---------
                eef_pos_obs = obs["policy"]["eef_pos"][0].cpu().numpy().astype(np.float32).reshape(-1)
                eef_quat_obs = obs["policy"]["eef_quat"][0].cpu().numpy().astype(np.float32).reshape(-1)
                grip_obs = obs["policy"]["gripper_pos"][0].cpu().numpy().astype(np.float32).reshape(-1)[:1]
                table_obs = obs["policy"]["table_cam"][0].cpu().numpy().astype(np.uint8)
                wrist_obs = obs["policy"]["wrist_cam"][0].cpu().numpy().astype(np.uint8)

                ep_actions.append(oracle_a.astype(np.float32))
                ep_eef_pos.append(eef_pos_obs)
                ep_eef_quat.append(eef_quat_obs)
                ep_grip.append(grip_obs)
                ep_table.append(table_obs)
                ep_wrist.append(wrist_obs)

                # ---- Step the env using the POLICY action ---------------
                # This is what makes it DAgger: the *policy's* state
                # distribution is what we collect labels from.
                cam_imgs = {}
                for k in cam_keys:
                    arr = obs["policy"][k][0].cpu().numpy().astype("uint8")
                    cam_imgs[k] = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                state_t = torch.from_numpy(np.concatenate([eef_pos_obs, eef_quat_obs, grip_obs], axis=0))
                policy_action = policy.select_action(cam_imgs, state_t)
                policy_action_t = (
                    torch.as_tensor(policy_action, dtype=torch.float32, device=device).unsqueeze(0)
                )
                obs, _, terminated, truncated, info = env.step(policy_action_t)

                if bool(terminated[0]) or bool(truncated[0]):
                    success = bool(success_term.func(env.unwrapped, **success_term.params)[0])
                    break

            T = len(ep_actions)
            total_steps += T
            successes += int(success)
            ep_grp = out_grp.create_group(f"demo_{ep}")
            ep_grp.attrs["success"] = success
            ep_grp.create_dataset("actions", data=np.stack(ep_actions, axis=0))
            obs_grp = ep_grp.create_group("obs")
            obs_grp.create_dataset("eef_pos", data=np.stack(ep_eef_pos, axis=0))
            obs_grp.create_dataset("eef_quat", data=np.stack(ep_eef_quat, axis=0))
            obs_grp.create_dataset("gripper_pos", data=np.stack(ep_grip, axis=0))
            obs_grp.create_dataset("table_cam", data=np.stack(ep_table, axis=0))
            obs_grp.create_dataset("wrist_cam", data=np.stack(ep_wrist, axis=0))

            logger.info("ep %3d: T=%3d %-7s policy_succ=%s",
                        ep, T, "SUCCESS" if success else "FAIL", success)

        out_grp.attrs["total"] = total_steps
        out_h5.close()

        logger.info("wrote %s (%d episodes, %d steps, policy success %d/%d = %.1f%%)",
                    out_path, args.num_episodes, total_steps, successes,
                    args.num_episodes, 100 * successes / max(1, args.num_episodes))

        try:
            env.close()
        except (RuntimeError, AttributeError, AssertionError):
            logger.warning("env.close() raised during teardown (ignored)", exc_info=True)
    except Exception:
        logger.exception("dagger collection failed")
        return 1
    finally:
        app.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
