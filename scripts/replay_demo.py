"""Diagnostic: replay a recorded demo's exact action sequence and check
whether success_term fires. If success fails on a *recorded* trajectory
the env is non-deterministic and the BC policy can never succeed."""

from __future__ import annotations
import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--task", default="put_plant_back",
               help="task name under src/arm_act/tasks/ (used to look up the gym id + env cfg)")
p.add_argument("--hdf5", default="data/augmented/put_plant_back/demos.hdf5",
               help="HDF5 demo file to replay (raw or mimic-augmented; same schema)")
p.add_argument("--num-demos", type=int, default=5,
               help="number of demos to replay; the success rate is printed at the end")
args = p.parse_args()

app = AppLauncher(headless=True, enable_cameras=True).app

try:
    import importlib
    import h5py
    import numpy as np
    import torch
    import gymnasium as gym

    import arm_act.tasks
    arm_act.tasks.register()

    from arm_act.config import load
    cfg = load(args.task)
    task_cfg = cfg["task"]
    gym_id = task_cfg["gym_id"]

    env_cfg_spec = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    cfg_mod_path, cfg_cls = env_cfg_spec.split(":")
    env_cfg = getattr(importlib.import_module(cfg_mod_path), cfg_cls)()
    env_cfg.scene.num_envs = 1
    env = gym.make(gym_id, cfg=env_cfg)
    device = env.unwrapped.device
    success_term = env.unwrapped.cfg.terminations.success

    f = h5py.File(args.hdf5, "r")
    demos = list(f["data"])
    print(f"loaded {len(demos)} demos from {args.hdf5}")

    n_success = 0
    for i, name in enumerate(demos[: args.num_demos]):
        ep = f["data"][name]
        actions = ep["actions"][:]
        T = actions.shape[0]
        obs, _ = env.reset()
        success = False
        for t in range(T):
            a = torch.as_tensor(actions[t], dtype=torch.float32, device=device).unsqueeze(0)
            obs, _, term, trunc, _ = env.step(a)
            if bool(term[0]) or bool(trunc[0]):
                success = bool(success_term.func(env.unwrapped, **success_term.params)[0])
                break
        if not success:
            # Check at end of trajectory in case it didn't terminate
            success = bool(success_term.func(env.unwrapped, **success_term.params)[0])
        n_success += int(success)
        print(f"demo {name}: T={T} success={success}")
    print(f"\n== REPLAY RESULT: {n_success}/{args.num_demos} =="
          f" ({100*n_success/args.num_demos:.0f}%)")
    env.close()
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    app.close()
