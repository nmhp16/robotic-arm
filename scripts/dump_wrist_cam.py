"""Dump the wrist_cam image after reset for visual inspection.

Verifies the camera is pointed at the scene. Saves a PNG to
/tmp/wristcam_reset.png. No env stepping — just reset and grab.
"""
from __future__ import annotations

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PutPlantBack-T3-IK-Rel-RL-Vision-v0")
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import importlib
import gymnasium as gym
import torch
import numpy as np
from PIL import Image

import arm_act.tasks
arm_act.tasks.register()

print(f"[dump] making {args.task}", flush=True)
env_cfg_entry = gym.spec(args.task).kwargs["env_cfg_entry_point"]
cfg_mod, cfg_cls = env_cfg_entry.rsplit(":", 1)
env_cfg = getattr(importlib.import_module(cfg_mod), cfg_cls)()
env_cfg.scene.num_envs = args.num_envs
env = gym.make(args.task, cfg=env_cfg)
print(f"[dump] env created", flush=True)
obs, _ = env.reset()
print(f"[dump] obs after reset:", flush=True)
for k, v in obs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape} {v.dtype}", flush=True)
    elif hasattr(v, "items"):
        for kk, vv in v.items():
            print(f"  {k}/{kk}: {vv.shape if hasattr(vv,'shape') else type(vv)}", flush=True)

# Try multiple keys to find the wrist_cam image
img = None
for key in ("rgb_camera", "policy"):
    o = obs.get(key)
    if o is None:
        continue
    if hasattr(o, "shape") and len(o.shape) == 4:
        img = o
        print(f"[dump] found image at obs[{key!r}] shape={tuple(o.shape)}", flush=True)
        break
    if hasattr(o, "items"):
        for k2, v2 in o.items():
            if hasattr(v2, "shape") and len(v2.shape) == 4:
                img = v2
                print(f"[dump] found image at obs[{key!r}][{k2!r}] shape={tuple(v2.shape)}", flush=True)
                break
    if img is not None:
        break

if img is None:
    print("[dump] WARN: no 4D image obs found", flush=True)
else:
    arr = img[0].detach().cpu()
    if arr.shape[0] in (1, 3):  # channels-first
        arr = arr.permute(1, 2, 0)
    arr = arr.numpy()
    if arr.dtype != np.uint8:
        # normalize for viewing
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save("/tmp/wristcam_reset.png")
    print("[dump] saved /tmp/wristcam_reset.png", flush=True)

print("DONE", flush=True)
env.close()
sim_app.close()
