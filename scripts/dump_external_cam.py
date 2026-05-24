"""Dump the table_cam (external view of the whole arm + scene) at reset.

Counterpart to dump_wrist_cam.py — shows what the full robot assembly
looks like from outside, so we can verify the T3-401 SCARA arm visuals
actually render. Saves a PNG to /tmp/extcam_reset.png.
"""
from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlantOutOfVialZimmer-T3-IK-Rel-v0")
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import importlib
import gymnasium as gym
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
print(f"[dump] obs keys: {list(obs.keys())}", flush=True)

# Find a 4D image — IL flow keeps cameras on the "policy" group.
img = None
key_found = None
for key in ("policy",):
    o = obs.get(key)
    if not hasattr(o, "items"):
        continue
    for k2, v2 in o.items():
        if hasattr(v2, "shape") and len(v2.shape) == 4:
            print(f"  obs[{key!r}][{k2!r}]: shape={tuple(v2.shape)}", flush=True)
            if "table" in k2.lower():
                img = v2
                key_found = f"{key}/{k2}"
                break
    if img is not None:
        break

if img is None:
    # Fallback: take any 4D image
    for key in ("policy",):
        o = obs.get(key)
        if not hasattr(o, "items"):
            continue
        for k2, v2 in o.items():
            if hasattr(v2, "shape") and len(v2.shape) == 4:
                img = v2
                key_found = f"{key}/{k2}"
                break

if img is None:
    print("[dump] NO 4D image found in obs", flush=True)
else:
    print(f"[dump] using {key_found}", flush=True)
    arr = img[0].detach().cpu()
    if arr.shape[0] in (1, 3):
        arr = arr.permute(1, 2, 0)
    arr = arr.numpy()
    if arr.dtype != np.uint8:
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save("/tmp/extcam_reset.png")
    print(f"[dump] saved /tmp/extcam_reset.png shape={arr.shape}", flush=True)

print("DONE", flush=True)
env.close()
sim_app.close()
