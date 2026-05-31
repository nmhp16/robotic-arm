"""TRUE closed-loop deployable eval: camera -> green-plant detector -> state
policy -> action. No ground-truth plant pose enters the actor; the plant's xy
comes from the detector reading the table_cam, exactly as it would on the real
robot. Success is judged by the env's real physics (true pose), as it should be.

How it differs from the noise-injection proxy (PLANT_POS_NOISE): there the actor
saw true_pose + Gaussian(sigma); here it sees true_pose + (detector_estimate -
true_pose) = the detector's ACTUAL per-episode localization error, derived from a
real rendered image. The detector is run ONCE per episode on the settled,
unoccluded view (matching its training distribution and realistic deployment:
localize, then execute), and that estimate's error is held for the episode.

Run (vision env needs cameras; PLANT_POS_NOISE forced 0 so the env's privileged
obs is the TRUE pose we measure the detector against):

  PLANT_POS_NOISE=0 env -u VIRTUAL_ENV -u CONDA_PREFIX PYTHONPATH="$PWD/src" \
    HDF5_USE_FILE_LOCKING=FALSE ~/IsaacLab/isaaclab.sh -p \
    scripts/eval_detector_closed_loop.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PLANT_POS_NOISE", "0.0")  # env priv obs = TRUE pose

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True, enable_cameras=True).app

NUM_ENVS = 64
TARGET_EPISODES = 128          # 2 batches of 64 -> tighter CI than the 60-ep proxy
CKPT = "checkpoints/arm_act_rl/2026-05-26_17-18-56/model_1199.pt"
DET = "/tmp/plant_detector.pt"
SETTLE = 6                     # let the spawn settle before the one detection


def main():
    import gymnasium as gym
    import importlib
    import numpy as np
    import torch
    import torch.nn as nn

    import arm_act.tasks
    from arm_act.config import load

    arm_act.tasks.register()
    spec = load("pick_plant_out_of_vial_zimmer")
    il = spec["task"]["gym_id"]
    vis_id = il[: -len("-v0")] + "-RL-Vision-v0"
    # STATE_ONLY=1 -> run this exact batched/settle rollout on the STATE env
    # (no camera, e=0). Isolates "my rollout method" from "the vision env":
    # if this matches eval_ppo_full's 55%, the method is fine.
    state_only = os.environ.get("STATE_ONLY", "0") == "1"
    env_id = il if state_only else vis_id
    mod, cls = gym.spec(env_id).kwargs["env_cfg_entry_point"].split(":")
    env_cfg = getattr(importlib.import_module(mod), cls)()
    env_cfg.scene.num_envs = NUM_ENVS
    env = gym.make(env_id, cfg=env_cfg)
    print(f"RESULT env={env_id} state_only={state_only}", flush=True)
    uenv = env.unwrapped
    dev = uenv.device
    adim = int(uenv.action_manager.total_action_dim)
    term_mgr = uenv.termination_manager
    active = term_mgr.active_terms
    print(f"RESULT termination terms: {active}", flush=True)
    max_steps = int(uenv.max_episode_length)   # full episode so time_out/success resolve
    print(f"RESULT max_episode_length={max_steps} -> stepping full episode", flush=True)

    # ---- green-plant detector (must match train_detector.py exactly) ----
    class PlantDetector(nn.Module):
        def __init__(self, in_ch=4):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, 32, 5, 2, 2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                nn.Flatten(),
            )
            self.head = nn.Sequential(nn.Linear(64 * 11 * 11, 128), nn.ReLU(), nn.Linear(128, 2))

        def forward(self, x):
            return self.head(self.conv(x))

    if not state_only:
        dck = torch.load(DET, map_location=dev, weights_only=False)
        det = PlantDetector().to(dev)
        det.load_state_dict(dck["state_dict"])
        det.eval()
        ymu = torch.as_tensor(dck["ymu"], device=dev, dtype=torch.float32)
        ysd = torch.as_tensor(dck["ysd"], device=dev, dtype=torch.float32)
        dmu, dsd = float(dck["dmu"]), float(dck["dsd"])

    def detect():
        cam = uenv.scene["table_cam"].data.output
        rgb = cam["rgb"][..., :3].float() / 255.0          # (N,H,W,3)
        d = cam["distance_to_image_plane"].float()
        if d.dim() == 4:
            d = d[..., 0]
        d = (d - dmu) / dsd
        x = torch.cat([rgb, d.unsqueeze(-1)], dim=-1).permute(0, 3, 1, 2)  # (N,4,H,W)
        with torch.no_grad():
            return det(x) * ysd + ymu                      # (N,2) origin-rel meters

    # ---- state actor (rebuilt from actor_state_dict; deterministic mean) ----
    sd = torch.load(CKPT, map_location=dev, weights_only=False)["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].to(dev)              # (1,50)
    std = sd["obs_normalizer._std"].to(dev)                # (1,50)
    mlp = nn.Sequential(
        nn.Linear(50, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 64), nn.ELU(),
        nn.Linear(64, 4),
    ).to(dev)
    with torch.no_grad():
        for i in (0, 2, 4, 6):
            mlp[i].weight.copy_(sd[f"mlp.{i}.weight"])
            mlp[i].bias.copy_(sd[f"mlp.{i}.bias"])
    mlp.eval()

    def act(proprio, priv):
        x = torch.cat([proprio, priv], dim=-1)             # (N,50) [proprio|privileged]
        x = (x - mean) / (std + 1e-2)                      # rsl_rl EmpiricalNormalization (eps=1e-2)
        with torch.no_grad():
            return mlp(x)

    zeros = torch.zeros((NUM_ENVS, adim), device=dev)
    counts = {"success": 0, "time_out": 0, "pickable_dropping": 0, "total": 0}
    sanity_done = False

    def gobs():
        # Authoritative obs-group dict — exactly what rsl_rl concatenates during
        # training. The gym reset/step return is NOT consistent across env variants
        # (the state env omits 'privileged' there), so read the obs manager directly.
        return uenv.observation_manager.compute()

    while counts["total"] < TARGET_EPISODES:
        env.reset()
        obs = gobs()
        for _ in range(SETTLE):
            env.step(zeros)
            obs = gobs()
        if not sanity_done:
            print("RESULT obs groups: " + ", ".join(
                f"{k}:{tuple(v.shape)}" for k, v in obs.items() if hasattr(v, "shape")),
                flush=True)

        # one detection on the clean settled view; freeze its error for the episode
        true_xy = obs["privileged"][:, 0:2]                # obj_pos-origin xy (TRUE; noise=0)
        if state_only:
            e = torch.zeros((NUM_ENVS, 2), device=dev)
        else:
            est_xy = detect()
            # DET_ERR_SCALE=0 -> control (perfect pose in the vision env, isolates the
            # actor/env wiring); =1 -> the real detector error (deployable).
            e = (est_xy - true_xy) * float(os.environ.get("DET_ERR_SCALE", "1.0"))

        if not sanity_done:
            # confirm index map: priv[:,0:2] must equal scene plant xy rel origin
            sc_xy = (uenv.scene["pickable"].data.root_pos_w[:, :2]
                     - uenv.scene.env_origins[:, :2])
            idx_err = (true_xy - sc_xy).abs().max().item()
            err_mm = (e.norm(dim=-1) * 1000.0)
            print(f"RESULT index-check max|priv0:2 - scene_xy| = {idx_err*1000:.3f}mm "
                  f"(should be ~0)", flush=True)
            print(f"RESULT detector err in-env: median={err_mm.median():.2f}mm "
                  f"mean={err_mm.mean():.2f}mm 90th={np.percentile(err_mm.cpu().numpy(),90):.2f}mm "
                  f"(train val was 1.7/2.3/3.6)", flush=True)
            sanity_done = True

        done = torch.zeros(NUM_ENVS, dtype=torch.bool, device=dev)
        succ = torch.zeros_like(done)
        tout = torch.zeros_like(done)
        drop = torch.zeros_like(done)

        for t in range(max_steps):
            priv = obs["privileged"].clone()
            # make the actor believe plant_xy = true + detector error, propagated
            # to every plant-xy-derived component of object_obs + pickable_pos:
            priv[:, 0:2] += e            # obj_pos - origin
            priv[:, 10:12] += e          # obj_pos - ee
            priv[:, 13:15] -= e          # target - obj  (target fixed, obj moved +e)
            priv[:, 16:18] += e          # pickable_pos
            a = act(obs["proprio"], priv)
            if counts["total"] == 0 and t in (0, 60, 200, max_steps - 1):
                eez = uenv.scene["ee_frame"].data.target_pos_w[:, 0, 2]
                print(f"RESULT diag t={t}: mean|a|={a.abs().mean():.3f} "
                      f"ee_z[min/mean/max]={eez.min():.3f}/{eez.mean():.3f}/{eez.max():.3f} "
                      f"done={int(done.sum())}", flush=True)
            out = env.step(a)
            term, trunc = out[2], out[3]
            obs = gobs()
            dmask = (term | trunc).bool()
            new = dmask & (~done)
            if new.any():
                if "success" in active:
                    succ |= term_mgr.get_term("success").bool() & new
                if "time_out" in active:
                    tout |= term_mgr.get_term("time_out").bool() & new
                if "pickable_dropping" in active:
                    drop |= term_mgr.get_term("pickable_dropping").bool() & new
                done |= new
            if done.all():
                break

        counts["success"] += int(succ.sum())
        counts["time_out"] += int(tout.sum())
        counts["pickable_dropping"] += int(drop.sum())
        counts["total"] += NUM_ENVS
        print(f"RESULT batch: episodes={counts['total']} success={counts['success']} "
              f"time_out={counts['time_out']} drop={counts['pickable_dropping']} "
              f"rate={counts['success']/counts['total']:.2%}", flush=True)

    rate = counts["success"] / max(counts["total"], 1)
    print(f"RESULT FINAL CLOSED-LOOP episodes={counts['total']} "
          f"success={counts['success']} ({rate:.2%}) "
          f"time_out={counts['time_out']} drop={counts['pickable_dropping']}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("CLOSED-LOOP EVAL FAILED:\n" + traceback.format_exc(), flush=True)
    finally:
        _app.close()
