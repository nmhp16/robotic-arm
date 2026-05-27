"""Record an mp4 of the multi-node form-closure grasp (state PPO policy),
captured from the fixed table_cam — the proven headless-render path (mirrors
src/arm_act/eval/rollout.py), avoiding eval_ppo_full's --video render-mode snag.

  env -u VIRTUAL_ENV -u CONDA_PREFIX PYTHONPATH="$PWD/src" \
    ~/IsaacLab/isaaclab.sh -p scripts/record_grasp_video.py
"""
from __future__ import annotations

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True, enable_cameras=True).app

NUM_ENVS = 4
STEPS = 600                                                        # full pick->carry->place is ~400 steps; 600 catches an attempt + reset
CKPT = "checkpoints/arm_act_rl/2026-05-27_02-42-37/model_1499.pt"  # WALLED vial->vial (real containment)
OUT = "eval/runs/vial2vial_walled.mp4"


def main():
    import gymnasium as gym
    import importlib
    import pathlib
    import torch
    import torch.nn as nn

    import arm_act.tasks
    from arm_act.config import load
    from arm_act.eval.common import save_video

    arm_act.tasks.register()
    spec = load("pick_plant_out_of_vial_zimmer")
    il = spec["task"]["gym_id"]
    vis_id = il[: -len("-v0")] + "-RL-Vision-v0"   # has table_cam + proprio + privileged
    mod, cls = gym.spec(vis_id).kwargs["env_cfg_entry_point"].split(":")
    env_cfg = getattr(importlib.import_module(mod), cls)()
    env_cfg.scene.num_envs = NUM_ENVS
    # Hi-res the table_cam so the video is watchable. The vision env renders it at
    # 84x84 for the CNN; the actor uses ONLY proprio+privileged, so raising the
    # camera resolution changes the rendered video, not the policy.
    if hasattr(env_cfg.scene, "table_cam"):
        env_cfg.scene.table_cam.height = 512
        env_cfg.scene.table_cam.width = 512
    env = gym.make(vis_id, cfg=env_cfg)
    uenv = env.unwrapped
    dev = uenv.device
    adim = int(uenv.action_manager.total_action_dim)
    term_mgr = uenv.termination_manager
    active = term_mgr.active_terms

    # bit-exact actor (== exported JIT, verified earlier)
    sd = torch.load(CKPT, map_location=dev, weights_only=False)["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].to(dev)
    std = sd["obs_normalizer._std"].to(dev)
    mlp = nn.Sequential(nn.Linear(50, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                        nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 4)).to(dev)
    with torch.no_grad():
        for i in (0, 2, 4, 6):
            mlp[i].weight.copy_(sd[f"mlp.{i}.weight"])
            mlp[i].bias.copy_(sd[f"mlp.{i}.bias"])
    mlp.eval()

    def gobs():
        return uenv.observation_manager.compute()   # authoritative obs groups

    def act(o):
        x = torch.cat([o["proprio"], o["privileged"]], dim=-1)
        x = (x - mean) / (std + 1e-2)
        with torch.no_grad():
            return mlp(x)

    env.reset()
    obs = gobs()
    print("RESULT obs groups: " + ", ".join(k for k, v in obs.items() if hasattr(v, "shape")), flush=True)

    import imageio.v2 as imageio, numpy as _np
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    # Buffer each env's current episode; save the FIRST episode (any env) that
    # actually PLACES the plant into the dest vial, so the clip shows the task
    # working rather than a random env that happened to miss (~21% rate).
    bufs = [[] for _ in range(NUM_ENVS)]
    MAXS, saved, n_done, n_succ = 3000, False, 0, 0
    for t in range(MAXS):
        a = act(obs)
        out = env.step(a)
        term, trunc = out[2], out[3]
        obs = gobs()
        rgb = uenv.scene["table_cam"].data.output["rgb"][:, ..., :3].detach().cpu().numpy().astype("uint8")
        dm = (term | trunc).bool()
        sflag = term_mgr.get_term("success").bool() if "success" in active else torch.zeros_like(dm)
        for e in range(NUM_ENVS):
            bufs[e].append(rgb[e])
            if bool(dm[e]):
                n_done += 1
                n_succ += int(bool(sflag[e]))
                if bool(sflag[e]) and len(bufs[e]) >= 20:
                    save_video(pathlib.Path(OUT), bufs[e], fps=20)
                    for i, idx in enumerate(_np.linspace(0, len(bufs[e]) - 1, 8).astype(int)):
                        imageio.imwrite(f"/tmp/v2v_{i}_t{idx}.png", bufs[e][idx])
                    print(f"RESULT wrote SUCCESS clip {OUT} (env{e}, {len(bufs[e])} frames) @step {t}; "
                          f"{n_succ}/{n_done} episodes succeeded so far", flush=True)
                    saved = True
                    break
                bufs[e] = []
        if saved:
            break
    if not saved:
        print(f"RESULT NO placed-in-vial success in {MAXS} steps ({n_succ}/{n_done} eps)", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("VIDEO FAILED:\n" + traceback.format_exc(), flush=True)
    finally:
        _app.close()
