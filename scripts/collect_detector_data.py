"""Collect (table_cam RGBD, true plant xy) pairs to train the green-plant
detector. Runs the vision env (table_cam, tiled) with intra-vial plant DR,
resets many times, and records each env's table_cam image + the plant's xy
(relative to its env origin — the same frame the privileged obs uses). Saves to
/tmp/detector_data.npz.

  env -u VIRTUAL_ENV -u CONDA_PREFIX ~/IsaacLab/isaaclab.sh -p scripts/collect_detector_data.py
"""
from __future__ import annotations
from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True, enable_cameras=True).app

N_ENVS = 256
N_RESETS = 48      # 256 * 48 ~= 12k samples
SETTLE_STEPS = 6


def main():
    import gymnasium as gym, importlib, torch, numpy as np
    import arm_act.tasks
    from arm_act.config import load
    arm_act.tasks.register()
    spec = load("pick_plant_out_of_vial_zimmer")
    il = spec["task"]["gym_id"]
    vis_id = il[: -len("-v0")] + "-RL-Vision-v0"
    mod, cls = gym.spec(vis_id).kwargs["env_cfg_entry_point"].split(":")
    env_cfg = getattr(importlib.import_module(mod), cls)()
    env_cfg.scene.num_envs = N_ENVS
    env = gym.make(vis_id, cfg=env_cfg)
    sc = env.unwrapped.scene
    dev = env.unwrapped.device
    adim = int(env.unwrapped.action_manager.total_action_dim)

    rgbs, depths, xys = [], [], []
    for it in range(N_RESETS):
        env.reset()
        for _ in range(SETTLE_STEPS):
            env.step(torch.zeros((N_ENVS, adim), device=dev))
        cam = sc["table_cam"].data.output
        rgb = cam["rgb"][..., :3].detach().cpu().numpy().astype("uint8")          # (N,H,W,3)
        d = cam["distance_to_image_plane"].detach().cpu().numpy().astype("float32")
        if d.ndim == 4:
            d = d[..., 0]
        origin = sc.env_origins.detach().cpu().numpy()
        pxy = (sc["pickable"].data.root_pos_w[:, :2].detach().cpu().numpy()
               - origin[:, :2]).astype("float32")                                 # (N,2)
        rgbs.append(rgb); depths.append(d); xys.append(pxy)
        print(f"RESULT reset {it + 1}/{N_RESETS}: +{N_ENVS}", flush=True)

    rgb = np.concatenate(rgbs); depth = np.concatenate(depths); xy = np.concatenate(xys)
    np.savez_compressed("/tmp/detector_data.npz", rgb=rgb, depth=depth, xy=xy)
    print(f"RESULT saved {len(xy)} samples to /tmp/detector_data.npz", flush=True)
    print(f"RESULT plant xy range: x[{xy[:,0].min():.3f},{xy[:,0].max():.3f}] "
          f"y[{xy[:,1].min():.3f},{xy[:,1].max():.3f}] (m, rel env origin)", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; print("COLLECT FAILED:\n" + traceback.format_exc())
    finally:
        _app.close()
