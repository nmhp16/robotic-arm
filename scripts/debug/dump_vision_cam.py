"""Dump what the VISION-RL policy actually sees: the 84x84 wrist_cam of the
-RL-Vision-v0 env (debug markers off, RGBD). Saves an upscaled PNG so we can
judge whether the thin recessed stem is even resolvable at that resolution.

  env -u VIRTUAL_ENV -u CONDA_PREFIX ~/IsaacLab/isaaclab.sh -p scripts/dump_vision_cam.py
"""
from __future__ import annotations
import numpy as np
from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True, enable_cameras=True).app


def main():
    import gymnasium as gym, importlib, torch
    import arm_act.tasks
    from arm_act.config import load
    arm_act.tasks.register()
    spec = load("pick_plant_out_of_vial_zimmer")
    il = spec["task"]["gym_id"]
    vis_id = il[:-len("-v0")] + "-RL-Vision-v0"
    cfg_ep = gym.spec(vis_id).kwargs["env_cfg_entry_point"]
    mod, cls = cfg_ep.split(":")
    env_cfg = getattr(importlib.import_module(mod), cls)()
    env_cfg.scene.num_envs = 1
    # Optional hi-res override (RES env var) to SEE the plant clearly — the
    # vision env downsizes wrist_cam to 84x84; at hi-res we can tell whether the
    # green plant is merely sub-pixel (resolution problem) or occluded in the vial.
    import os as _os
    _res = int(_os.environ.get("RES", "0"))
    if _res and env_cfg.scene.wrist_cam is not None:
        env_cfg.scene.wrist_cam.height = _res
        env_cfg.scene.wrist_cam.width = _res
    env = gym.make(vis_id, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        dev = env.unwrapped.device
        adim = int(env.unwrapped.action_manager.total_action_dim)
        for _ in range(10):
            obs, *_ = env.step(torch.zeros((1, adim), device=dev))
        # Ground-truth runtime poses (written to file; Isaac logging eats stdout).
        sc = env.unwrapped.scene
        lines = []
        for nm in ("pickable", "vial"):
            try:
                a = sc[nm]
                p = a.data.root_pos_w[0].cpu().numpy()
                q = a.data.root_quat_w[0].cpu().numpy()
                origin = sc.env_origins[0].cpu().numpy()
                lines.append(f"{nm}: pos(rel env)=({(p[0]-origin[0])*1000:.1f},{(p[1]-origin[1])*1000:.1f},{(p[2]-origin[2])*1000:.1f})mm quat(wxyz)=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f})")
            except Exception as e:
                lines.append(f"{nm}: ERR {e}")
        try:
            tcp = sc["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
            origin = sc.env_origins[0].cpu().numpy()
            lines.append(f"tcp: pos(rel env)=({(tcp[0]-origin[0])*1000:.1f},{(tcp[1]-origin[1])*1000:.1f},{(tcp[2]-origin[2])*1000:.1f})mm")
        except Exception as e:
            lines.append(f"tcp: ERR {e}")
        try:
            wc = sc["wrist_cam"]
            cp = wc.data.pos_w[0].cpu().numpy()
            cq = wc.data.quat_w_ros[0].cpu().numpy()
            origin = sc.env_origins[0].cpu().numpy()
            # camera looks down its -Z (ros) / +Z (usd); report optical axis in world
            import numpy as _np
            w, x, y, z = cq
            # rotate the camera local -Z (ros convention forward) into world
            fwd = _np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])  # +Z col of R
            lines.append(f"wrist_cam: pos(rel env)=({(cp[0]-origin[0])*1000:.1f},{(cp[1]-origin[1])*1000:.1f},{(cp[2]-origin[2])*1000:.1f})mm  optical_axis(world)=({fwd[0]:.2f},{fwd[1]:.2f},{fwd[2]:.2f})")
        except Exception as e:
            lines.append(f"wrist_cam: ERR {e}")
        open("/tmp/plant_pose.txt", "w").write("\n".join(lines) + "\n")
        # raw camera render (84x84) the CNN ingests
        cam = env.unwrapped.scene["wrist_cam"].data.output
        rgb = cam["rgb"][0].cpu().numpy()[..., :3].astype("uint8")
        print(f"RESULT wrist_cam rgb shape={rgb.shape} dtype={rgb.dtype} min={rgb.min()} max={rgb.max()}")
        # green-ness check: how many pixels are clearly plant-green?
        r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
        greenish = ((g > r + 20) & (g > b + 20)).sum()
        print(f"RESULT green-ish pixels (plant): {greenish} / {rgb.shape[0]*rgb.shape[1]}")
        from PIL import Image
        img = Image.fromarray(rgb)
        if rgb.shape[0] < 200:  # upscale tiny 84x84 for viewing
            img = img.resize((336, 336), Image.NEAREST)
        out = "/tmp/vision_wrist_cam_hires.png" if rgb.shape[0] >= 200 else "/tmp/vision_wrist_cam.png"
        img.save(out)
        print(f"RESULT wrote {out}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; print("DUMP FAILED:\n" + traceback.format_exc())
    finally:
        _app.close()
