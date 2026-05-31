"""Settle-smoke for the COMPLIANT (jointed) plant (CLAUDE.md #9 dynamics look).

Loads the pick_plant_out_of_vial_zimmer env (which now spawns the 2-link
spring-jointed plant articulation in its vial well), holds the arm still, and
lets the plant settle under gravity. Reports:
  - env loaded at all  (validates the articulation + spring actuator wiring)
  - the pickable's joint names (confirms `leaf_sway` is there)
  - leaf_sway angle over time -> does the spring hold it ~upright (near 0) or
    does it flop to the limit (spring too soft)?
  - base/root z + xy drift -> does it rest in the well or fall through / slide?

Writes a summary to /tmp/compliant_smoke.txt and a timeline plot to
/tmp/compliant_smoke.png. Run:
  env -u VIRTUAL_ENV -u CONDA_PREFIX ~/IsaacLab/isaaclab.sh -p scripts/smoke_compliant_plant.py
"""
from __future__ import annotations

import numpy as np
from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True, enable_cameras=True).app

OUT_TXT = "/tmp/compliant_smoke.txt"
_log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    _log_lines.append(msg)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(_log_lines) + "\n")


def main() -> int:
    import gymnasium as gym
    import importlib
    import torch

    import arm_act.tasks
    from arm_act.config import load

    arm_act.tasks.register()
    spec = load("pick_plant_out_of_vial_zimmer")
    gym_id = spec["task"]["gym_id"]
    env_cfg_spec = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    mod_name, cls_name = env_cfg_spec.split(":")
    env_cfg = getattr(importlib.import_module(mod_name), cls_name)()
    env_cfg.scene.num_envs = 1

    env = gym.make(gym_id, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        scene = env.unwrapped.scene
        origin = scene.env_origins[0]
        plant = scene["pickable"]
        jnames = list(getattr(plant.data, "joint_names", []) or [])
        log(f"ENV LOADED. pickable type={type(plant).__name__}")
        log(f"pickable joint_names={jnames}")
        bend_idx = jnames.index("leaf_sway") if "leaf_sway" in jnames else None
        log(f"leaf_sway idx={bend_idx}")

        device = env.unwrapped.device
        action_dim = int(env.unwrapped.action_manager.total_action_dim)
        zero = torch.zeros((1, action_dim), device=device)

        steps, zs, xs, ys, angs = [], [], [], [], []
        N = 80
        for t in range(N):
            obs, *_ = env.step(zero)  # hold arm; let plant settle
            rp = (plant.data.root_pos_w[0] - origin).cpu().numpy()
            ang = float(plant.data.joint_pos[0, bend_idx].cpu()) if bend_idx is not None else float("nan")
            steps.append(t); zs.append(float(rp[2])); xs.append(float(rp[0])); ys.append(float(rp[1])); angs.append(ang)

        z0, zf = zs[0], zs[-1]
        xdrift = xs[-1] - xs[0]; ydrift = ys[-1] - ys[0]
        ang_f = angs[-1]; ang_max = max(abs(a) for a in angs)
        log("--- settled (last step) ---")
        log(f"root z: {z0:.4f} -> {zf:.4f} m   (base should rest near the well floor)")
        log(f"root xy drift: dx={xdrift*1000:.1f}mm dy={ydrift*1000:.1f}mm   (small = stays centred)")
        log(f"leaf_sway angle: final={ang_f:+.3f} rad ({np.degrees(ang_f):+.1f} deg), max|ang|={np.degrees(ang_max):.1f} deg")
        limit = 0.7
        if bend_idx is not None:
            if abs(ang_f) > 0.9 * limit:
                log("VERDICT: FLOPPED — spring too soft (joint pinned near limit). Raise stiffness.")
            elif abs(ang_f) < 0.15:
                log("VERDICT: stands upright (joint near 0). Good — check it still bends under contact.")
            else:
                log(f"VERDICT: leans {np.degrees(ang_f):.0f} deg at rest. Borderline; consider a bit more stiffness.")

        # timeline plot (the 'look')
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        a1.plot(steps, np.degrees(angs), "-o", ms=3, color="purple")
        a1.axhline(0, color="g", ls=":"); a1.axhline(np.degrees(limit), color="r", ls="--")
        a1.axhline(-np.degrees(limit), color="r", ls="--")
        a1.set_ylabel("leaf_sway (deg)"); a1.set_title("Compliant plant settling under gravity (arm held)")
        a2.plot(steps, np.array(zs) * 1000, "-o", ms=3, label="root z")
        a2.plot(steps, np.array(xs) * 1000, "-", label="root x"); a2.plot(steps, np.array(ys) * 1000, "-", label="root y")
        a2.set_ylabel("root pos (mm)"); a2.set_xlabel("sim step"); a2.legend(fontsize=8)
        plt.tight_layout(); plt.savefig("/tmp/compliant_smoke.png", dpi=110)
        log("wrote /tmp/compliant_smoke.png")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("SMOKE FAILED:\n" + traceback.format_exc())
    finally:
        _app.close()
