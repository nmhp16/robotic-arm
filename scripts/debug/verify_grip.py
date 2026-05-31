"""Verify the multi-node policy ACTUALLY grips+lifts (vs gaming the lifted
metric). Rolls out env 0, logs per step: plant lift (rel rest), tcp z, finger
gap, plant-tcp xy alignment, and the first-termination outcome. Dumps 8 table_cam
stills to /tmp/grip_*.png so the grasp can be eyeballed. Temp diagnostic.
"""
from __future__ import annotations
from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True, enable_cameras=True).app

NUM_ENVS = 4
STEPS = 250
CKPT = "checkpoints/arm_act_rl/2026-05-26_22-39-45/model_1199.pt"


def main():
    import gymnasium as gym, importlib, numpy as np, torch, torch.nn as nn
    import imageio.v2 as imageio
    import arm_act.tasks
    from arm_act.config import load
    arm_act.tasks.register()
    spec = load("pick_plant_out_of_vial_zimmer")
    il = spec["task"]["gym_id"]; vis = il[:-len("-v0")] + "-RL-Vision-v0"
    mod, cls = gym.spec(vis).kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(mod), cls)(); cfg.scene.num_envs = NUM_ENVS
    if hasattr(cfg.scene, "table_cam"):
        cfg.scene.table_cam.height = 512; cfg.scene.table_cam.width = 512
    env = gym.make(vis, cfg=cfg); uenv = env.unwrapped; dev = uenv.device
    adim = int(uenv.action_manager.total_action_dim); tm = uenv.termination_manager; sc = uenv.scene
    jn = list(sc["robot"].data.joint_names); fi = jn.index("finger_left_joint")

    sd = torch.load(CKPT, map_location=dev, weights_only=False)["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].to(dev); std = sd["obs_normalizer._std"].to(dev)
    mlp = nn.Sequential(nn.Linear(50,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,4)).to(dev)
    with torch.no_grad():
        for i in (0,2,4,6): mlp[i].weight.copy_(sd[f"mlp.{i}.weight"]); mlp[i].bias.copy_(sd[f"mlp.{i}.bias"])
    mlp.eval()
    g = lambda: uenv.observation_manager.compute()
    def act(o):
        x=torch.cat([o["proprio"],o["privileged"]],-1); x=(x-mean)/(std+1e-2)
        with torch.no_grad(): return mlp(x)

    env.reset(); o=g()
    z = torch.zeros((NUM_ENVS,adim),device=dev)
    for _ in range(6): env.step(z); o=g()
    z0 = sc["pickable"].data.root_pos_w[:,2].clone()       # rest height per env
    e=0; frames=[]; log=[]; done0=False; outcome="timeout"
    for t in range(STEPS):
        o=g(); a=act(o); out=env.step(a); term,trunc=out[2],out[3]
        p=sc["pickable"].data.root_pos_w; tcp=sc["ee_frame"].data.target_pos_w[:,0,:]
        fg=float(sc["robot"].data.joint_pos[e,fi])
        frames.append(sc["table_cam"].data.output["rgb"][e,...,:3].detach().cpu().numpy().astype("uint8"))
        log.append((t, float(p[e,2]-z0[e]), float(tcp[e,2]), fg, float((p[e,:2]-tcp[e,:2]).norm())))
        if not done0 and bool(term[e] or trunc[e]):
            done0=True
            outcome = "success" if bool(tm.get_term("success")[e]) else ("drop" if bool(tm.get_term("pickable_dropping")[e]) else "timeout")
            print(f"RESULT env0 FIRST outcome @step {t}: {outcome}", flush=True)
    L=np.array([r[1] for r in log]); A=np.array([r[4] for r in log]); G=np.array([r[3] for r in log])
    print(f"RESULT env0 lift cm: max={L.max()*100:.1f} final={L[-1]*100:.1f} | finger gap-cmd min/max={G.min():.4f}/{G.max():.4f} | plant-tcp xy mm min/max={A.min()*1000:.1f}/{A.max()*1000:.1f}", flush=True)
    print("RESULT t / lift_cm / tcp_z / fingerjoint / plant-tcp_mm:", flush=True)
    for r in log[::20]:
        print(f"   t={r[0]:3d}  lift={r[1]*100:6.1f}  tcp_z={r[2]:.3f}  fj={r[3]:.4f}  ptcp={r[4]*1000:6.1f}", flush=True)
    for i,idx in enumerate(np.linspace(0,len(frames)-1,8).astype(int)):
        imageio.imwrite(f"/tmp/grip_{i}_t{idx}.png", frames[idx])
    print("RESULT wrote /tmp/grip_0..7 stills", flush=True)
    env.close()


if __name__ == "__main__":
    try: main()
    except Exception:
        import traceback; print("VERIFY FAILED:\n"+traceback.format_exc(), flush=True)
    finally: _app.close()
