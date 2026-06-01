"""Why does form-closure RL plateau at ~30% success despite ~90% lift?
Load a trained actor, roll out deterministically (mean action) across N envs, and
bucket every episode by its FINAL state so we know what to fix:

  success      : dist_dest < 0.03 and lifted (z>0.03)         -> the win
  reached_low  : dist_dest < 0.06 but z < 0.03                -> carried, but LOWERED
                 (success metric needs lifted; would be a placement, not a fail)
  stalled_far  : z > 0.03 but dist_dest > 0.06                -> lifted, never navigated
  dropped      : z < 0.015                                    -> lost the plant en route

Also reports the distribution of final dist_dest so we see if the policy gets
*close* (navigation precision) or not at all (navigation/reach).

  ~/newton-probe/bin/python scripts/newton/diagnose_carry.py checkpoints/newton_rl_fc3/actor.pt [max_delta]
"""
import sys, pathlib
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from arm_act.newton.rl_env import NewtonVecEnv
from arm_act.newton.ppo import ActorCritic, RunNorm

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/newton_rl_fc3/actor.pt"
max_delta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.015
N, T = 128, 180

env = NewtonVecEnv(num_envs=N, spawn_jitter=0.006, max_ctrl=T, init_mode="grasped",
                   form_closure=True, max_delta=[max_delta, max_delta, 0.02])
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
ac = ActorCritic(env.obs_dim, env.act_dim); ac.load_state_dict(ck["actor"]); ac.eval()
rn = RunNorm(env.obs_dim); rn.mean = ck["obs_mean"]; rn.var = ck["obs_var"]

o = env.reset()
ever_lifted = np.zeros(N, bool)
for _ in range(T):
    with torch.no_grad():
        mu, _, _ = ac(torch.from_numpy(rn.norm(o)))
    o, r, d, info = env.step(mu.numpy())
    ever_lifted |= info["lifted"]

plant = env._plant(); eef = env._eef()
dist = np.linalg.norm(plant[:, :2] - env._dest_xy, axis=1)
z = plant[:, 2]
gripc = env.s0.joint_q.numpy()[env._q["fl"]] > 0.0015

success = (dist < 0.030) & (z > 0.030)
dropped = z < 0.015
reached_low = (dist < 0.060) & (z <= 0.030) & ~dropped
stalled_far = (dist >= 0.060) & (z > 0.030)
other = ~(success | dropped | reached_low | stalled_far)

print(f"ckpt={ckpt}  max_delta={max_delta}  N={N} T={T}  ckpt_train_succ={ck.get('succ','?')}")
print(f"  ever_lifted (any step)   : {ever_lifted.mean()*100:5.1f}%")
print(f"  gripc at end             : {gripc.mean()*100:5.1f}%")
print(f"  --- FINAL-STATE BUCKETS ---")
print(f"  success (near+lifted)    : {success.mean()*100:5.1f}%")
print(f"  reached_low (near, z<30) : {reached_low.mean()*100:5.1f}%  <- carried but lowered")
print(f"  stalled_far (lifted,far) : {stalled_far.mean()*100:5.1f}%  <- never navigated")
print(f"  dropped (z<15mm)         : {dropped.mean()*100:5.1f}%  <- lost grip")
print(f"  other                    : {other.mean()*100:5.1f}%")
print(f"  --- final dist_dest distribution (mm) ---")
for p in (10, 25, 50, 75, 90):
    print(f"    p{p:02d} = {np.percentile(dist, p)*1000:6.1f}")
print(f"  final z (mm): p25={np.percentile(z,25)*1000:.0f} p50={np.percentile(z,50)*1000:.0f} p75={np.percentile(z,75)*1000:.0f}")
