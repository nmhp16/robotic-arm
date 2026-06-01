"""Compact PPO for the batched Newton state env (NewtonVecEnv).

Self-contained (torch, runs in ~/newton-probe): Gaussian MLP actor-critic, GAE,
clipped surrogate, running obs normalization. Episodic: each iteration collects
one fixed-length episode across all N worlds, bootstraps 0 at the terminal step.
Reports rolling vial->vial success (env computes it) and checkpoints the actor so
eval/deploy can load it. Target: beat the scripted oracle's 84%.

  ~/newton-probe/bin/python -m arm_act.newton.ppo --num-envs 128 --iters 300 \
      --out checkpoints/newton_rl
"""
from __future__ import annotations
import argparse, pathlib, time, json
import numpy as np
import torch
import torch.nn as nn

from arm_act.newton.rl_env import NewtonVecEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, init_log_std=-0.5):
        super().__init__()
        def mlp():
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = mlp(); self.mu = nn.Linear(hidden, act_dim)
        self.critic = mlp(); self.v = nn.Linear(hidden, 1)
        # lower init_log_std + low entropy coef forces the policy to commit the
        # behavior to the MEAN instead of relying on exploration noise (the v8/v9
        # failure: ~40% sampled succ but 0% deterministic — noise did the carrying).
        self.log_std = nn.Parameter(init_log_std * torch.ones(act_dim))

    def forward(self, obs):
        mu = self.mu(self.actor(obs))
        val = self.v(self.critic(obs)).squeeze(-1)
        # robustness: a rare diverged env / gradient spike can NaN these over a long
        # run; sanitize + bound so the Normal dist + update never crash.
        mu = torch.nan_to_num(mu, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)
        std = torch.nan_to_num(self.log_std, nan=-0.5).clamp(-4.0, 1.0).exp()
        val = torch.nan_to_num(val, nan=0.0, posinf=1e3, neginf=-1e3)
        return mu, std, val


class RunNorm:
    """Running mean/std obs normalizer (Welford)."""
    def __init__(self, dim):
        self.mean = np.zeros(dim, np.float64); self.var = np.ones(dim, np.float64); self.n = 1e-4
    def update(self, x):
        b = x.shape[0]; m = x.mean(0); v = x.var(0)
        nt = self.n + b; d = m - self.mean
        self.mean += d * b / nt
        self.var = (self.var * self.n + v * b + d * d * self.n * b / nt) / nt
        self.n = nt
    def norm(self, x):
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--ep-len", type=int, default=200)
    ap.add_argument("--spawn-jitter", type=float, default=0.006)
    ap.add_argument("--init-mode", default="full", help="full | grasped (reverse-curriculum)")
    ap.add_argument("--form-closure", action="store_true", help="necked stem + finger ridges (geometric grip)")
    ap.add_argument("--max-delta", type=float, default=None,
                    help="per-step TCP xy delta clamp (smaller=gentler exploration; default 0.04)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--init-log-std", type=float, default=-0.5,
                    help="lower (e.g. -1.2) starts the policy committed; pairs with low --ent")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("checkpoints/newton_rl"))
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/actor.pt (actor+opt+obsnorm)")
    ap.add_argument("--seed", type=int, default=0, help="torch+env seed (net init / learning speed is seed-sensitive)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    md = None if args.max_delta is None else [args.max_delta, args.max_delta, 0.02]
    env = NewtonVecEnv(num_envs=args.num_envs, spawn_jitter=args.spawn_jitter,
                       max_ctrl=args.ep_len, init_mode=args.init_mode,
                       form_closure=args.form_closure, max_delta=md)
    N, T = env.N, args.ep_len
    ac = ActorCritic(env.obs_dim, env.act_dim, init_log_std=args.init_log_std).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    rn = RunNorm(env.obs_dim)
    best_sr = -1.0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume and (args.out / "actor.pt").exists():
        ck = torch.load(args.out / "actor.pt", map_location=dev, weights_only=False)
        ac.load_state_dict(ck["actor"]); rn.mean = ck["obs_mean"]; rn.var = ck["obs_var"]
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        if "rn_n" in ck: rn.n = ck["rn_n"]
        print(f"[ppo] resumed from {args.out/'actor.pt'} (prior succ={ck.get('succ','?')})", flush=True)
    print(f"[ppo] N={N} T={T} obs={env.obs_dim} act={env.act_dim} dev={dev}", flush=True)

    for it in range(args.iters):
        t0 = time.time()
        obs_b = np.zeros((T, N, env.obs_dim), np.float32)
        act_b = np.zeros((T, N, env.act_dim), np.float32)
        logp_b = np.zeros((T, N), np.float32)
        val_b = np.zeros((T, N), np.float32)
        rew_b = np.zeros((T, N), np.float32)
        succ_ep = np.zeros(N, bool)

        o = env.reset()
        grasp_ep = np.zeros(N, bool); lift_ep = np.zeros(N, bool); near_ep = np.zeros(N, bool)
        for t in range(T):
            on = rn.norm(o)
            with torch.no_grad():
                mu, std, val = ac(torch.from_numpy(on).to(dev))
                dist = torch.distributions.Normal(mu, std)
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1)
            an = a.cpu().numpy()
            obs_b[t] = on; act_b[t] = an; logp_b[t] = logp.cpu().numpy(); val_b[t] = val.cpu().numpy()
            o, r, d, info = env.step(an)
            rew_b[t] = r
            succ_ep |= info["success"]
            grasp_ep |= info["grasped"]; lift_ep |= info["lifted"]; near_ep |= info["near_dest"]
        rn.update(obs_b.reshape(-1, env.obs_dim))

        # GAE (terminal at T: bootstrap 0)
        adv = np.zeros((T, N), np.float32); last = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nextv = val_b[t + 1] if t + 1 < T else np.zeros(N, np.float32)
            delta = rew_b[t] + args.gamma * nextv - val_b[t]
            last = delta + args.gamma * args.lam * last
            adv[t] = last
        ret = adv + val_b
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # flatten + PPO update
        bo = torch.from_numpy(obs_b.reshape(-1, env.obs_dim)).to(dev)
        ba = torch.from_numpy(act_b.reshape(-1, env.act_dim)).to(dev)
        blp = torch.from_numpy(logp_b.reshape(-1)).to(dev)
        bret = torch.from_numpy(ret.reshape(-1)).to(dev)
        badv = torch.from_numpy(adv.reshape(-1)).to(dev)
        n = bo.shape[0]; mb = n // args.minibatches
        for _ in range(args.epochs):
            idx = torch.randperm(n, device=dev)
            for s in range(0, n, mb):
                j = idx[s:s + mb]
                mu, std, val = ac(bo[j])
                dist = torch.distributions.Normal(mu, std)
                lp = dist.log_prob(ba[j]).sum(-1)
                ratio = (lp - blp[j]).exp()
                s1 = ratio * badv[j]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * badv[j]
                pol_loss = -torch.min(s1, s2).mean()
                v_loss = ((val - bret[j]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pol_loss + 0.5 * v_loss - args.ent * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0); opt.step()

        sr = succ_ep.mean() * 100
        print(f"[ppo] it {it:3d}  succ={sr:5.1f}%  grasp={grasp_ep.mean()*100:4.0f}% "
              f"lift={lift_ep.mean()*100:4.0f}% near={near_ep.mean()*100:4.0f}%  "
              f"ep_rew={rew_b.sum(0).mean():7.2f}  v_loss={v_loss.item():7.1f}  {time.time()-t0:.0f}s", flush=True)
        ck = {"actor": ac.state_dict(), "obs_mean": rn.mean, "obs_var": rn.var,
              "obs_dim": env.obs_dim, "act_dim": env.act_dim,
              "opt": opt.state_dict(), "rn_n": rn.n, "succ": float(sr), "it": it}
        if it > 0 and it % args.save_every == 0 or it == args.iters - 1:
            torch.save(ck, args.out / "actor.pt")
            print(f"[ppo] saved {args.out/'actor.pt'} (succ={sr:.1f}%)", flush=True)
        # keep the best-so-far separately so a late regression / overwrite can't lose
        # the peak (note: train sr is a rolling proxy; confirm with diagnose_carry.py).
        if sr > best_sr:
            best_sr = sr
            torch.save(ck, args.out / "actor_best.pt")
    print(f"[ppo] DONE (best train succ={best_sr:.1f}%)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
