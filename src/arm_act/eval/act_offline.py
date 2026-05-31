"""Offline action-prediction eval for a trained ACT checkpoint.

Loads a checkpoint written by ``train_act`` and, for every frame of a demo
HDF5, compares the policy's predicted action to the recorded oracle action.
This is the pure-offline counterpart to the Isaac sim rollout (``rollout.py``):
no simulator, no Isaac Lab, no Newton — just "did BC learn the oracle's
state->action mapping". Useful when the closed-loop env can't run cheaply
(e.g. the Newton state demos: PhysX can't reproduce the grasp, and a Newton
policy rollout is a separate harness).

Reports, per action dim, the mean-|err| and RMSE in raw action units, plus the
binary gripper accuracy (sign match on the last dim). The state vector is
rebuilt from the checkpoint's recorded ``state_keys`` so it matches training.

    ./scripts/eval_offline_act.sh --checkpoint checkpoints/.../final \
        --hdf5 data/raw/pick_plant_out_of_vial_zimmer/demos_newton.hdf5
  (or:  python -m arm_act.eval.act_offline --checkpoint <dir> --hdf5 <file>)
"""

from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np
import torch

from arm_act.training.act_policy import load_policy, normalize_states


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=pathlib.Path, required=True,
                   help="checkpoint dir (model.pt + config.json + norm_stats.json)")
    p.add_argument("--hdf5", type=pathlib.Path, required=True,
                   help="demo HDF5 to evaluate against (held-out set ideally)")
    p.add_argument("--max-demos", type=int, default=0, help="0 = all demos in the file")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    policy = load_policy(args.checkpoint, device=args.device)
    model = policy.model
    state_keys = tuple(model.cfg.state_keys)
    action_dim = model.cfg.action_dim
    print(f"[offline] checkpoint: {args.checkpoint}")
    print(f"[offline] state_keys={state_keys}  state_dim={model.cfg.state_dim}  "
          f"action_dim={action_dim}  cameras={model.cfg.camera_keys or '(none)'}")

    abs_err = np.zeros(action_dim, dtype=np.float64)
    sq_err = np.zeros(action_dim, dtype=np.float64)
    n_frames = 0
    grip_correct = 0
    n_demos = 0

    with h5py.File(str(args.hdf5), "r") as f:
        demo_ids = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if args.max_demos:
            demo_ids = demo_ids[: args.max_demos]
        for did in demo_ids:
            d = f["data"][did]
            if "actions" not in d or "obs" not in d:
                continue
            obs = d["obs"]
            actions = np.asarray(d["actions"], dtype=np.float32)
            T = actions.shape[0]
            if T == 0:
                continue
            state = np.concatenate(
                [np.asarray(obs[k], dtype=np.float32).reshape(T, -1) for k in state_keys],
                axis=1,
            )
            st = torch.from_numpy(state).to(args.device)
            st_norm = normalize_states(st, policy.state_stats)
            empty_imgs: dict[str, torch.Tensor] = {}
            with torch.inference_mode():
                # First action of the predicted chunk at each frame == the
                # action the policy would actually emit at that state.
                pred_norm = model(empty_imgs, st_norm)[:, 0, :]  # (T, action_dim)
            pred = policy._unnormalize_action(pred_norm).cpu().numpy()

            err = pred - actions
            abs_err += np.abs(err).sum(axis=0)
            sq_err += (err ** 2).sum(axis=0)
            grip_correct += int(np.sum(np.sign(pred[:, -1]) == np.sign(actions[:, -1])))
            n_frames += T
            n_demos += 1

    if n_frames == 0:
        print("[offline] no usable frames")
        return 1

    mae = abs_err / n_frames
    rmse = np.sqrt(sq_err / n_frames)
    print(f"[offline] evaluated {n_demos} demos, {n_frames} frames")
    print(f"[offline] per-dim MAE  : {np.array2string(mae, precision=5, floatmode='fixed')}")
    print(f"[offline] per-dim RMSE : {np.array2string(rmse, precision=5, floatmode='fixed')}")
    print(f"[offline] mean MAE (pos dims 0..{action_dim-2}): {mae[:-1].mean():.5f}")
    print(f"[offline] gripper sign accuracy: {grip_correct}/{n_frames} = "
          f"{100.0 * grip_correct / n_frames:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
