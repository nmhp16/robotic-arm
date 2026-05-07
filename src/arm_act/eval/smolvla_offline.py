"""Offline action-prediction eval for a fine-tuned SmolVLA checkpoint.

Loads a checkpoint produced by ``./scripts/train.sh`` (the SmolVLA path)
and iterates over a slice of the LeRobotDataset, computing per-frame
L1 / MSE between the policy's predicted action chunk's first action and
the oracle's logged action. This is the cheapest sanity check for a
fine-tune:

  - if loss is high after 5k+ steps, training pipeline is broken
  - if per-dimension losses for the rotation deltas (drx/dry/drz) are
    high, the model failed to learn the always-zero pose constraint
  - if the gripper dimension's L1 is >0.5 (where the dim is in [-1, 1]),
    the model can't decide when to open/close — usually means too few
    training steps or a bad action normalization choice

This is purely offline — we don't execute actions in a sim. For closed-
loop sim rollout you need a SmolVLA-aware version of ``eval/rollout.py``,
which doesn't exist yet; the cross-venv split (lerobot lives in the
training venv, Isaac Lab uses its bundled python) is the blocker.

Usage:

    ./scripts/smolvla_eval.sh --task pick_vial_from_holder
    ./scripts/smolvla_eval.sh --task pick_vial_from_holder \\
        --checkpoint checkpoints/pick_vial_from_holder/smolvla/checkpoints/005000/pretrained_model \\
        --num-frames 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        default="pick_vial_from_holder",
        help="Task name; default reads dataset from data/lerobot/local/<task>/.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a SmolVLA checkpoint dir (the ``pretrained_model`` subfolder). "
            "Default: checkpoints/<task>/smolvla/checkpoints/last/pretrained_model"
        ),
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override LeRobotDataset root; defaults to data/lerobot/local/<task>/.",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=200,
        help="Number of randomly-sampled frames to evaluate (0 = full dataset).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.checkpoint is None:
        args.checkpoint = (
            Path("checkpoints") / args.task / "smolvla" / "checkpoints" / "last" / "pretrained_model"
        )
    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if args.dataset_root is None:
        args.dataset_root = Path("data") / "lerobot" / "local" / args.task
    if not args.dataset_root.exists():
        print(f"dataset not found: {args.dataset_root}", file=sys.stderr)
        return 1

    # Heavy imports are deferred so --help works without lerobot installed.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    repo_id = f"local/{args.task}"
    logger.info("loading dataset %s from %s", repo_id, args.dataset_root)
    dataset = LeRobotDataset(repo_id, root=args.dataset_root)
    logger.info("dataset: %d episodes / %d frames", dataset.num_episodes, len(dataset))

    logger.info("loading checkpoint %s", args.checkpoint)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    # Saved alongside the model: input preprocessor (rename + tokenize +
    # normalize) and output postprocessor (un-normalize action chunk).
    pre = DataProcessorPipeline.from_pretrained(
        args.checkpoint, config_filename="policy_preprocessor.json"
    )
    post = DataProcessorPipeline.from_pretrained(
        args.checkpoint, config_filename="policy_postprocessor.json"
    )

    n = args.num_frames if args.num_frames > 0 else len(dataset)
    n = min(n, len(dataset))
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(dataset))[:n]

    action_dim = dataset.features["action"]["shape"][0]
    action_names = dataset.features["action"].get("names") or [f"a{i}" for i in range(action_dim)]

    abs_err_sum = np.zeros(action_dim, dtype=np.float64)
    sq_err_sum = np.zeros(action_dim, dtype=np.float64)
    count = 0

    log_every = max(1, n // 10)
    with torch.inference_mode():
        for i, idx in enumerate(indices):
            sample = dataset[int(idx)]

            # SmolVLA's action queue would otherwise carry over predictions
            # from the previous (unrelated) sample, biasing this evaluator
            # toward whichever episode the chunk was generated for. Reset.
            policy.reset()

            batch = pre(sample)
            pred = policy.select_action(batch)
            pred_unnorm = post({"action": pred})["action"]
            pred_arr = pred_unnorm.squeeze(0).cpu().numpy().astype(np.float64)
            true_arr = sample["action"].numpy().astype(np.float64)

            err = pred_arr - true_arr
            abs_err_sum += np.abs(err)
            sq_err_sum += err**2
            count += 1

            if (i + 1) % log_every == 0:
                logger.info("evaluated %d / %d frames", i + 1, n)

    abs_err_mean = abs_err_sum / count
    mse = sq_err_sum / count
    rmse = np.sqrt(mse)

    print()
    print(f"Frames evaluated:   {count}")
    print(f"Checkpoint:         {args.checkpoint}")
    print(f"Dataset:            {args.dataset_root}")
    print()
    print(f"Overall L1 mean:    {abs_err_mean.mean():.4f}")
    print(f"Overall MSE:        {mse.mean():.4f}")
    print(f"Overall RMSE:       {rmse.mean():.4f}")
    print()
    print(f"{'dim':<8} {'L1 mean':>9} {'RMSE':>9}")
    print("-" * 30)
    for name, l1, r in zip(action_names, abs_err_mean, rmse, strict=False):
        print(f"{name:<8} {l1:>9.4f} {r:>9.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
