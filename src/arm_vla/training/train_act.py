"""Train ACT on UR5 pick-and-place demos.

Single-config trainer — replaces the OpenVLA LoRA pipeline. Reads
``data/augmented/demos.hdf5`` directly (no RLDS step), constructs an
action-chunked dataloader, trains an ACT model with L1 loss, and saves
the policy + normalization stats every ``save_every`` steps.

    ./scripts/train.sh                       # uses src/arm_vla/training/config.yaml
    ./scripts/train.sh --max-steps 1000      # ad-hoc override
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from arm_vla.training.act_policy import (
    ACTConfig,
    ACTModel,
    normalize_actions,
    normalize_states,
    save_policy,
)
from arm_vla.training.dataset import HDF5DemoDataset, collate

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=pathlib.Path, default=pathlib.Path("src/arm_vla/training/config.yaml"))
    p.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("checkpoints/act-ur5-pickplace"))
    # Optional ad-hoc overrides (don't have to edit the YAML for a quick test).
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def _cosine_lr(step: int, warmup: int, total: int, peak: float, min_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    span = 1.0 - min_frac
    return peak * (min_frac + span * 0.5 * (1.0 + float(np.cos(np.pi * min(progress, 1.0)))))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    # --- Data ----------------------------------------------------------------
    data_cfg = cfg["data"]
    dataset = HDF5DemoDataset(
        hdf5_path=data_cfg["hdf5_path"],
        camera_keys=tuple(data_cfg["camera_keys"]),
        chunk_size=cfg["policy"]["chunk_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=cfg["training"].get("num_workers", 4) > 0,
        drop_last=True,
    )

    # --- Model ---------------------------------------------------------------
    pol_cfg = cfg["policy"]
    model_cfg = ACTConfig(
        camera_keys=tuple(data_cfg["camera_keys"]),
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        chunk_size=pol_cfg["chunk_size"],
        hidden_dim=pol_cfg.get("hidden_dim", 256),
        n_heads=pol_cfg.get("n_heads", 8),
        n_encoder_layers=pol_cfg.get("n_encoder_layers", 4),
        n_decoder_layers=pol_cfg.get("n_decoder_layers", 1),
        dim_feedforward=pol_cfg.get("dim_feedforward", 1024),
        dropout=pol_cfg.get("dropout", 0.1),
        pretrained_backbone=pol_cfg.get("pretrained_backbone", True),
        action_min=dataset.action_stats.min.tolist(),
        action_max=dataset.action_stats.max.tolist(),
        state_mean=dataset.state_stats.mean.tolist(),
        state_std=dataset.state_stats.std.tolist(),
    )
    model = ACTModel(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("ACT model: %.1f M params", n_params)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    # --- Train loop ----------------------------------------------------------
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    max_steps = cfg["training"]["max_steps"]
    log_every = cfg["training"].get("log_every", 50)
    save_every = cfg["training"].get("save_every", 5000)
    warmup = cfg["training"].get("warmup_steps", 500)
    peak_lr = cfg["training"]["learning_rate"]

    model.train()
    step = 0
    t0 = time.time()
    losses_window: list[float] = []
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
            state = batch["state"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            pad = batch["action_pad"].to(device, non_blocking=True)

            state_norm = normalize_states(state, dataset.state_stats)
            action_norm = normalize_actions(action, dataset.action_stats)

            pred = model(images, state_norm)  # (B, chunk, action_dim) in [-1, 1]
            loss_per_step = F.l1_loss(pred, action_norm, reduction="none").mean(dim=-1)  # (B, chunk)
            mask = (~pad).to(loss_per_step.dtype)
            loss = (loss_per_step * mask).sum() / mask.sum().clamp_min(1.0)

            for g in optim.param_groups:
                g["lr"] = _cosine_lr(step, warmup, max_steps, peak_lr)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            losses_window.append(loss.item())
            if step % log_every == 0:
                avg = sum(losses_window) / len(losses_window)
                losses_window.clear()
                elapsed = time.time() - t0
                lr_now = optim.param_groups[0]["lr"]
                logger.info(
                    "step %6d  loss=%.4f  lr=%.2e  elapsed=%.0fs",
                    step,
                    avg,
                    lr_now,
                    elapsed,
                )

            if step > 0 and step % save_every == 0:
                save_policy(out_dir / f"step_{step:06d}", model, dataset.action_stats, dataset.state_stats)
                logger.info("saved checkpoint: step %d", step)

            step += 1

    save_policy(out_dir / "final", model, dataset.action_stats, dataset.state_stats)
    logger.info("done. final checkpoint at %s", out_dir / "final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
