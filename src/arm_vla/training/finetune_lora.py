"""OpenVLA LoRA fine-tune on a UR5 + 2F-85 pick-and-place RLDS dataset.

Delegates VLA-specific plumbing (RLDS data loader, action tokenizer, prompt
formatting) to the upstream ``openvla`` / ``prismatic`` package. This driver
registers the local dataset with OpenVLA's OXE config registry at runtime,
computes action quantile stats for inference-time unnormalization, and runs
a standard HF + peft training loop.

    ./scripts/train.sh
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass, field

import draccus
import numpy as np
import torch
import wandb
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "src/arm_vla/training/config.yaml"
_DEFAULT_OUTPUT_DIR = "checkpoints/openvla-ur5-pickplace-lora"


def _env_path(name: str, fallback: str) -> pathlib.Path:
    return pathlib.Path(os.environ.get(name, fallback))


@dataclass
class Config:
    # Override via CLI (--config-path) or env (ARM_VLA_TRAIN_CONFIG).
    config_path: pathlib.Path = field(
        default_factory=lambda: _env_path("ARM_VLA_TRAIN_CONFIG", _DEFAULT_CONFIG_PATH)
    )
    # Override via CLI (--output-dir) or env (ARM_VLA_CHECKPOINT_DIR).
    output_dir: pathlib.Path = field(
        default_factory=lambda: _env_path("ARM_VLA_CHECKPOINT_DIR", _DEFAULT_OUTPUT_DIR)
    )


def _load_yaml(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _register_dataset(dataset_name: str) -> None:
    """Inject the dataset entry into OpenVLA's OXE registry.

    Avoids forking upstream ``openvla``. Schema drift surfaces as a
    ``KeyError`` at data-loader construction time.
    """
    from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
    from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES

    OXE_DATASET_CONFIGS[dataset_name] = {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": "pos_euler",
        "action_encoding": "eef_pos",
        "control_freq": 20,
    }
    OXE_NAMED_MIXTURES[dataset_name] = [(dataset_name, 1.0)]


def _compute_action_stats(data_root_dir: pathlib.Path, dataset_name: str) -> dict:
    """Scan the training split to get q01/q99 + mean/std per action dim."""
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(data_root_dir / dataset_name / "1.0.0"))
    ds = builder.as_dataset(split="train")

    all_actions = []
    for ep in tfds.as_numpy(ds):
        for step in ep["steps"]:
            all_actions.append(step["action"])
    actions = np.stack(all_actions, axis=0)
    return {
        "q01": np.quantile(actions, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).astype(np.float32).tolist(),
        "mean": actions.mean(axis=0).astype(np.float32).tolist(),
        "std": actions.std(axis=0).astype(np.float32).tolist(),
        "min": actions.min(axis=0).astype(np.float32).tolist(),
        "max": actions.max(axis=0).astype(np.float32).tolist(),
        "num_transitions": int(actions.shape[0]),
        # Last dim is the binary gripper command; don't normalize it.
        "mask": [True] * 6 + [False],
    }


# Cosine schedule decays LR from 1.0 to _LR_MIN_FRAC of peak, not to 0 —
# matches OpenVLA's reference finetune config and avoids vanishing-gradient
# behavior in the tail.
_LR_MIN_FRAC: float = 0.1


def _cosine_schedule(warmup: int, total: int):
    span = 1.0 - _LR_MIN_FRAC

    def fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return _LR_MIN_FRAC + span * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    return fn


def main(cfg: Config) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    yaml_cfg = _load_yaml(cfg.config_path)
    data_cfg = yaml_cfg["data"]
    model_cfg = yaml_cfg["model"]
    lora_cfg = yaml_cfg["lora"]
    train_cfg = yaml_cfg["training"]
    wb_cfg = yaml_cfg["wandb"]

    data_root_dir = pathlib.Path(data_cfg["data_root_dir"]).resolve()
    dataset_name = data_cfg["dataset_name"]

    _register_dataset(dataset_name)

    stats = _compute_action_stats(data_root_dir, dataset_name)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_cfg["vla_path"], trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_cfg["vla_path"],
        torch_dtype=torch.bfloat16 if train_cfg["bf16"] else torch.float32,
        attn_implementation=model_cfg["attn_implementation"],
        trust_remote_code=True,
    ).to("cuda")

    lora = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    from prismatic.vla.datasets import RLDSDataset
    from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

    save_dataset_statistics({dataset_name: {"action": stats}}, cfg.output_dir)

    train_dataset = RLDSDataset(
        data_root_dir=data_root_dir,
        data_mix=dataset_name,
        batch_transform=None,
        resize_resolution=(224, 224),
        shuffle_buffer_size=data_cfg["shuffle_buffer_size"],
        image_aug=data_cfg["image_aug"],
        train=True,
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        sampler=None,
        num_workers=0,
        collate_fn=None,
    )

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train_cfg["learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=_cosine_schedule(train_cfg["warmup_steps"], train_cfg["max_steps"]),
    )

    if wb_cfg["enabled"]:
        wandb.init(project=wb_cfg["project"], mode=wb_cfg["mode"], config=yaml_cfg)

    model.train()
    step = 0
    grad_accum = train_cfg["grad_accum"]
    for batch in dataloader:
        batch = {
            k: v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = outputs.loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            optim.step()
            scheduler.step()
            optim.zero_grad()

        if step % train_cfg["log_every"] == 0:
            logger.info(
                "step %6d  loss=%.4f  lr=%.2e",
                step,
                loss.item() * grad_accum,
                scheduler.get_last_lr()[0],
            )
            if wb_cfg["enabled"]:
                wandb.log(
                    {"loss": loss.item() * grad_accum, "lr": scheduler.get_last_lr()[0]},
                    step=step,
                )

        if step > 0 and step % train_cfg["save_every"] == 0:
            ckpt = cfg.output_dir / f"step_{step:06d}"
            model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)

        step += 1
        if step >= train_cfg["max_steps"]:
            break

    model.save_pretrained(cfg.output_dir / "final")
    processor.save_pretrained(cfg.output_dir / "final")


if __name__ == "__main__":
    main(draccus.parse(Config))
