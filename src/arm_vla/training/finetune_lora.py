"""LoRA fine-tune OpenVLA-7B on our UR10 pick-and-place RLDS dataset.

This is a thin, educational driver script. The heavy lifting — the VLA model,
the RLDS data pipeline, the continuous→discrete action tokenization — is
delegated to the upstream ``openvla`` package. Our job here is:

1. Register our TFDS dataset with OpenVLA's OXE config registry.
2. Compute action normalization stats (q01/q99 per-dim) from the training
   split, because OpenVLA expects normalized actions in [-1, 1] so the 256
   discrete bins cover the real action range.
3. Build the model + LoRA adapters.
4. Run a standard HF training loop.
5. Save the LoRA checkpoint + the action stats (the stats are needed at
   inference to unnormalize predictions — keeping them next to the checkpoint
   is the usual convention).

Run inside the training venv (``./.venv``). OpenVLA must be installed from
git — see README.md.

    ./scripts/train.sh
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import draccus
import numpy as np
import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import wandb


@dataclass
class Config:
    config_path: pathlib.Path = pathlib.Path("src/arm_vla/training/config.yaml")
    output_dir: pathlib.Path = pathlib.Path("checkpoints/openvla-ur10-pickplace-lora")


def _load_yaml(p: pathlib.Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


def _register_dataset(data_root_dir: pathlib.Path, dataset_name: str):
    """Inject our TFDS dataset into OpenVLA's OXE registry.

    OpenVLA looks up a dataset by name in ``OXE_DATASET_CONFIGS``; that dict
    maps name → per-dim feature spec (image keys, proprio dims, action dims,
    control freq). We fabricate an entry for our dataset that matches what
    the RLDS builder wrote.

    This is a runtime patch rather than a fork. If/when the schema drifts,
    the mismatch will surface as a KeyError when the data loader tries to
    read an expected field.
    """
    from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
    from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES

    OXE_DATASET_CONFIGS[dataset_name] = {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": "pos_euler",  # we store quat but OpenVLA treats the 8-D vector as opaque state
        "action_encoding": "eef_pos",   # 7-D delta pose + gripper
        "control_freq": 20,
    }
    OXE_NAMED_MIXTURES[dataset_name] = [(dataset_name, 1.0)]
    print(f"registered {dataset_name} in OXE_DATASET_CONFIGS")


def _compute_action_stats(data_root_dir: pathlib.Path, dataset_name: str) -> dict:
    """Scan the training split once to get q01/q99 + mean/std per action dim.

    OpenVLA normalizes actions to [-1, 1] using the 1st/99th percentiles so
    outlier demonstrations don't squash the bin range. Stats are saved with
    the checkpoint for inference-time unnormalization.
    """
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(data_root_dir / dataset_name / "1.0.0"))
    ds = builder.as_dataset(split="train")

    all_actions = []
    for ep in tfds.as_numpy(ds):
        for step in ep["steps"]:
            all_actions.append(step["action"])
    actions = np.stack(all_actions, axis=0)  # (N, 7)
    stats = {
        "q01": np.quantile(actions, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).astype(np.float32).tolist(),
        "mean": actions.mean(axis=0).astype(np.float32).tolist(),
        "std": actions.std(axis=0).astype(np.float32).tolist(),
        "min": actions.min(axis=0).astype(np.float32).tolist(),
        "max": actions.max(axis=0).astype(np.float32).tolist(),
        "num_transitions": int(actions.shape[0]),
        # Last dim is the binary gripper command; mask it so OpenVLA doesn't
        # normalize a binary channel into nonsense.
        "mask": [True] * 6 + [False],
    }
    print(f"action stats computed over {stats['num_transitions']} transitions")
    return stats


def main(cfg: Config):
    print(f"loading training config from {cfg.config_path}")
    yaml_cfg = _load_yaml(cfg.config_path)

    data_cfg = yaml_cfg["data"]
    model_cfg = yaml_cfg["model"]
    lora_cfg = yaml_cfg["lora"]
    train_cfg = yaml_cfg["training"]
    wb_cfg = yaml_cfg["wandb"]

    data_root_dir = pathlib.Path(data_cfg["data_root_dir"]).resolve()
    dataset_name = data_cfg["dataset_name"]

    # 1. Register the dataset before any OpenVLA import that would freeze the
    #    config dict.
    _register_dataset(data_root_dir, dataset_name)

    # 2. Action stats (cheap, one scan).
    stats = _compute_action_stats(data_root_dir, dataset_name)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cfg.output_dir / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump({dataset_name: {"action": stats}}, f, indent=2)
    print(f"wrote {stats_path}")

    # 3. Model + processor.
    print(f"loading {model_cfg['vla_path']} (attn={model_cfg['attn_implementation']})")
    processor = AutoProcessor.from_pretrained(model_cfg["vla_path"], trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_cfg["vla_path"],
        torch_dtype=torch.bfloat16 if train_cfg["bf16"] else torch.float32,
        attn_implementation=model_cfg["attn_implementation"],
        trust_remote_code=True,
    )
    model.to("cuda")

    # 4. LoRA. Freezes base weights; trains low-rank adapters on attention &
    #    MLP projections.
    lora = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 5. Data loader. OpenVLA's RLDSDataset handles all the VLA-specific bits:
    #    image aug, action quantile bucketing, language tokenization, prompt
    #    formatting. We just need to give it the right args.
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSDataset
    from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

    # Save stats in the format OpenVLA expects (per-dataset nesting) so the
    # RLDSDataset loader can read them instead of rescanning.
    save_dataset_statistics({dataset_name: {"action": stats}}, cfg.output_dir)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    train_dataset = RLDSDataset(
        data_root_dir=data_root_dir,
        data_mix=dataset_name,
        batch_transform=None,   # RLDSDataset supplies a default for VLA training
        resize_resolution=(224, 224),
        shuffle_buffer_size=data_cfg["shuffle_buffer_size"],
        image_aug=data_cfg["image_aug"],
        train=True,
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        sampler=None,  # RLDSDataset is an IterableDataset under the hood
        num_workers=0,
        collate_fn=None,
    )

    # 6. Optimizer + schedule.
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train_cfg["learning_rate"],
    )
    total_steps = train_cfg["max_steps"]
    warmup = train_cfg["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        # Cosine decay to 10% of peak.
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # 7. wandb.
    if wb_cfg["enabled"]:
        wandb.init(project=wb_cfg["project"], mode=wb_cfg["mode"], config=yaml_cfg)

    # 8. Training loop. Straightforward: forward, loss on action tokens, step.
    model.train()
    step = 0
    grad_accum = train_cfg["grad_accum"]
    for batch in dataloader:
        batch = {k: v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            optim.step()
            scheduler.step()
            optim.zero_grad()

        if step % train_cfg["log_every"] == 0:
            print(f"step {step:>6d}  loss={loss.item() * grad_accum:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
            if wb_cfg["enabled"]:
                wandb.log({"loss": loss.item() * grad_accum, "lr": scheduler.get_last_lr()[0]}, step=step)

        if step > 0 and step % train_cfg["save_every"] == 0:
            ckpt = cfg.output_dir / f"step_{step:06d}"
            model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)
            print(f"saved {ckpt}")

        step += 1
        if step >= total_steps:
            break

    # Final save.
    model.save_pretrained(cfg.output_dir / "final")
    processor.save_pretrained(cfg.output_dir / "final")
    print(f"training done — final checkpoint at {cfg.output_dir / 'final'}")


if __name__ == "__main__":
    cfg = draccus.parse(Config)
    main(cfg)
