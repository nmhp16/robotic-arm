"""Validate the bundled training YAML config."""

from __future__ import annotations

import pathlib

import pytest
import yaml

CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "arm_vla"
    / "training"
    / "config.yaml"
)


@pytest.fixture
def config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_config_file_exists() -> None:
    assert CONFIG_PATH.exists(), f"missing: {CONFIG_PATH}"


def test_has_required_top_level_sections(config: dict) -> None:
    for section in ("model", "data", "lora", "training", "wandb"):
        assert section in config, f"missing section: {section}"


def test_model_fields(config: dict) -> None:
    assert "vla_path" in config["model"]
    assert "attn_implementation" in config["model"]


def test_data_fields(config: dict) -> None:
    for field in ("data_root_dir", "dataset_name", "image_aug", "shuffle_buffer_size"):
        assert field in config["data"], f"missing data.{field}"


def test_training_fields(config: dict) -> None:
    required = (
        "batch_size",
        "grad_accum",
        "learning_rate",
        "warmup_steps",
        "max_steps",
        "save_every",
        "log_every",
        "bf16",
    )
    for field in required:
        assert field in config["training"], f"missing training.{field}"


def test_lora_target_modules_non_empty(config: dict) -> None:
    targets = config["lora"]["target_modules"]
    assert isinstance(targets, list) and len(targets) > 0


def test_training_step_budgets_are_positive(config: dict) -> None:
    train = config["training"]
    assert train["max_steps"] > train["warmup_steps"] >= 0
    assert train["batch_size"] > 0
    assert train["grad_accum"] > 0
