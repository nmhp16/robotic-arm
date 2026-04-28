"""Validate the bundled defaults.yaml and per-task task.yaml files."""

from __future__ import annotations

import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULTS_PATH = REPO_ROOT / "src" / "arm_act" / "training" / "defaults.yaml"
TASKS_ROOT = REPO_ROOT / "src" / "arm_act" / "tasks"


def _read(p: pathlib.Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


@pytest.fixture
def defaults() -> dict:
    return _read(DEFAULTS_PATH)


def test_defaults_file_exists() -> None:
    assert DEFAULTS_PATH.is_file(), f"missing: {DEFAULTS_PATH}"


def test_defaults_has_required_sections(defaults: dict) -> None:
    for section in ("data", "policy", "training", "eval", "collect", "mimic"):
        assert section in defaults, f"missing section: {section}"


def test_defaults_does_not_pin_task_specific_fields(defaults: dict) -> None:
    # Task-specific stuff (gym ids, output paths, checkpoint paths) belongs
    # in tasks/<task>.yaml — not the shared defaults.
    assert "task" not in defaults
    assert "hdf5_path" not in defaults.get("data", {})
    assert "output_dir" not in defaults.get("training", {})
    assert "checkpoint" not in defaults.get("eval", {})


def test_defaults_policy_fields(defaults: dict) -> None:
    required = (
        "chunk_size", "hidden_dim", "n_heads", "n_encoder_layers",
        "n_decoder_layers", "dim_feedforward", "dropout", "pretrained_backbone",
    )
    for f in required:
        assert f in defaults["policy"], f"missing policy.{f}"


def test_defaults_training_fields(defaults: dict) -> None:
    required = (
        "max_steps", "batch_size", "learning_rate", "warmup_steps",
        "log_every", "save_every",
    )
    for f in required:
        assert f in defaults["training"], f"missing training.{f}"


def test_defaults_step_budgets_positive(defaults: dict) -> None:
    t = defaults["training"]
    assert t["max_steps"] > t["warmup_steps"] >= 0
    assert t["batch_size"] > 0
    e = defaults["eval"]
    assert e["num_episodes"] > 0
    assert e["max_steps_per_episode"] > 0


def test_only_task_yamls_under_tasks_root() -> None:
    """Tasks should be YAML-only — flat ``tasks/<name>.yaml`` files."""
    yamls = sorted(p.name for p in TASKS_ROOT.glob("*.yaml"))
    assert yamls, "no task yaml files under tasks/"
    # No straggler .yml or per-task subfolders.
    subdirs = sorted(
        p.name for p in TASKS_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith("_") and p.name != "__pycache__"
    )
    assert subdirs == [], (
        f"unexpected task subfolders {subdirs}; tasks should be flat YAML files"
    )


@pytest.mark.parametrize("task_yaml", sorted(TASKS_ROOT.glob("*.yaml")))
def test_task_yaml_has_required_top_level_sections(task_yaml: pathlib.Path) -> None:
    cfg = _read(task_yaml)
    for section in ("task", "robot", "objects", "cameras", "success",
                    "grasp_check", "oracle", "mimic", "data", "training", "eval"):
        assert section in cfg, f"{task_yaml}: missing section {section!r}"


@pytest.mark.parametrize("task_yaml", sorted(TASKS_ROOT.glob("*.yaml")))
def test_task_yaml_has_pickable_and_target(task_yaml: pathlib.Path) -> None:
    cfg = _read(task_yaml)
    roles = [obj.get("role") for obj in cfg["objects"].values()]
    assert "pickable" in roles, f"{task_yaml}: no object with role=pickable"
    assert "target" in roles, f"{task_yaml}: no object with role=target"


@pytest.mark.parametrize("task_yaml", sorted(TASKS_ROOT.glob("*.yaml")))
def test_task_yaml_robot_lookup_key_is_known(task_yaml: pathlib.Path) -> None:
    # Don't import the runtime (would pull in isaaclab); just check the
    # robot.type field is non-empty and looks like a valid identifier.
    cfg = _read(task_yaml)
    rtype = cfg["robot"]["type"]
    assert isinstance(rtype, str) and rtype.replace("_", "").isalnum()


@pytest.mark.parametrize("task_yaml", sorted(TASKS_ROOT.glob("*.yaml")))
def test_task_yaml_gym_ids_distinct(task_yaml: pathlib.Path) -> None:
    cfg = _read(task_yaml)
    assert cfg["task"]["gym_id"] != cfg["task"]["mimic_gym_id"]
