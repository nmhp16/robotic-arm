"""Tests for ``arm_act.config.load`` and the deep-merge overlay."""

from __future__ import annotations

import textwrap


def test_list_tasks_includes_plant_tasks() -> None:
    from arm_act.config import list_tasks

    tasks = list_tasks()
    assert "pick_plant_out" in tasks
    assert "put_plant_back" in tasks


def test_load_default_task_returns_dict() -> None:
    from arm_act.config import load

    cfg = load()  # default task = pick_plant_out
    assert isinstance(cfg, dict)
    for section in (
        "task", "robot", "objects", "cameras", "success", "grasp_check",
        "oracle", "mimic", "data", "policy", "training", "eval",
        "collect",
    ):
        assert section in cfg, f"missing section: {section}"


def test_load_pick_plant_out_overlays_task_fields() -> None:
    from arm_act.config import load

    cfg = load("pick_plant_out")
    # Overlay (task.yaml) wins on these:
    assert cfg["task"]["name"] == "pick_plant_out"
    assert cfg["task"]["gym_id"] == "Isaac-PickPlantOut-T3-IK-Rel-v0"
    assert cfg["robot"]["type"] == "t3_401_simple_gripper"
    assert cfg["data"]["hdf5_path"]
    assert cfg["training"]["output_dir"]
    assert cfg["eval"]["checkpoint"]
    # Defaults survive when overlay doesn't override:
    assert cfg["policy"]["chunk_size"] > 0
    assert cfg["data"]["camera_keys"]


def test_load_unknown_task_raises() -> None:
    import pytest

    from arm_act.config import load

    with pytest.raises(FileNotFoundError):
        load("does_not_exist_task_name_12345")


def test_load_custom_defaults_path(tmp_path) -> None:
    from arm_act.config import load

    custom = tmp_path / "alt_defaults.yaml"
    custom.write_text(
        textwrap.dedent(
            """
            policy:
              chunk_size: 7
            training:
              max_steps: 11
            """
        ).strip()
    )
    cfg = load("pick_plant_out", defaults_path=custom)
    assert cfg["policy"]["chunk_size"] == 7
    assert cfg["training"]["max_steps"] == 11
    # task.yaml overlay still applied on top:
    assert cfg["task"]["name"] == "pick_plant_out"
    assert cfg["training"]["output_dir"]
