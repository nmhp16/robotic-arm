"""Smoke tests for the shared eval task registry.

These run without Isaac Sim — ``arm_vla.eval.common`` is intentionally
kept free of runtime-only deps.
"""

from __future__ import annotations

import pytest

TASK_NAMES = ["pick_place", "pick_place_ur10", "stack"]
REQUIRED_FIELDS = ("gym_id", "module", "cfg_path", "instruction", "unnorm_key")


def test_registry_has_expected_tasks() -> None:
    from arm_vla.eval.common import TASK_REGISTRY

    assert set(TASK_REGISTRY) == set(TASK_NAMES)


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_task_entry_has_required_fields(task_name: str) -> None:
    from arm_vla.eval.common import TASK_REGISTRY

    entry = TASK_REGISTRY[task_name]
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    assert not missing, f"{task_name} missing: {missing}"


def test_cfg_paths_have_module_colon_class_form() -> None:
    from arm_vla.eval.common import TASK_REGISTRY

    for name, entry in TASK_REGISTRY.items():
        assert ":" in entry["cfg_path"], f"{name} cfg_path must be 'module:Class'"
        mod, cls = entry["cfg_path"].split(":")
        assert mod and cls


def test_gym_ids_are_unique() -> None:
    from arm_vla.eval.common import TASK_REGISTRY

    gym_ids = [e["gym_id"] for e in TASK_REGISTRY.values()]
    assert len(gym_ids) == len(set(gym_ids)), "duplicate gym ids in registry"
