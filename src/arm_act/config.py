"""Single source of truth for the ACT pipeline config.

Two-layer YAML:

* ``src/arm_vla/training/defaults.yaml``  — shared hyperparams (policy,
  training, eval, collect, mimic). You usually don't touch this.
* ``src/arm_vla/tasks/<task>.yaml``       — per-task spec (one file per
  task — gym ids, objects, cameras, success thresholds, oracle waypoints,
  data paths, checkpoint location).

``load(task)`` returns the deep-merged dict — overlay wins on conflicts.
The module also exposes a tiny CLI so shell scripts can read fields:

    python -m arm_vla.config get task.gym_id --task pick_place
    python -m arm_vla.config list                # all task names

Used by ``arm_vla.training.train_act``, ``arm_vla.eval.rollout``, and the
``arm_vla.cli.*`` shell wrappers.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
from typing import Any

import yaml

_HERE = pathlib.Path(__file__).resolve().parent
DEFAULTS_PATH = _HERE / "training" / "defaults.yaml"
TASKS_ROOT = _HERE / "tasks"
DEFAULT_TASK = "pick_place"


def load(task: str = DEFAULT_TASK, defaults_path: pathlib.Path | None = None) -> dict[str, Any]:
    """Return defaults.yaml deep-merged with ``tasks/<task>.yaml``.

    Raises FileNotFoundError if no such task spec exists.
    """
    base = _read(defaults_path or DEFAULTS_PATH)
    task_yaml = TASKS_ROOT / f"{task}.yaml"
    if not task_yaml.is_file():
        available = ", ".join(list_tasks()) or "(none)"
        raise FileNotFoundError(
            f"no task spec at {task_yaml} — available tasks: {available}"
        )
    overlay = _read(task_yaml)
    return _deep_merge(base, overlay)


def list_tasks() -> list[str]:
    """Return all task names — every ``<name>.yaml`` directly under ``tasks/``."""
    return sorted(p.stem for p in TASKS_ROOT.glob("*.yaml"))


def _read(p: pathlib.Path) -> dict[str, Any]:
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge: dict values are merged, scalars/lists in overlay win."""
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _cli() -> int:
    p = argparse.ArgumentParser(description="Read values from the merged ACT pipeline config.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("get", help="Print one config value.")
    g.add_argument("key", help="Dotted path, e.g. task.gym_id or eval.num_episodes")
    g.add_argument("--task", default=DEFAULT_TASK)

    sub.add_parser("list", help="List all available task names.")

    args = p.parse_args()
    if args.cmd == "list":
        for name in list_tasks():
            print(name)
        return 0

    cfg = load(args.task)
    val: Any = cfg
    for k in args.key.split("."):
        if not isinstance(val, dict) or k not in val:
            print(f"missing key: {args.key}", file=sys.stderr)
            return 2
        val = val[k]
    if isinstance(val, list):
        print("\n".join(str(x) for x in val))
    elif val is None:
        print("")
    else:
        print(val)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
