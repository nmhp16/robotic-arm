"""Run the parametric scripted-oracle demo collector for a task.

Loads the task config and forwards control to the oracle module of the
template the task selects. Default is ``_runtime`` (standard pick-and-
place archetype); tasks can opt into a sibling template such as
``_runtime_jar`` by setting ``template:`` in their YAML. All flags
after ``--task`` pass through to the oracle's own argparser.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from arm_act.config import DEFAULT_TASK, load
from arm_act.tasks import template_for


def main() -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--task", default=DEFAULT_TASK)
    args, remainder = p.parse_known_args()

    spec = load(args.task)
    template = template_for(spec)
    oracle_mod = importlib.import_module(f"arm_act.tasks.{template}.oracle")

    sys.argv = ["oracle"] + remainder
    return int(oracle_mod.main(spec))


if __name__ == "__main__":
    raise SystemExit(main())
