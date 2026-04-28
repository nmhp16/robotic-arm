"""Run the parametric scripted-oracle demo collector for a task.

Loads the task config and forwards control to
:func:`arm_act.tasks._runtime.oracle.main`. All flags after ``--task``
pass through to the oracle's own argparser.
"""

from __future__ import annotations

import argparse
import sys

from arm_act.config import DEFAULT_TASK, load
from arm_act.tasks._runtime.oracle import main as run_oracle


def main() -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--task", default=DEFAULT_TASK)
    args, remainder = p.parse_known_args()

    sys.argv = ["oracle"] + remainder
    return int(run_oracle(load(args.task)))


if __name__ == "__main__":
    raise SystemExit(main())
