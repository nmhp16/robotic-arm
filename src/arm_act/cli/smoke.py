"""Run the parametric scene-preview / smoke check for a task.

Loads the task config and forwards control to
:func:`arm_act.tasks._runtime.smoke.main`. All flags after ``--task``
pass through to the smoke check's own argparser.
"""

from __future__ import annotations

import argparse
import sys

from arm_act.config import DEFAULT_TASK, load
from arm_act.tasks._runtime.smoke import main as run_smoke


def main() -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--task", default=DEFAULT_TASK)
    args, remainder = p.parse_known_args()

    sys.argv = ["smoke"] + remainder
    return int(run_smoke(load(args.task)))


if __name__ == "__main__":
    raise SystemExit(main())
