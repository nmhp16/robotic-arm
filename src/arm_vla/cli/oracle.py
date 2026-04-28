"""Run the parametric scripted-oracle for a given task.

Loads the task config, then hands it to ``arm_vla.tasks._runtime.oracle.main``.
"""

from __future__ import annotations

import argparse
import sys

from arm_vla.config import DEFAULT_TASK, load
from arm_vla.tasks._runtime.oracle import main as oracle_main


def main() -> int:
    # Pre-parse just --task; remaining args go to the oracle's own argparser.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--task", default=DEFAULT_TASK)
    args, remainder = p.parse_known_args()

    sys.argv = ["oracle"] + remainder
    return int(oracle_main(load(args.task)))


if __name__ == "__main__":
    raise SystemExit(main())
