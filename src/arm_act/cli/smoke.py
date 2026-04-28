"""Run the parametric scene-preview / smoke check for a given task."""

from __future__ import annotations

import argparse
import sys

from arm_vla.config import DEFAULT_TASK, load
from arm_vla.tasks._runtime.smoke import main as smoke_main


def main() -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--task", default=DEFAULT_TASK)
    args, remainder = p.parse_known_args()

    sys.argv = ["smoke"] + remainder
    return int(smoke_main(load(args.task)))


if __name__ == "__main__":
    raise SystemExit(main())
