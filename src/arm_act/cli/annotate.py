"""Annotate raw demos with mimic ``datagen_info`` fields.

Wraps Isaac Lab's
``scripts/imitation_learning/isaaclab_mimic/annotate_demos.py``. Replays
each raw demo inside the env to compute the ``DatagenInfo`` fields
(``eef_pose``, ``object_pose``, ``target_eef_pose``,
``subtask_term_signals``) that the mimic generator requires.

Default I/O paths come from ``data.raw_path`` and ``data.annotated_path``
in ``tasks/<task>.yaml``.
"""

from __future__ import annotations

import argparse
import pathlib
import runpy
import sys

from arm_act.cli import isaaclab_script, register_tasks
from arm_act.config import DEFAULT_TASK, load


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--input", type=pathlib.Path, default=None,
                   help="default: data.raw_path in tasks/<task>.yaml")
    p.add_argument("--output", type=pathlib.Path, default=None,
                   help="default: data.annotated_path in tasks/<task>.yaml")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load(args.task)

    annotate = isaaclab_script("scripts/imitation_learning/isaaclab_mimic/annotate_demos.py")
    register_tasks()

    in_path = pathlib.Path(args.input or cfg["data"]["raw_path"])
    out_path = pathlib.Path(args.output or cfg["data"]["annotated_path"])
    if not in_path.is_file():
        print(
            f"input not found: {in_path} — run ./scripts/teleop.sh or ./scripts/oracle.sh first",
            file=sys.stderr,
        )
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "annotate_demos.py",
        "--task", cfg["task"]["mimic_gym_id"],
        "--input_file", str(in_path),
        "--output_file", str(out_path),
        "--auto",
        "--headless",
        "--enable_cameras",
    ]
    runpy.run_path(str(annotate), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
