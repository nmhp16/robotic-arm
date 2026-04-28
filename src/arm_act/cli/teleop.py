"""Record keyboard demonstrations for a task.

Wraps Isaac Lab's ``scripts/tools/record_demos.py``. Resolves the task's
gym id and default raw HDF5 path from :func:`arm_act.config.load`.

Keyboard (Se3Keyboard):

    W/S   +x / -x     A/D  +y / -y     Q/E  +z / -z
    Z/X   roll        T/G  pitch       C/V  yaw
    K     toggle gripper (open/close)
    R     reset and discard current episode
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
    p.add_argument("--num-demos", type=int, default=None,
                   help="default: collect.num_demos in defaults.yaml")
    p.add_argument("--step-hz", type=int, default=None,
                   help="default: collect.step_hz in defaults.yaml")
    p.add_argument("--dataset", type=pathlib.Path, default=None,
                   help="default: data.raw_path in tasks/<task>.yaml")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load(args.task)

    record_demos = isaaclab_script("scripts/tools/record_demos.py")
    register_tasks()

    dataset = pathlib.Path(args.dataset or cfg["data"]["raw_path"])
    dataset.parent.mkdir(parents=True, exist_ok=True)

    num_demos = args.num_demos or cfg["collect"]["num_demos"]
    step_hz = args.step_hz or cfg["collect"]["step_hz"]

    sys.argv = [
        "record_demos.py",
        "--task", cfg["task"]["gym_id"],
        "--teleop_device", "keyboard",
        "--dataset_file", str(dataset),
        "--step_hz", str(step_hz),
        "--num_demos", str(num_demos),
        "--enable_cameras",
    ]
    runpy.run_path(str(record_demos), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
