"""Augment annotated demos via curobo (isaaclab_mimic).

Wraps Isaac Lab's ``scripts/imitation_learning/isaaclab_mimic/generate_dataset.py``.
Runs segment replay across randomized scenes and writes a much larger
HDF5 (``data.augmented_path``) suitable for ACT training.

Defaults: ``data.annotated_path`` → ``data.augmented_path`` from task.yaml,
plus ``mimic.num_demos`` / ``mimic.num_envs`` from defaults.yaml.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import runpy
import sys

from arm_vla.config import DEFAULT_TASK, load


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--input", type=pathlib.Path, default=None, help="default: data.annotated_path")
    p.add_argument("--output", type=pathlib.Path, default=None, help="default: data.augmented_path")
    p.add_argument("--num-demos", type=int, default=None, help="default: mimic.num_demos")
    p.add_argument("--num-envs", type=int, default=None, help="default: mimic.num_envs")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load(args.task)

    isaaclab = pathlib.Path(os.environ.get("ISAACLAB", os.path.expanduser("~/IsaacLab")))
    generate = isaaclab / "scripts" / "imitation_learning" / "isaaclab_mimic" / "generate_dataset.py"
    if not generate.is_file():
        print(f"could not find {generate} — set ISAACLAB=...", file=sys.stderr)
        return 2

    __import__("arm_vla.tasks")  # auto-registers every task's gym ids

    in_path = pathlib.Path(args.input or cfg["data"]["annotated_path"])
    out_path = pathlib.Path(args.output or cfg["data"]["augmented_path"])
    if not in_path.is_file():
        print(f"input not found: {in_path} — run ./scripts/annotate.sh first", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_demos = args.num_demos or cfg["mimic"]["num_demos"]
    num_envs = args.num_envs or cfg["mimic"]["num_envs"]

    sys.argv = [
        "generate_dataset.py",
        "--task", cfg["task"]["mimic_gym_id"],
        "--input_file", str(in_path),
        "--output_file", str(out_path),
        "--generation_num_trials", str(num_demos),
        "--num_envs", str(num_envs),
        "--enable_cameras",
    ]
    runpy.run_path(str(generate), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
