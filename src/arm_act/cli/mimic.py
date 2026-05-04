"""Augment annotated demos via curobo (``isaaclab_mimic``).

Wraps Isaac Lab's
``scripts/imitation_learning/isaaclab_mimic/generate_dataset.py``. Runs
segment replay across randomized scenes and writes a much larger HDF5
suitable for ACT training.

Defaults: ``data.annotated_path`` → ``data.augmented_path`` from
``tasks/<task>.yaml``, plus ``mimic.num_demos`` and ``mimic.num_envs``
from ``defaults.yaml``.
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
                   help="default: data.annotated_path in tasks/<task>.yaml")
    p.add_argument("--output", type=pathlib.Path, default=None,
                   help="default: data.augmented_path in tasks/<task>.yaml")
    p.add_argument("--num-demos", type=int, default=None,
                   help="default: mimic.num_demos in defaults.yaml")
    p.add_argument("--num-envs", type=int, default=None,
                   help="default: mimic.num_envs in defaults.yaml")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load(args.task)

    generate = isaaclab_script("scripts/imitation_learning/isaaclab_mimic/generate_dataset.py")
    # Note: arm_act task gym registration happens INSIDE generate_dataset.py
    # (we patched it to import arm_act.tasks after AppLauncher init).
    # See the 4-line block at the top of that file.

    in_path = pathlib.Path(args.input or cfg["data"]["annotated_path"])
    out_path = pathlib.Path(args.output or cfg["data"]["augmented_path"])
    if not in_path.is_file():
        print(
            f"input not found: {in_path} — run ./scripts/annotate.sh first",
            file=sys.stderr,
        )
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
        "--headless",       # without this, AppLauncher defaults to windowed
                            # mode and env.reset() hangs forever waiting on a
                            # display surface that never appears
        "--enable_cameras",
        "--rendering_mode", "performance",  # matches the docs example;
                            # without it, gym.make() crashes inside
                            # ViewportCameraController.__init__ trying to read
                            # `omni:kit:centerOfInterest` on a viewport prim
                            # that doesn't exist in our headless+CUDA setup.
    ]
    runpy.run_path(str(generate), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
