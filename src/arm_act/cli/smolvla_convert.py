"""CLI: convert raw oracle demos for a task into LeRobotDataset format.

Reads the same task YAML as the rest of the pipeline so the dataset
inherits the canonical raw_path and instruction without per-task
redundant config. Output goes under ``data/lerobot/<repo_id>/`` so it
sits next to ``data/raw/`` and ``data/annotated/`` and is gitignored
by the existing ``/data/`` rule.

Usage:

    ./scripts/smolvla_convert.sh --task pick_vial_from_holder
    ./scripts/smolvla_convert.sh --task pick_vial_from_holder --overwrite
    ./scripts/smolvla_convert.sh --task pick_vial_from_holder \\
        --repo-id local/pick_vial_v2 --fps 20

After conversion, fine-tune SmolVLA via ``./scripts/smolvla_train.sh``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from arm_act.config import DEFAULT_TASK, load
from arm_act.training.dataset_lerobot import convert_hdf5_to_lerobot


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/lerobot"),
        help="root dir under which the LeRobotDataset will be created.",
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot dataset identifier; defaults to local/<task>.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=20,
        help="control frequency. Default 20 = Isaac Lab sim.dt=0.01 / decimation=5.",
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="obs.* keys to include as image tracks. Default: table_cam wrist_cam.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace any existing dataset at the target path.",
    )
    args = p.parse_args()

    spec = load(args.task)
    hdf5_path = Path(spec["data"]["raw_path"])
    if not hdf5_path.is_absolute():
        # Same convention as the rest of the pipeline: paths are relative
        # to the repo root, so resolve from CWD when invoked via the bash
        # wrapper which cd's to REPO_ROOT.
        hdf5_path = hdf5_path.resolve()
    if not hdf5_path.exists():
        print(f"raw demos file not found: {hdf5_path}", file=sys.stderr)
        return 1

    instruction = spec["task"].get("instruction")
    if not instruction:
        print(
            f"task {args.task!r} has no task.instruction; SmolVLA needs a "
            "language string. Add one to the task YAML and re-run.",
            file=sys.stderr,
        )
        return 1

    repo_id = args.repo_id or f"local/{args.task}"

    output_dir = convert_hdf5_to_lerobot(
        hdf5_path=hdf5_path,
        output_root=args.output_root,
        repo_id=repo_id,
        instruction=instruction,
        fps=args.fps,
        cameras=args.cameras,
        overwrite=args.overwrite,
    )
    print(f"wrote LeRobotDataset to {output_dir}")
    print(
        f"\nNext step:\n"
        f"  ./scripts/smolvla_train.sh --task {args.task}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
