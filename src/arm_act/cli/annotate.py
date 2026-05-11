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
import shutil
import sys
import tempfile

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


def _strip_placeholders(src: pathlib.Path, dst: pathlib.Path) -> tuple[int, int]:
    """Copy episodes with an ``actions`` group from src to dst, dropping
    placeholder demos that only have ``initial_state``. Isaac Lab's
    annotate_demos.py crashes with KeyError on the first such placeholder.

    Returns (kept, dropped).
    """
    import h5py  # local import: keeps import time off the main path

    kept = 0
    dropped = 0
    with h5py.File(src, "r") as f_in, h5py.File(dst, "w") as f_out:
        for attr_name, attr_val in f_in.attrs.items():
            f_out.attrs[attr_name] = attr_val
        out_group = f_out.create_group("data")
        in_group = f_in["data"]
        for attr_name, attr_val in in_group.attrs.items():
            out_group.attrs[attr_name] = attr_val
        for ep_name in in_group:
            if "actions" not in in_group[ep_name]:
                dropped += 1
                continue
            f_in.copy(in_group[ep_name], out_group, name=ep_name)
            kept += 1
    return kept, dropped


def main() -> int:
    args = _parse_args()
    cfg = load(args.task)

    annotate = isaaclab_script("scripts/imitation_learning/isaaclab_mimic/annotate_demos.py")
    # Note: arm_act task gym registration happens INSIDE annotate_demos.py
    # (we patched it to import arm_act.tasks after AppLauncher init).
    # See the 4-line block at the top of that file.

    in_path = pathlib.Path(args.input or cfg["data"]["raw_path"])
    out_path = pathlib.Path(args.output or cfg["data"]["annotated_path"])
    if not in_path.is_file():
        print(
            f"input not found: {in_path} — run ./scripts/teleop.sh or ./scripts/oracle.sh first",
            file=sys.stderr,
        )
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Oracle writes placeholder episodes (only ``initial_state``, no
    # ``actions``) for slots where recording started but didn't complete.
    # annotate_demos.py's ``episode.data["actions"]`` access KeyErrors on
    # the first one, so pre-strip them into a temp HDF5.
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="annotate_input_"))
    cleaned = tmp_dir / "demos_no_placeholders.hdf5"
    kept, dropped = _strip_placeholders(in_path, cleaned)
    print(f"[annotate] stripped placeholders: kept={kept} dropped={dropped} -> {cleaned}", flush=True)

    sys.argv = [
        "annotate_demos.py",
        "--task", cfg["task"]["mimic_gym_id"],
        "--input_file", str(cleaned),
        "--output_file", str(out_path),
        "--auto",
        "--headless",
        "--enable_cameras",
    ]
    try:
        runpy.run_path(str(annotate), run_name="__main__")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
