"""Backfill missing /initial_state groups in an oracle demo HDF5.

The parallel oracle's manual env._reset_idx() bypassed the recorder's
record_post_reset() hook, so episodes after an env's first one lack the
/initial_state group that isaaclab_mimic's annotate step requires.

initial_state is exactly the first frame of /states (both recorded at
the post-reset initial pose), so we reconstruct it as states[0:1] for
every sub-key. Idempotent: demos that already have initial_state are
skipped.

Usage:
    isaaclab.sh -p scripts/backfill_initial_state.py <demos.hdf5>
"""
from __future__ import annotations

import sys

import h5py


def _copy_first_frame(src_grp: h5py.Group, dst_grp: h5py.Group) -> None:
    for k in src_grp.keys():
        child = src_grp[k]
        if isinstance(child, h5py.Dataset):
            dst_grp.create_dataset(k, data=child[0:1])
        else:
            sub = dst_grp.create_group(k)
            _copy_first_frame(child, sub)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: backfill_initial_state.py <demos.hdf5>", file=sys.stderr)
        return 2
    path = sys.argv[1]

    with h5py.File(path, "a") as f:
        data = f["data"]
        fixed = 0
        skipped = 0
        for dk in data.keys():
            g = data[dk]
            if "initial_state" in g:
                skipped += 1
                continue
            if "states" not in g:
                print(f"  [warn] {dk} has no /states — cannot backfill", file=sys.stderr)
                continue
            init = g.create_group("initial_state")
            _copy_first_frame(g["states"], init)
            fixed += 1
        print(f"backfilled initial_state for {fixed} demos ({skipped} already had it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
