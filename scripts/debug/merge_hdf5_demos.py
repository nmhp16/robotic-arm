"""Merge two oracle demo HDF5 files into one, renumbering demo keys.

The recorder writes episodes as /data/demo_0, demo_1, ... Each file
starts numbering at 0, so a naive copy would collide. This script
appends src into dst, renumbering src's demos to continue after dst's
highest index.

Usage:
    isaaclab.sh -p scripts/merge_hdf5_demos.py <dst.hdf5> <src.hdf5>
The merged result is written back to <dst.hdf5> (src is left intact).
"""
from __future__ import annotations

import sys

import h5py


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: merge_hdf5_demos.py <dst.hdf5> <src.hdf5>", file=sys.stderr)
        return 2
    dst_path, src_path = sys.argv[1], sys.argv[2]

    with h5py.File(dst_path, "a") as dst, h5py.File(src_path, "r") as src:
        dst_data = dst.require_group("data")
        src_data = src["data"]

        existing = [k for k in dst_data.keys() if k.startswith("demo_")]
        next_idx = max((int(k.split("_")[1]) for k in existing), default=-1) + 1

        # Preserve dataset-level attrs (env_args etc.) only if dst lacks them.
        for attr_k, attr_v in src_data.attrs.items():
            if attr_k not in dst_data.attrs:
                dst_data.attrs[attr_k] = attr_v

        src_demos = sorted(src_data.keys(), key=lambda k: int(k.split("_")[1]))
        copied = 0
        for demo_k in src_demos:
            new_key = f"demo_{next_idx}"
            src.copy(src_data[demo_k], dst_data, name=new_key)
            next_idx += 1
            copied += 1

        total = len([k for k in dst_data.keys() if k.startswith("demo_")])
        print(f"merged {copied} demos from {src_path} -> {dst_path} (total now {total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
