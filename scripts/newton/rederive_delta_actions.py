"""Rewrite demo actions as eef-frame lookahead deltas (closed-loop-valid).

The collected demos record action = clip(oracle_waypoint - tcp): a residual to
the oracle's YAML-height waypoints. That's fine for offline action-matching, but
the z waypoint is in a DIFFERENT frame than eef_pos (lift_height 0.095 vs actual
tcp lift z 0.150, non-constant offset), so applying it IK-rel (ee_des = ee+action)
drives z to the wrong height and never lifts. xy frames match; only z is broken.

This re-derives action[k] = clip(eef_pos[k+L] - eef_pos[k], +/-[dxy,dxy,dz]) in
the SAME frame as eef_pos (the control frame), keeping the recorded gripper col.
Self-limiting (shrinks at trajectory end) and reproduces the path under
ee_des = ee + action. No re-simulation — reads stored eef_pos only.

  ~/newton-probe/bin/python scripts/newton/rederive_delta_actions.py IN.hdf5 OUT.hdf5 [--lookahead 5]
"""
import argparse, shutil, pathlib, numpy as np, h5py

MAXES = np.array([0.04, 0.04, 0.02], dtype=np.float32)  # max_dxy, max_dxy, max_dz (YAML)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--lookahead", type=int, default=5)
    a = ap.parse_args()
    shutil.copy(a.inp, a.out)
    L = a.lookahead
    with h5py.File(a.out, "r+") as f:
        for did in f["data"].keys():
            g = f["data"][did]
            if "actions" not in g or "obs" not in g:
                continue
            ee = np.asarray(g["obs"]["eef_pos"], dtype=np.float32)   # (T,3) control frame
            grip = np.asarray(g["actions"], dtype=np.float32)[:, 3:4]  # keep recorded gripper
            T = ee.shape[0]
            tgt = ee[np.minimum(np.arange(T) + L, T - 1)]            # lookahead position
            delta = np.clip(tgt - ee, -MAXES, MAXES)
            new = np.concatenate([delta, grip], axis=1).astype(np.float32)
            g["actions"][...] = new
            if "actions" in g["obs"]:                                # obs/actions = last action
                oa = np.zeros_like(new); oa[1:] = new[:-1]
                g["obs"]["actions"][...] = oa
    print(f"rewrote actions as eef-frame lookahead-{L} deltas -> {a.out}")


if __name__ == "__main__":
    main()
