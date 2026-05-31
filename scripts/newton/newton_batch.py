#!/usr/bin/env python3
"""Measure Newton scripted-oracle vial->vial success rate over N trials."""
import sys, pathlib, statistics as st
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import newton_urdf_test as T

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
succ = 0
dists = []
for i in range(1, N + 1):
    ok, dist, _ = T.main(record=False, quiet=True)
    dists.append(dist)
    succ += int(ok)
    print(f"trial {i:2d}: {'SUCCESS' if ok else 'FAIL   '}  dist={dist:7.1f}mm   "
          f"running {succ}/{i} = {100*succ/i:.0f}%", flush=True)
print(f"\n>>> FULL vial->vial SUCCESS: {succ}/{N} = {100*succ/N:.1f}%", flush=True)
print(f">>> median dist {st.median(dists):.0f}mm | min {min(dists):.1f}mm | "
      f"<60mm (near-dest) {sum(d<60 for d in dists)}/{N}", flush=True)
