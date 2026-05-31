#!/usr/bin/env python3
"""Tip-vs-plant friction isolation: is the high-friction grip from the engineerable
fingertip PADS (deployable) or from unrealistic PLANT-surface friction (a sim freebie)?

Sets per-surface friction (MU_TIP on the fingers, MU_PLANT on the stem) and reports
the vial->vial success rate. Usage: newton_isolation.py TIP PLANT OTHER N
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import newton_urdf_test as T

tip   = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
plant = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
other = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
N     = int(sys.argv[4]) if len(sys.argv) > 4 else 12

T.MU_TIP, T.MU_PLANT, T.MU_OTHER = tip, plant, other
print(f"=== ISOLATION: tip_mu={tip} plant_mu={plant} other_mu={other}  N={N} ===", flush=True)
succ = 0
for i in range(1, N + 1):
    ok, dist, _ = T.main(record=False, quiet=True)
    succ += int(ok)
    print(f"trial {i:2d}: {'SUCCESS' if ok else 'FAIL   '}  dist={dist:7.1f}mm  "
          f"running {succ}/{i} = {100*succ/i:.0f}%", flush=True)
print(f">>> tip={tip}/plant={plant}: {succ}/{N} = {100*succ/N:.1f}%", flush=True)
