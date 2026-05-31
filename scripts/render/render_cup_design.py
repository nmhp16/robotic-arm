"""Design check for the PROPER vial->vial: real walled well + cup/cage fingertips
+ larger stem. Validates the showstoppers CHEAPLY (matplotlib, seconds) before any
converter/retrain (CLAUDE.md #9/#15):

  SIDE (XZ): can the SCARA lift the plant base CLEAR of the well rim within its
             ~70mm lift ceiling? (the reach question)
  TOP  (XY): does the OPEN gripper fit inside the well bore (reach-in clearance)?
             does the CLOSED cage wrap the stem >180deg (form-closure)?
All mm.
"""
from __future__ import annotations
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge

# --- vial well ---
WELL_H = 70.0           # 7cm tall (user spec)
BORE_R = 14.0           # inner radius (28mm bore) — wide for reach-in clearance
WALL_T = 2.0            # wall thickness -> 32mm OD
# --- plant ---
STEM_R = 4.0            # 8mm stem (enlarged from 6mm: easier to grip + see)
PLANT_BASE_Z = 0.0
PLANT_TOP_Z = 76.0      # sticks ~6mm above the rim so it's graspable near the rim
GRASP_Z = 64.0          # grip just below the rim (minimal reach-in)
# --- arm reach ---
LIFT_CEIL = 70.0        # oracle lift_height ceiling ~0.07m (SCARA long-tip max up)
# --- gripper ---
FINGER_T = 3.0          # finger blade thickness
OPEN_GAP = 14.0         # inner-face gap when open
CAGE_WRAP_DEG = 220.0   # target wrap of the closed cage around the stem

fig, (axs, axt) = plt.subplots(1, 2, figsize=(13, 7))

# ---------- SIDE (XZ): reach check ----------
axs.add_patch(Rectangle((-BORE_R - WALL_T, 0), WALL_T, WELL_H, fc="lightsteelblue", ec="navy"))
axs.add_patch(Rectangle((BORE_R, 0), WALL_T, WELL_H, fc="lightsteelblue", ec="navy"))
axs.add_patch(Rectangle((-BORE_R - WALL_T, 0), 2 * (BORE_R + WALL_T), 4, fc="lightsteelblue", ec="navy"))  # floor
axs.add_patch(Rectangle((-STEM_R, PLANT_BASE_Z, ), 2 * STEM_R, PLANT_TOP_Z, fc="green", alpha=0.4, label="stem (in source well)"))
axs.axhline(WELL_H, color="navy", ls="--", lw=1, label=f"rim z={WELL_H:.0f}")
axs.axhline(GRASP_Z, color="orange", ls=":", lw=1, label=f"grasp z={GRASP_Z:.0f}")
base_after = PLANT_BASE_Z + LIFT_CEIL
axs.axhline(base_after, color="red", ls=":", lw=1, label=f"plant base after max lift = {base_after:.0f}")
axs.set_title(f"SIDE: lift base {PLANT_BASE_Z:.0f} -> {base_after:.0f} (ceil {LIFT_CEIL:.0f}); "
              f"rim {WELL_H:.0f}\n{'CLEARS rim ✓' if base_after >= WELL_H else 'does NOT clear rim ✗ (vial too tall for reach)'}")
axs.set_xlim(-30, 30); axs.set_ylim(-5, 145); axs.set_xlabel("X (mm)"); axs.set_ylabel("Z (mm)")
axs.legend(fontsize=7, loc="upper right")

# ---------- TOP (XY): bore clearance + cage wrap ----------
axt.add_patch(Circle((0, 0), BORE_R + WALL_T, fc="none", ec="navy", lw=1.5))
axt.add_patch(Circle((0, 0), BORE_R, fc="aliceblue", ec="navy", lw=1, label=f"bore r={BORE_R:.0f}"))
axt.add_patch(Circle((0, 0), STEM_R, fc="green", alpha=0.5, label=f"stem r={STEM_R:.0f}"))
# open gripper: two blades at +-(OPEN_GAP/2 + FINGER_T/2), span check vs bore
half_open = OPEN_GAP / 2 + FINGER_T
axt.add_patch(Rectangle((-half_open - 1, -6), 2, 12, fc="gray", alpha=0.5))
axt.add_patch(Rectangle((half_open - 1, -6), 2, 12, fc="gray", alpha=0.5, label=f"open finger outer +-{half_open:.0f}"))
# closed cage: two wedges wrapping the stem from +-X
half_wrap = CAGE_WRAP_DEG / 2
axt.add_patch(Wedge((0, 0), STEM_R + 1.5, 180 - half_wrap, 180 + half_wrap, width=1.5, fc="darkorange", alpha=0.8))
axt.add_patch(Wedge((0, 0), STEM_R + 1.5, -half_wrap, half_wrap, width=1.5, fc="darkorange", alpha=0.8, label=f"closed cage wrap {CAGE_WRAP_DEG:.0f}°"))
axt.set_title(f"TOP: open finger +-{half_open:.0f} vs bore {BORE_R:.0f} -> "
              f"{'FITS ✓' if half_open < BORE_R else 'HITS WALL ✗'}; cage wrap {CAGE_WRAP_DEG:.0f}° "
              f"{'>180 caged ✓' if CAGE_WRAP_DEG > 180 else '<180 ✗'}")
axt.set_xlim(-22, 22); axt.set_ylim(-22, 22); axt.set_aspect("equal"); axt.set_xlabel("X (mm)"); axt.set_ylabel("Y (mm)")
axt.legend(fontsize=7, loc="upper right")

plt.tight_layout(); plt.savefig("/tmp/cup_design.png", dpi=110)
print("wrote /tmp/cup_design.png")
print(f"REACH: base {PLANT_BASE_Z:.0f} -> {base_after:.0f} mm at max lift; rim {WELL_H:.0f} -> "
      f"{'CLEARS' if base_after >= WELL_H else 'FAILS (need shorter well or taller grasp)'}")
print(f"BORE:  open finger outer +-{half_open:.0f} vs bore r {BORE_R:.0f} -> {'FITS' if half_open < BORE_R else 'HITS WALL'}")
print(f"CAGE:  wrap {CAGE_WRAP_DEG:.0f}° -> {'form-closure (caged)' if CAGE_WRAP_DEG > 180 else 'not caged'}")
