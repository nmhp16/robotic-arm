"""Standalone matplotlib preview of the leaf-disk addition to the plant
collision proxy. CLAUDE.md #9: validate geometry BEFORE the full Isaac Sim
rebake, since the rebake needs the DGX Spark.

Renders SIDE (XZ) and TOP (XY @ disk-height) views with:
  - Vial walls + bore at z=0..0.045
  - Plant collision (grasp cylinder + 3 nodes + NEW leaf disk)
  - Zimmer gripper jaws fully open at DESCEND target TCP=0.030

Numeric checks printed at the bottom:
  - Does the disk block the gripper's open-jaw descent path?
  - Is the disk inside the vial bore?
  - At what plant lift does the disk clear the rim?

Everything in millimetres in the rendered figure. No Isaac required.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# --- vial (wide_vial, walled well) ---
VIAL_OUTER_R = 15.0
VIAL_BORE_R = 10.0
VIAL_H = 45.0  # current 45mm walled well in zimmer task

# --- plant collision proxy, UNSCALED URDF mm → with task z-scale 0.5 ---
Z_SCALE = 0.5
# grasp cylinder
GRASP_CYL_Z = 58.0 * Z_SCALE      # 29 mm
GRASP_CYL_LEN = 40.0 * Z_SCALE    # 20 mm
GRASP_CYL_R = 3.0                 # x/y not scaled
# three form-closure nodes
NODE_Z = [50.0 * Z_SCALE, 62.0 * Z_SCALE, 74.0 * Z_SCALE]  # 25, 31, 37 mm
NODE_LEN = 5.0 * Z_SCALE          # 2.5 mm
NODE_R = 4.0
# NEW leaf disk
DISK_Z = 70.0 * Z_SCALE           # 35 mm
DISK_LEN = 3.0 * Z_SCALE          # 1.5 mm
DISK_R = 4.5

# --- visual leaves (no collision in current asset — drawn for reference) ---
VLEAF_Z_LO = 65.0 * Z_SCALE       # 32.5 mm
VLEAF_Z_HI = 75.0 * Z_SCALE       # 37.5 mm
VLEAF_R = 11.0

# --- gripper jaws (Zimmer GEP2010IL custom blades) at DESCEND TCP=30 ---
TCP_Z = 30.0
TCP_Z_OFFSET = 160.0
TOOL0_Z = TCP_Z + TCP_Z_OFFSET            # 190
FINGER_LINK_Z = TOOL0_Z - 80.0            # 110 (finger link origin in world)
BLADE_Z_TOP = FINGER_LINK_Z - 10.0        # 100 (finger-local z=0.010)
BLADE_Z_BOT = FINGER_LINK_Z - 76.0        # 34  (finger-local z=0.076)
JAW_INNER_X = 5.0    # at fully open, inner face at gripper x = ±5
JAW_OUTER_X = 8.0
PRONG_Z = FINGER_LINK_Z - 70.0            # 40  (V-groove prong centre)
PRONG_LEN = 40.0                          # mm

fig, (axs, axt) = plt.subplots(1, 2, figsize=(13, 8))

# ================= SIDE (XZ) =================
axs.set_title("SIDE (XZ): leaf disk vs gripper jaws @ DESCEND target (TCP=30)\n"
              "disk r=4.5mm centred at z=35; jaw inner face at x=±5 → 0.5mm clearance",
              fontsize=10)
# vial walls + bore
for sgn in (-1, 1):
    x0 = VIAL_BORE_R if sgn > 0 else -VIAL_OUTER_R
    axs.add_patch(Rectangle((x0, 0), VIAL_OUTER_R - VIAL_BORE_R, VIAL_H,
                            fc="0.85", ec="0.5",
                            label="vial wall" if sgn < 0 else None))
# grasp cylinder
axs.add_patch(Rectangle((-GRASP_CYL_R, GRASP_CYL_Z - GRASP_CYL_LEN / 2),
                        2 * GRASP_CYL_R, GRASP_CYL_LEN,
                        fc="darkgreen", alpha=0.8, label="grasp cyl (r=3, z=29)"))
# three nodes
for i, nz in enumerate(NODE_Z):
    axs.add_patch(Rectangle((-NODE_R, nz - NODE_LEN / 2), 2 * NODE_R, NODE_LEN,
                            fc="forestgreen", alpha=0.7,
                            label="form-closure nodes (r=4)" if i == 0 else None))
# NEW leaf disk
axs.add_patch(Rectangle((-DISK_R, DISK_Z - DISK_LEN / 2),
                        2 * DISK_R, DISK_LEN,
                        fc="limegreen", ec="darkgreen", lw=1.5,
                        label=f"NEW leaf disk (r={DISK_R}, z={DISK_Z:.0f})"))
# visual leaves (no collision) drawn as outline for reference
axs.add_patch(Rectangle((-VLEAF_R, VLEAF_Z_LO), 2 * VLEAF_R, VLEAF_Z_HI - VLEAF_Z_LO,
                        fill=False, ec="green", ls=":", lw=1,
                        label=f"visual leaves (r={VLEAF_R}, no collision)"))
# gripper jaws (open)
for sgn in (-1, 1):
    x0 = JAW_INNER_X if sgn > 0 else -JAW_OUTER_X
    axs.add_patch(Rectangle((x0, BLADE_Z_BOT),
                            JAW_OUTER_X - JAW_INNER_X, BLADE_Z_TOP - BLADE_Z_BOT,
                            fc="steelblue", alpha=0.55,
                            label="finger blade (open)" if sgn < 0 else None))
# horizontal reference lines
axs.axhline(VIAL_H, color="r", ls="--", lw=1, label=f"vial rim z={VIAL_H:.0f}")
axs.axhline(TCP_Z, color="g", ls=":", lw=1, label=f"TCP z={TCP_Z:.0f}")
axs.axhline(BLADE_Z_BOT, color="b", ls=":", lw=0.5, label=f"jaw bottom z={BLADE_Z_BOT:.0f}")
axs.set_xlim(-20, 20); axs.set_ylim(-5, 130)
axs.set_xlabel("X (mm) — finger close direction")
axs.set_ylabel("Z (mm)")
axs.legend(fontsize=6.5, loc="upper right")
axs.set_aspect("equal")

# ================= TOP (XY @ disk height) =================
axt.set_title(f"TOP (XY @ z={DISK_Z:.0f}): disk inside the vial bore;\n"
              f"gripper jaws (open) clear the disk by {JAW_INNER_X - DISK_R:.1f}mm",
              fontsize=10)
axt.add_patch(Circle((0, 0), VIAL_OUTER_R, fill=False, ec="0.5"))
axt.add_patch(Circle((0, 0), VIAL_BORE_R, fill=False, ec="0.5", ls="--",
                     label=f"bore r={VIAL_BORE_R:.0f}"))
# NEW leaf disk
axt.add_patch(Circle((0, 0), DISK_R, fc="limegreen", alpha=0.7,
                     label=f"NEW disk r={DISK_R}"))
# visual leaves outline
axt.add_patch(Circle((0, 0), VLEAF_R, fill=False, ec="green", ls=":",
                     label=f"visual leaves r={VLEAF_R}"))
# gripper jaws projected at this z
for sgn in (-1, 1):
    x0 = JAW_INNER_X if sgn > 0 else -JAW_OUTER_X
    axt.add_patch(Rectangle((x0, -2.5), JAW_OUTER_X - JAW_INNER_X, 5,
                            fc="steelblue", alpha=0.6,
                            label="finger (open)" if sgn < 0 else None))
axt.set_xlim(-18, 18); axt.set_ylim(-18, 18); axt.set_aspect("equal")
axt.set_xlabel("X (mm)"); axt.set_ylabel("Y (mm)")
axt.legend(fontsize=7)

plt.tight_layout()
out = "/tmp/leaf_disk_check.png"
plt.savefig(out, dpi=110)
print(f"wrote {out}")

# --- numeric checks ---
print()
print("=== geometry checks (mm, world frame, DESCEND TCP=30) ===")
gap_jaw_disk = JAW_INNER_X - DISK_R
print(f"jaw-inner→disk-edge gap (X): {gap_jaw_disk:+.2f} "
      f"[{'OK descent clear' if gap_jaw_disk > 0 else 'DISK BLOCKS JAW DESCENT'}]")
gap_disk_bore = VIAL_BORE_R - DISK_R
print(f"disk-edge→bore-wall gap   : {gap_disk_bore:+.2f} "
      f"[{'fits' if gap_disk_bore > 0 else 'TOO BIG FOR BORE'}, "
      f"plant can drift ±{gap_disk_bore:.1f} before disk hits wall]")
print(f"disk z-range: {DISK_Z - DISK_LEN/2:.2f} .. {DISK_Z + DISK_LEN/2:.2f}  "
      f"(jaw bottom at {BLADE_Z_BOT:.0f}, overlap {max(0, BLADE_Z_BOT - (DISK_Z - DISK_LEN/2)):.2f} mm — but X-clear above)")
print(f"disk vs vial rim: disk_top={DISK_Z + DISK_LEN/2:.2f}, rim={VIAL_H:.0f} → "
      f"disk is {VIAL_H - (DISK_Z + DISK_LEN/2):.1f}mm BELOW the rim at rest")
lift_clear_disk = VIAL_H - (DISK_Z - DISK_LEN/2)
print(f"lift required for disk to clear rim: plant must rise {lift_clear_disk:.1f}mm "
      f"(oracle.lift_height=95mm → plant base rises ~65mm → disk clears with {65 - lift_clear_disk:.0f}mm margin)")
print()
print(f"visual leaves (r={VLEAF_R}) extend BEYOND disk (r={DISK_R}) by "
      f"{VLEAF_R - DISK_R:.1f}mm — visible clipping reduced but not eliminated")
