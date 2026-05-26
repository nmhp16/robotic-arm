"""Render the COMPLIANT (jointed) plant design before building the URDF
(CLAUDE.md #9/#15 — reliable matplotlib, not the Isaac scene-dump).

The rigid plant gets knocked away like a domino when a finger touches it
off-centre (the ~38% oracle ceiling). Real plants don't: the stem BENDS and
springs back. We model that with a 2-link articulation:

    base (root)  --[revolute Y, spring]--  stem_top
    z 0..22mm                              nub z22..44 + leaf mesh

KEY: same root-frame convention as the current rigid plant (root/base at
z=0, graspable region at z~35mm) so the oracle heights (hover 0.05,
grasp_z_offset 0.035, lift 0.07) transfer unchanged. Only the asset +
spring tuning are new.

This render answers:
  - at REST, is the nub at the grasp height (~35mm) and inside the bore?
  - when BENT by the closing finger, does the nub stay graspable and do the
    leaves stay within the 20mm bore (not poke through the wall)?
  - is the bend joint low enough that the plant gives compliantly instead of
    toppling as a rigid body?
All mm.
"""
from __future__ import annotations
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow

# --- vial well (unchanged fixture) ---
VIAL_OUTER_R = 15.0
VIAL_BORE_R  = 10.0
VIAL_H       = 70.0

# --- compliant plant links (final dims, link-local mm) ---
BASE_R   = 3.0          # rooted lower stem (cylinder)
BASE_TOP = 22.0         # base spans z 0..22
JOINT_Z  = 22.0         # revolute-Y bend joint here (axis perp to finger close = X)
NUB_W    = 6.0          # graspable box nub X/Y width
NUB_TOP  = 44.0         # nub spans z 22..44 (in world, when straight)
LEAF_BASE = 12.0        # leaf-mesh silhouette base (overlaps stub, hides joint seam)
LEAF_TOP  = 60.0        # leaf-mesh top (~48mm tall mesh = 80mm * 0.6 z-squash)
LEAF_HALFW = 9.0        # leaves spread ~17-20mm wide (mesh bounds) -> ~9mm half

GRASP_Z  = 35.0         # oracle grasp target = root_z(0) + grasp_z_offset(0.035)

# --- gripper (Zimmer, long tip) ---
FINGER_THICK = 3.0
FINGER_OPEN_INNER = 5.0
FINGER_LEN = 80.0
BEND_DEG = 14.0         # illustrate the stem bending under an off-centre close


def rot_about_joint(x, z, deg):
    """Rotate point (x,z) about the joint (0, JOINT_Z) by deg (about Y -> in XZ)."""
    t = np.deg2rad(deg)
    dx, dz = x - 0.0, z - JOINT_Z
    rx = dx * np.cos(t) + dz * np.sin(t)
    rz = -dx * np.sin(t) + dz * np.cos(t)
    return rx, JOINT_Z + rz


fig, (axs, axt) = plt.subplots(1, 2, figsize=(13, 7))

# ===================== SIDE (XZ) =====================
axs.set_title("SIDE (XZ): compliant 2-link plant — REST (solid) vs BENT (faded)\n"
              "stem flexes about the low joint instead of being knocked away")
# vial walls
for sgn in (-1, 1):
    x0 = VIAL_BORE_R if sgn > 0 else -VIAL_OUTER_R
    axs.add_patch(Rectangle((x0, 0), VIAL_OUTER_R - VIAL_BORE_R, VIAL_H,
                            fc="0.85", ec="0.5", label="vial wall" if sgn < 0 else None))

# --- REST pose (straight) ---
# base stub (root, rooted in well)
axs.add_patch(Rectangle((-BASE_R, 0), 2 * BASE_R, BASE_TOP, fc="darkgreen", alpha=0.8,
                        label="base/root link (z0-22)"))
# leaf silhouette (on stem_top) — triangle-ish blob
leaf = plt.Polygon([(-LEAF_HALFW, LEAF_TOP), (LEAF_HALFW, LEAF_TOP),
                    (NUB_W / 2, LEAF_BASE), (-NUB_W / 2, LEAF_BASE)],
                   closed=True, fc="forestgreen", alpha=0.45, label="leaf mesh (stem_top)")
axs.add_patch(leaf)
# graspable nub (on stem_top)
axs.add_patch(Rectangle((-NUB_W / 2, JOINT_Z), NUB_W, NUB_TOP - JOINT_Z,
                        fc="limegreen", ec="darkgreen", alpha=0.9, label="graspable nub (z22-44)"))
# joint marker
axs.plot(0, JOINT_Z, "o", color="red", ms=9, label="bend joint (revolute-Y, spring)")

# --- BENT pose (faded) ---
def bent_rect(x0, z0, w, h, deg, **kw):
    corners = [(x0, z0), (x0 + w, z0), (x0 + w, z0 + h), (x0, z0 + h)]
    rc = [rot_about_joint(x, z, deg) for x, z in corners]
    axs.add_patch(plt.Polygon(rc, closed=True, **kw))

bent_rect(-NUB_W / 2, JOINT_Z, NUB_W, NUB_TOP - JOINT_Z, BEND_DEG,
          fc="limegreen", alpha=0.3)
lb = [rot_about_joint(x, z, BEND_DEG) for x, z in
      [(-LEAF_HALFW, LEAF_TOP), (LEAF_HALFW, LEAF_TOP), (NUB_W / 2, LEAF_BASE), (-NUB_W / 2, LEAF_BASE)]]
axs.add_patch(plt.Polygon(lb, closed=True, fc="forestgreen", alpha=0.18))

# gripper fingers (open) descending; the +X finger is what pushes the stem -> bend
for sgn in (-1, 1):
    inner = sgn * FINGER_OPEN_INNER
    axs.add_patch(Rectangle((min(inner, inner + sgn * FINGER_THICK), GRASP_Z),
                            FINGER_THICK, FINGER_LEN, fc="steelblue", alpha=0.55,
                            label="finger (80mm)" if sgn < 0 else None))
axs.add_patch(Rectangle((-12, GRASP_Z + FINGER_LEN), 24, 15, fc="dimgray", alpha=0.7, label="gripper body"))

axs.axhline(VIAL_H, color="r", ls="--", lw=1, label=f"vial rim z={VIAL_H:.0f}")
axs.axhline(GRASP_Z, color="g", ls=":", lw=1, label=f"grasp z={GRASP_Z:.0f}")
axs.set_xlim(-20, 20); axs.set_ylim(-5, 150)
axs.set_xlabel("X (mm) — finger close direction"); axs.set_ylabel("Z (mm)")
axs.legend(fontsize=6.5, loc="upper right")

# ===================== TOP (XY @ grasp height) =====================
axt.set_title("TOP (XY @ grasp z=35): nub centred in bore;\nbent nub stays inside the 20mm bore")
axt.add_patch(Circle((0, 0), VIAL_OUTER_R, fill=False, ec="0.5"))
axt.add_patch(Circle((0, 0), VIAL_BORE_R, fill=False, ec="0.5", ls="--", label="bore 20mm"))
# nub at rest (centred)
axt.add_patch(Rectangle((-NUB_W / 2, -NUB_W / 2), NUB_W, NUB_W, fc="limegreen", alpha=0.7, label="nub @rest"))
# nub when bent: its centre shifts in +X by the bend
nub_cx_bent, _ = rot_about_joint(0, (JOINT_Z + NUB_TOP) / 2, BEND_DEG)
axt.add_patch(Rectangle((nub_cx_bent - NUB_W / 2, -NUB_W / 2), NUB_W, NUB_W,
                        fc="orange", alpha=0.5, label=f"nub @bent {BEND_DEG:.0f}deg"))
for sgn in (-1, 1):
    inner = sgn * FINGER_OPEN_INNER
    axt.add_patch(Rectangle((min(inner, inner + sgn * FINGER_THICK), -2.5), FINGER_THICK, 5,
                            fc="steelblue", alpha=0.6, label="finger (open)" if sgn < 0 else None))
axt.set_xlim(-18, 18); axt.set_ylim(-18, 18); axt.set_aspect("equal")
axt.set_xlabel("X (mm)"); axt.set_ylabel("Y (mm)"); axt.legend(fontsize=7)

plt.tight_layout(); plt.savefig("/tmp/compliant_plant.png", dpi=110)
print("wrote /tmp/compliant_plant.png")

# --- numeric checks ---
nub_cx_bent, nub_cz_bent = rot_about_joint(0, (JOINT_Z + NUB_TOP) / 2, BEND_DEG)
leaf_tip_x, _ = rot_about_joint(LEAF_HALFW, LEAF_TOP, BEND_DEG)
print(f"REST: nub centre z={(JOINT_Z+NUB_TOP)/2:.0f}mm, grasp target z={GRASP_Z:.0f} -> "
      f"{'INSIDE nub' if JOINT_Z<=GRASP_Z<=NUB_TOP else 'OUTSIDE nub!'}")
print(f"BENT {BEND_DEG:.0f}deg: nub centre shifts to x={nub_cx_bent:.1f}mm "
      f"(finger inner face at x={FINGER_OPEN_INNER:.1f}) -> "
      f"{'still graspable' if abs(nub_cx_bent)<FINGER_OPEN_INNER else 'nub past finger - would slip'}")
print(f"BENT leaf tip x={leaf_tip_x:.1f}mm vs bore {VIAL_BORE_R:.0f}mm -> "
      f"{'leaves rub wall (helps centre)' if abs(leaf_tip_x)>=VIAL_BORE_R else 'leaves clear of wall'}")
print(f"base/root frame at z=0 (sits on well floor); pickable_pos[2]~0 -> oracle grasp_z_offset 0.035 lands at z=35 = nub. heights UNCHANGED.")
