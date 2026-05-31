"""Validate the V-groove fingertip geometry BEFORE editing the URDF
(CLAUDE.md #9/#15). Draws the EXACT angled-box params that will go into the
URDF, so the render matches the asset (the prior attempt failed because the
mock != URDF and the groove ended up past the fingertip).

Finger (link frame, metres): blade box 0.003(X) x 0.005(Y) x 0.050(Z),
inner face at X=0, tip z in [0.010,0.060]. Fingers close along X.
V-groove = two 45deg plates per finger protruding from the inner face toward
the stem, forming a V that funnels an off-centre 3mm stem to the centreline.

Outputs /tmp/vgroove_design.png:
  (left)  XY @ tip, fingers near-closed -> does the V cradle + centre the stem?
  (right) XZ side -> V at the tip (z<=0.060), not floating.
"""
from __future__ import annotations
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

BLADE_X, BLADE_Y, BLADE_Z = 0.003, 0.005, 0.050
TIP_Z = 0.060
STEM_R = 0.0015
GAP = 0.004                      # near-closed gripper gap (inner faces +-0.002)
# V-wall plate params (per finger, link frame)
W_LEN, W_THICK, W_HZ = 0.008, 0.0012, 0.025
W_CZ = 0.0475                    # wall centre z (tip zone 0.035..0.060)
W_OFF_X, W_OFF_Y = 0.0018, 0.0018
W_ANG = np.radians(45)

def rot_rect(cx, cy, lx, ly, ang):
    c, s = np.cos(ang), np.sin(ang)
    corners = np.array([[-lx/2,-ly/2],[lx/2,-ly/2],[lx/2,ly/2],[-lx/2,ly/2]])
    R = np.array([[c,-s],[s,c]])
    return (corners @ R.T) + np.array([cx, cy])

fig, (axc, axs) = plt.subplots(1, 2, figsize=(13, 6))

# ---- XY @ tip, near-closed ----
axc.set_title("XY @ tip (gripper near-closed)\nV-walls funnel + cradle the stem?")
half = GAP/2
for sign, sgn_ang in ((-1, +1), (+1, -1)):   # left: +ang, right: -ang
    inner = sign*half
    # blade (faint)
    axc.add_patch(Rectangle((min(inner, inner-sign*BLADE_X), -BLADE_Y/2), BLADE_X, BLADE_Y,
                            fill=False, ec="0.7", ls=":", lw=1))
    # two V-wall plates protruding toward stem (+X for left)
    for ysign in (+1, -1):
        cx = inner + sign*W_OFF_X
        cy = ysign*W_OFF_Y
        ang = sgn_ang*ysign*W_ANG
        axc.add_patch(Polygon(rot_rect(cx, cy, W_LEN, W_THICK, ang),
                              closed=True, fc="steelblue", ec="b", alpha=0.7))
axc.add_patch(Circle((0,0), STEM_R, fc="green", ec="darkgreen", alpha=0.65, label="3mm stem"))
axc.axhline(0, color="0.85", lw=.5); axc.axvline(0, color="0.85", lw=.5)
axc.set_xlabel("X close axis (m)"); axc.set_ylabel("Y depth (m)")
axc.set_xlim(-0.008,0.008); axc.set_ylim(-0.008,0.008); axc.set_aspect("equal"); axc.legend(fontsize=8)

# ---- XZ side ----
axs.set_title("XZ side: V at tip (z<=0.060)?")
for sign in (-1,+1):
    inner=sign*half
    axs.add_patch(Rectangle((min(inner,inner-sign*BLADE_X),0.010), BLADE_X, BLADE_Z, fc="0.85", ec="0.5"))
    axs.add_patch(Rectangle((min(inner+sign*W_OFF_X-0.004, inner+sign*W_OFF_X+0.004), W_CZ-W_HZ/2),
                            0.008, W_HZ, fc="steelblue", alpha=0.5,
                            label="V-wall zone" if sign<0 else None))
axs.add_patch(Rectangle((-STEM_R,0.005),2*STEM_R,0.075, fc="green", alpha=0.35, label="stem"))
axs.axhline(TIP_Z, color="k", ls="--", lw=1, label=f"tip z={TIP_Z}")
axs.axhline(0.104, color="red", ls=":", lw=1.2, label="OLD BUG z=0.104")
axs.set_xlabel("X (m)"); axs.set_ylabel("Z (m)"); axs.set_xlim(-0.012,0.012); axs.set_ylim(0,0.12); axs.legend(fontsize=8)

plt.tight_layout(); plt.savefig("/tmp/vgroove_design.png", dpi=110)
print("wrote /tmp/vgroove_design.png")
# quick numeric sanity
vtip_x = half - W_OFF_X - (W_LEN/2)*np.cos(W_ANG)   # innermost reach of wall toward centre
print(f"near-closed gap={GAP*1000:.1f}mm, stem dia={2*STEM_R*1000:.1f}mm, V-wall z-centre={W_CZ} (tip {TIP_Z})")
print(f"wall plates rotated +-45deg, protrude ~{(W_OFF_X)*1000:.1f}mm past inner face toward stem")
