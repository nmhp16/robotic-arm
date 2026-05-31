"""V-groove fingertip + round stem, in-vial Zimmer geometry (CLAUDE.md #9/#15).

The flat-jaw friction grip is non-deterministic (29% oracle). A V-notch in each
fingertip CRADLES a round stem (form-closure: 4 tangent lines, captured in X+Y)
so the lift no longer relies on marginal friction. Round stem (not box) because
a cylinder self-centres in a V (flat jaws roll a cylinder out — a V traps it).

This validates, in the XY plane at the grasp height:
  - the V-faces are tangent to the 6mm stem when the jaws close (true cradle)
  - the V mouth <= the 5mm finger width
  - the closed gripper still fits the 20mm vial bore
and in XZ that the notch sits at the fingertip (reaches into the vial). All mm.
"""
from __future__ import annotations
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- params (mm) ---
STEM_R       = 3.0     # 6mm round stem (switched from box to nest in the V)
FINGER_THICK = 3.0     # Zimmer finger X-thickness (unchanged spec)
FINGER_W     = 5.0     # finger Y-width
OPEN_INNER   = 5.0     # inner face at +-5 when open (10mm gap)
VIAL_BORE_R  = 10.0
V_HALF_DEG   = 45.0    # V half-angle from the closing (X) axis -> 90deg included
V_DEPTH      = 2.5     # notch depth into the 3mm finger (0.5mm solid behind apex)

a = np.radians(V_HALF_DEG)
# Cylinder radius R cradled in a V (half-angle a): centre sits R/sin(a) from apex
# along the bisector. So when the stem centre is at X=0, the +X finger apex is:
APEX_X  = STEM_R / np.sin(a)          # 4.243 for R=3, a=45
MOUTH_X = APEX_X - V_DEPTH            # inner-face (mouth) X when gripping
MOUTH_HW = V_DEPTH * np.tan(a)        # mouth half-width
OUTER_X = MOUTH_X + FINGER_THICK      # finger outer face (closed)

fig, (axt, axs) = plt.subplots(1, 2, figsize=(13, 6.5))

# ===== TOP (XY) at grasp height: the capture validation =====
axt.set_title("TOP (XY @ grasp): round stem cradled in the V-notches\n(form-closure: 4 tangent lines, captured in X+Y)")
axt.add_patch(Circle((0, 0), VIAL_BORE_R, fill=False, ec="0.6", ls="--", label="vial bore 20mm"))
axt.add_patch(Circle((0, 0), STEM_R, fc="limegreen", ec="darkgreen", label="round stem 6mm"))

def vfinger(sgn):
    # V-notched finger on the +/-X side, gripping position.
    mx, ax_, ox = sgn*MOUTH_X, sgn*APEX_X, sgn*OUTER_X
    pts = [(mx, FINGER_W/2), (ox, FINGER_W/2), (ox, -FINGER_W/2),
           (mx, -FINGER_W/2), (ax_, 0)]   # box minus the V-notch (apex points inward)
    return plt.Polygon(pts, closed=True, fc="steelblue", ec="navy", alpha=0.7)

for sgn in (-1, 1):
    axt.add_patch(vfinger(sgn))
# mark the 4 tangent (contact) points: foot of perpendicular from origin to each face
for sgn in (-1, 1):
    for sy in (-1, 1):
        # face from mouth-corner (sgn*MOUTH_X, sy*MOUTH_HW... use FINGER_W/2) to apex (sgn*APEX_X,0)
        p1 = np.array([sgn*MOUTH_X, sy*FINGER_W/2]); p2 = np.array([sgn*APEX_X, 0.0])
        d = p2 - p1; d = d/np.linalg.norm(d)
        t = -np.dot(p1, d); foot = p1 + t*d
        axt.plot(*foot, "rx", ms=9, mew=2)
axt.plot([], [], "rx", label="contact (tangent) point")
# open finger inner faces for reference
for sgn in (-1, 1):
    axt.axvline(sgn*OPEN_INNER, color="0.8", ls=":", lw=1)
axt.text(OPEN_INNER, -9, "open\ninner", fontsize=7, ha="center", color="0.5")
axt.set_xlim(-13, 13); axt.set_ylim(-13, 13); axt.set_aspect("equal")
axt.set_xlabel("X (mm) — jaw close"); axt.set_ylabel("Y (mm)"); axt.legend(fontsize=7, loc="upper right")

# ===== SIDE (XZ): notch at the fingertip, reaching into the vial =====
axs.set_title("SIDE (XZ): V-notch at the fingertip (reaches into vial)")
VIAL_H, FINGER_LEN, GRASP_Z = 70.0, 80.0, 35.0
for s in (-1, 1):  # vial walls
    x0 = (10 if s > 0 else -15)
    axs.add_patch(plt.Rectangle((x0, 0), 5, VIAL_H, fc="0.85", ec="0.5"))
axs.add_patch(plt.Rectangle((-STEM_R, 0), 2*STEM_R, 58, fc="green", alpha=0.4, label="plant (stem+leaves)"))
# finger as a long bar from grasp_z up, with a notch hint at the tip
for sgn in (-1, 1):
    axs.add_patch(plt.Rectangle((sgn*MOUTH_X if sgn>0 else sgn*OUTER_X, GRASP_Z), FINGER_THICK, FINGER_LEN,
                                fc="steelblue", alpha=0.6, label="finger (80mm)" if sgn<0 else None))
axs.axhline(GRASP_Z, color="g", ls=":", label=f"grasp/notch z={GRASP_Z:.0f}")
axs.axhline(VIAL_H, color="r", ls="--", label=f"vial rim z={VIAL_H:.0f}")
axs.set_xlim(-16, 16); axs.set_ylim(-5, 130); axs.set_xlabel("X (mm)"); axs.set_ylabel("Z (mm)"); axs.legend(fontsize=7)

plt.tight_layout(); plt.savefig("/tmp/vgroove_invial.png", dpi=110)
print("wrote /tmp/vgroove_invial.png")

# numeric checks
# distance from origin to a V-face (should equal STEM_R => tangent/cradle)
p1 = np.array([MOUTH_X, FINGER_W/2]); p2 = np.array([APEX_X, 0.0])
nrm = np.array([p2[1]-p1[1], -(p2[0]-p1[0])]); nrm = nrm/np.linalg.norm(nrm)
dist = abs(np.dot(p1, nrm))
print(f"V: half-angle={V_HALF_DEG:.0f}deg depth={V_DEPTH}mm  apex_X={APEX_X:.2f} mouth_X={MOUTH_X:.2f} mouth_HW={MOUTH_HW:.2f}")
print(f"face-to-stem-centre dist={dist:.2f}mm vs stem R={STEM_R:.2f} -> {'TANGENT (cradles)' if abs(dist-STEM_R)<0.05 else 'NOT tangent — adjust'}")
print(f"mouth half-width {MOUTH_HW:.2f} vs finger half-width {FINGER_W/2:.2f} -> {'OK fits finger' if MOUTH_HW<=FINGER_W/2+1e-6 else 'TOO WIDE'}")
print(f"closed gripper outer span = +-{OUTER_X:.2f} ({2*OUTER_X:.1f}mm) vs bore 20mm -> {'FITS' if OUTER_X<VIAL_BORE_R else 'TOO WIDE for bore'} ({VIAL_BORE_R-OUTER_X:.1f}mm/side)")
print(f"open->close travel per side = {OPEN_INNER-MOUTH_X:.2f}mm (stroke ok if < ~5mm)")
