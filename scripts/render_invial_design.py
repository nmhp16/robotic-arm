"""Render the in-vial grasp geometry (side + top cross-sections) to validate
clearances BEFORE any Isaac/oracle cycle (CLAUDE.md #9/#15, reliable matplotlib).

Real task: a short plant standing INSIDE an empty vial (plant < vial height),
the long gripper tip reaching DOWN into the vial bore to grasp it and lift out.

Key clearance questions this answers:
  - does the 16mm-span gripper FIT in the vial bore? (need bore > gripper span)
  - do the 80mm fingers REACH the recessed plant while the body clears the rim?
  - is the plant recessed (top below the rim)?
  - how much can the plant be off-centre before the gripper hits the bore wall?
All in mm.
"""
from __future__ import annotations
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# --- geometry (mm) ---
VIAL_OUTER_R = 15.0
VIAL_BORE_R  = 10.0          # 20mm bore (collision well we will build)
VIAL_H       = 70.0
PLANT_R      = 3.0           # 6mm stem
PLANT_H      = 50.0          # < VIAL_H -> recessed
FINGER_THICK = 3.0           # current Zimmer finger thickness (X)
FINGER_OPEN_INNER = 5.0      # inner face 5mm from centre at open
FINGER_LEN   = 80.0          # lengthened tip
GRASP_Z      = 40.0          # grasp near plant top (recessed, inside vial)
BODY_BOT_Z   = GRASP_Z + FINGER_LEN   # gripper body bottom = tips + finger length

gripper_outer = FINGER_OPEN_INNER + FINGER_THICK   # 8mm -> 16mm span
gripper_reach = VIAL_BORE_R - gripper_outer        # how far gripper centre can shift in bore
plant_play    = VIAL_BORE_R - PLANT_R              # how far plant can sit off-centre

fig,(axs,axt)=plt.subplots(1,2,figsize=(13,7))

# ===== SIDE (XZ) =====
axs.set_title("SIDE (XZ): long tip reaching into vial\n(does it reach the recessed plant, body clear of rim?)")
# vial walls
for sgn in (-1,1):
    axs.add_patch(Rectangle((sgn*VIAL_BORE_R if sgn>0 else -VIAL_OUTER_R,0),
                            VIAL_OUTER_R-VIAL_BORE_R,VIAL_H, fc="0.8",ec="0.5",label="vial wall" if sgn<0 else None))
# plant (recessed, standing at vial floor)
axs.add_patch(Rectangle((-PLANT_R,0),2*PLANT_R,PLANT_H, fc="green",alpha=0.5,label=f"plant ({PLANT_H:.0f}mm < vial {VIAL_H:.0f})"))
# gripper fingers (open) descending to grasp
for sgn in (-1,1):
    inner=sgn*FINGER_OPEN_INNER
    axs.add_patch(Rectangle((min(inner,inner+sgn*FINGER_THICK),GRASP_Z),FINGER_THICK,FINGER_LEN,
                            fc="steelblue",alpha=0.6,label="finger (80mm)" if sgn<0 else None))
# gripper body
axs.add_patch(Rectangle((-12,BODY_BOT_Z),24,15,fc="dimgray",alpha=0.7,label="gripper body"))
axs.axhline(VIAL_H,color="r",ls="--",lw=1,label=f"vial rim z={VIAL_H:.0f}")
axs.axhline(GRASP_Z,color="g",ls=":",lw=1,label=f"grasp z={GRASP_Z:.0f}")
axs.set_xlim(-20,20); axs.set_ylim(-5,150); axs.set_xlabel("X (mm)"); axs.set_ylabel("Z (mm)"); axs.legend(fontsize=7,loc="upper right")

# ===== TOP (XY at grasp height) =====
axt.set_title(f"TOP (XY @ grasp): does the {2*gripper_outer:.0f}mm gripper fit the {2*VIAL_BORE_R:.0f}mm bore?")
axt.add_patch(Circle((0,0),VIAL_OUTER_R,fill=False,ec="0.5"))
axt.add_patch(Circle((0,0),VIAL_BORE_R,fill=False,ec="0.5",ls="--",label=f"bore {2*VIAL_BORE_R:.0f}mm"))
axt.add_patch(Circle((0,0),PLANT_R,fc="green",alpha=0.5,label=f"plant {2*PLANT_R:.0f}mm"))
for sgn in (-1,1):
    inner=sgn*FINGER_OPEN_INNER
    axt.add_patch(Rectangle((min(inner,inner+sgn*FINGER_THICK),-2.5),FINGER_THICK,5,fc="steelblue",alpha=0.7,label="finger (open)" if sgn<0 else None))
axt.set_xlim(-18,18); axt.set_ylim(-18,18); axt.set_aspect("equal"); axt.set_xlabel("X close (mm)"); axt.set_ylabel("Y (mm)"); axt.legend(fontsize=7)

plt.tight_layout(); plt.savefig("/tmp/invial_design.png",dpi=110)
print("wrote /tmp/invial_design.png")
print(f"gripper span={2*gripper_outer:.0f}mm, bore={2*VIAL_BORE_R:.0f}mm -> {'FITS' if gripper_outer<VIAL_BORE_R else 'TOO WIDE'} (clearance {VIAL_BORE_R-gripper_outer:.1f}mm/side)")
print(f"gripper can shift +-{gripper_reach:.1f}mm in bore; plant can sit +-{plant_play:.1f}mm off-centre")
print(f"=> {'OK if plant stays centred' if gripper_reach>=plant_play else f'CONFLICT: plant can be +-{plant_play:.1f}mm but gripper only reaches +-{gripper_reach:.1f}mm -> need thinner fingers or tighter bore'}")
print(f"fingers {FINGER_LEN:.0f}mm reach z={GRASP_Z:.0f}; body bottom z={BODY_BOT_Z:.0f} vs rim {VIAL_H:.0f} -> {'body clears rim' if BODY_BOT_Z>VIAL_H else 'BODY HITS RIM'}")
