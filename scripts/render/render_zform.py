"""Z-form-closure design: stem NODE + fingertip LEDGE (CLAUDE.md #9/#15).

The lift-slip (stem slides DOWN out of the grip) is friction-limited and no
lateral grasp shape fixes it. Z-form-closure does: a wider NODE on the stem
rests on a horizontal LEDGE at the fingertip, so the lift carries the plant by
geometry (like a bucket handle), not friction — works at realistic mu.

Side view (XZ), +X finger only (mirror on -X). Validates:
  - DESCENT: open finger's ledge clears the node (ledge gap > node width)
  - CLOSED: ledge sits UNDER the node rim (node rests on ledge top)
  - RELIEF: shank recessed above the ledge so the node fits (no interpenetration)
  - the V-tip still grips the 6mm stem below the node (lateral stability)
All mm. Realistic: grasp the stem just under a leaf node, node carries the lift.
"""
from __future__ import annotations
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- stem + node (mm, world) ---
STEM_R   = 3.0          # 6mm stem
NODE_R   = 4.0          # 8mm node (leaf-node collar)
NODE_Z0, NODE_Z1 = 43.0, 49.0   # node spans z 43..49
GRASP_Z  = 32.0         # V-tip grips the stem here (below the node)

# --- finger features (+X side), CLOSED (gripping) positions ---
VTIP_INNER  = 3.0       # V-tip inner face grips the 6mm stem at +-3
VTIP_Z0, VTIP_Z1 = 25.0, 39.0   # V-notch tip region
LEDGE_INNER = 3.0       # ledge inner edge FLUSH with stem surface (+3) — NOT inside it; supports only the node RIM (+3..+4 overhang)
LEDGE_Z0, LEDGE_Z1 = 40.0, 43.0 # ledge just under the node (top at 43 = node bottom)
SHANK_INNER = 5.0       # shank RECESSED to +5 so the +-4 node clears (relief)
SHANK_Z0, SHANK_Z1 = 43.0, 110.0
OPEN_SHIFT  = 2.0       # finger shifts out this much when open (for descent)

fig, (axc, axo) = plt.subplots(1, 2, figsize=(12, 7))

def draw_finger(ax, shift, alpha):
    # V-tip (grips stem), ledge (under node), recessed shank (relief)
    ax.add_patch(Rectangle((VTIP_INNER+shift, VTIP_Z0), 3, VTIP_Z1-VTIP_Z0, fc="steelblue", alpha=alpha, ec="navy"))
    ax.add_patch(Rectangle((LEDGE_INNER+shift, LEDGE_Z0), 3, LEDGE_Z1-LEDGE_Z0, fc="darkorange", alpha=alpha, ec="navy"))
    ax.add_patch(Rectangle((SHANK_INNER+shift, SHANK_Z0), 3, SHANK_Z1-SHANK_Z0, fc="steelblue", alpha=alpha, ec="navy"))

def draw_plant(ax):
    ax.add_patch(Rectangle((-STEM_R, 0), 2*STEM_R, 58, fc="green", alpha=0.35, label="stem 6mm"))
    ax.add_patch(Rectangle((-NODE_R, NODE_Z0), 2*NODE_R, NODE_Z1-NODE_Z0, fc="forestgreen", alpha=0.8, label="node 8mm (leaf node)"))

# CLOSED (gripping + lifting): node rests on ledge
axc.set_title("CLOSED (grip+lift): node rests on the ledge\n-> lifted by GEOMETRY, not friction")
draw_plant(axc)
draw_finger(axc, 0.0, 0.75)
axc.annotate("ledge UNDER node rim", (LEDGE_INNER, LEDGE_Z1), (8, 52), fontsize=8,
             arrowprops=dict(arrowstyle="->", color="darkorange"))
axc.annotate("V-tip grips stem", (VTIP_INNER, 32), (8, 24), fontsize=8,
             arrowprops=dict(arrowstyle="->", color="navy"))
axc.axhline(GRASP_Z, color="g", ls=":", lw=1)
axc.set_xlim(-8, 12); axc.set_ylim(0, 70); axc.set_xlabel("X (mm)"); axc.set_ylabel("Z (mm)"); axc.legend(fontsize=8, loc="upper right")

# OPEN (descent): ledge clears the node
axo.set_title("OPEN (descent): ledge gap clears the node")
draw_plant(axo)
draw_finger(axo, OPEN_SHIFT, 0.55)
# mirror finger (-X) to show the gap
def draw_finger_mirror(ax, shift, alpha):
    ax.add_patch(Rectangle((-(VTIP_INNER+shift)-3, VTIP_Z0), 3, VTIP_Z1-VTIP_Z0, fc="steelblue", alpha=alpha))
    ax.add_patch(Rectangle((-(LEDGE_INNER+shift)-3, LEDGE_Z0), 3, LEDGE_Z1-LEDGE_Z0, fc="darkorange", alpha=alpha))
    ax.add_patch(Rectangle((-(SHANK_INNER+shift)-3, SHANK_Z0), 3, SHANK_Z1-SHANK_Z0, fc="steelblue", alpha=alpha))
draw_finger_mirror(axo, OPEN_SHIFT, 0.55)
axo.set_xlim(-12, 12); axo.set_ylim(0, 70); axo.set_xlabel("X (mm)"); axo.set_ylabel("Z (mm)")

plt.tight_layout(); plt.savefig("/tmp/zform_design.png", dpi=110)
print("wrote /tmp/zform_design.png")

# --- numeric checks ---
ledge_open = LEDGE_INNER + OPEN_SHIFT
print(f"DESCENT: open ledge inner = +-{ledge_open:.1f}mm vs node +-{NODE_R:.1f}mm -> {'CLEARS' if ledge_open >= NODE_R else 'BLOCKS node (bad)'}")
print(f"CLOSED:  ledge inner = +-{LEDGE_INNER:.1f}mm vs node rim +-{NODE_R:.1f}mm -> {'ledge UNDER node ('+str(NODE_R-LEDGE_INNER)+'mm overlap to rest on)' if LEDGE_INNER < NODE_R else 'ledge misses node (bad)'}")
print(f"RELIEF:  shank inner = +-{SHANK_INNER:.1f}mm vs node +-{NODE_R:.1f}mm -> {'node CLEARS shank' if SHANK_INNER >= NODE_R else 'node hits shank (bad)'}")
print(f"GRIP:    V-tip inner = +-{VTIP_INNER:.1f}mm vs stem +-{STEM_R:.1f}mm -> {'grips stem' if VTIP_INNER<=STEM_R+0.2 else 'misses stem'}")
print(f"node bottom z={NODE_Z0} vs ledge top z={LEDGE_Z1} -> {'node rests on ledge' if abs(NODE_Z0-LEDGE_Z1)<1.5 else 'GAP/overlap '+str(NODE_Z0-LEDGE_Z1)+'mm'}")
print(f"node top z={NODE_Z1} < vial rim 70 -> {'recessed OK' if NODE_Z1 < 70 else 'node above rim'}")
