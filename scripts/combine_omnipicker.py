"""Combine the 3 OmniPicker visual STLs into one ``omnipicker_full.STL``.

Inputs (already on disk):
    assets/ur5_omnipicker/meshes/omnipicker_housing.STL      — housing alone
    assets/ur5_omnipicker/meshes/omnipicker_outer_finger.STL — outer finger
    assets/ur5_omnipicker/meshes/omnipicker_inner_finger.STL — inner finger

Frame of the input STLs is the merge script's "gripper_l_base_link local
frame" — which is rotated relative to URDF's gripper_base frame. Inspecting
the housing geometry shows:
  - The wrist-mating CIRCULAR face is at the housing's Y=0
  - The body extends in +Y from there up to Y=+83 mm
  - A rectangular finger-slot opening goes through the housing at
    Y=50–70 mm (in housing-local), running parallel to the Z axis

URDF gripper_base convention (per existing finger joint origins) is:
  - Z = approach axis (away from the wrist)
  - Y = finger spread axis (fingers at ±0.0265 m)
  - X = lateral

So we rotate every input STL +90° about +X to map:
    housing-local Y -> URDF Z   (housing extension into URDF approach axis)
    housing-local Z -> URDF Y   (lateral becomes finger spread)
    housing-local X -> URDF X   (unchanged)

After this rotation the fingers naturally sit at URDF Y=±0.0265 (which
maps from housing-local Z=∓0.0265, well inside the housing's old Z range
±0.04 m — i.e. inside the body of the housing and lined up with the
rectangular finger slot).

Inner finger goes on the +Y side of the URDF (finger_left), outer on -Y
(finger_right). The previous merge_omnipicker_3parts.py output had this
swapped, which made the fingers cross the centerline.

Output:
    assets/ur5_omnipicker/meshes/omnipicker_full.STL — single rigid mesh
    of the assembled gripper at OPEN pose, in URDF gripper_base frame,
    with circular wrist-mating face at Z=0 and fingers slotted into the
    housing's rectangular opening.

Run after editing any input STL:
    $ARM_ACT_VENV/bin/python scripts/combine_omnipicker.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent
MESH_DIR = REPO / "assets" / "ur5_omnipicker" / "meshes"

# +90° about +X: maps (x, y, z) -> (x, -z, y).  Brings the housing into
# URDF gripper_base orientation (wrist face at Z=0, body in +Z).
ROT_X90 = np.array([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0],
])

# Outer finger: rotate -90° about +Z so its jaw face points in +Y
# (toward centerline, facing finger_LEFT). Maps (x, y, z) -> (y, -x, z).
ROT_Z_OUTER = np.array([
    [ 0, 1, 0],
    [-1, 0, 0],
    [ 0, 0, 1],
])

# Inner finger: mirror the inner-finger STL across X-Z plane (negate Y)
# BEFORE applying the same -90° Z rotation as outer. The inner mesh was
# generated as the (-Y of outer) but the merge script's frame had it
# pointing backwards; flipping Y first makes the jaw face point toward
# the centerline like outer's does.
MIRROR_Y = np.array([
    [1,  0, 0],
    [0, -1, 0],
    [0,  0, 1],
])
ROT_Z_INNER = ROT_Z_OUTER @ MIRROR_Y

# URDF joint origins for the fingers, in gripper_base local frame.
FL_ORIGIN = np.array([0.0, +0.0265, +0.07])      # finger_left  (+Y)
FR_ORIGIN = np.array([0.0, -0.0265, +0.07])      # finger_right (-Y)


def load_rotate(name, post_offset=None):
    """Load STL, rotate +90° about X, optionally translate."""
    m = trimesh.load(MESH_DIR / name, force="mesh")
    V = np.asarray(m.vertices) @ ROT_X90.T
    if post_offset is not None:
        V = V + post_offset
    return V, np.asarray(m.faces)


def report(name, V):
    b = V * 1000
    print(f"  {name:25s} bbox X[{b[:,0].min():+7.1f},{b[:,0].max():+7.1f}] "
          f"Y[{b[:,1].min():+7.1f},{b[:,1].max():+7.1f}] "
          f"Z[{b[:,2].min():+7.1f},{b[:,2].max():+7.1f}]")


def write_stl(path, V, F, header):
    with open(path, "wb") as f:
        h = header.encode("utf-8")
        f.write(h + b" " * (80 - len(h)))
        f.write(struct.pack("<I", len(F)))
        for tri in F:
            v0, v1, v2 = V[tri[0]], V[tri[1]], V[tri[2]]
            n = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                n = n / ln
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(b"\x00\x00")


    # Housing's long axis (Y in its native frame) needs to become URDF +Z.
    # The fingers' long axis (Z in their native frame) is ALREADY URDF +Z,
    # so they don't need rotation — only translation to the joint origins.

def _load(name, rotation=None, post_offset=None):
    m = trimesh.load(MESH_DIR / name, force="mesh")
    V = np.asarray(m.vertices)
    if rotation is not None:
        V = V @ rotation.T
    if post_offset is not None:
        V = V + post_offset
    return V, np.asarray(m.faces)

def main():
    V_h, F_h = _load("omnipicker_housing.STL", rotation=ROT_X90)
    V_i, F_i = _load("omnipicker_inner_finger.STL", rotation=ROT_Z_INNER, post_offset=FL_ORIGIN)
    V_o, F_o = _load("omnipicker_outer_finger.STL", rotation=ROT_Z_OUTER, post_offset=FR_ORIGIN)

    print("After +90° X-rotation, in URDF gripper_base frame (mm):")
    report("housing",                  V_h)
    report("inner @ finger_LEFT  +Y",  V_i)
    report("outer @ finger_RIGHT -Y",  V_o)

    V = np.vstack([V_h, V_i, V_o])
    F = np.vstack([F_h, F_i + len(V_h), F_o + len(V_h) + len(V_i)])

    out = MESH_DIR / "omnipicker_full.STL"
    write_stl(out, V, F, "omnipicker full (rotated +90X, fingers slotted)")
    b = V * 1000
    print(f"\nCombined bbox: X[{b[:,0].min():+.1f},{b[:,0].max():+.1f}] "
          f"Y[{b[:,1].min():+.1f},{b[:,1].max():+.1f}] "
          f"Z[{b[:,2].min():+.1f},{b[:,2].max():+.1f}] mm")
    print(f"Wrote {out}: {len(V)} verts, {len(F)} tris")


if __name__ == "__main__":
    main()
