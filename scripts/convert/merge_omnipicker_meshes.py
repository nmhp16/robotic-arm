"""Merge all 9 OmniPicker visual meshes from the G1 USD into one
combined STL, with each link transformed into gripper_l_base_link's
local frame. The result is a single mesh capturing the whole gripper
at its default-open pose — drop-in replacement for our hand-clipped
omnipicker_base.STL.
"""
import os
import struct
import numpy as np
from pxr import Usd, UsdGeom, Gf

USD_PATH = "/tmp/g1_omnipicker/configuration/robot_base.usd"
OUT_STL = "/Volumes/System/Projects/robotic-arm/assets/ur5_omnipicker/meshes/omnipicker_full.STL"

stage = Usd.Stage.Open(USD_PATH)


def world_xform(prim):
    """Get the local-to-world transform of a prim (assumes the
    parent chain has identity world transform — true here since
    /genie at world origin)."""
    xf = UsdGeom.Xformable(prim)
    if not xf:
        return Gf.Matrix4d(1)
    # GetLocalTransformation respects the ordered xform ops
    m, _ = xf.GetLocalTransformation()
    parent = prim.GetParent()
    if parent and parent.GetPath() != prim.GetPath().GetParentPath().GetParentPath():
        # cheap parent-walk; works because /genie is the root xform
        return world_xform(parent) * m
    return m


def get_world_pose(prim_path):
    """Read xformOp:translate + xformOp:orient from a /genie/X prim."""
    p = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(p)
    ops = xf.GetOrderedXformOps()
    t = Gf.Vec3d(0)
    q = Gf.Quatd(1)
    s = Gf.Vec3d(1)
    for op in ops:
        n = op.GetOpName()
        if "translate" in n:
            t = Gf.Vec3d(op.Get())
        elif "orient" in n:
            q = Gf.Quatd(op.Get())
        elif "scale" in n:
            s = Gf.Vec3d(op.Get())
    m = Gf.Matrix4d(1)
    m.SetScale(s)
    m = m * Gf.Matrix4d().SetRotate(q) * Gf.Matrix4d().SetTranslate(t)
    # USD column-major; for row-vec Python, use Transform()
    M = Gf.Matrix4d(1)
    M.SetTranslateOnly(t)
    M.SetRotateOnly(Gf.Rotation(q))
    return M


def read_mesh(prim_path):
    p = stage.GetPrimAtPath(prim_path)
    mesh = UsdGeom.Mesh(p)
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    verts = np.array([(v[0], v[1], v[2]) for v in pts], dtype=np.float64)
    tris = []
    idx = 0
    for n in counts:
        if n == 3:
            tris.append([indices[idx], indices[idx+1], indices[idx+2]])
        else:
            for i in range(1, n - 1):
                tris.append([indices[idx], indices[idx+i], indices[idx+i+1]])
        idx += n
    return verts, np.array(tris, dtype=np.int64)


# 9 gripper-left link names — in the order we'll merge.
LINKS = [
    "gripper_l_base_link",
    "gripper_l_inner_link1",
    "gripper_l_inner_link2",
    "gripper_l_inner_link3",
    "gripper_l_inner_link4",
    "gripper_l_outer_link1",
    "gripper_l_outer_link2",
    "gripper_l_outer_link3",
    "gripper_l_outer_link4",
]

T_base = get_world_pose(f"/genie/{LINKS[0]}")
T_base_inv = T_base.GetInverse()

merged_verts = []
merged_tris = []

for link in LINKS:
    mesh_path = f"/visuals/{link}/{link[len('gripper_l_'):]}/mesh"
    # path style we saw: /visuals/gripper_l_base_link/gripper_base_link/mesh
    # the middle dir drops the "gripper_l_" prefix
    # but for inner_link1 it's /visuals/gripper_l_inner_link1/inner_link1/mesh
    short = link.replace("gripper_l_", "")
    mesh_path = f"/visuals/{link}/{short}/mesh"
    if short == "base_link":
        mesh_path = f"/visuals/{link}/gripper_base_link/mesh"
    p = stage.GetPrimAtPath(mesh_path)
    if not p.IsValid():
        print(f"  SKIP {link}: no mesh at {mesh_path}")
        continue

    verts, tris = read_mesh(mesh_path)
    T_link = get_world_pose(f"/genie/{link}")
    # Transform link verts: V_base = T_base⁻¹ · T_link · V_local
    T = T_link * T_base_inv  # row-vec convention: V·T = V·T_link·T_base_inv
    # USD Gf.Matrix4d uses row-vector convention with .Transform(point)
    transformed = np.array(
        [tuple(T.Transform(Gf.Vec3d(*v))) for v in verts],
        dtype=np.float64,
    )

    offset = sum(len(arr) for arr in merged_verts)
    merged_verts.append(transformed)
    merged_tris.append(tris + offset)
    print(f"  {link}: {len(verts):6d} verts, {len(tris):6d} tris transformed")

V = np.vstack(merged_verts)
T = np.vstack(merged_tris)
print(f"\nmerged: {len(V):,} verts, {len(T):,} tris")
print(f"bbox (m): min={V.min(0)} max={V.max(0)} size={V.max(0)-V.min(0)}")

# Write binary STL.
with open(OUT_STL, "wb") as f:
    header = b"omnipicker merged from G1_omnipicker USD"
    f.write(header + b" " * (80 - len(header)))
    f.write(struct.pack("<I", len(T)))
    for tri in T:
        v0, v1, v2 = V[tri[0]], V[tri[1]], V[tri[2]]
        a = v1 - v0
        b = v2 - v0
        n = np.cross(a, b)
        ln = np.linalg.norm(n)
        if ln > 1e-12:
            n = n / ln
        f.write(struct.pack("<fff", *n))
        f.write(struct.pack("<fff", *v0))
        f.write(struct.pack("<fff", *v1))
        f.write(struct.pack("<fff", *v2))
        f.write(b"\x00\x00")
print(f"\n-> {OUT_STL}  ({os.path.getsize(OUT_STL)/1e6:.1f} MB)")
