"""Merge G1 OmniPicker visuals into 3 STLs matching the URDF's 3 rigid bodies:

  omnipicker_housing.STL       attaches to URDF gripper_base   (1 link)
  omnipicker_outer_finger.STL  attaches to URDF finger_left    (4 outer links)
  omnipicker_inner_finger.STL  attaches to URDF finger_right   (4 inner links)

Each finger group is baked at the G1's default-open pose and expressed in its
target URDF link's LOCAL frame, so the URDF prismatic joint translates the
visual correctly as the fingers close. The 4-bar linkage's small rotational
component is lost (visual fingers will only translate, not rotate slightly).

Run from the robotic-arm repo root:
    python3 scripts/merge_omnipicker_3parts.py
"""
import os
import struct
import numpy as np
from pxr import Usd, UsdGeom, Gf

USD_PATH = "/tmp/g1_omnipicker/configuration/robot_base.usd"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "ur5_omnipicker", "meshes")

# URDF finger_left/finger_right joint origins in gripper_base frame (from URDF).
# The merged finger STLs will be expressed in these local frames so the prismatic
# joints carry them correctly.
FINGER_LEFT_ORIGIN_IN_BASE = np.array([0.0, +0.0265, +0.07])
FINGER_RIGHT_ORIGIN_IN_BASE = np.array([0.0, -0.0265, +0.07])

stage = Usd.Stage.Open(USD_PATH)


def get_world_pose(prim_path):
    p = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(p)
    t = Gf.Vec3d(0)
    q = Gf.Quatd(1)
    for op in xf.GetOrderedXformOps():
        n = op.GetOpName()
        if "translate" in n:
            t = Gf.Vec3d(op.Get())
        elif "orient" in n:
            q = Gf.Quatd(op.Get())
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


def mesh_path(link):
    short = link.replace("gripper_l_", "")
    if short == "base_link":
        return f"/visuals/{link}/gripper_base_link/mesh"
    return f"/visuals/{link}/{short}/mesh"


def merge_links_into_frame(links, target_frame_in_base):
    """Merge the given links' visual meshes into a single (verts, tris) pair,
    expressed in a frame defined by target_frame_in_base (translation only,
    in base_link's local frame). Returns (verts, tris) with tri indices
    already offset.
    """
    T_base = get_world_pose(f"/genie/{links[0] if links[0] == 'gripper_l_base_link' else 'gripper_l_base_link'}")
    T_base_inv = T_base.GetInverse()

    merged_verts = []
    merged_tris = []
    for link in links:
        mp = mesh_path(link)
        p = stage.GetPrimAtPath(mp)
        if not p.IsValid():
            print(f"  SKIP {link}: no mesh at {mp}")
            continue
        verts, tris = read_mesh(mp)
        T_link = get_world_pose(f"/genie/{link}")
        # vertices into base_link's local frame
        T = T_link * T_base_inv
        in_base = np.array(
            [tuple(T.Transform(Gf.Vec3d(*v))) for v in verts],
            dtype=np.float64,
        )
        # then shift into target frame (subtract target origin in base)
        in_target = in_base - target_frame_in_base
        offset = sum(len(arr) for arr in merged_verts)
        merged_verts.append(in_target)
        merged_tris.append(tris + offset)
        print(f"  {link}: {len(verts):6d} verts, {len(tris):6d} tris")

    if not merged_verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    V = np.vstack(merged_verts)
    T = np.vstack(merged_tris)
    return V, T


def write_stl(path, V, T, header_str):
    with open(path, "wb") as f:
        header = header_str.encode("utf-8")
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


def report(name, V):
    if len(V) == 0:
        print(f"  {name}: EMPTY")
        return
    mn, mx = V.min(0) * 1000, V.max(0) * 1000
    print(f"  {name}: bbox X[{mn[0]:+.1f},{mx[0]:+.1f}] Y[{mn[1]:+.1f},{mx[1]:+.1f}] Z[{mn[2]:+.1f},{mx[2]:+.1f}] mm")


print("Housing (gripper_base):")
V_h, T_h = merge_links_into_frame(["gripper_l_base_link"], np.zeros(3))
report("housing", V_h)

print("\nOuter finger group (finger_left, in finger_left local frame):")
V_o, T_o = merge_links_into_frame(
    ["gripper_l_outer_link1", "gripper_l_outer_link2",
     "gripper_l_outer_link3", "gripper_l_outer_link4"],
    FINGER_LEFT_ORIGIN_IN_BASE,
)
report("outer_finger", V_o)

print("\nInner finger group (finger_right, in finger_right local frame):")
V_i, T_i = merge_links_into_frame(
    ["gripper_l_inner_link1", "gripper_l_inner_link2",
     "gripper_l_inner_link3", "gripper_l_inner_link4"],
    FINGER_RIGHT_ORIGIN_IN_BASE,
)
report("inner_finger", V_i)

os.makedirs(OUT_DIR, exist_ok=True)
write_stl(os.path.join(OUT_DIR, "omnipicker_housing.STL"), V_h, T_h, "omnipicker housing")
write_stl(os.path.join(OUT_DIR, "omnipicker_outer_finger.STL"), V_o, T_o, "omnipicker outer finger")
write_stl(os.path.join(OUT_DIR, "omnipicker_inner_finger.STL"), V_i, T_i, "omnipicker inner finger")
print(f"\nWrote 3 STLs to {os.path.abspath(OUT_DIR)}")
