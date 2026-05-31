"""Split the Epson T3-401S STEP assembly into per-link STL meshes.

Reads:  assets/t3_401_simple_gripper/cad/epson649561eu.stp
Writes: assets/t3_401_simple_gripper/meshes/<part>.stl   (assembly placement applied)

The Z-axis shaft (`link_3`) is trimmed to the bottom 100 mm so the visible
geometry matches the URDF's joint_3 -> joint_4 distance. The full physical
shaft is ~330 mm in CAD; the upper portion lives inside the arm 2 housing
and the column on the real machine, neither of which the URDF models, so
keeping the full mesh would just dangle below arm 2 in sim.

Run with the repo's local venv that has cadquery-ocp + trimesh installed:
    ~/.venv-step/bin/python scripts/convert_step_to_meshes.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

from OCP.BRep import BRep_Builder
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.StlAPI import StlAPI_Writer
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCP.XCAFDoc import XCAFDoc_DocumentTool

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STEP_PATH = os.path.join(REPO_ROOT, "assets", "t3_401_simple_gripper", "cad", "epson649561eu.stp")
MESH_DIR = os.path.join(REPO_ROOT, "assets", "t3_401_simple_gripper", "meshes")

# Map STEP product names to URDF link names. Anything not in this map is
# either an assembly wrapper or a non-kinematic decoration we skip.
PART_TO_LINK: Dict[str, str] = {
    "T3-401S_base": "base_link",
    "T3-401S_arm1": "link_1",
    "T3-401S_arm2": "link_2",
    "T3-401S_shaft": "link_3",
    "T3-401S_cable": "cable",  # cosmetic; URDF can ignore
}

# STEP units are millimetres; URDF/USD use metres.
SCALE_TO_M = 0.001

# After splitting, the shaft mesh is trimmed to its bottom 100 mm so the
# visible geometry matches the URDF's joint_3 -> joint_4 spacing.
SHAFT_VISIBLE_LENGTH_MM = 100.0


def _label_name(label: TDF_Label) -> str:
    from OCP.TDataStd import TDataStd_Name

    name_attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), name_attr):
        return name_attr.Get().ToExtString()
    return ""


def _walk_free_shapes(shape_tool, label: TDF_Label, parent_loc: TopLoc_Location, out: List[Tuple[str, TopoDS_Shape, TopLoc_Location]]) -> None:
    """Recurse the assembly: emit (name, shape, world_location) for every leaf."""
    if shape_tool.IsAssembly_s(label):
        comps = TDF_LabelSequence()
        shape_tool.GetComponents_s(label, comps)
        for i in range(1, comps.Length() + 1):
            comp_lbl = comps.Value(i)
            comp_loc = shape_tool.GetLocation_s(comp_lbl)
            world_loc = TopLoc_Location(parent_loc.Transformation() * comp_loc.Transformation())
            ref_lbl = TDF_Label()
            if shape_tool.GetReferredShape_s(comp_lbl, ref_lbl):
                _walk_free_shapes(shape_tool, ref_lbl, world_loc, out)
            else:
                _walk_free_shapes(shape_tool, comp_lbl, world_loc, out)
    else:
        shp = shape_tool.GetShape_s(label)
        if shp.IsNull():
            return
        out.append((_label_name(label), shp, parent_loc))


def _bbox_m(shape: TopoDS_Shape, loc: TopLoc_Location) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    located = shape.Moved(loc)
    box = Bnd_Box()
    BRepBndLib.Add_s(located, box, True)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (
        (xmin * SCALE_TO_M, ymin * SCALE_TO_M, zmin * SCALE_TO_M),
        (xmax * SCALE_TO_M, ymax * SCALE_TO_M, zmax * SCALE_TO_M),
    )


def _write_stl(shape: TopoDS_Shape, loc: TopLoc_Location, out_path: str) -> None:
    located = shape.Moved(loc)
    BRepMesh_IncrementalMesh(located, 0.5, False, 0.5, True).Perform()
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    writer.Write(located, out_path)


def _trim_shaft_in_place(stl_path: str) -> None:
    """Keep only the bottom SHAFT_VISIBLE_LENGTH_MM of the shaft mesh.

    The full shaft is ~330 mm in CAD but the URDF treats link_3 as a 100 mm
    cylinder hanging below joint_3. Trimming the upper portion (which
    physically lives inside the arm 2 housing / column on the real machine)
    keeps the URDF's joint_3 -> joint_4 spacing consistent with the visible
    geometry.
    """
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh")
    cutoff_y = float(mesh.bounds[0][1]) + SHAFT_VISIBLE_LENGTH_MM
    trimmed = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=[0.0, -1.0, 0.0],
        plane_origin=[0.0, cutoff_y, 0.0],
    )
    if trimmed is None or trimmed.is_empty:
        raise RuntimeError(f"shaft trim produced empty mesh; cutoff_y={cutoff_y}")
    trimmed.export(stl_path)


def main() -> int:
    if not os.path.isfile(STEP_PATH):
        print(f"STEP file not found: {STEP_PATH}", file=sys.stderr)
        return 2
    os.makedirs(MESH_DIR, exist_ok=True)

    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    reader = STEPCAFControl_Reader()
    if reader.ReadFile(STEP_PATH) != 1:
        print("STEP read failed", file=sys.stderr)
        return 1
    if not reader.Transfer(doc):
        print("STEP transfer failed", file=sys.stderr)
        return 1

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    free_shapes = TDF_LabelSequence()
    shape_tool.GetFreeShapes(free_shapes)

    # Aggregate every leaf instance per part name (some assemblies have multiple
    # instances of the same product — for T3-401S each kinematic part is unique).
    leaves: List[Tuple[str, TopoDS_Shape, TopLoc_Location]] = []
    for i in range(1, free_shapes.Length() + 1):
        _walk_free_shapes(shape_tool, free_shapes.Value(i), TopLoc_Location(), leaves)

    grouped: Dict[str, List[Tuple[TopoDS_Shape, TopLoc_Location]]] = {}
    for name, shape, loc in leaves:
        grouped.setdefault(name, []).append((shape, loc))

    print(f"discovered parts: {sorted(grouped)}")

    for part_name, link_name in PART_TO_LINK.items():
        instances = grouped.get(part_name)
        if not instances:
            print(f"  [skip] {part_name}: not found in assembly")
            continue
        if len(instances) == 1:
            shape, loc = instances[0]
        else:
            builder = BRep_Builder()
            comp = TopoDS_Compound()
            builder.MakeCompound(comp)
            for shp, lc in instances:
                builder.Add(comp, shp.Moved(lc))
            shape, loc = comp, TopLoc_Location()
        out_path = os.path.join(MESH_DIR, f"{link_name}.stl")
        _write_stl(shape, loc, out_path)
        if link_name == "link_3":
            _trim_shaft_in_place(out_path)
        bb_min, bb_max = _bbox_m(shape, loc)
        size = tuple(bb_max[i] - bb_min[i] for i in range(3))
        print(
            f"  [ok ] {part_name:<22s} -> {link_name+'.stl':<20s} "
            f"min=({bb_min[0]:+.4f},{bb_min[1]:+.4f},{bb_min[2]:+.4f}) "
            f"max=({bb_max[0]:+.4f},{bb_max[1]:+.4f},{bb_max[2]:+.4f}) "
            f"size=({size[0]:.3f},{size[1]:.3f},{size[2]:.3f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
