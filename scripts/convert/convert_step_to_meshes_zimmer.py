"""Split the Zimmer GEP2010IL-00-B STEP assembly into per-link STL meshes.

Reads:  assets/t3_401_zimmer/cad/t3_401_zimmer.stp
Writes: assets/t3_401_zimmer/meshes/<part>.stl   (assembly placement applied)

The STEP assembly (downloaded from 3DFindIt — Zimmer's PartCommunity portal)
contains 5 parts (German naming):

    GEP2010IL-00-B1_ZYL   — Zylinder, the gripper body (→ gripper_base link)
    GEP2010IL-00-B_bac 1  — Backe, the jaw / finger (appears twice for L+R)
    CSTE00846_Stecker-    — Stecker, electrical connector (cosmetic, optional)
    CTYP00096_Etikett     — Etikett, label sticker (drop entirely)

Per CLAUDE.md #13, STEP part-name conventions vary between assemblies; this
PART_TO_LINK map is Zimmer-specific. The two jaw instances share the same
product name `GEP2010IL-00-B_bac 1` — we emit one combined mesh for each by
spatial split (negative-X jaw → finger_left, positive-X jaw → finger_right)
after inspecting the assembly placement.

Run with the repo's local venv that has cadquery-ocp + trimesh installed:
    ~/.venv-step/bin/python scripts/convert_step_to_meshes_zimmer.py
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
STEP_PATH = os.path.join(REPO_ROOT, "assets", "t3_401_zimmer", "cad", "t3_401_zimmer.stp")
MESH_DIR = os.path.join(REPO_ROOT, "assets", "t3_401_zimmer", "meshes")

# Map STEP product names to URDF link names. Stecker (connector) is kept as
# a cosmetic decoration since it carries the cable that visually identifies
# this is a real gripper rather than a generic stand-in. Etikett (label
# sticker) is dropped entirely — adds polygons without semantic value.
PART_TO_LINK: Dict[str, str] = {
    "GEP2010IL-00-B1_ZYL": "gripper_base",
    "GEP2010IL-00-B_bac 1": "finger_pair",  # split into L/R below by xy sign
    "CSTE00846_Stecker-": "stecker",        # cosmetic only
}

# Parts to drop entirely (cosmetic + non-load-bearing).
PARTS_TO_SKIP = {"CTYP00096_Etikett"}

# STEP units are millimetres; URDF/USD use metres.
SCALE_TO_M = 0.001


def _label_name(label: TDF_Label) -> str:
    from OCP.TDataStd import TDataStd_Name

    name_attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), name_attr):
        return name_attr.Get().ToExtString()
    return ""


def _walk_free_shapes(
    shape_tool,
    label: TDF_Label,
    parent_loc: TopLoc_Location,
    out: List[Tuple[str, TopoDS_Shape, TopLoc_Location]],
) -> None:
    """Recurse the assembly: emit (name, shape, world_location) for every leaf."""
    if shape_tool.IsAssembly_s(label):
        comps = TDF_LabelSequence()
        shape_tool.GetComponents_s(label, comps)
        for i in range(1, comps.Length() + 1):
            comp_lbl = comps.Value(i)
            comp_loc = shape_tool.GetLocation_s(comp_lbl)
            world_loc = TopLoc_Location(
                parent_loc.Transformation() * comp_loc.Transformation()
            )
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


def _bbox_m(
    shape: TopoDS_Shape, loc: TopLoc_Location
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    located = shape.Moved(loc)
    box = Bnd_Box()
    BRepBndLib.Add_s(located, box, True)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (
        (xmin * SCALE_TO_M, ymin * SCALE_TO_M, zmin * SCALE_TO_M),
        (xmax * SCALE_TO_M, ymax * SCALE_TO_M, zmax * SCALE_TO_M),
    )


def _bbox_center_x_m(shape: TopoDS_Shape, loc: TopLoc_Location) -> float:
    """X-axis bounding-box center in metres. Used to disambiguate the two
    `_bac` jaw instances by their assembly placement."""
    bb_min, bb_max = _bbox_m(shape, loc)
    return 0.5 * (bb_min[0] + bb_max[0])


def _write_stl(shape: TopoDS_Shape, loc: TopLoc_Location, out_path: str) -> None:
    located = shape.Moved(loc)
    BRepMesh_IncrementalMesh(located, 0.5, False, 0.5, True).Perform()
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    writer.Write(located, out_path)


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

    leaves: List[Tuple[str, TopoDS_Shape, TopLoc_Location]] = []
    for i in range(1, free_shapes.Length() + 1):
        _walk_free_shapes(shape_tool, free_shapes.Value(i), TopLoc_Location(), leaves)

    grouped: Dict[str, List[Tuple[TopoDS_Shape, TopLoc_Location]]] = {}
    for name, shape, loc in leaves:
        grouped.setdefault(name, []).append((shape, loc))

    print(f"discovered parts: {sorted(grouped)}")
    for name, insts in sorted(grouped.items()):
        print(f"  - {name!r:<40s}  instances={len(insts)}")

    for part_name in PARTS_TO_SKIP:
        if part_name in grouped:
            print(f"  [skip] {part_name}: explicitly dropped (cosmetic)")

    for part_name, link_name in PART_TO_LINK.items():
        instances = grouped.get(part_name)
        if not instances:
            print(f"  [skip] {part_name}: not found in assembly")
            continue

        # Special case: the two jaws share a product name. Split them by
        # their assembly placement's x-coordinate — left jaw has the more
        # negative x-bbox-center, right jaw the more positive.
        if link_name == "finger_pair" and len(instances) == 2:
            sorted_by_x = sorted(
                instances, key=lambda inst: _bbox_center_x_m(inst[0], inst[1])
            )
            for sub_name, (shape, loc) in zip(("finger_left", "finger_right"), sorted_by_x):
                out_path = os.path.join(MESH_DIR, f"{sub_name}.stl")
                _write_stl(shape, loc, out_path)
                bb_min, bb_max = _bbox_m(shape, loc)
                size = tuple(bb_max[i] - bb_min[i] for i in range(3))
                print(
                    f"  [ok ] {part_name:<26s} -> {sub_name+'.stl':<20s} "
                    f"min=({bb_min[0]:+.4f},{bb_min[1]:+.4f},{bb_min[2]:+.4f}) "
                    f"max=({bb_max[0]:+.4f},{bb_max[1]:+.4f},{bb_max[2]:+.4f}) "
                    f"size=({size[0]:.3f},{size[1]:.3f},{size[2]:.3f})"
                )
            continue

        # Combine all instances into one compound mesh (for stecker which
        # may appear once, or any other multi-instance non-jaw part).
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
        bb_min, bb_max = _bbox_m(shape, loc)
        size = tuple(bb_max[i] - bb_min[i] for i in range(3))
        print(
            f"  [ok ] {part_name:<26s} -> {link_name+'.stl':<20s} "
            f"min=({bb_min[0]:+.4f},{bb_min[1]:+.4f},{bb_min[2]:+.4f}) "
            f"max=({bb_max[0]:+.4f},{bb_max[1]:+.4f},{bb_max[2]:+.4f}) "
            f"size=({size[0]:.3f},{size[1]:.3f},{size[2]:.3f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
