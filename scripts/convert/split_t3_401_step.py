"""Convert Epson T3-B401S STEP CAD into per-link meshes for the SCARA URDF.

Pipeline phases (each is a subcommand):

    inspect   list every solid in the STEP file (bbox / centroid / volume)
              + summary histograms over centroid Z and radial distance, so we
              can see how Epson grouped the assembly. Optionally exports each
              solid as an individual STL for visual review.

    group     (next) assigns each solid to a URDF link via a kinematic
              spatial heuristic, writes a `grouping.yaml` that can be hand-
              edited if any solid lands in the wrong bucket.

    build     (next) merges the grouped solids per link, exports a visual OBJ
              + a CoACD collision OBJ, computes mass+inertia from the mesh,
              rewrites the URDF to reference the meshes, and re-runs the
              existing USD converter.

Run in the project venv (NOT IsaacLab's python — that's only for the final
USD step). One-time install:

    .venv/bin/pip install cadquery cadquery-ocp trimesh coacd

Then:

    .venv/bin/python scripts/split_t3_401_step.py inspect
    .venv/bin/python scripts/split_t3_401_step.py inspect --export-individuals
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("t3_step")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CAD_DIR = REPO_ROOT / "assets" / "t3_401_simple_gripper" / "cad"
DEFAULT_STEP = CAD_DIR / "T3-B401S.stp"
INVENTORY_JSON = CAD_DIR / "inventory.json"
INVENTORY_TXT = CAD_DIR / "inventory.txt"
INDIVIDUALS_DIR = CAD_DIR / "individuals"


@dataclass
class SolidInfo:
    index: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_size: tuple[float, float, float]
    centroid: tuple[float, float, float]
    volume_mm3: float
    radial_mm: float


def _require_cadquery():
    try:
        import cadquery as cq
    except ImportError:
        sys.exit(
            "cadquery is not installed in this venv. Install with:\n"
            "    .venv/bin/pip install cadquery cadquery-ocp trimesh coacd"
        )
    return cq


def load_solids(step_path: Path):
    if not step_path.exists():
        sys.exit(
            f"STEP file not found: {step_path}\n"
            f"Drop the Epson T3-B401S CAD (.stp) at that path and re-run."
        )
    cq = _require_cadquery()
    logger.info("Reading STEP: %s", step_path)
    wp = cq.importers.importStep(str(step_path))
    compound = wp.val()
    solids = compound.Solids()
    logger.info("Found %d solids", len(solids))
    return cq, solids


def collect_infos(solids) -> list[SolidInfo]:
    infos: list[SolidInfo] = []
    for i, s in enumerate(solids):
        bb = s.BoundingBox()
        c = s.Center()
        infos.append(
            SolidInfo(
                index=i,
                bbox_min=(bb.xmin, bb.ymin, bb.zmin),
                bbox_max=(bb.xmax, bb.ymax, bb.zmax),
                bbox_size=(bb.xmax - bb.xmin, bb.ymax - bb.ymin, bb.zmax - bb.zmin),
                centroid=(c.x, c.y, c.z),
                volume_mm3=float(s.Volume()),
                radial_mm=math.hypot(c.x, c.y),
            )
        )
    return infos


def histogram(values: list[float], n: int = 20) -> list[tuple[float, float, int]]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [(lo, hi, len(values))]
    width = (hi - lo) / n
    counts = [0] * n
    for v in values:
        idx = min(int((v - lo) / width), n - 1)
        counts[idx] += 1
    return [(lo + i * width, lo + (i + 1) * width, c) for i, c in enumerate(counts)]


def render_report(infos: list[SolidInfo]) -> str:
    if not infos:
        return "STEP file contained no solids."

    lines: list[str] = []
    lines.append(f"=== STEP inventory: {len(infos)} solids ===")
    lines.append("")

    all_min = [min(i.bbox_min[k] for i in infos) for k in range(3)]
    all_max = [max(i.bbox_max[k] for i in infos) for k in range(3)]
    lines.append(
        "Overall bbox (mm): "
        f"X[{all_min[0]:8.1f}, {all_max[0]:8.1f}]  "
        f"Y[{all_min[1]:8.1f}, {all_max[1]:8.1f}]  "
        f"Z[{all_min[2]:8.1f}, {all_max[2]:8.1f}]"
    )
    lines.append(
        "Overall size (mm): "
        f"{all_max[0] - all_min[0]:.1f} x "
        f"{all_max[1] - all_min[1]:.1f} x "
        f"{all_max[2] - all_min[2]:.1f}"
    )
    lines.append("")

    z_centers = [i.centroid[2] for i in infos]
    lines.append("Centroid Z histogram (height layers - hints J1/J2 axis heights):")
    for z_lo, z_hi, count in histogram(z_centers, n=20):
        bar = "#" * count
        lines.append(f"  Z[{z_lo:8.1f}, {z_hi:8.1f}]  {count:4d}  {bar}")
    lines.append("")

    radials = [i.radial_mm for i in infos]
    lines.append("Centroid radial distance from Z axis (hints arm reach):")
    for r_lo, r_hi, count in histogram(radials, n=20):
        bar = "#" * count
        lines.append(f"  r[{r_lo:8.1f}, {r_hi:8.1f}]  {count:4d}  {bar}")
    lines.append("")

    by_vol = sorted(infos, key=lambda x: -x.volume_mm3)
    top_n = min(50, len(by_vol))
    lines.append(f"=== Top {top_n} solids by volume ===")
    lines.append("")
    lines.append(
        f"{'idx':>4}  {'vol_cm3':>10}  "
        f"{'cx':>8} {'cy':>8} {'cz':>8}  {'radial':>8}  "
        f"{'sx':>7} {'sy':>7} {'sz':>7}"
    )
    for s in by_vol[:top_n]:
        lines.append(
            f"{s.index:>4}  {s.volume_mm3 / 1000.0:>10.2f}  "
            f"{s.centroid[0]:>8.1f} {s.centroid[1]:>8.1f} {s.centroid[2]:>8.1f}  "
            f"{s.radial_mm:>8.1f}  "
            f"{s.bbox_size[0]:>7.1f} {s.bbox_size[1]:>7.1f} {s.bbox_size[2]:>7.1f}"
        )
    lines.append("")
    return "\n".join(lines)


def export_individuals(cq, solids, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting %d individual STLs to %s", len(solids), out_dir)
    for i, s in enumerate(solids):
        path = out_dir / f"solid_{i:03d}.stl"
        cq.exporters.export(s, str(path))
    logger.info("Done exporting individuals.")


def cmd_inspect(args: argparse.Namespace) -> None:
    cq, solids = load_solids(args.step)
    infos = collect_infos(solids)

    CAD_DIR.mkdir(parents=True, exist_ok=True)
    INVENTORY_JSON.write_text(json.dumps([asdict(i) for i in infos], indent=2))
    report = render_report(infos)
    INVENTORY_TXT.write_text(report)
    print(report)
    logger.info("Wrote %s", INVENTORY_TXT)
    logger.info("Wrote %s", INVENTORY_JSON)

    if args.export_individuals:
        export_individuals(cq, solids, INDIVIDUALS_DIR)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="List every solid in the STEP file.")
    pi.add_argument(
        "--step",
        type=Path,
        default=DEFAULT_STEP,
        help=f"Path to the STEP file (default: {DEFAULT_STEP.relative_to(REPO_ROOT)}).",
    )
    pi.add_argument(
        "--export-individuals",
        action="store_true",
        help="Also export each solid as an STL under cad/individuals/ for visual review.",
    )
    pi.set_defaults(func=cmd_inspect)

    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
