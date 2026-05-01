# CAD → URDF (visual-swap) playbook

When the vendor ships only CAD (no URDF), and an existing primitive URDF in
the project already works for kinematics, follow this method to upgrade
visuals without breaking the training pipeline. Done once for the Epson
T3-401S; the same shape applies to any STEP-only arm.

## When to use this method

- Vendor ships only CAD (T-series Epson, low-end SCARAs, custom arms).
- An existing primitive URDF works for the kinematic skeleton.
- You want sim-to-real visual fidelity but **don't** want to retune the
  oracle / ACT pipeline by changing joint origins.

## When **not** to use it

- Vendor or community ships a URDF — use that. For Epson **GX4/GX8/RS/C/CX**
  see `Epson-Robots/epson-robot-ros2`. For UR/Franka/KUKA there are
  canonical packages.
- Robot has parallel / closed-loop kinematics — URDF can't represent these.
- You need physically accurate collision envelopes — do convex
  decomposition (CoACD) instead and replace `<collision>` blocks too.

## Prerequisites

One-time venv outside the project (don't install full `cadquery`, the
aarch64 wheel chain breaks on `nlopt`):

```bash
python3 -m venv ~/.venv-step
~/.venv-step/bin/pip install cadquery-ocp typing_extensions trimesh numpy scipy shapely
```

## The pipeline

### 1. Inspect the STEP — what parts exist?

```bash
grep -oE "PRODUCT\('[^']+','[^']+'" assets/.../cad/*.stp | head
```

Map each named product to a URDF link. T3-401S example:

| STEP product | URDF link |
|---|---|
| `_T3-B401S_B` | `base_link` |
| `_T3-B401S_A1` | `link_1` |
| `_T3-B401S_A2` | `link_2` |
| `_T3-B401S_SHAFT` | `link_3` (trimmed) |
| `_T3-B401S_CABLE` | skipped (cosmetic) |

If parts aren't named, enumerate solids and group by centroid Z + radial
distance heuristics — there's a stub for that approach in
`scripts/split_t3_401_step.py` (commit `c52f72f`).

### 2. Split STEP → STLs

`scripts/convert_step_to_meshes.py` uses raw `OCP` (cadquery's OCCT
binding):

1. `STEPCAFControl_Reader` reads the assembly into an `XCAF` doc.
2. `XCAFDoc_DocumentTool.ShapeTool_s` gives the assembly tree.
3. Walk it; group leaves by product name; aggregate via `BRep_Builder` if
   a part has multiple instances.
4. Apply the world placement (`shape.Moved(loc)`) so each STL is in CAD
   world coords.
5. Tessellate with `BRepMesh_IncrementalMesh(shape, 0.5, False, 0.5, True)`.
6. Write with `StlAPI_Writer` (binary).

Run: `~/.venv-step/bin/python scripts/convert_step_to_meshes.py`

The script also prints per-part bboxes in metres. **Copy those numbers** —
they drive step 4.

### 3. Reconcile CAD frame ↔ URDF frame

Frame mismatch is the #1 bug source. Compare:

|  | CAD (Epson STEP) | URDF (this project) |
|---|---|---|
| Vertical axis | `+Y` | `+Z` |
| Arm extends along | `-X` | `+X` |
| Units | mm | m |

Fix once, in every `<visual>` origin:

```xml
<origin xyz="..." rpy="1.5707963 0 3.1415927"/>      <!-- pi/2 roll, pi yaw -->
<mesh filename="meshes/<link>.stl" scale="0.001 0.001 0.001"/>
```

Rotation matrix `R = R_z(π) · R_x(π/2)` maps CAD `(x,y,z) → URDF (-x, z, y)`.
For a different vendor, replace this with whatever maps their up-axis
to URDF Z and their forward to URDF +X.

### 4. Per-link visual translation

For each child link, find the joint axis position in CAD coords, apply `R`,
negate to get the translation that puts the joint at the URDF link origin.

T3-401S joint axes (derived from spec sheet arm-lengths + bbox):

| Joint | CAD coord (m) | After `R` | Translation in `<visual>` |
|---|---|---|---|
| joint_1 | `(0, 0.18, 0)` | `(0, 0, 0.18)` | `xyz="0 0 -0.18"` |
| joint_2 | `(-0.225, 0.20, 0)` | `(0.225, 0, 0.20)` | `xyz="-0.225 0 -0.20"` |
| shaft top (trimmed, J3=0) | `(-0.40, 0.26, 0)` | `(0.40, 0, 0.26)` | `xyz="-0.40 0 -0.26"` |

For `base_link` (no parent rotation), the translation is just whatever
centers the column on the URDF z-axis.

### 5. Trim meshes that conflict with URDF kinematic abstractions

The CAD shaft is 330 mm; the URDF treats `link_3` as a 100 mm cylinder
hanging from joint_3 (with joint_4 at z=-0.10). Don't change the URDF —
trim the mesh:

```python
trimmed = trimesh.intersections.slice_mesh_plane(
    mesh, plane_normal=[0, -1, 0], plane_origin=[0, cutoff_y, 0]
)
```

Note: `plane_normal` points to the **kept** side. The splitter has this
baked in via `_trim_shaft_in_place`.

### 6. Bridge mesh ↔ URDF dimensional gaps

When URDF spec dimensions differ from CAD (e.g., column 0.500 in URDF vs
0.437 in CAD — intentional headroom choice), add a small primitive visual
to fill the gap rather than scaling/distorting the mesh:

```xml
<visual>
  <origin xyz="0 0 0.4685"/>
  <geometry><cylinder length="0.063" radius="0.05"/></geometry>
</visual>
```

### 7. Keep `<collision>` and `<inertial>` as primitives

PhysX prefers convex primitives. Mesh collisions are slow and prone to
contact-pair instability. Only swap collisions if you need accurate
envelope **and** you've done convex decomposition (CoACD).

### 8. Regenerate USD

```bash
rm -f assets/<robot>/{*.usd,configuration/*.usd,.asset_hash}
env -u VIRTUAL_ENV -u CONDA_PREFIX ~/IsaacLab/isaaclab.sh -p \
    scripts/convert_t3_401_simple_gripper.py
```

The `env -u VIRTUAL_ENV -u CONDA_PREFIX` is required because `~/.bashrc`
auto-activates a venv that otherwise hijacks `isaaclab.sh -p`.

### 9. Verify

Two checks; both must pass.

```bash
# Zero-action stillness — joint velocities must all be 0.0000 across N ticks.
# If anything moves, the conversion is wrong (collision overlap, init-pose
# fight, gravity enabled, etc.) — fix before iterating on visuals.
PYTHONPATH=src env -u VIRTUAL_ENV -u CONDA_PREFIX \
    ~/IsaacLab/isaaclab.sh -p scripts/joint_trace.py --task pick_plant_out --steps 30

# Bounded random walk — should produce smooth motion, no joint-limit thrash.
env -u VIRTUAL_ENV -u CONDA_PREFIX ./scripts/smoke.sh \
    --task pick_plant_out --steps 60 --random --video-out media/check.mp4
```

## Reusing this for a different arm

The skeleton transfers; the per-arm parts that change:

1. **`PART_TO_LINK` map** in `convert_step_to_meshes.py` — depends on
   vendor naming.
2. **Rotation `rpy`** in the URDF visual origins — depends on the
   vendor's up-axis and chosen forward direction.
3. **Per-link translations** — derive from joint axes (spec sheet +
   bbox). Sanity-check by checking total arm reach matches spec.
4. **Mesh trimming** — only needed when the vendor mesh exceeds what
   the URDF kinematic stub represents (typical for prismatic Z shafts).
5. **Gap-fill primitives** — only needed when the URDF intentionally
   diverges from CAD dimensions.

## Common gotchas, in order of frequency

1. **STL is in mm, URDF is in m.** Set `scale="0.001 0.001 0.001"` in
   the `<mesh>` tag.
2. **`isaaclab.sh -p` runs the wrong python** when an outer venv is
   active. Always `env -u VIRTUAL_ENV -u CONDA_PREFIX` it.
3. **`trimesh.slice_mesh_plane` keeps the side the normal points
   toward**, not away from. Easy to invert.
4. **Smoke `--random` shaking ≠ broken conversion.** Always check
   zero-action first; random can drive the IK through SCARA
   singularities.
5. **`cadquery-ocp` works on aarch64; full `cadquery` doesn't** because
   `nlopt` has no aarch64 wheel.
