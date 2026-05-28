"""Convert vial + leaf-plant STL meshes from test/cad output to USDs the
task env can spawn. Uses URDF wrapping + UrdfConverter (more reliable
physics schema attachment than MeshConverter on aarch64 Isaac builds).

Reads mm-unit STLs produced by build123d in ~/test/cad/output/ and writes
metre-unit USDs into assets/<name>/<name>.usd.

Run with:
    ~/IsaacLab/isaaclab.sh -p scripts/convert_cad_assets.py
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile

from isaaclab.app import AppLauncher

_app_launcher = AppLauncher(headless=True)
_simulation_app = _app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAD_OUTPUT = os.path.expanduser("~/test/cad/output")

# COMPLIANCE-ABOVE-GRASP plant URDF (2026-05-25). A first attempt put the
# spring joint LOW (grasped stem compliant) — but the sim's rigid, position-
# controlled jaws FOLDED the compliant stem (joint hit 2 rad) and slid off
# instead of lifting it: 14% oracle vs 38% rigid. Real force-controlled
# grippers conform; this sim's don't. So put the compliance ABOVE the grasp:
#   - `stem` (root): RIGID lower stem, solid 6mm box collider z 0..0.040,
#     heavy. The gripper grasps+lifts THIS (at z~0.035, oracle heights
#     unchanged) so it lifts as a unit like the 38% rigid plant.
#   - `leaves`: the floppy leafy crown ABOVE the grasp, on a revolute spring
#     joint (`leaf_sway`) at z=0.040. Bends when the arm/fingers brush it and
#     springs back -> the *visible* plant behaves like a real plant (leaves
#     give) WITHOUT the compliance breaking the grasp. Carries the full leaf
#     mesh visual (still a plant). Soft spring set in env_cfg.
# {s}=xy scale (m/mm), {sz}=z scale. The full mesh (base back at world z=0 via
# the -0.040 visual offset) overlaps the rigid stem in its lower region and
# shows the leafy crown above the joint, which is the part that sways.
PLANT_COMPLIANT_URDF = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="stem">
    <visual>
      <origin xyz="0 0 0.020" rpy="0 0 0"/>
      <geometry><cylinder radius="0.003" length="0.040"/></geometry>
      <material name="stem"><color rgba="{r} {g} {b} {a}"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.020" rpy="0 0 0"/>
      <geometry><box size="0.006 0.006 0.040"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.020" rpy="0 0 0"/>
      <mass value="0.0017"/>
      <inertia ixx="2.3e-7" ixy="0" ixz="0" iyy="2.3e-7" iyz="0" izz="1e-8"/>
    </inertial>
  </link>
  <link name="leaves">
    <visual>
      <origin xyz="0 0 -0.040" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_path}" scale="{s} {s} {sz}"/></geometry>
      <material name="leaf"><color rgba="{r} {g} {b} {a}"/></material>
    </visual>
    <!-- NO leaf collider: the 80mm fingers must pass THROUGH the leaf zone
         (z 40-58) to reach the recessed stem at z~35, so any collider here
         blocks the reach-in grasp (verified: leaf box -> 8%, approach-phase
         fails). Visual-only leaves (like the 38% rigid plant) let the fingers
         pass. Trade-off: the leaves then don't physically sway. -->
    <inertial>
      <origin xyz="0 0 0.010" rpy="0 0 0"/>
      <mass value="0.0003"/>
      <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/>
    </inertial>
  </link>
  <joint name="leaf_sway" type="revolute">
    <parent link="stem"/>
    <child link="leaves"/>
    <origin xyz="0 0 0.040" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.9" upper="0.9" effort="2.0" velocity="10.0"/>
    <dynamics damping="0.0" friction="0.0"/>
  </joint>
</robot>
"""

# Per-asset config dicts. Fields:
#   stl_basename      — input STL in test/cad/output/
#   asset_dir         — output directory under assets/
#   mass_kg           — kg
#   kinematic         — True for static props (vial, pedestal)
#   scale             — m per stl unit
#   collider_type     — "convex_hull", "convex_decomposition", or "primitive"
#   friction          — (static, dynamic, restitution); None to skip
#   collision_override — None to use mesh; otherwise a `<collision>` snippet
#                        substituted into the URDF (visual stays as the mesh).
#                        Use this when the convex hull is too "fat" to grasp,
#                        e.g. a thin stem hidden inside a leafy plant mesh.
ASSETS = [
    {
        "stl": "wide_vial",
        "asset_dir": "wide_vial",
        "mass": 0.018,
        "kinematic": True,
        "scale": 0.001,
        "collider": "convex_decomposition",
        "friction": (0.5, 0.4, 0.0),
        # HOLLOW WELL collision (4 walls) instead of the solid convex-decomp
        # (which filled the bore and is why collision was disabled). The well
        # has a ~20mm square bore that holds the plant centred (can't be
        # knocked far) yet admits the 16mm gripper, and 70mm tall walls. The
        # plant stands inside; the long tip reaches down past the rim. Visual
        # stays the round vial mesh. This is the fixture that makes the grasp
        # repeatable (the real "pick out of a vial" task).
        "collision_override": (
            '<collision>\n'  # +X wall
            '      <origin xyz="0.0125 0 0.035" rpy="0 0 0"/>\n'
            '      <geometry><box size="0.005 0.030 0.070"/></geometry>\n'
            '    </collision>\n'
            '    <collision>\n'  # -X wall
            '      <origin xyz="-0.0125 0 0.035" rpy="0 0 0"/>\n'
            '      <geometry><box size="0.005 0.030 0.070"/></geometry>\n'
            '    </collision>\n'
            '    <collision>\n'  # +Y wall
            '      <origin xyz="0 0.0125 0.035" rpy="0 0 0"/>\n'
            '      <geometry><box size="0.020 0.005 0.070"/></geometry>\n'
            '    </collision>\n'
            '    <collision>\n'  # -Y wall
            '      <origin xyz="0 -0.0125 0.035" rpy="0 0 0"/>\n'
            '      <geometry><box size="0.020 0.005 0.070"/></geometry>\n'
            '    </collision>'
        ),
        "color_rgba": (0.72, 0.82, 0.92, 0.12),   # GLASS: low alpha → transparent, so the recessed stem shows THROUGH the vial walls (real vial is glass). The opaque grey vial was hiding the stem from every camera — likely the core perception fix.
    },
    {
        "stl": "leaf_plant",
        "asset_dir": "leaf_plant",
        "mass": 0.002,   # reverted to 2g: the 10g experiment made the IK too slow during MOVE phase (SCARA couldn't traverse to dest vial in 400 sim steps; 5/5 timeouts). Swinging is a smaller problem than no-transport-at-all.
        "kinematic": False,
        "scale": 0.001,
        "collider": "convex_decomposition",
        "friction": (6.0, 5.0, 0.0),   # 3->6: a real plant stem isn't slick; higher mu raises the static-friction limit so the grip holds against gravity through the LIFT (the dominant slip failure). Safe now that the V-groove stops lateral ejection.
        # Saturated green so the policy's vision backbone has a strong colour
        # signal to localize the pickable against the grey table/vial/tray.
        # Vivid low-R/B green: the RTX render desaturates green ~5x (bright dome
        # light + OmniPBR specular wash), so a moderate green came out pale mint
        # (greenest pixel G/R only 1.09). Pushing R,B near 0 keeps it clearly
        # green through the wash so the wrist-cam vision policy gets a strong cue.
        "color_rgba": (0.04, 0.85, 0.06, 1.0),
        # Stem collision proxy for ACTUAL friction grip (2026-05-25).
        # Previously a 10 mm base stub (built for kinematic_attach, which grips
        # by root-to-TCP distance, not contact) — so there was nothing to close
        # on at the grasp height (~z=0.022) and the gripper closed past the
        # visual stem every time (0% honest lift). This replaces it with a
        # clean cylinder over the lower stem region (z 0..0.045) that the spec
        # parallel gripper can actually pinch. Capped at 45 mm (not the full
        # 80 mm) so it does NOT reach up and overlap the wrist (link_4) during
        # DESCEND — that overlap was the original reason the stem collision was
        # removed (IK stall at z~0.106). Radius 4 mm = 8 mm graspable stem.
        # The leafy VISUAL mesh is unchanged — it's still a plant; this is just
        # a standard simple collision proxy for a complex mesh.
        # Stem collision = a thin BOX (6x6x45mm), not a cylinder. A rigid round
        # cylinder squeezed by flat parallel jaws is unstable — it rolls/squirts
        # out (verified: hold-vs-grip-force window is narrow + alignment-
        # sensitive; 80N held 2/41, 120N held 0/64). A real plant stem is
        # COMPLIANT and flattens under the jaws into a stable face-to-face
        # contact, which a thin box models better than a rigid cylinder AND
        # grips reliably regardless of grip force / small misalignment. 6mm
        # wide so the spec 10mm open jaws clear it on descent (~2mm/side).
        # Leafy VISUAL mesh unchanged — still a plant.
        # Box nub for the graspable stem. Origin z=0.058 in the UNSCALED frame
        # so that with the task's [1,1,0.6] z-squash (plant shortened to fit
        # inside the 70mm vial) it lands at world z~0.035 — a reach-in grasp
        # height near the top of the recessed plant (not deep at the vial
        # floor). Box (not cylinder) so flat jaws get a stable face grip.
        # ROUND stem (6mm cyl) for the V-GROOVE grip: a cylinder nests + self-
        # centres in the fingertip V (form-closure), where a box doesn't seat
        # cleanly. (A cylinder in FLAT jaws rolled out — 2/41 — which is why
        # this was a box before; the V-groove fixes that by cradling it.)
        # length 0.040 * spawn z-scale 0.6 = 0.024 m world, centred at world
        # z=0.058*0.6=0.035 (the grasp height). radius 0.003 = 6mm dia.
        # ROUND 6mm stem for the V-groove grip, over the recessed lower-stem
        # region (z 0.058 center, 40mm long → world ~0.035 grasp height at the
        # task's 0.6 z-squash). This is the PROVEN recessed-grasp collider (~80%
        # state PPO). A full-stem collider for a tall visible plant was tried but
        # the tall plant is ungraspable (1.56%, lever-arm lift-slip), so we keep
        # the recessed plant + fixed table_cam for vision (vision_plant_renders_grey
        # memory). Z-form ledge tried twice (33%/25%), never beat 40% V-groove+mu6.
        # MULTI-NODE stem (2026-05-26): smooth 6mm grasp cylinder + 3 collision-
        # only 8mm "leaf nodes" (unscaled z 0.050/0.062/0.074, ×0.6 squash →
        # world 0.030/0.037/0.044) bracketing the grasp valley. FORM-CLOSURE vs
        # lift-slip: an 8mm node can't pass the ~6mm closed jaws, so on any axial
        # slip the next node JAMS — a hard geometric stop at any grasp height
        # (μ-INDEPENDENT, so it sidesteps the friction ceiling that caps the
        # smooth-stem grip ~60%). Unlike the single node+ledge (25/33%, needed
        # ~1mm precise seating), nodes throughout catch regardless of grasp z.
        # Nodes are COLLISION-ONLY; the leafy VISUAL mesh is unchanged so it
        # still reads as a plant (node swellings hidden in the foliage).
        "collision_override": (
            '<collision>\n'
            '      <origin xyz="0 0 0.058" rpy="0 0 0"/>\n'
            '      <geometry>\n'
            '        <cylinder radius="0.003" length="0.040"/>\n'
            '      </geometry>\n'
            '    </collision>\n'
            '    <collision>\n'
            '      <origin xyz="0 0 0.050" rpy="0 0 0"/>\n'
            '      <geometry><cylinder radius="0.004" length="0.005"/></geometry>\n'
            '    </collision>\n'
            '    <collision>\n'
            '      <origin xyz="0 0 0.062" rpy="0 0 0"/>\n'
            '      <geometry><cylinder radius="0.004" length="0.005"/></geometry>\n'
            '    </collision>\n'
            '    <collision>\n'
            '      <origin xyz="0 0 0.074" rpy="0 0 0"/>\n'
            '      <geometry><cylinder radius="0.004" length="0.005"/></geometry>\n'
            '    </collision>\n'
            # Leaf-canopy disk: gives the foliage a thin physical presence so it
            # can contact the vial walls (stops visual leaves from clipping
            # through) and so wall friction resists rotation. Radius 4.5 mm
            # leaves 0.5 mm clearance from the gripper's open-jaw inner face
            # (gripper-x = ±5 mm), so the disk doesn't block DESCEND. With the
            # task z-scale 0.5 the disk lands at world z ≈ 0.035 (mid-leaves)
            # and is 1.5 mm thick — inside the vial bore (r=10 mm) at the
            # grasp height, then clears the rim during LIFT.
            '    <collision>\n'
            '      <origin xyz="0 0 0.070" rpy="0 0 0"/>\n'
            '      <geometry><cylinder radius="0.0045" length="0.003"/></geometry>\n'
            '    </collision>'
        ),
    },
    {
        # COMPLIANT (jointed) plant — same leaf_plant.stl, but built as a
        # 2-link articulation (base + spring-jointed stem_top) so the stem
        # BENDS when touched instead of being knocked away. New asset dir so
        # the rigid leaf_plant.usd that other tasks use stays untouched
        # (CLAUDE.md #2). Only pick_plant_out_of_vial_zimmer.yaml points here.
        # Dims baked into the URDF at FINAL size (no spawn scale on an
        # articulation — scale=0.001 xy, z_squash 0.6 -> 48mm tall plant).
        "stl": "leaf_plant",
        "asset_dir": "leaf_plant_compliant",
        "mass": 0.002,                 # informational; per-link masses are in the URDF
        "kinematic": False,            # FLOATING base (free root) so it lifts out
        "scale": 0.001,
        "z_squash": 0.725,             # mesh -> 58mm: 40mm rigid stem + ~18mm leafy crown above the joint, < 70mm vial
        "collider": "convex_decomposition",
        "friction": (3.0, 2.5, 0.0),
        "color_rgba": (0.15, 0.75, 0.20, 1.0),
        "collision_override": None,    # the multi-link URDF defines its own colliders
        "urdf_text": PLANT_COMPLIANT_URDF,
    },
]

# Minimal one-link URDF wrapper around a mesh. The collision block is
# substituted in from the asset's config — it can be the same mesh
# (default) or an override primitive (e.g. thin cylinder for the stem
# of a leafy plant).
URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="{s} {s} {s}"/>
      </geometry>
      <material name="mat"><color rgba="{r} {g} {b} {a}"/></material>
    </visual>
    {collision_block}
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
  </link>
</robot>
"""

DEFAULT_COLLISION_BLOCK = """<collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="{s} {s} {s}"/>
      </geometry>
    </collision>"""


def _bake_friction_into_usd(usd_path: str, friction: tuple) -> None:
    """Open the converted USD and bind a PhysicsMaterial with the given
    static/dynamic/restitution to all collision prims under the asset.

    Isaac Lab's UsdFileCfg in this build doesn't accept physics_material
    in the spawner, so we set it once at conversion time. The material
    survives subsequent runtime spawns.
    """
    from pxr import Sdf, Usd, UsdPhysics, UsdShade  # type: ignore

    static, dynamic, restitution = friction
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"could not open USD: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    root_path = default_prim.GetPath() if default_prim else Sdf.Path("/")

    # Create a PhysicsMaterial under the asset root.
    mat_path = root_path.AppendChild("physicsMaterial")
    mat_prim = UsdShade.Material.Define(stage, mat_path)
    pm = UsdPhysics.MaterialAPI.Apply(mat_prim.GetPrim())
    pm.CreateStaticFrictionAttr().Set(static)
    pm.CreateDynamicFrictionAttr().Set(dynamic)
    pm.CreateRestitutionAttr().Set(restitution)

    # Bind the material to (a) any prim with CollisionAPI applied, plus
    # (b) any "collisions" Xform — which is what UrdfConverter produces
    # for nested mesh collision references. Material binding propagates
    # to descendants, so binding the parent Xform is enough.
    bound = 0
    for prim in stage.Traverse():
        is_collision = prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetName() == "collisions"
        if not is_collision:
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat_prim, materialPurpose="physics"
        )
        bound += 1

    stage.Save()
    print(f">>>   baked friction (s={static} d={dynamic} r={restitution}) on {bound} collision prim(s)", flush=True)


def _bake_visual_color_into_usd(usd_path: str, color_rgba: tuple) -> None:
    """Author a visible VISUAL material on the converted mesh.

    Isaac Lab's UrdfConverter drops the URDF ``<material><color>`` — it binds
    an EMPTY ``DefaultMaterial`` (shaderId=None, no diffuse) to the visual
    mesh, so the renderer falls back to a flat grey. That silently killed the
    "saturated green so the vision backbone has a strong colour cue" intent:
    the wrist cam saw a grey plant on a grey table (0 green pixels), which is
    a big part of why the vision policy can't localise it.

    Mirror ``_bake_friction_into_usd``: open the geometry layer
    (``configuration/<name>_base.usd``, where the Mesh + DefaultMaterial
    actually live — the top .usd doesn't expose them) and (a) fill every
    Shader as a UsdPreviewSurface with diffuseColor = color, and (b) set the
    mesh ``displayColor`` primvar as a fallback. Re-applied every conversion.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # type: ignore

    r, g, b, a = color_rgba
    base = os.path.join(
        os.path.dirname(usd_path), "configuration",
        os.path.splitext(os.path.basename(usd_path))[0] + "_base.usd",
    )
    target = base if os.path.isfile(base) else usd_path
    stage = Usd.Stage.Open(target)
    if stage is None:
        raise RuntimeError(f"could not open USD for visual bake: {target}")

    # Material depends on alpha. RTX ignores UsdPreviewSurface OPACITY (renders
    # opaque — verified), so for a transparent asset (a<1) we author OmniGlass.mdl
    # (RTX-native glass) so the real vial reads as GLASS and the recessed stem is
    # visible THROUGH its walls (the perception fix). For opaque assets (a==1) we
    # use a UsdPreviewSurface bound strongerThanDescendants — light + RTX honors
    # its diffuseColor (the earlier grey render was a binding-strength loss, not a
    # shader issue; see vision_plant_renders_grey memory).
    UsdGeom.Scope.Define(stage, "/_VisualLooks")
    if a < 1.0:
        mat = UsdShade.Material.Define(stage, "/_VisualLooks/GlassMat")
        shader = UsdShade.Shader.Define(stage, "/_VisualLooks/GlassMat/Shader")
        shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
        shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
        shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.1)        # near-1 → minimal refraction so the stem behind isn't warped
        shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)      # thin vial wall
        shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.0)  # clear, not frosted
        mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        mat_kind = "OmniGlass(MDL)"
    else:
        mat = UsdShade.Material.Define(stage, "/_VisualLooks/PreviewColor")
        shader = UsdShade.Shader.Define(stage, "/_VisualLooks/PreviewColor/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        mat_kind = "UsdPreviewSurface"

    meshes = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI(prim).Bind(
                mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
            UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(r, g, b)])  # viewport fallback
            meshes += 1

    stage.Save()
    print(
        f">>>   baked {mat_kind} visual color rgba={color_rgba} on {meshes} mesh(es) in {os.path.basename(target)}",
        flush=True,
    )


def convert(
    stl_basename: str,
    asset_dir_name: str,
    mass: float,
    kinematic: bool,
    scale: float,
    collider_type: str = "convex_hull",
    friction: tuple | None = None,
    collision_override: str | None = None,
    color_rgba: tuple = (0.5, 0.6, 0.5, 1.0),
    urdf_override: str | None = None,
    z_squash: float = 1.0,
) -> None:
    """Wrap one STL in a minimal URDF and convert it to USD under ``assets/<asset_dir_name>/``.

    Writes ``assets/<asset_dir_name>/<asset_dir_name>.usd`` plus a copy of
    the STL under ``meshes/`` so the URDF can reference it by relative path.

    Args:
        stl_basename: STL filename (without extension) inside ``CAD_OUTPUT``.
        asset_dir_name: Subdirectory under ``assets/`` and the URDF link name.
        mass: Link mass in kg.
        kinematic: If True, sets ``fix_base`` on the URDF importer — pin the
            asset in place. Use for static props (vials, pedestals).
        scale: m per STL unit. STLs from build123d are mm, so pass ``0.001``.
        collider_type: One of ``"convex_hull"``, ``"convex_decomposition"``,
            ``"primitive"``. Decomposition is heavier but lets concave geometry
            (e.g. a vial cavity) collide accurately.
        friction: ``(static, dynamic, restitution)`` baked into the USD via a
            PhysicsMaterial. ``None`` skips the bake — Isaac Lab will use its
            default material at spawn time.
        collision_override: Raw URDF ``<collision>...</collision>`` snippet to
            substitute for the default mesh collider. Use when the mesh's
            convex hull is unsuitable (e.g. a thin stem hidden inside leaves);
            the visual mesh stays untouched.
        color_rgba: Visual material RGBA in [0, 1].

    Raises:
        FileNotFoundError: If ``CAD_OUTPUT/<stl_basename>.stl`` is missing.
    """
    stl_path = os.path.join(CAD_OUTPUT, f"{stl_basename}.stl")
    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"missing STL: {stl_path}")

    usd_dir = os.path.join(REPO_ROOT, "assets", asset_dir_name)
    mesh_dir = os.path.join(usd_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # Copy the STL into the asset dir so the URDF can reference it via a
    # relative path the converter resolves consistently.
    local_mesh = os.path.join(mesh_dir, f"{asset_dir_name}.stl")
    shutil.copy2(stl_path, local_mesh)

    mesh_rel = f"meshes/{asset_dir_name}.stl"
    r, g, b, a = color_rgba
    if urdf_override is not None:
        # Multi-link (articulated) asset: caller supplies the full URDF. {sz}
        # is the z-axis mesh scale (z-squash baked in) vs {s} for x/y.
        urdf_text = urdf_override.format(
            name=asset_dir_name, mesh_path=mesh_rel,
            s=scale, sz=scale * z_squash, r=r, g=g, b=b, a=a,
        )
    else:
        if collision_override is not None:
            collision_block = collision_override
        else:
            collision_block = DEFAULT_COLLISION_BLOCK.format(mesh_path=mesh_rel, s=scale)
        urdf_text = URDF_TEMPLATE.format(
            name=asset_dir_name,
            mesh_path=mesh_rel,
            s=scale,
            mass=mass,
            collision_block=collision_block,
            r=r, g=g, b=b, a=a,
        )
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", dir=usd_dir, delete=False) as f:
        urdf_path = f.name
        f.write(urdf_text)

    try:
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=usd_dir,
            usd_file_name=f"{asset_dir_name}.usd",
            fix_base=kinematic,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=False,
            force_usd_conversion=True,
            collider_type=collider_type,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        )
        converter = UrdfConverter(cfg)
        print(f">>> {asset_dir_name}: {converter.usd_path}", flush=True)
        if friction is not None:
            _bake_friction_into_usd(converter.usd_path, friction)
        _bake_visual_color_into_usd(converter.usd_path, color_rgba)
    finally:
        os.unlink(urdf_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    print(f">>> CONVERT_CAD_ASSETS START. CAD_OUTPUT={CAD_OUTPUT}", flush=True)
    for a in ASSETS:
        print(
            f">>> {a['stl']} -> {a['asset_dir']}/  "
            f"(mass={a['mass']}, kinematic={a['kinematic']}, collider={a['collider']}, "
            f"friction={a['friction']}, override={a['collision_override'] is not None})",
            flush=True,
        )
        try:
            convert(
                stl_basename=a["stl"],
                asset_dir_name=a["asset_dir"],
                mass=a["mass"],
                kinematic=a["kinematic"],
                scale=a["scale"],
                collider_type=a["collider"],
                friction=a["friction"],
                collision_override=a["collision_override"],
                color_rgba=a.get("color_rgba", (0.5, 0.6, 0.5, 1.0)),
                urdf_override=a.get("urdf_text"),
                z_squash=a.get("z_squash", 1.0),
            )
        except FileNotFoundError as e:
            print(f">>> SKIP {a['stl']}: {e}", flush=True)
        except Exception as e:
            usd_path = os.path.join(REPO_ROOT, "assets", a['asset_dir'], f"{a['asset_dir']}.usd")
            if os.path.isfile(usd_path):
                print(f">>> {a['stl']}: USD written despite '{type(e).__name__}: {e}'", flush=True)
                continue
            print(f">>> ERROR {a['stl']}: {type(e).__name__}: {e}", flush=True)
            raise


if __name__ == "__main__":
    try:
        main()
    finally:
        _simulation_app.close()
