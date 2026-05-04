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
        "collision_override": None,
    },
    {
        "stl": "leaf_plant",
        "asset_dir": "leaf_plant",
        "mass": 0.002,
        "kinematic": False,
        "scale": 0.001,
        "collider": "convex_decomposition",
        "friction": (3.0, 2.5, 0.0),
        # Override collision with a stout cylinder for the stem. The
        # 4 mm-radius collision is fatter than the 1.5 mm visual stem,
        # but PhysX needs enough contact area for the tweezer pinch to
        # transmit normal force without slipping — at sub-2 mm we get
        # glancing contacts. Plant still fits cleanly through the 27 mm
        # vial mouth.
        "collision_override": (
            '<collision>\n'
            '      <origin xyz="0 0 0.040" rpy="0 0 0"/>\n'
            '      <geometry>\n'
            '        <cylinder length="0.080" radius="0.0040"/>\n'
            '      </geometry>\n'
            '    </collision>'
        ),
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
      <material name="grey"><color rgba="0.5 0.6 0.5 1.0"/></material>
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


def convert(
    stl_basename: str,
    asset_dir_name: str,
    mass: float,
    kinematic: bool,
    scale: float,
    collider_type: str = "convex_hull",
    friction: tuple | None = None,
    collision_override: str | None = None,
) -> None:
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
