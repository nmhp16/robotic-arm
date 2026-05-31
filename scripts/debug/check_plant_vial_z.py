"""Is the plant recessed below the vial rim (so a top-down wrist cam can't see
it)? Compute world-aligned Z extents of plant vs vial at their task spawn Z.

  env -u VIRTUAL_ENV -u CONDA_PREFIX ~/IsaacLab/isaaclab.sh -p scripts/check_plant_vial_z.py
"""
from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True).app
from pxr import Usd, UsdGeom, Gf


def zext(path, scale_z=1.0):
    stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    zmin, zmax = rng.GetMin()[2] * scale_z, rng.GetMax()[2] * scale_z
    return zmin, zmax


# task: plant scale [1,1,1.0] spawn z=0.0 ; vial scale [1,1,0.6] spawn z=0.0
pz0, pz1 = zext("assets/leaf_plant/leaf_plant.usd", scale_z=1.0)
vz0, vz1 = zext("assets/wide_vial/wide_vial.usd", scale_z=1.0)
PLANT_SPAWN_Z, VIAL_SPAWN_Z = 0.0, 0.0
p_lo, p_hi = pz0 + PLANT_SPAWN_Z, pz1 + PLANT_SPAWN_Z
v_lo, v_hi = vz0 + VIAL_SPAWN_Z, vz1 + VIAL_SPAWN_Z
print(f"RESULT plant world Z: [{p_lo*1000:.1f}, {p_hi*1000:.1f}] mm  (height {(pz1-pz0)*1000:.1f} mm @0.6 scale)")
print(f"RESULT vial  world Z: [{v_lo*1000:.1f}, {v_hi*1000:.1f}] mm  (rim at {v_hi*1000:.1f} mm)")
gap = (v_hi - p_hi) * 1000
if gap > 0:
    print(f"RESULT plant top is {gap:.1f} mm BELOW the vial rim -> RECESSED, occluded from a top-down wrist cam")
else:
    print(f"RESULT plant top is {-gap:.1f} mm ABOVE the vial rim -> visible above rim")
