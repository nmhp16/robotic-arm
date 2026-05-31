"""Inspect the COMPOSED leaf_plant.usd as the renderer sees it: descend through
instance proxies (Isaac marks imported geometry instanceable, so a plain
Traverse misses it) and report each visual Mesh's resolved displayColor + the
diffuseColor of its bound material's surface shader.

  env -u VIRTUAL_ENV -u CONDA_PREFIX ~/IsaacLab/isaaclab.sh -p scripts/inspect_plant_color.py
"""
from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True).app
from pxr import Usd, UsdGeom, UsdShade

USD = "assets/leaf_plant/leaf_plant.usd"
stage = Usd.Stage.Open(USD, Usd.Stage.LoadAll)
print(f"=== {USD} (instance proxies included) ===")
rng = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
nmesh = 0
for prim in rng:
    if prim.IsA(UsdGeom.Mesh):
        nmesh += 1
        gp = UsdGeom.Gprim(prim)
        dc = gp.GetDisplayColorAttr().Get()
        # resolve bound material -> surface shader -> diffuseColor
        binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        diff = None
        matpath = None
        if binding:
            matpath = binding.GetPath().pathString
            src = binding.GetSurfaceOutput().GetConnectedSource()
            if src:
                sh = UsdShade.Shader(src[0].GetPrim())
                di = sh.GetInput("diffuseColor")
                diff = tuple(di.Get()) if di and di.Get() is not None else None
        print(f"  Mesh {prim.GetPath()}")
        print(f"     displayColor={[tuple(round(c,3) for c in v) for v in dc] if dc else None}")
        print(f"     boundMaterial={matpath}  diffuseColor={diff}")
print(f"  ({nmesh} visual mesh(es))")
