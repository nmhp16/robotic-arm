"""Convert assets/wide_vial/wide_vial.urdf to USD.

Free-floating rigid body — no fix_base, no zero-density link. The
previously shipped wide_vial.usd was generated with fix_base=true and
link_density=0.0; that made the runtime treat it as an articulated
zero-mass body, and PhysX shoved the vial sideways 50 cm at spawn from
a tiny depenetration impulse. This rebuild uses convex_hull collision
(single piece, clean rigid body), real mass + inertia from the URDF,
and articulation disabled — so the vial behaves as a normal physics
prop the OmniPicker can grasp.

Run with:
    ~/IsaacLab/isaaclab.sh -p scripts/convert_wide_vial.py
"""

from __future__ import annotations

import logging
import os

from isaaclab.app import AppLauncher

_app_launcher = AppLauncher(headless=True)
_simulation_app = _app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF_PATH = os.path.join(REPO_ROOT, "assets", "wide_vial", "wide_vial.urdf")
USD_DIR = os.path.join(REPO_ROOT, "assets", "wide_vial")
USD_NAME = "wide_vial.usd"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = UrdfConverterCfg(
        asset_path=URDF_PATH,
        usd_dir=USD_DIR,
        usd_file_name=USD_NAME,
        fix_base=False,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        force_usd_conversion=True,
        collider_type="sdf",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    converter = UrdfConverter(cfg)
    logger.info("USD written to: %s", converter.usd_path)


if __name__ == "__main__":
    try:
        main()
    finally:
        _simulation_app.close()
