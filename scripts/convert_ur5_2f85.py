"""Convert assets/ur5_2f85/ur5_2f85.urdf to USD for Isaac Lab.

The 2F-85 has URDF mimic joints; PhysX has no native mimic, so we let the
importer split them into independent joints we drive in lockstep from the
ArticulationCfg. The stock convert_urdf.py CLI doesn't expose that flag,
so this script wraps UrdfConverterCfg directly.

Run with:
    ~/IsaacLab/isaaclab.sh -p scripts/convert_ur5_2f85.py
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
URDF_PATH = os.path.join(REPO_ROOT, "assets", "ur5_2f85", "ur5_2f85.urdf")
USD_DIR = os.path.join(REPO_ROOT, "assets", "ur5_2f85")
USD_NAME = "ur5_2f85.usd"


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
        fix_base=True,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=True,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=1.0,
            ),
        ),
    )
    converter = UrdfConverter(cfg)
    logger.info("USD written to: %s", converter.usd_path)


if __name__ == "__main__":
    try:
        main()
    finally:
        _simulation_app.close()
