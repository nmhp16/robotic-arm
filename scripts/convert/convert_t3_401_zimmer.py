"""Convert assets/t3_401_zimmer/t3_401_zimmer.urdf to USD.

Same arm kinematics as t3_401_simple_gripper / t3_401_tweezer; the gripper
section uses the Zimmer GEP2010IL-00-B parallel-jaw gripper extracted from
its STEP source (via convert_step_to_meshes_zimmer.py).

Run with:
    ~/IsaacLab/isaaclab.sh -p scripts/convert_t3_401_zimmer.py
"""

from __future__ import annotations

import logging
import os

from isaaclab.app import AppLauncher

_app_launcher = AppLauncher(headless=True)
_simulation_app = _app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF_PATH = os.path.join(REPO_ROOT, "assets", "t3_401_zimmer", "t3_401_zimmer.urdf")
USD_DIR = os.path.join(REPO_ROOT, "assets", "t3_401_zimmer")
USD_NAME = "t3_401_zimmer.usd"


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
        convert_mimic_joints_to_normal_joints=False,
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
