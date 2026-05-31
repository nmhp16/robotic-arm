"""Convert assets/ur5_omnipicker/ur5_omnipicker.urdf to USD.

UR5 6-axis arm + AgiBot OmniPicker parallel-jaw gripper. Two independent
prismatic finger joints (no URDF mimic); ``robot_cfg.py`` drives both
with the same command via BinaryJointPositionActionCfg, matching the
t3_401_simple_gripper pattern.

Run with:
    ~/IsaacLab/isaaclab.sh -p scripts/convert_ur5_omnipicker.py
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
URDF_PATH = os.path.join(REPO_ROOT, "assets", "ur5_omnipicker", "ur5_omnipicker.urdf")
USD_DIR = os.path.join(REPO_ROOT, "assets", "ur5_omnipicker")
USD_NAME = "ur5_omnipicker.usd"


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
