"""Convert assets/ur5_simple_gripper/ur5_simple_gripper.urdf to USD.

Simpler than the 2F-85 variant — just two prismatic fingers, no mimic
joints, no 4-bar linkage — so ``convert_mimic_joints_to_normal_joints``
is irrelevant here. All joints are already independent.

Run with:
    ~/IsaacLab/isaaclab.sh -p scripts/convert_ur5_simple_gripper.py
"""

from __future__ import annotations

import os

from isaaclab.app import AppLauncher

_app_launcher = AppLauncher(headless=True)
_simulation_app = _app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF_PATH = os.path.join(REPO_ROOT, "assets", "ur5_simple_gripper", "ur5_simple_gripper.urdf")
USD_DIR = os.path.join(REPO_ROOT, "assets", "ur5_simple_gripper")
USD_NAME = "ur5_simple_gripper.usd"


def main() -> None:
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
    print(f"USD written to: {converter.usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        _simulation_app.close()
