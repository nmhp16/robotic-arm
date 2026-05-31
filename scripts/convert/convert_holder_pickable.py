"""Convert assets/holder/holder.urdf to a free-floating rigid USD.

Companion to scripts/convert_holder.py — that one produces a kinematic
fixed-base holder for use as a SCENE FIXTURE (vial-pick task). This one
produces a free-floating dynamic rigid body for use as a PICKABLE in
the holder-move task. Same mesh, different physics.
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
URDF_PATH = os.path.join(REPO_ROOT, "assets", "holder", "holder.urdf")
USD_DIR = os.path.join(REPO_ROOT, "assets", "holder")
USD_NAME = "holder_pickable.usd"


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
        # Free-floating rigid body, NOT a fixed-base scene fixture.
        fix_base=False,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        force_usd_conversion=True,
        # SDF gives clean contact normals on the laid-down rack walls when
        # the gripper closes on the rim — important for stable grip.
        collider_type="sdf",
        # No joints in this asset (single rigid link), but the cfg still
        # requires a joint_drive block to be populated. Stub values.
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
