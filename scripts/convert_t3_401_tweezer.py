"""Convert assets/t3_401_tweezer/t3_401_tweezer.urdf to USD.

Same arm kinematics as the simple-gripper variant; the gripper section
swaps the two paddle fingers for a pair of long pointed tweezer halves
split from cadlib's tweezers_pointed mesh.

Run with:
    ~/IsaacLab/isaaclab.sh -p scripts/convert_t3_401_tweezer.py
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
URDF_PATH = os.path.join(REPO_ROOT, "assets", "t3_401_tweezer", "t3_401_tweezer.urdf")
USD_DIR = os.path.join(REPO_ROOT, "assets", "t3_401_tweezer")
USD_NAME = "t3_401_tweezer.usd"


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
                # B-iter3 (2026-05-18): K=800 for strong clamping (was 100
                # pre-friction-grip), D=1.0 because critical damping for
                # finger mass 0.005 kg is 2*sqrt(K*m) ≈ 4.0; D=5.0 from
                # iter2 was overdamped and may have stalled the close.
                stiffness=800.0,
                damping=1.0,
            ),
        ),
    )
    converter = UrdfConverter(cfg)
    logger.info("USD written to: %s", converter.usd_path)

    # Bake high friction onto the tweezer collisions (default PhysX μ=0.5
    # otherwise). When the friction-combine mode is AVERAGE, plant μ=8.0 +
    # tweezer μ=0.5 averages to 4.25; with MIN combine it'd collapse to
    # 0.5 and the grip would just slip. Force the tweezer side to match
    # the plant's high μ so the contact has high effective friction
    # regardless of which combine mode PhysX/Isaac Lab uses.
    from convert_cad_assets import _bake_friction_into_usd  # type: ignore
    _bake_friction_into_usd(converter.usd_path, (8.0, 6.0, 0.0))


if __name__ == "__main__":
    try:
        main()
    finally:
        _simulation_app.close()
