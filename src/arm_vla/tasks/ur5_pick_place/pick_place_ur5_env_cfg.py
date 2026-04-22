"""UR5 + simple two-prismatic-finger gripper pick-and-place env.

Robot is the locally converted USD at
``assets/ur5_simple_gripper/ur5_simple_gripper.usd`` (see
``scripts/convert_ur5_simple_gripper.py``). The two prismatic fingers are
driven by ``BinaryJointPositionActionCfg`` — both joints slide 0→0.04 m
symmetrically, so there's no mimic/4-bar to synchronize. Kept compatible
at the controller level (7-D Delta-pose + binary gripper) with a real
Robotiq 2F-85, so policies fine-tuned here transfer to the physical robot.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import events, mdp
from .pick_place_env_cfg import ObservationsCfg, PickPlaceEnvCfg
from .robot_cfg import UR5_SIMPLE_GRIPPER_CFG

# tool0_aligned → TCP between fingertips, along z+ (tool axis after our
# realignment of tool0). Sum: gripper base offset (0.02 m) + finger reach
# (~0.06 m) ≈ 0.08 m. Verify with the FrameTransformer marker and tweak
# if grasps land high/low.
_TCP_Z_OFFSET: float = 0.08

# Simple prismatic gripper travel: 0.0 m (fully open) → 0.04 m (fingers
# touching at the gripper centerline). Both joints commanded identically
# so closure is symmetric — no mimic relationship to enforce.
_GRIPPER_OPEN_M: float = 0.0
_GRIPPER_CLOSE_M: float = 0.04
_GRIPPER_JOINTS = ["finger_left_joint", "finger_right_joint"]
_GRIPPER_OPEN_CMD = {n: _GRIPPER_OPEN_M for n in _GRIPPER_JOINTS}
_GRIPPER_CLOSE_CMD = {n: _GRIPPER_CLOSE_M for n in _GRIPPER_JOINTS}

# Cube spawn range — tuned to keep the arm's elbow-up IK reliably
# reachable. Broader XY hits reach-envelope edges where TCP descent
# saturates ~5 cm above the cube.
_CUBE_X_RANGE: tuple[float, float] = (0.40, 0.50)
_CUBE_Y_RANGE: tuple[float, float] = (-0.08, 0.08)
_CUBE_Z: float = 0.0403                 # 4 cm cube resting on table (z=0)
_CUBE_YAW_RANGE: tuple[float, float] = (-0.5, 0.5)

# Target spawn range — x > _CUBE_X_RANGE.max so cube and target never
# spawn overlapped (would fire the success termination at t=0 and
# produce junk demos).
_TARGET_X_RANGE: tuple[float, float] = (0.55, 0.65)
_TARGET_Y_RANGE: tuple[float, float] = (-0.10, 0.10)
_TARGET_Z: float = 0.0103               # flat pad thickness on table

# Arm joint reset jitter (radians).
_ARM_JITTER_STD: float = 0.02


@configclass
class EventCfg:
    """Reset events: small joint jitter on arm only, randomized cube + target."""

    randomize_joint_state = EventTerm(
        func=events.randomize_arm_joints_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": _ARM_JITTER_STD, "asset_cfg": SceneEntityCfg("robot")},
    )

    randomize_cube = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": _CUBE_X_RANGE,
                "y": _CUBE_Y_RANGE,
                "z": (_CUBE_Z, _CUBE_Z),
                "yaw": _CUBE_YAW_RANGE,
            },
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("cube")],
        },
    )

    randomize_target = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": _TARGET_X_RANGE,
                "y": _TARGET_Y_RANGE,
                "z": (_TARGET_Z, _TARGET_Z),
                "yaw": (0.0, 0.0),
            },
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("target")],
        },
    )


@configclass
class VisuomotorObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        table_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_depth = ObsTerm(
            func=mdp.wrist_center_depth,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "window": 5},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class UR5PickPlaceEnvCfg(PickPlaceEnvCfg):
    """UR5 pick-and-place, IK-relative arm action + binary 2F-85 gripper."""

    observations: VisuomotorObservationsCfg = VisuomotorObservationsCfg()

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    image_obs_list = ["table_cam", "wrist_cam"]

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()
        self.scene.robot = UR5_SIMPLE_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool0_aligned",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, _TCP_Z_OFFSET]),
                ),
            ],
        )

        # Joint regex restricted to arm joints — the gripper is driven by a
        # separate action term below, not by IK.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_joint", "wrist_.*"],
            body_name="tool0_aligned",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, _TCP_Z_OFFSET]),
        )

        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=_GRIPPER_JOINTS,
            open_command_expr=_GRIPPER_OPEN_CMD,
            close_command_expr=_GRIPPER_CLOSE_CMD,
        )

        # Wrist cam mounted on flange (z+ along the tool axis). With
        # convention="ros" the camera optical frame uses z forward, so
        # identity quaternion ⇒ optical z = flange z = approach axis.
        # Placed 10 cm past flange along z+ to clear the gripper body.
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tool0_aligned/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=14.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.18),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="ros",
            ),
        )

        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 4.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.4, 0.0, 1.0),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )

        self.sim.render.antialiasing_mode = "DLAA"
        self.num_rerenders_on_reset = 3

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
