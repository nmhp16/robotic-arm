"""UR5 + Robotiq 2F-85 pick-and-place env, IK-relative actions, RGB cameras.

Robot is the locally converted USD at ``assets/ur5_2f85/ur5_2f85.usd``
(see ``scripts/convert_ur5_2f85.py``). All six 2F-85 joints are driven in
lockstep by ``BinaryJointPositionActionCfg`` with the URDF mimic
multipliers baked in, since PhysX has no native mimic support.
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
from .robot_cfg import UR5_ROBOTIQ_2F_85_CFG

# tool0 → TCP between fingertips. Sum: ur_to_robotiq adapter (~0.011 m), 2F-85
# base + finger reach (~0.13 m). Verify with FrameTransformer marker; tweak
# if grasps land high or low.
_TCP_Z_OFFSET: float = 0.14

# 2F-85 driver joint travel: ~0.0 (open) → ~0.7 rad (fingers parallel-closed
# on a thin object). Mimic-joint multipliers from the URDF, baked in here so
# all six joints move together.
_GRIPPER_OPEN_RAD: float = 0.0
_GRIPPER_CLOSE_RAD: float = 0.7
_GRIPPER_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
_GRIPPER_MULTIPLIERS = {
    "robotiq_85_left_knuckle_joint": 1.0,
    "robotiq_85_right_knuckle_joint": -1.0,
    "robotiq_85_left_inner_knuckle_joint": 1.0,
    "robotiq_85_right_inner_knuckle_joint": -1.0,
    "robotiq_85_left_finger_tip_joint": -1.0,
    "robotiq_85_right_finger_tip_joint": 1.0,
}
_GRIPPER_OPEN_CMD = {n: _GRIPPER_OPEN_RAD * m for n, m in _GRIPPER_MULTIPLIERS.items()}
_GRIPPER_CLOSE_CMD = {n: _GRIPPER_CLOSE_RAD * m for n, m in _GRIPPER_MULTIPLIERS.items()}


@configclass
class EventCfg:
    """Reset events: small joint jitter on arm only, randomized cube + target."""

    randomize_joint_state = EventTerm(
        func=events.randomize_arm_joints_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )

    randomize_cube = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.35, 0.50), "y": (-0.15, 0.15), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("cube")],
        },
    )

    randomize_target = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.45, 0.65), "y": (-0.20, 0.20), "z": (0.0103, 0.0103), "yaw": (0.0, 0.0)},
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
        self.scene.robot = UR5_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool0",
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
            body_name="tool0",
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

        # Wrist cam mounted at the gripper base, looking down the approach
        # axis. tool0 z+ points "out of the wrist along the tool"; +0.05 m
        # offset places the cam just above the gripper base. Rotation
        # ros-convention (w,x,y,z) rotates the cam to look along +z.
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tool0/wrist_cam",
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
                pos=(0.0, 0.05, 0.05),
                rot=(0.7071, 0.0, 0.7071, 0.0),
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
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.5),
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
