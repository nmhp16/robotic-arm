"""UR5e + Robotiq 2F-85 pick-and-place env with IK-rel actions + RGB cameras.

Concrete subclass of ``PickPlaceEnvCfg``:
- UR5e arm with Robotiq 2F-85 parallel-jaw gripper
- IK-relative 6-DoF Δpose action + binary gripper action
- Wrist + third-person 224×224 RGB cameras (OpenVLA input size)
- Events: fixed home joint pose + randomized cube/target starts
- Keyboard teleop device wired in
- Runs on GPU (parallel-jaw, unlike the suction-forced-CPU path)
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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from arm_vla.assets.ur5e_cfg import (
    UR5E_GRIPPER_CLOSE_VAL,
    UR5E_GRIPPER_JOINT_NAME,
    UR5E_GRIPPER_OPEN_VAL,
    UR5E_ROBOTIQ_2F_85_CFG,
)

from . import mdp
from .pick_place_env_cfg import ObservationsCfg, PickPlaceEnvCfg


@configclass
class EventCfg:
    """Reset-time events: home the arm, jitter joints slightly, randomize cube
    and target within a workspace box on the table."""

    init_ur5_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={"default_pose": [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0]},
    )

    randomize_ur5_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )

    randomize_cube = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            # UR5e has a smaller reach than UR10 — tighter workspace box.
            "pose_range": {"x": (0.35, 0.50), "y": (-0.12, 0.12), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("cube")],
        },
    )

    randomize_target = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.45, 0.60), "y": (-0.15, 0.15), "z": (0.0103, 0.0103), "yaw": (0.0, 0.0)},
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

    policy: PolicyCfg = PolicyCfg()


@configclass
class UR5PickPlaceEnvCfg(PickPlaceEnvCfg):
    """UR5e + Robotiq 2F-85 pick-and-place, IK-rel actions, RGB cameras."""

    observations: VisuomotorObservationsCfg = VisuomotorObservationsCfg()

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    image_obs_list = ["table_cam", "wrist_cam"]

    # Exposed to mdp helpers via ``env.cfg.<name>``.
    gripper_joint_name: str = UR5E_GRIPPER_JOINT_NAME
    gripper_open_val: float = UR5E_GRIPPER_OPEN_VAL
    gripper_close_val: float = UR5E_GRIPPER_CLOSE_VAL

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()

        # Robot with 2F-85 parallel-jaw gripper. Parallel-jaw works on GPU —
        # no device=cpu forcing like the suction variant.
        self.scene.robot = UR5E_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # EE frame transformer. ``tool0`` is UR's flange frame; we offset along
        # the tool z-axis by the 2F-85 fingertip offset (~0.155 m) so the TCP
        # tracked by ee_frame sits where the fingers actually grasp.
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool0",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.155]),
                ),
            ],
        )

        # Arm action: IK-relative 6-DoF Δpose. ``tool0`` is the body we drive;
        # the body_offset puts the IK target at the fingertip center so Δpose
        # is expressed in that useful frame.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*_joint", "elbow_joint", "wrist_.*_joint"],
            body_name="tool0",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.155]),
        )

        # Gripper action: binary, drives ``finger_joint`` between open/close
        # values defined in ur5e_cfg.py.
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[self.gripper_joint_name],
            open_command_expr={self.gripper_joint_name: self.gripper_open_val},
            close_command_expr={self.gripper_joint_name: self.gripper_close_val},
        )

        # Wrist camera — mounted on tool0 looking along the approach axis
        # (+z of tool0). 224×224 for OpenVLA input.
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tool0/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.03, 2.0),
            ),
            # Camera z looks along tool0's +z (approach axis).
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.0),
                rot=(0.0, 0.707, 0.707, 0.0),
                convention="ros",
            ),
        )

        # Third-person camera at a fixed world pose framing the UR5 workspace.
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

        # Keyboard teleop. SpaceMouse slot intentionally omitted — user only has
        # keyboard. Plugging one in later = one extra Se3SpaceMouseCfg entry.
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
