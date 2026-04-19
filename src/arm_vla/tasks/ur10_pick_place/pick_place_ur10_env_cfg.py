"""UR10 + long-suction pick-and-place env with IK-relative actions + RGB cameras.

Concrete subclass of ``PickPlaceEnvCfg``. Adds:
- UR10 robot with long-suction gripper (SurfaceGripper)
- Differential IK relative action (6-DoF Δpose) + binary suction action (1-DoF)
- Wrist + third-person cameras at 224×224 RGB (matches OpenVLA input size)
- Events: fixed home joint pose + randomized cube/target starts
- Teleop device config (keyboard today; SpaceMouse slot kept for future)

Suction grippers in Isaac Lab currently require CPU physics — ``device="cpu"``
is set in ``__post_init__``. This is slow but fine for demo collection (few
parallel envs) and for eval rollouts (single env).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import SurfaceGripperCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    SurfaceGripperBinaryActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.universal_robots import UR10_LONG_SUCTION_CFG
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import mdp
from .pick_place_env_cfg import ObservationsCfg, PickPlaceEnvCfg


@configclass
class EventCfg:
    """Reset-time events: home the arm, jitter its joints a bit, randomize cube
    and target poses within a box on the table."""

    init_ur10_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={"default_pose": [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0]},
    )

    randomize_ur10_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )

    randomize_cube = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.40, 0.55), "y": (-0.15, 0.15), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("cube")],
        },
    )

    randomize_target = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.55, 0.70), "y": (-0.20, 0.20), "z": (0.0103, 0.0103), "yaw": (0.0, 0.0)},
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("target")],
        },
    )


@configclass
class VisuomotorObservationsCfg(ObservationsCfg):
    """Adds wrist + third-person RGB terms to the policy obs group."""

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
class UR10PickPlaceEnvCfg(PickPlaceEnvCfg):
    """UR10 long-suction pick-and-place, IK-relative actions, RGB cameras."""

    observations: VisuomotorObservationsCfg = VisuomotorObservationsCfg()

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    # List read by some Isaac Lab data-collection helpers.
    image_obs_list = ["table_cam", "wrist_cam"]

    def __post_init__(self):
        super().__post_init__()

        # Suction grippers require CPU sim as of Isaac Lab 2.3.2
        self.device = "cpu"

        self.events = EventCfg()

        # Robot
        self.scene.robot = UR10_LONG_SUCTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Suction gripper attached to the tool flange
        self.scene.surface_gripper = SurfaceGripperCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/SurfaceGripper",
            max_grip_distance=0.0075,
            shear_force_limit=5000.0,
            coaxial_force_limit=5000.0,
            retry_interval=0.05,
        )

        # EE frame transformer so mdp.ee_frame_pos / ee_frame_quat work.
        # The 0.22 m offset is the long-suction extension along the tool axis.
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.22, 0.0, 0.0]),
                ),
            ],
        )

        # Actions: 6-DoF IK-relative pose + binary suction. Total 7-D, matching
        # OpenVLA's expected action vector shape.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="ee_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.22]),
        )
        self.actions.gripper_action = SurfaceGripperBinaryActionCfg(
            asset_name="surface_gripper",
            open_command=-1.0,
            close_command=1.0,
        )

        # Wrist camera — mounted on the tool flange looking along the approach
        # axis. 224×224 to match OpenVLA's expected input; distance_to_image_plane
        # is kept in case we want to train depth-aware policies later.
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 2.0),
            ),
            # Look forward along the tool axis (+x of ee_link), slightly offset.
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.05),
                rot=(0.5, -0.5, 0.5, -0.5),  # rotate camera Z to align with ee +X
                convention="ros",
            ),
        )

        # Third-person camera — fixed in world frame, over-the-shoulder view of
        # the workspace. Pose chosen to frame the ~0.4–0.7 m x-band of the table.
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
                pos=(1.2, 0.0, 0.6),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),  # look toward table center
                convention="ros",
            ),
        )

        # Antialiasing helps the RGB observations not look like mush at 224².
        self.sim.render.antialiasing_mode = "DLAA"
        self.num_rerenders_on_reset = 3

        # Teleop devices. Keyboard only for now; SpaceMouse slot is kept wired
        # so plugging one in later is a one-line change.
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
