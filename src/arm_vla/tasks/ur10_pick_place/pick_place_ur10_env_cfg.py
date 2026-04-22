"""UR10 pick-and-place env, IK-relative actions, RGB cameras, long-suction gripper.

Uses Isaac Lab's shipped ``UR10_LONG_SUCTION_CFG``, which references a UR10
USD variant that ships with a pre-authored ``SurfaceGripper`` schema. This
is the shortest path to a working end-to-end env in Isaac Lab today; see
README's "Design notes" for the trade-offs.
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

# ee_link flange → suction TCP along the tool axis (UR10 long suction extension).
_TCP_X_OFFSET: float = 0.22


@configclass
class EventCfg:
    """Reset events: home pose, small joint jitter, randomized cube + target."""

    # Matches Isaac Lab's shipped UR10 + long-suction stack config. Wrist_2
    # negated from UR10_LONG_SUCTION_CFG's default so the suction approach
    # axis points down through the table's workspace without flipping
    # through the wrist singularity during descent.
    init_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={"default_pose": [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0]},
    )

    randomize_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )

    randomize_cube = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.40, 0.60), "y": (-0.15, 0.15), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("cube")],
        },
    )

    randomize_target = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.55, 0.75), "y": (-0.20, 0.20), "z": (0.0103, 0.0103), "yaw": (0.0, 0.0)},
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
class UR10PickPlaceEnvCfg(PickPlaceEnvCfg):
    """UR10 pick-and-place, IK-relative arm action + binary long-suction gripper."""

    observations: VisuomotorObservationsCfg = VisuomotorObservationsCfg()

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    image_obs_list = ["table_cam", "wrist_cam"]

    def __post_init__(self):
        super().__post_init__()

        # SurfaceGripper currently requires CPU physics in Isaac Lab 2.3.x.
        self.device = "cpu"
        self.sim.device = "cpu"
        self.sim.use_gpu_pipeline = False
        self.sim.physx.use_gpu = False

        self.events = EventCfg()
        self.scene.robot = UR10_LONG_SUCTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.surface_gripper = SurfaceGripperCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/SurfaceGripper",
            max_grip_distance=0.0075,
            shear_force_limit=5000.0,
            coaxial_force_limit=5000.0,
            retry_interval=0.05,
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[_TCP_X_OFFSET, 0.0, 0.0]),
                ),
            ],
        )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="ee_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=1.0,
            # Matches Isaac Lab's shipped stack IK-rel config for UR10 long
            # suction: ee_link local -Z points down the tool axis at this
            # home pose. Using the stack's value keeps the controlled point
            # and the suction tip aligned.
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -_TCP_X_OFFSET]),
        )

        self.actions.gripper_action = SurfaceGripperBinaryActionCfg(
            asset_name="surface_gripper",
            open_command=-1.0,
            close_command=1.0,
        )

        # Wrist cam mounted near the suction tip, pointing along the tool
        # approach axis. Orientation is calibrated for the UR10 long-suction
        # home pose; expect some drift when the arm is far from home.
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/wrist_cam",
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
                pos=(0.30, 0.0, 0.0),
                rot=(0.7071, 0.0, -0.7071, 0.0),
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
                pos=(1.2, 0.0, 0.6),
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
