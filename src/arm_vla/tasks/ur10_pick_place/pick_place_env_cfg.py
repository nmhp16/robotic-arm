"""Abstract base env cfg for UR10 pick-and-place.

Defines the scene (table + cube + target + lighting), observation groups, and
termination terms. Robot, actions, events, and cameras are filled in by the
concrete subclass (``pick_place_ur10_env_cfg.py``). Matches the structure of
``isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg`` so mimic and
downstream tooling see a familiar shape.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene: UR10 + table + cube + target zone + ground + dome light.

    ``robot`` and ``ee_frame`` are left MISSING and filled in by the robot-specific
    subclass.
    """

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Pick object (standard Isaac blue block USD, known 5 cm cube).
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0203], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Target zone: same block asset (green), kinematic + gravity-disabled so it
    # stays where we put it on reset and isn't pushed around by the cube.
    # Randomized pose lives in the events cfg.
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0.1, 0.0103], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
            scale=(1.0, 1.0, 0.2),  # thin pad, not a full cube
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )


@configclass
class ActionsCfg:
    """Set by the robot-specific subclass."""

    arm_action: ActionTermCfg = MISSING
    gripper_action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """State-only obs. Camera obs are added by the visuomotor subclass."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_pos = ObsTerm(func=mdp.cube_position)
        cube_quat = ObsTerm(func=mdp.cube_orientation)
        target_pos = ObsTerm(func=mdp.target_position)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Populated by visuomotor subclass — left empty here so state-only
        variants can omit camera rendering entirely."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask annotations for mimic demo augmentation.

        Mimic splits demonstrations into segments at boundaries where these flip
        from False → True. Two subtasks for pick-and-place: ``grasp`` (cube in
        gripper) and ``place`` (cube on target).
        """

        grasp = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )
        place = ObsTerm(
            func=mdp.cube_on_target,
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube")},
    )
    success = DoneTerm(func=mdp.cube_on_target)


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    rewards = None  # pure imitation, no RL reward
    events = None  # filled by robot-specific subclass
    curriculum = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 20.0
        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = 5

        # Bounds needed for the suction gripper physics
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
