"""Generic base configclasses for parametric pick-and-place tasks.

Defines the *shape* of the env (scene structure, observation groups,
action terms, terminations). All task-specific values — object specs,
camera placements, success thresholds — are filled in by the builder in
``env_cfg.py`` from the task's task.yaml.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene with a robot, ee_frame, table, light — plus pickable + target spawned by the builder."""

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

    # Filled in by the builder from `objects:` in task.yaml.
    pickable: RigidObjectCfg = MISSING
    target: RigidObjectCfg = MISSING


@configclass
class ActionsCfg:
    arm_action: ActionTermCfg = MISSING
    gripper_action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Default observation groups. The builder also adds task-specific
    image obs (table_cam / wrist_cam) onto the policy group."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # The builder fills in `params` so these point at the right scene entities.
        object = ObsTerm(func=mdp.object_obs)
        cube_pos = ObsTerm(func=mdp.object_position)
        cube_quat = ObsTerm(func=mdp.object_orientation)
        target_pos = ObsTerm(func=mdp.object_position)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        # Builder fills `params` to point at the right object names + thresholds.
        grasp = ObsTerm(func=mdp.object_grasped)
        place = ObsTerm(func=mdp.object_on_target)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Default terminations: timeout + object-dropped + success.
    Builder fills in `success` params so the function points at the right
    pickable/target with the right thresholds."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pickable_dropping = DoneTerm(func=mdp.root_height_below_minimum)
    success = DoneTerm(func=mdp.object_on_target)


@configclass
class PickPlaceEnvCfgBase(ManagerBasedRLEnvCfg):
    """Shape-only env cfg. The builder in env_cfg.py subclasses this and
    fills in task-specific values via __post_init__."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 20.0
        self.sim.dt = 0.01  # 100 Hz
        self.sim.render_interval = 5
