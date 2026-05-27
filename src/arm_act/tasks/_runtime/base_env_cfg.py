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
from isaaclab.managers import RewardTermCfg as RewTerm
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

    # Filled in by the builder from `objects:` in task.yaml. The pickable may
    # be a rigid body OR an articulation (the compliant jointed plant) — both
    # expose .data.root_pos_w so obs/oracle/metrics are agnostic.
    pickable: RigidObjectCfg | ArticulationCfg = MISSING
    target: RigidObjectCfg = MISSING


@configclass
class ActionsCfg:
    arm_action: ActionTermCfg = MISSING
    gripper_action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation groups.

    Two layouts coexist on the same env_cfg:

    - **IL pipelines** (oracle, mimic, dagger, ACT, SmolVLA) read the
      ``policy`` group — robot state + object ground truth bundled
      together, matching the demo HDF5 schema.

    - **RL pipelines** (``Isaac-…-RL-v0`` / ``-RL-Vision-v0``) use the
      asymmetric ``proprio`` + ``privileged`` split. The actor sees only
      ``proprio`` (deployable on the real robot); the critic sees
      ``proprio + privileged`` (sim ground truth, training-time-only,
      stable value learning). The ``_apply_reward_params`` helper strips
      ``policy`` and instantiates the split for RL envs only.

    The split exists because the bundled ``policy`` group leaks object
    ground truth to the actor — that's fine for IL (demos record what
    the oracle saw) but fatal for sim-to-real RL: the real robot has no
    plant-pose sensor."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # The builder fills in `params` so these point at the right scene entities.
        object = ObsTerm(func=mdp.object_obs)
        pickable_pos = ObsTerm(func=mdp.object_position)
        pickable_quat = ObsTerm(func=mdp.object_orientation)
        target_pos = ObsTerm(func=mdp.object_position)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class ProprioCfg(ObsGroup):
        """Robot-state-only obs (deployable on the real arm).

        Everything in this group can be read from real-robot encoders +
        forward kinematics + a known target pose at machine startup. The
        RL actor reads ONLY this group, so the trained policy can be
        deployed without any external object-perception system.

        ``target_pos`` is included because real fixtures have known fixed
        poses (e.g. the vial slot in a holder rack) — that's not sim
        cheating, it's calibrated workcell geometry."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        target_pos = ObsTerm(func=mdp.object_position)

        def __post_init__(self):
            # enable_corruption=True lets per-term ``noise`` configs fire.
            # We attach GaussianNoiseCfg in env_cfg builder for sim-to-real
            # robustness against real encoder noise.
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Sim-only ground-truth obs for the asymmetric critic.

        The pickable's pose is sampled from PhysX state — there's no
        equivalent measurement on the real robot. Including it in the
        critic obs accelerates value-function learning (the critic
        always knows exactly where the plant is, regardless of how well
        the actor's vision encoder is doing) without contaminating the
        deployable actor."""

        object = ObsTerm(func=mdp.object_obs)
        pickable_pos = ObsTerm(func=mdp.object_position)
        pickable_quat = ObsTerm(func=mdp.object_orientation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        # Empty by default — IL pipelines add the camera obs to the
        # `policy` group instead. Vision RL re-routes the wrist_cam
        # obs here via env_cfg's _apply_reward_params(enable_vision=True)
        # so the CNN actor-critic can route the 2D image through its
        # encoder without colliding with the 1D state obs.
        wrist_cam: ObsTerm | None = None

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
    # proprio + privileged default to None; the RL env_cfg builder
    # instantiates them and strips `policy`. IL paths leave them None
    # so ObservationManager skips them entirely.
    proprio: ProprioCfg | None = None
    privileged: PrivilegedCfg | None = None
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
class RewardsCfg:
    """Shaped rewards for closed-loop RL fine-tuning.

    Two modes, picked via task yaml:

      Full pick-and-place RL (default):
        - approach_pickable + approach_target + grasp_bonus +
          success_bonus + action_l2_penalty
        - lift_bonus weight = 0 (disabled)
        - Use when the full task — pick, transport, release — must be
          learned end-to-end.

      Factored pick-only RL:
        - approach_pickable + grasp_bonus + lift_bonus +
          action_l2_penalty
        - approach_target + success_bonus weights = 0 (disabled)
        - Set termination to ``mdp.pickable_lifted`` instead of
          ``mdp.object_on_target``. RL ends as soon as the policy has
          grasped + lifted the pickable; a scripted controller handles
          the deterministic transport-to-target phase.
        - Use when the target pose is known at runtime (e.g. fixed vial
          location) — much shorter training horizon.

    Per-task weight overrides go through the ``rewards:`` block in the
    task yaml. Use 0.0 to disable a term.
    """

    approach_pickable = RewTerm(func=mdp.reward_tcp_to_pickable, weight=1.0)
    approach_target = RewTerm(func=mdp.reward_pickable_to_target, weight=1.0)
    grasp_bonus = RewTerm(func=mdp.reward_grasp_at_pickable, weight=1.0)
    lift_bonus = RewTerm(func=mdp.reward_lift_to_height, weight=0.0)
    lift_shaping = RewTerm(func=mdp.reward_lift_progress_dense, weight=0.0)
    success_bonus = RewTerm(func=mdp.reward_object_on_target, weight=50.0)
    action_l2_penalty = RewTerm(func=mdp.reward_action_penalty, weight=-0.001)
    # Penalty for DROPPING the plant (root fell below minimum_height). Weight set
    # negative via the task yaml `rewards:` block. A dropped leaf = contamination
    # in the real lab, so this must dominate: dropping should be worse than failing
    # to complete, so the policy learns to hold or abort rather than drop.
    drop_penalty = RewTerm(func=mdp.reward_pickable_dropped, weight=0.0)


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
