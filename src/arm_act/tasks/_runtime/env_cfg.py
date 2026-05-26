"""Build a parametric env_cfg class from a task.yaml spec.

``build_env_cfg(spec, class_name)`` returns a freshly-decorated
configclass that subclasses ``PickPlaceEnvCfgBase`` and applies the
spec in ``__post_init__``: spawns the pickable + target objects, places
the cameras, wires the IK + gripper actions, and fills in the success /
grasp-check / subtask thresholds.

The runtime registry calls this once per task at import time, then
``gym.register`` exposes the generated class as the env_cfg_entry_point.
"""

from __future__ import annotations

import os
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from isaaclab.envs.mdp import randomize_rigid_body_mass, randomize_rigid_body_material
from isaaclab.utils.noise import GaussianNoiseCfg

from . import events as events_mod
from . import mdp
from .base_env_cfg import PickPlaceEnvCfgBase
from .robot_cfg import build_robot_cfg

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def build_env_cfg(
    spec: dict[str, Any],
    class_name: str,
    enable_rewards: bool = False,
    enable_vision: bool = False,
) -> type:
    """Return a configclass-decorated subclass of PickPlaceEnvCfgBase
    with all task-specific fields baked in from ``spec``.

    Args:
        enable_rewards: when True, attach a RewardsCfg (shaped rewards for
            RL fine-tuning). The IL/oracle/mimic/dagger pipeline does NOT
            need rewards, so this defaults to False to keep those flows
            untouched.
        enable_vision: when True, keep the wrist_cam alive in the RL env
            instead of stripping it. Pair with a CNN-based PPO config
            (VisionPPORunnerCfg). Has no effect when enable_rewards is
            False (IL path always keeps all cameras).
    """
    from .base_env_cfg import RewardsCfg

    class _Cfg(PickPlaceEnvCfgBase):
        def __post_init__(self):
            super().__post_init__()
            _apply_spec(self, spec)
            if enable_rewards:
                self.rewards = RewardsCfg()
                _apply_reward_params(self, spec, enable_vision=enable_vision)

    _Cfg.__name__ = class_name
    _Cfg.__qualname__ = class_name
    return configclass(_Cfg)


def _apply_reward_params(
    env_cfg: PickPlaceEnvCfgBase, spec: dict[str, Any], enable_vision: bool = False
) -> None:
    """Fill the per-reward-term ``params`` dict from the task spec.

    Robot/task agnostic: pulls ``gripper_driver_joint`` and
    ``gripper_closed_threshold`` from ``spec["robot"]`` (the same keys
    used by the IL pipeline), pulls success thresholds from the
    terminations block (kept in sync with the env's success_term).

    Per-task weight overrides:
      Add a ``rewards:`` block to the task yaml with any subset of:
        rewards:
          approach_pickable: 1.0
          approach_target: 1.0
          grasp_bonus: 1.0
          success_bonus: 50.0
          action_l2_penalty: -0.001
      Missing keys keep the class default. Use 0.0 to disable a term."""
    from isaaclab.managers import SceneEntityCfg

    driver_joint = spec["robot"]["gripper_driver_joint"]
    closed_threshold = float(spec["robot"]["gripper_closed_threshold"])
    # success_bonus rewards "object_on_target" — keep its params bound to
    # the placement geometry from the task yaml regardless of which
    # termination mode (placed/lifted) is active. Reading directly from
    # the spec means the success_bonus is still meaningful even if the
    # user has switched terminations.success to pickable_lifted.
    placement_params = dict(
        object_cfg=SceneEntityCfg("pickable"),
        target_cfg=SceneEntityCfg("target"),
        xy_threshold=float(spec["success"]["xy_threshold"]),
        height_threshold=float(spec["success"]["height_threshold"]),
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    )
    rew = env_cfg.rewards

    rew.approach_pickable.params = dict(
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg("pickable"),
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    )
    rew.approach_target.params = dict(
        object_cfg=SceneEntityCfg("pickable"),
        target_cfg=SceneEntityCfg("target"),
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    )
    rew.grasp_bonus.params = dict(
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg("pickable"),
        # Tighter than the historical 0.06 — at 0.03 (3 cm) the grasp
        # bonus only fires inside / barely outside kin_attach's 2 cm
        # capture range, so the policy can't park outside the capture
        # zone and farm +reward by closing-and-camping. Tasks that
        # still want the looser bonus can override via the rewards
        # block (not currently parsed; would need extra plumbing).
        diff_threshold=0.03,
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    )
    rew.success_bonus.params = placement_params
    # Lift bonus — read lift_height from the task yaml's rewards block
    # (default 0.10 m). Only meaningful if the user enables this term
    # via a non-zero weight in the rewards: block.
    rewards_block = spec.get("rewards") if isinstance(spec.get("rewards"), dict) else {}
    lift_height = float(rewards_block.get("lift_height", 0.10))
    min_stable_steps = int(rewards_block.get("lift_stable_steps", 10))
    rew.lift_bonus.params = dict(
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg("pickable"),
        lift_height=lift_height,
        diff_threshold=0.06,
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    )
    # Dense partial-lift shaping (off by default; enable via a non-zero
    # lift_shaping weight in the rewards: block). Bridges the grasp -> lift
    # gradient gap that the binary lift_bonus leaves open under friction grip.
    rew.lift_shaping.params = dict(
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg("pickable"),
        lift_height=lift_height,
        diff_threshold=0.06,
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    )
    # Re-wire the lifted-mode reward stack when the task selects
    # success_termination=lifted. Two changes from the default
    # (placed-mode) wiring:
    #
    # 1. lift_bonus → reward_lift_hold_progress. The default
    #    reward_lift_to_height fires +1 per step on every step the
    #    lift condition holds, regardless of stability. PPO reward-
    #    hacks this by firing-and-dropping repeatedly. The hold-
    #    progress reward scales with how long the policy holds, so
    #    a 5-step hold gives 1+2+3+4+5=15 while 5 flicks give 5 —
    #    the policy is paid more to hold.
    #
    # 2. success_bonus → reward_terminal_stable_lift. The default
    #    success_bonus fires on placement (object_on_target), which
    #    never happens in lifted mode. Replace it with a one-shot
    #    bonus on the success step — strong terminal incentive
    #    that, combined with the small per-step hold-progress
    #    reward, makes holding strictly better than flicking.
    success_mode = str(rewards_block.get("success_termination", "placed")).lower()
    if success_mode == "lifted":
        rew.lift_bonus.func = mdp.reward_lift_hold_progress
        rew.lift_bonus.params = dict(
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg("pickable"),
            lift_height=lift_height,
            diff_threshold=0.06,
            driver_joint=driver_joint,
            closed_threshold=closed_threshold,
        )
        rew.success_bonus.func = mdp.reward_terminal_stable_lift
        rew.success_bonus.params = dict(
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg("pickable"),
            lift_height=lift_height,
            diff_threshold=0.06,
            driver_joint=driver_joint,
            closed_threshold=closed_threshold,
            min_stable_steps=min_stable_steps,
        )
    # action_l2_penalty needs no params.

    # Per-task weight overrides from the task yaml. Non-weight keys
    # (lift_height, success_termination) are consumed elsewhere and
    # silently skipped here.
    NON_WEIGHT_KEYS = {"lift_height", "lift_stable_steps", "success_termination"}
    weights = (spec.get("rewards") or {})
    for term_name, w in weights.items():
        if term_name in NON_WEIGHT_KEYS:
            continue
        term = getattr(rew, term_name, None)
        if term is None:
            raise KeyError(
                f"rewards.{term_name!r} in task spec doesn't match any "
                f"reward term in RewardsCfg (known: approach_pickable, "
                f"approach_target, grasp_bonus, lift_bonus, lift_shaping, "
                f"success_bonus, action_l2_penalty)"
            )
        term.weight = float(w)

    # --- Camera / image-obs handling for RL ----------------------------------
    # Two modes:
    #
    # 1. State-only (enable_vision=False, default): strip ALL cameras and
    #    image obs. The default actor-critic in DefaultPPORunnerCfg is a
    #    state-only MLP. Keeping cameras spawned at RL time costs one
    #    CameraCfg per env (= one Vulkan viewport), which blows past the
    #    GPU's descriptor pool (~1024 viewports) at large num_envs.
    #
    # 2. Vision (enable_vision=True): keep wrist_cam but strip
    #    table_cam. The CNN actor-critic in VisionPPORunnerCfg reads from
    #    the wrist_cam to learn vision-conditioned picking. We keep only
    #    one camera per env to halve the descriptor cost.
    cameras = spec.get("cameras", {}) or {}
    pol = env_cfg.observations.policy
    # Which cameras to strip: ALL in state-only mode; only non-wrist in
    # vision mode.
    cams_to_strip = list(cameras.keys()) if not enable_vision else [
        c for c in cameras if c != "wrist_cam"
    ]
    for cam_name in cams_to_strip:
        if hasattr(pol, cam_name) and getattr(pol, cam_name) is not None:
            setattr(pol, cam_name, None)
        if hasattr(env_cfg.scene, cam_name) and getattr(env_cfg.scene, cam_name) is not None:
            setattr(env_cfg.scene, cam_name, None)
    if hasattr(pol, "wrist_depth") and pol.wrist_depth is not None:
        pol.wrist_depth = None

    # Vision RL: downsize the surviving wrist_cam from the IL default
    # 224x224 to 84x84 (Atari-style CNN input). 224x224 is overkill for
    # PPO + 4-DOF arm and triples the GPU memory / step time.
    if enable_vision and hasattr(env_cfg.scene, "wrist_cam"):
        wc = env_cfg.scene.wrist_cam
        if wc is not None:
            wc.height = 84
            wc.width = 84
        # Disable the ee_frame debug visualization marker. It renders
        # an XYZ coordinate cross right at the TCP — when the wrist
        # cam is mounted on the wrist, this cross dominates the frame
        # in giant primary colors and the actual scene (plant, table)
        # is hard to see. The state-only PPO didn't care because it
        # never saw images.
        if hasattr(env_cfg.scene, "ee_frame") and env_cfg.scene.ee_frame is not None:
            env_cfg.scene.ee_frame.debug_vis = False
        # IL adds the wrist_cam ObsTerm to the policy group (where the
        # state obs also live). For PPO with a CNN actor-critic that
        # needs to route the image through an encoder, we must move it
        # to the rgb_camera group so the concatenate_terms=True on
        # policy doesn't try to stack the 84x84x3 image with the 1D
        # state obs (which raises a shape mismatch).
        #
        # Also swap mdp.image → mdp.image_chw: Isaac Lab returns the
        # image channels-last (B, H, W, C), but rsl_rl's CNNModel
        # expects channels-first (B, C, H, W). The CHW wrapper
        # permutes; without it the CNN's conv math reads the wrong
        # axis and crashes with "Trying to create tensor with
        # negative dimension".
        cam_obs = getattr(pol, "wrist_cam", None)
        if cam_obs is not None and hasattr(env_cfg.observations, "rgb_camera"):
            # When the task yaml has wrist_cam.depth: true, the camera
            # renders both rgb + distance_to_image_plane and we route a
            # 4-channel RGBD image to the CNN — depth gives the actor an
            # unambiguous "plant is X mm below" signal that pure RGB
            # struggles to extract from a dark workspace. Pure RGB path
            # (image_chw) is kept for tasks without depth.
            has_depth = bool(cameras.get("wrist_cam", {}).get("depth"))
            cam_obs.func = mdp.image_rgbd_chw if has_depth else mdp.image_chw
            # Force normalize=True for the CNN path so the image comes
            # through as float in [-mean, 1-mean] instead of raw uint8.
            # Conv2d requires float input; otherwise we get the
            # "Input type (unsigned char) and bias type (float) should
            # be the same" runtime error.
            cam_obs.params = dict(cam_obs.params or {})
            cam_obs.params["normalize"] = True
            env_cfg.observations.rgb_camera.wrist_cam = cam_obs
            pol.wrist_cam = None
    # NOTE: leave env_cfg.observations.rgb_camera as the empty
    # configclass group — setting the whole group to None makes
    # ObservationManager dim-summing fail on construction.

    # --- Asymmetric actor-critic obs split for RL ---------------------------
    # Replace the bundled IL `policy` group with two RL-specific groups:
    #
    #   proprio    — robot state + known fixed target_pos. The ACTOR reads
    #                only this (+ rgb_camera in vision mode), so the policy
    #                is deployable on the real arm.
    #   privileged — object ground-truth pose. Only the CRITIC reads this;
    #                gives the value head stable supervision while keeping
    #                the actor sim-to-real-clean.
    #
    # See ppo_cfg.py's `obs_groups` dict for the actor/critic wiring.
    from .base_env_cfg import ObservationsCfg as _Obs
    obs = env_cfg.observations
    obs.proprio = _Obs.ProprioCfg()
    obs.privileged = _Obs.PrivilegedCfg()
    obs.proprio.gripper_pos.params = {"driver_joint": driver_joint}
    obs.proprio.target_pos.params = {"object_cfg": SceneEntityCfg("target")}
    obs.privileged.object.params = {
        "object_cfg": SceneEntityCfg("pickable"),
        "target_cfg": SceneEntityCfg("target"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    }
    obs.privileged.pickable_pos.params = {"object_cfg": SceneEntityCfg("pickable")}
    obs.privileged.pickable_quat.params = {"object_cfg": SceneEntityCfg("pickable")}

    # Per-term gaussian noise on the proprio side — calibrated against real
    # encoder/IK error budgets. Values are SI units (radians for revolute
    # joints, meters for prismatic / EE pose). Action noise is handled
    # separately by the action manager (not implemented here yet).
    obs.proprio.joint_pos.noise = GaussianNoiseCfg(mean=0.0, std=0.005)
    obs.proprio.joint_vel.noise = GaussianNoiseCfg(mean=0.0, std=0.01)
    obs.proprio.eef_pos.noise = GaussianNoiseCfg(mean=0.0, std=0.002)
    obs.proprio.eef_quat.noise = GaussianNoiseCfg(mean=0.0, std=0.005)
    obs.proprio.gripper_pos.noise = GaussianNoiseCfg(mean=0.0, std=0.0005)

    # Strip the legacy bundled groups. rsl_rl's runner does check_nan(obs)
    # which expects every obs entry to be a leaf Tensor; removing unused
    # groups keeps the obs dict clean for the runner's iteration.
    extras_to_strip = ["policy", "subtask_terms"]
    if not enable_vision:
        extras_to_strip.append("rgb_camera")
    else:
        env_cfg.observations.rgb_camera.concatenate_terms = True
    for extra_group in extras_to_strip:
        if hasattr(env_cfg.observations, extra_group):
            try:
                delattr(env_cfg.observations, extra_group)
            except AttributeError:
                grp = getattr(env_cfg.observations, extra_group)
                if grp is not None:
                    import dataclasses as _dc
                    for f in _dc.fields(grp):
                        v = getattr(grp, f.name, None)
                        if v is not None and getattr(v, "func", None) is not None:
                            setattr(grp, f.name, None)


def _apply_spec(env_cfg: PickPlaceEnvCfgBase, spec: dict[str, Any]) -> None:
    """Mutate an env_cfg instance: spawn objects, place cameras, wire
    actions + events + terminations from the YAML spec."""

    robot_cfg = spec["robot"]
    objects = spec["objects"]
    cameras = spec["cameras"]
    success = spec["success"]
    grasp_check = spec["grasp_check"]

    pickable_name, pickable = _named_role(objects, "pickable")
    target_name, target = _named_role(objects, "target")

    driver_joint = robot_cfg["gripper_driver_joint"]
    closed_threshold = float(robot_cfg["gripper_closed_threshold"])
    # Separate, optionally-lower trigger for the kinematic_attach event.
    # closed_threshold is used semantically for "the gripper has firmly
    # closed on the payload" (grasp-success, place-open checks); the
    # attach trigger should fire much earlier in the close cycle so the
    # payload is rigidly snapped to the TCP before the closing fingers
    # bump it sideways. Falls back to closed_threshold when not set.
    attach_threshold = float(robot_cfg.get(
        "gripper_kinematic_attach_threshold", closed_threshold,
    ))
    # use_kinematic_attach: when True (default), the kinematic_attach event
    # term welds the vial to the TCP on grasp — pipeline shortcut that
    # bypasses friction physics for replayable demos but blocks sim-to-real.
    # Set False in the task YAML to require real friction-based grasping;
    # the trade-off is the oracle must learn to maintain grip via contact
    # and the demos become physics-realistic (closer to real robot behavior).
    use_kinematic_attach = bool(robot_cfg.get("use_kinematic_attach", True))
    tcp_z = float(robot_cfg["tcp_z_offset"])
    arm_joints: list[str] = list(robot_cfg["arm_joints"])
    ee_body: str = str(robot_cfg.get("ee_body", "tool0_aligned"))

    # --- Robot + ee frame ---------------------------------------------------
    # robot.init_joint_pos: optional per-task override of the articulation's
    # home joint positions. Used by RL tasks that want the arm to spawn near
    # the workspace (e.g. "hover above the vial") instead of the default
    # all-zeros home pose, so the policy only has to learn the precision
    # pick-and-lift rather than the gross approach. arm_joint_jitter_std
    # still adds gaussian noise around these values per-reset.
    robot_art = build_robot_cfg(robot_cfg["type"]).replace(prim_path="{ENV_REGEX_NS}/Robot")
    init_overrides = robot_cfg.get("init_joint_pos") or {}
    if init_overrides:
        merged = {
            **robot_art.init_state.joint_pos,
            **{k: float(v) for k, v in init_overrides.items()},
        }
        robot_art = robot_art.replace(
            init_state=robot_art.init_state.replace(joint_pos=merged)
        )
    env_cfg.scene.robot = robot_art

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    env_cfg.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{ee_body}",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, tcp_z]),
            ),
        ],
    )

    # --- Pickable + target spawn -------------------------------------------
    env_cfg.scene.pickable = _build_object_cfg(pickable_name, pickable, prim_suffix="Pickable")
    env_cfg.scene.target = _build_object_cfg(target_name, target, prim_suffix="Target")

    # --- Distractor spawn (kinematic scene props the policy doesn't grasp) ---
    # Anything with role: distractor in objects: (e.g., a pedestal under the
    # vial, or a foreground prop) gets attached to the scene by name. They
    # don't appear in observations or success terms — they're physical.
    for distractor_name, distractor in objects.items():
        if distractor.get("role") != "distractor":
            continue
        prim_suffix = "".join(part.capitalize() for part in distractor_name.split("_"))
        cfg_obj = _build_object_cfg(distractor_name, distractor, prim_suffix=prim_suffix)
        setattr(env_cfg.scene, distractor_name, cfg_obj)

    # --- Reset events: arm jitter + pickable/target pose randomization +
    #                   sim2real domain randomization (physics + visual)
    enable_dr = bool(robot_cfg.get("enable_domain_randomization", True))
    @configclass
    class _Events:
        randomize_joint_state = EventTerm(
            func=events_mod.randomize_arm_joints_by_gaussian_offset,
            mode="reset",
            params={
                "mean": 0.0,
                "std": float(robot_cfg["arm_joint_jitter_std"]),
                "arm_joint_names": list(arm_joints),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # If the task has a `vial` distractor, co-randomize plant+vial by the
        # SAME offset (so the plant stays in the well while its world position
        # varies — forces a vision actor to localize). Else randomize the
        # pickable alone (generic behaviour for other tasks).
        randomize_pickable = EventTerm(
            func=(
                events_mod.randomize_plant_and_vial_together
                if "vial" in objects
                else franka_stack_events.randomize_object_pose
            ),
            mode="reset",
            params=(
                {
                    "pose_range": _spawn_pose_range(pickable["spawn"]),
                    "plant_cfg": SceneEntityCfg("pickable"),
                    "vial_cfg": SceneEntityCfg("vial"),
                    "plant_z": float(_mid(pickable["spawn"]["z"])),
                    "vial_z": float(_mid(objects["vial"]["spawn"]["z"])),
                }
                if "vial" in objects
                else {
                    "pose_range": _spawn_pose_range(pickable["spawn"]),
                    "min_separation": 0.0,
                    "asset_cfgs": [SceneEntityCfg("pickable")],
                }
            ),
        )
        randomize_target = EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": _spawn_pose_range(target["spawn"]),
                "min_separation": 0.0,
                "asset_cfgs": [SceneEntityCfg("target")],
            },
        )
        # ---- sim2real domain randomization ----
        # Pickable friction ±20% — the real-robot's actual friction
        # coefficient against the plant stem will differ from sim's
        # nominal 3.0/2.5. Randomizing during training forces the policy
        # to learn grasping that doesn't depend on exact mu.
        if enable_dr:
            randomize_pickable_friction = EventTerm(
                func=randomize_rigid_body_material,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("pickable"),
                    "static_friction_range": (2.4, 3.6),    # 3.0 ± 20%
                    "dynamic_friction_range": (2.0, 3.0),   # 2.5 ± 20%
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 32,
                },
            )
        # Pickable mass ±15% — plant species + hydration vary mass.
        if enable_dr:
            randomize_pickable_mass = EventTerm(
                func=randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("pickable"),
                    "mass_distribution_params": (0.85, 1.15),
                    "operation": "scale",
                    "distribution": "uniform",
                },
            )
        # Dome light intensity ±50% — real-world lighting varies far
        # more than physics or geometry. Largest sim2real-gap factor
        # for vision policies.
        if enable_dr:
            randomize_lighting = EventTerm(
                func=events_mod.randomize_dome_light_intensity,
                mode="reset",
                params={"intensity_range": (1500.0, 4500.0)},
            )
        # Interval mode with a zero-second window means the event manager
        # fires this term every env step (after physics, before next obs).
        # See ``mdp.kinematic_attach_payload`` for the rationale — bypasses
        # GPU-PhysX friction non-determinism by snapping the payload to the
        # gripper while the jaws are closed.
        # Conditional: when robot_cfg["use_kinematic_attach"] is False, the
        # event is omitted and the policy must learn real friction-based
        # grasping. Set False in the task YAML for sim-to-real-friendly demos.
        if use_kinematic_attach:
            kinematic_attach = EventTerm(
                func=mdp.kinematic_attach_payload,
                mode="interval",
                interval_range_s=(0.0, 0.0),
                is_global_time=False,
                params={
                    "payload_cfg": SceneEntityCfg("pickable"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "driver_joint": driver_joint,
                    # Use the lower attach_threshold (typically ~1/4 of
                    # closed_threshold) so the payload is captured on the
                    # first physics sub-step of finger closure, before the
                    # closing fingers can knock the vial laterally.
                    "closed_threshold": attach_threshold,
                    # 2 cm capture distance — tight enough to force the
                    # policy to position TCP near the stem before
                    # closing, so the learned behavior transfers
                    # meaningfully to real friction grasping. Override
                    # via robot.capture_distance in the task yaml.
                    "capture_distance": float(robot_cfg.get("capture_distance", 0.02)),
                },
            )

    env_cfg.events = _Events()

    # --- IK arm action + binary gripper action -----------------------------
    # SCARA T3-401 has 4 DOF (J1+J2 planar, J3 prismatic Z, J4 yaw); roll
    # and pitch are mechanically pinned to 0. A 6-DOF "pose" target is
    # over-determined and DLS under-drives Z when it tries to balance
    # unrealizable rotation residuals (we observed INSERT-phase descent
    # locking after ~1mm). Position-only IK (3-DOF target) lets the
    # solver focus on tracking xyz cleanly; J4 yaw is left at its home
    # value, which is fine for tasks where post-grasp orientation
    # doesn't matter. Set robot.ik_command_type: pose in task.yaml to
    # restore the 6-DOF behavior.
    ik_command_type = str(robot_cfg.get("ik_command_type", "position"))
    env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(arm_joints),
        body_name=ee_body,
        controller=DifferentialIKControllerCfg(
            command_type=ik_command_type, use_relative_mode=True, ik_method="dls"
        ),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, tcp_z]),
    )
    gripper_joints = list(robot_cfg["gripper_joints"])
    open_cmd = {n: float(robot_cfg["gripper_open_m"]) for n in gripper_joints}
    close_cmd = {n: float(robot_cfg["gripper_close_m"]) for n in gripper_joints}
    env_cfg.actions.gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=gripper_joints,
        open_command_expr=open_cmd,
        close_command_expr=close_cmd,
    )

    # --- Cameras ------------------------------------------------------------
    for cam_name, cam in cameras.items():
        cam_cfg = _build_camera_cfg(cam_name, cam)
        setattr(env_cfg.scene, cam_name, cam_cfg)

    # --- Image obs (one per camera) + wrist depth -------------------------
    pol = env_cfg.observations.policy
    for cam_name in cameras:
        setattr(pol, cam_name, ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg(cam_name), "data_type": "rgb", "normalize": False},
        ))
    if "wrist_cam" in cameras and cameras["wrist_cam"].get("depth"):
        pol.wrist_depth = ObsTerm(
            func=mdp.wrist_center_depth,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "window": 5},
        )

    # --- Wire object/target names + thresholds into observation params -----
    pol.object.params = {
        "object_cfg": SceneEntityCfg("pickable"),
        "target_cfg": SceneEntityCfg("target"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    }
    pol.pickable_pos.params = {"object_cfg": SceneEntityCfg("pickable")}
    pol.pickable_quat.params = {"object_cfg": SceneEntityCfg("pickable")}
    pol.target_pos.params = {"object_cfg": SceneEntityCfg("target")}
    pol.gripper_pos.params = {"driver_joint": driver_joint}

    sub = env_cfg.observations.subtask_terms
    sub.grasp.params = {
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        "object_cfg": SceneEntityCfg("pickable"),
        "diff_threshold": float(grasp_check["diff_threshold"]),
        "driver_joint": driver_joint,
        "closed_threshold": closed_threshold,
    }
    sub.place.params = {
        "object_cfg": SceneEntityCfg("pickable"),
        "target_cfg": SceneEntityCfg("target"),
        "xy_threshold": float(success["xy_threshold"]),
        "height_threshold": float(success["height_threshold"]),
        "driver_joint": driver_joint,
        "closed_threshold": closed_threshold,
    }

    # --- Terminations -------------------------------------------------------
    env_cfg.terminations.pickable_dropping.params = {
        "minimum_height": -0.05,
        "asset_cfg": SceneEntityCfg("pickable"),
    }

    # Success termination has two modes (selected via task yaml):
    #   rewards.success_termination: "placed"  (default) — full pick-and-place
    #     RL ends when the object_on_target condition is satisfied.
    #   rewards.success_termination: "lifted"          — factored pick-only RL
    #     ends when the policy has grasped + lifted the pickable. A scripted
    #     controller takes over for transport.
    # See RewardsCfg docstring for the matching reward-weight pattern.
    rewards_block = spec.get("rewards") if isinstance(spec.get("rewards"), dict) else {}
    success_mode = str(rewards_block.get("success_termination", "placed")).lower()
    if success_mode == "lifted":
        lift_height = float(rewards_block.get("lift_height", 0.10))
        # min_stable_steps blocks the 1-frame-teleport exploit when
        # kinematic_attach is on. 10 steps ≈ 0.2 s at typical sim dt;
        # long enough that the policy must actually hold the lift,
        # short enough that it doesn't penalize a real grasp.
        min_stable_steps = int(rewards_block.get("lift_stable_steps", 10))
        env_cfg.terminations.success.func = mdp.pickable_lifted
        env_cfg.terminations.success.params = {
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("pickable"),
            "lift_height": lift_height,
            "diff_threshold": 0.06,
            "driver_joint": driver_joint,
            "closed_threshold": closed_threshold,
            "min_stable_steps": min_stable_steps,
        }
        # success_bonus rewire for lifted mode happens in
        # _apply_reward_params (only runs when rewards are attached for
        # RL envs).
    elif success_mode == "placed":
        env_cfg.terminations.success.params = dict(sub.place.params)
    else:
        raise ValueError(
            f"rewards.success_termination must be 'placed' or 'lifted', "
            f"got {success_mode!r}"
        )

    # --- Misc sim settings + teleop device --------------------------------
    env_cfg.sim.render.antialiasing_mode = "DLAA"
    env_cfg.num_rerenders_on_reset = 3
    # Apply gravity (and other external forces) every solver iteration. Without
    # this, GPU-PhysX can produce noticeably non-deterministic contact-rich
    # trajectories — recording-vs-replay of oracle demos diverges enough to
    # drop a 2 cm cuboid out of a parallel-jaw grip mid-trajectory.
    env_cfg.sim.physx.enable_external_forces_every_iteration = True
    env_cfg.teleop_devices = DevicesCfg(
        devices={
            "keyboard": Se3KeyboardCfg(
                pos_sensitivity=0.02,
                rot_sensitivity=0.05,
                sim_device=env_cfg.sim.device,
            ),
        }
    )


def _named_role(objects: dict[str, dict], role: str) -> tuple[str, dict]:
    matches = [(name, o) for name, o in objects.items() if o.get("role") == role]
    if not matches:
        raise KeyError(f"task spec has no object with role={role!r}")
    if len(matches) > 1:
        raise KeyError(f"task spec has multiple objects with role={role!r}: {[m[0] for m in matches]}")
    return matches[0]


def _build_object_cfg(name: str, obj: dict, prim_suffix: str) -> RigidObjectCfg:
    obj_type = obj["type"]
    spawn = obj["spawn"]
    init_pos = (
        float(_mid(spawn["x"])),
        float(_mid(spawn["y"])),
        float(_mid(spawn["z"])),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=list(init_pos), rot=[1, 0, 0, 0])
    prim_path = f"{{ENV_REGEX_NS}}/{prim_suffix}"

    if obj_type == "cuboid":
        kinematic = bool(obj.get("kinematic", False))
        rigid_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=kinematic,
            kinematic_enabled=kinematic,
        )
        friction = obj.get("friction") or {"static": 0.5, "dynamic": 0.5, "restitution": 0.0}
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(friction["static"]),
            dynamic_friction=float(friction["dynamic"]),
            restitution=float(friction.get("restitution", 0.0)),
        )
        color = tuple(float(c) for c in obj.get("color", (0.5, 0.5, 0.5)))
        return RigidObjectCfg(
            prim_path=prim_path,
            init_state=init_state,
            spawn=CuboidCfg(
                size=tuple(obj["size"]),
                rigid_props=rigid_props,
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                physics_material=material,
                visual_material=PreviewSurfaceCfg(diffuse_color=color, roughness=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=float(obj["mass"])),
            ),
        )

    if obj_type == "usd":
        rigid_props = RigidBodyPropertiesCfg(
            kinematic_enabled=bool(obj.get("kinematic", False)),
            disable_gravity=bool(obj.get("kinematic", False)),
        )
        # Optional `collision: false` in YAML disables collision for this
        # asset. Useful for kinematic decoration that the policy shouldn't
        # interact with physically — e.g. a hollow vial whose convex
        # decomposition collides with anything inside it instead of being
        # truly hollow. Default is collision enabled (matches old behavior).
        collision_props = None
        if obj.get("collision", True) is False:
            collision_props = CollisionPropertiesCfg(collision_enabled=False)
        # usd_path may be either a Nucleus-relative path (Props/Blocks/...) or
        # a project-relative path. ``local:`` prefix marks a repo-relative path.
        raw = str(obj["usd_path"])
        if raw.startswith("local:"):
            usd_full = os.path.join(_REPO_ROOT, raw[len("local:") :])
        elif raw.startswith("/") or raw.startswith("file://"):
            usd_full = raw
        else:
            usd_full = f"{ISAAC_NUCLEUS_DIR}/{raw}"
        # Articulated pickable (e.g. the compliant jointed plant): build an
        # ArticulationCfg with a passive spring on the bend joint instead of a
        # rigid body. The spring (ImplicitActuator holding joint target 0)
        # makes the stem FLEX under a finger push and spring back, instead of
        # being knocked away like a rigid body (the ~38% rigid-plant ceiling).
        # Floating base (free root, gravity on) so it rests in the vial well
        # and the grasp lifts the whole plant out. obs/oracle/lift-metric all
        # read .data.root_pos_w (the base link) — same convention as rigid.
        if obj.get("articulated"):
            spring = obj.get("spring") or {}
            spring_joint = str(spring.get("joint", "stem_bend"))
            return ArticulationCfg(
                prim_path=prim_path,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=list(init_pos),
                    rot=[1, 0, 0, 0],
                    joint_pos={spring_joint: 0.0},
                ),
                spawn=UsdFileCfg(
                    usd_path=usd_full,
                    scale=tuple(obj.get("scale", (1.0, 1.0, 1.0))),
                    rigid_props=RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=1.0,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        articulation_enabled=True,
                        enabled_self_collisions=False,
                    ),
                    collision_props=collision_props,
                ),
                actuators={
                    "stem_spring": ImplicitActuatorCfg(
                        joint_names_expr=[spring_joint],
                        stiffness=float(spring.get("stiffness", 0.05)),
                        damping=float(spring.get("damping", 0.005)),
                        effort_limit=float(spring.get("effort_limit", 5.0)),
                        velocity_limit=float(spring.get("velocity_limit", 10.0)),
                    ),
                },
            )

        # Locally-converted USDs are produced via UrdfConverter which marks
        # the root with ArticulationRootAPI even for single-link assets.
        # That conflicts with RigidObjectCfg, which expects a plain rigid
        # body. Disable the articulation root for ``local:`` assets.
        articulation_props = None
        if raw.startswith("local:"):
            articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            )
        # Note: friction is baked into the converted USD by
        # convert_cad_assets.py at conversion time (UsdFileCfg in this
        # Isaac Lab build does not accept physics_material directly).
        return RigidObjectCfg(
            prim_path=prim_path,
            init_state=init_state,
            spawn=UsdFileCfg(
                usd_path=usd_full,
                scale=tuple(obj.get("scale", (1.0, 1.0, 1.0))),
                rigid_props=rigid_props,
                articulation_props=articulation_props,
                collision_props=collision_props,
            ),
        )

    raise ValueError(f"unsupported object type: {obj_type!r}")


def _build_camera_cfg(cam_name: str, cam: dict) -> CameraCfg:
    parent = cam.get("parent")
    if parent:
        prim_path = f"{{ENV_REGEX_NS}}/Robot/{parent}/{cam_name}"
    else:
        prim_path = f"{{ENV_REGEX_NS}}/{cam_name}"

    data_types = ["rgb"]
    if cam.get("depth"):
        data_types.append("distance_to_image_plane")

    # Use TiledCamera if this is the wrist_cam (the one vision-RL reads).
    # TiledCamera renders ALL envs into one big tiled viewport (1 Vulkan
    # descriptor for thousands of envs), as opposed to plain CameraCfg
    # which spawns one viewport per env and blows the descriptor pool
    # past ~64 envs.
    cls = TiledCameraCfg if cam_name == "wrist_cam" else CameraCfg
    return cls(
        prim_path=prim_path,
        update_period=0.0,
        height=int(cam.get("height", 224)),
        width=int(cam.get("width", 224)),
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(cam.get("focal_length", 18.0)),
            focus_distance=400.0,
            horizontal_aperture=float(cam.get("horizontal_aperture", 20.955)),
            clipping_range=tuple(cam.get("clipping_range", (0.1, 4.0))),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(cam["pos"]),
            rot=tuple(cam["rot"]),
            convention="ros",
        ),
    )


def _spawn_pose_range(spawn: dict) -> dict:
    return {
        "x": tuple(spawn["x"]),
        "y": tuple(spawn["y"]),
        "z": tuple(spawn["z"]),
        "yaw": tuple(spawn.get("yaw", (0.0, 0.0))),
    }


def _mid(rng) -> float:
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return 0.5 * (float(rng[0]) + float(rng[1]))
    return float(rng)
