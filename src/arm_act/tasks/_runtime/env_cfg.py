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

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
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
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import events as events_mod
from . import mdp
from .base_env_cfg import PickPlaceEnvCfgBase
from .robot_cfg import build_robot_cfg


def build_env_cfg(spec: dict[str, Any], class_name: str) -> type:
    """Return a configclass-decorated subclass of PickPlaceEnvCfgBase
    with all task-specific fields baked in from ``spec``."""

    class _Cfg(PickPlaceEnvCfgBase):
        def __post_init__(self):
            super().__post_init__()
            _apply_spec(self, spec)

    _Cfg.__name__ = class_name
    _Cfg.__qualname__ = class_name
    return configclass(_Cfg)


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
    tcp_z = float(robot_cfg["tcp_z_offset"])

    # --- Robot + ee frame ---------------------------------------------------
    env_cfg.scene.robot = build_robot_cfg(robot_cfg["type"]).replace(prim_path="{ENV_REGEX_NS}/Robot")

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    env_cfg.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/tool0_aligned",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, tcp_z]),
            ),
        ],
    )

    # --- Pickable + target spawn -------------------------------------------
    env_cfg.scene.pickable = _build_object_cfg(pickable_name, pickable, prim_suffix="Pickable")
    env_cfg.scene.target = _build_object_cfg(target_name, target, prim_suffix="Target")

    # --- Reset events: arm jitter + pickable/target pose randomization ----
    @configclass
    class _Events:
        randomize_joint_state = EventTerm(
            func=events_mod.randomize_arm_joints_by_gaussian_offset,
            mode="reset",
            params={"mean": 0.0, "std": float(robot_cfg["arm_joint_jitter_std"]), "asset_cfg": SceneEntityCfg("robot")},
        )
        randomize_pickable = EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": _spawn_pose_range(pickable["spawn"]),
                "min_separation": 0.0,
                "asset_cfgs": [SceneEntityCfg("pickable")],
            },
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

    env_cfg.events = _Events()

    # --- IK arm action + binary gripper action -----------------------------
    env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_joint", "wrist_.*"],
        body_name="tool0_aligned",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
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
    env_cfg.terminations.success.params = dict(sub.place.params)

    # --- Misc sim settings + teleop device --------------------------------
    env_cfg.sim.render.antialiasing_mode = "DLAA"
    env_cfg.num_rerenders_on_reset = 3
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
        rigid_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
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
        return RigidObjectCfg(
            prim_path=prim_path,
            init_state=init_state,
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/{obj['usd_path']}",
                scale=tuple(obj.get("scale", (1.0, 1.0, 1.0))),
                rigid_props=rigid_props,
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

    return CameraCfg(
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
