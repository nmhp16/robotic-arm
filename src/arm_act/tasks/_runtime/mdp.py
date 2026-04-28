"""Generic MDP terms for parametric pick-and-place tasks.

The functions take ``SceneEntityCfg`` references plus the gripper threshold
+ driver-joint name as kwargs, so the same code serves any parallel-jaw
task. The runtime builder wires the right names through ``params={...}``
on each ObsTerm / DoneTerm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp import (  # noqa: F401  re-exported for env cfg
    image,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    root_height_below_minimum,
    time_out,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _gripper_drive_pos(env: ManagerBasedRLEnv, joint_name: str) -> torch.Tensor | None:
    """Driver-joint position per env, shape (N,). None if the joint is missing."""
    robot: Articulation = env.scene["robot"]
    try:
        idx = robot.data.joint_names.index(joint_name)
    except ValueError:
        return None
    return robot.data.joint_pos[:, idx]


def ee_frame_pos(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins


def ee_frame_quat(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


def gripper_pos(
    env: ManagerBasedRLEnv,
    driver_joint: str = "finger_left_joint",
) -> torch.Tensor:
    """Driver-joint position per env, shape (N, 1). 0 ≈ open."""
    pos = _gripper_drive_pos(env, driver_joint)
    if pos is None:
        return torch.zeros((env.num_envs, 1), device=env.device)
    return pos.view(-1, 1)


def wrist_center_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    window: int = 5,
) -> torch.Tensor:
    cam: Camera = env.scene[sensor_cfg.name]
    depth = cam.data.output["distance_to_image_plane"]
    if depth.dim() == 4:
        depth = depth.squeeze(-1)
    N, H, W = depth.shape
    cy, cx = H // 2, W // 2
    half = max(1, window // 2)
    patch = depth[:, cy - half : cy + half + 1, cx - half : cx + half + 1]
    patch = torch.nan_to_num(patch, nan=2.0, posinf=2.0, neginf=2.0)
    patch = patch.clamp(max=2.0)
    flat = patch.reshape(N, -1)
    med = flat.median(dim=1).values
    return med.view(-1, 1)


def object_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w - env.scene.env_origins


def object_orientation(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_quat_w


def object_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w
    target_pos_w = target.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    return torch.cat(
        (
            obj_pos_w - env.scene.env_origins,
            obj_quat_w,
            target_pos_w - env.scene.env_origins,
            obj_pos_w - ee_pos_w,
            target_pos_w - obj_pos_w,
        ),
        dim=1,
    )


def object_grasped(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    diff_threshold: float = 0.06,
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.02,
) -> torch.Tensor:
    """True when the gripper is closed and the object is within
    ``diff_threshold`` m of the TCP."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    dist = torch.linalg.vector_norm(
        obj.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=1
    )
    close_to_obj = dist < diff_threshold

    pos = _gripper_drive_pos(env, driver_joint)
    if pos is None:
        return close_to_obj
    gripper_closed = pos > closed_threshold
    return torch.logical_and(close_to_obj, gripper_closed)


def object_on_target(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.06,
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.02,
) -> torch.Tensor:
    """True when the object sits within (xy, height) thresholds of the target
    AND the gripper is open."""
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    pos_diff = obj.data.root_pos_w - target.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    placed = torch.logical_and(xy_dist < xy_threshold, pos_diff[:, 2] < height_threshold)
    placed = torch.logical_and(placed, pos_diff[:, 2] > -0.02)

    pos = _gripper_drive_pos(env, driver_joint)
    if pos is None:
        return placed
    gripper_open = pos < closed_threshold
    return torch.logical_and(placed, gripper_open)
