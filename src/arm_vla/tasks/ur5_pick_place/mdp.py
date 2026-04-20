"""Task-specific MDP terms for UR5 + Robotiq 2F-85 pick-and-place."""

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

# 2F-85 left_knuckle_joint sweeps ~0 (open) → ~0.7 rad (closed at finger contact).
# Treat anything past this as "closed" for grasp/place predicates.
_GRIPPER_CLOSED_THRESHOLD = 0.35
_DRIVER_JOINT_NAME = "robotiq_85_left_knuckle_joint"


def _gripper_drive_pos(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Driver-joint position per env, shape (N,). None if joint missing."""
    robot: Articulation = env.scene["robot"]
    try:
        idx = robot.data.joint_names.index(_DRIVER_JOINT_NAME)
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


def gripper_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Driver-joint position per env, shape (N, 1). 0 ≈ open, ~0.7 ≈ closed."""
    pos = _gripper_drive_pos(env)
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


def cube_position(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w - env.scene.env_origins


def cube_orientation(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_quat_w


def target_position(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    target: RigidObject = env.scene[target_cfg.name]
    return target.data.root_pos_w - env.scene.env_origins


def object_obs(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = cube.data.root_pos_w
    cube_quat_w = cube.data.root_quat_w
    target_pos_w = target.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    return torch.cat(
        (
            cube_pos_w - env.scene.env_origins,
            cube_quat_w,
            target_pos_w - env.scene.env_origins,
            cube_pos_w - ee_pos_w,
            target_pos_w - cube_pos_w,
        ),
        dim=1,
    )


def object_grasped(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """True when the gripper is closed and the cube is within
    ``diff_threshold`` m of the TCP."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]

    dist = torch.linalg.vector_norm(
        cube.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=1
    )
    close_to_cube = dist < diff_threshold

    pos = _gripper_drive_pos(env)
    if pos is None:
        return close_to_cube
    gripper_closed = pos > _GRIPPER_CLOSED_THRESHOLD
    return torch.logical_and(close_to_cube, gripper_closed)


def cube_on_target(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.06,
) -> torch.Tensor:
    """True when the cube is within ``xy_threshold`` m of the target in xy,
    within ``height_threshold`` m in z, and the gripper is open."""
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    pos_diff = cube.data.root_pos_w - target.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    placed = torch.logical_and(xy_dist < xy_threshold, pos_diff[:, 2] < height_threshold)
    placed = torch.logical_and(placed, pos_diff[:, 2] > -0.02)

    pos = _gripper_drive_pos(env)
    if pos is None:
        return placed
    gripper_open = pos < _GRIPPER_CLOSED_THRESHOLD
    return torch.logical_and(placed, gripper_open)
