"""Task-specific MDP terms for UR10 + surface-suction pick-and-place."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
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


def _suction_state(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Return the surface-gripper state tensor, or None if not present.

    State values: 1 = closed (vacuum holding), 0 = closing, -1 = open.
    """
    if not hasattr(env.scene, "surface_grippers") or not env.scene.surface_grippers:
        return None
    gripper = next(iter(env.scene.surface_grippers.values()))
    return gripper.state.view(-1)


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
    """Suction state as a scalar per env, shape (N, 1). 1 = closed, -1 = open."""
    state = _suction_state(env)
    if state is None:
        return torch.zeros((env.num_envs, 1), device=env.device)
    return state.view(-1, 1)


def wrist_center_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    window: int = 5,
) -> torch.Tensor:
    """Median depth over a small window around the wrist-cam center pixel.

    Returns meters along the camera's forward axis to the nearest surface
    in view. Invalid/infinite values are clipped to the camera's far plane.
    Shape (N, 1).
    """
    cam: Camera = env.scene[sensor_cfg.name]
    depth = cam.data.output["distance_to_image_plane"]  # (N, H, W) or (N, H, W, 1)
    if depth.dim() == 4:
        depth = depth.squeeze(-1)
    N, H, W = depth.shape
    cy, cx = H // 2, W // 2
    half = max(1, window // 2)
    patch = depth[:, cy - half : cy + half + 1, cx - half : cx + half + 1]
    patch = torch.nan_to_num(patch, nan=2.0, posinf=2.0, neginf=2.0)
    patch = patch.clamp(max=2.0)
    # Median is robust to stray pixels on thin edges.
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
    """16-D vector: cube_pos(3) + cube_quat(4) + target_pos(3) + ee_to_cube(3) + cube_to_target(3)."""
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
    """True when the suction is closed and the cube is within
    ``diff_threshold`` m of the TCP."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]

    dist = torch.linalg.vector_norm(
        cube.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=1
    )
    close_to_cube = dist < diff_threshold

    state = _suction_state(env)
    if state is None:
        return close_to_cube
    suction_closed = state == 1
    return torch.logical_and(close_to_cube, suction_closed)


def cube_on_target(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.06,
) -> torch.Tensor:
    """True when the cube is within ``xy_threshold`` m of the target in xy,
    within ``height_threshold`` m in z, and the suction is released.

    The released-suction check rejects "success while still holding the cube".
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    pos_diff = cube.data.root_pos_w - target.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    placed = torch.logical_and(xy_dist < xy_threshold, pos_diff[:, 2] < height_threshold)
    placed = torch.logical_and(placed, pos_diff[:, 2] > -0.02)

    state = _suction_state(env)
    if state is None:
        return placed
    suction_open = state == -1
    return torch.logical_and(placed, suction_open)
