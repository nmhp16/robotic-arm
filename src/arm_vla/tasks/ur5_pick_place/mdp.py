"""Task-specific MDP terms for UR5e + Robotiq 2F-85 pick-and-place.

Generic terms are imported from ``isaaclab.envs.mdp``. Everything below is
single-cube + target-zone + parallel-jaw specific — we don't reuse the Isaac
Lab stack helpers because several of them assume a 2-joint parallel gripper
(2F-85 has one driving joint; the rest are mimics/passives).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp import (  # noqa: F401 (re-exported for env cfg)
    image,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    root_height_below_minimum,
    time_out,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Proprio / object observations
# ---------------------------------------------------------------------------

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
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Scalar ``finger_joint`` angle, (N, 1). 0 = fully open, ~0.8 = closed."""
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints([env.cfg.gripper_joint_name])
    return robot.data.joint_pos[:, joint_ids].view(-1, 1)


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
    """Compact state observation (env-local): cube_pos(3) + cube_quat(4)
    + target_pos(3) + ee_to_cube(3) + cube_to_target(3) = 16-D."""
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


# ---------------------------------------------------------------------------
# Subtask annotations (for mimic segmentation)
# ---------------------------------------------------------------------------

def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    diff_threshold: float = 0.06,
    gripper_close_threshold: float = 0.2,
) -> torch.Tensor:
    """Cube is grasped if the gripper is closed past a threshold *and* the
    cube is within ``diff_threshold`` of the TCP. The closed-past-threshold
    check avoids flagging the open gripper passing over the cube as a grasp.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]

    pose_diff = torch.linalg.vector_norm(
        cube.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=1
    )
    close_to_cube = pose_diff < diff_threshold

    joint_ids, _ = robot.find_joints([env.cfg.gripper_joint_name])
    finger_pos = robot.data.joint_pos[:, joint_ids].view(-1)
    gripper_closed = finger_pos > gripper_close_threshold

    return torch.logical_and(close_to_cube, gripper_closed)


def cube_on_target(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.06,
    gripper_open_threshold: float = 0.1,
) -> torch.Tensor:
    """Success: cube within ``xy_threshold`` in the xy plane *and* no more
    than ``height_threshold`` above the target in z, *and* the gripper is
    open (finger_joint below ``gripper_open_threshold``).

    The open-gripper check blocks "success while still holding the cube" —
    i.e. the arm passing over the target with the cube clamped would
    otherwise flip success on one frame.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    pos_diff = cube.data.root_pos_w - target.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    placed = torch.logical_and(xy_dist < xy_threshold, pos_diff[:, 2] < height_threshold)
    placed = torch.logical_and(placed, pos_diff[:, 2] > -0.02)

    joint_ids, _ = robot.find_joints([env.cfg.gripper_joint_name])
    finger_pos = robot.data.joint_pos[:, joint_ids].view(-1)
    gripper_open = finger_pos < gripper_open_threshold

    return torch.logical_and(placed, gripper_open)
