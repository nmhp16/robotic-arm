"""Task-specific MDP terms for UR10 pick-and-place.

Generic terms (joint_pos_rel, last_action, time_out, image, root_height_below_minimum)
are imported from ``isaaclab.envs.mdp``. Stack-task observation helpers we can reuse
(``ee_frame_pos``, ``ee_frame_quat``, ``gripper_pos``, ``object_grasped``) are
re-exported from the Isaac Lab stack package. Everything below is specific to the
single-cube + target-zone pick-and-place variant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
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
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import (  # noqa: F401
    ee_frame_pos,
    ee_frame_quat,
    gripper_pos,
    object_grasped,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    """Compact state observation (in env-local frame):

    cube_pos (3) + cube_quat (4) + target_pos (3) + ee_to_cube (3) + cube_to_target (3) = 16-D.
    """
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


def cube_on_target(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.06,
) -> torch.Tensor:
    """Success: cube is within ``xy_threshold`` of the target in the xy plane,
    no more than ``height_threshold`` above it in z, and the suction gripper is
    released.

    The released-suction check prevents "success while still holding the cube",
    which would otherwise trigger immediately on the first frame the arm passes
    over the target.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    pos_diff = cube.data.root_pos_w - target.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    placed = torch.logical_and(xy_dist < xy_threshold, pos_diff[:, 2] < height_threshold)
    placed = torch.logical_and(placed, pos_diff[:, 2] > -0.02)  # not below the target

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        sg = env.scene.surface_grippers["surface_gripper"]
        suction_open = (sg.state.view(-1) == -1)
        placed = torch.logical_and(placed, suction_open)

    return placed
