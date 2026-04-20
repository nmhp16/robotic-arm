"""Reset events scoped to UR5 + 2F-85.

`isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events`'s
randomizer hard-codes "skip last 2 joints" to leave the Panda's two finger
joints alone. UR5 + 2F-85 has six gripper joints, so we filter by name
instead of by index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_ARM_JOINT_PREFIXES = ("shoulder_", "elbow_", "wrist_")


def randomize_arm_joints_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Add gaussian noise to arm joints only; gripper joints stay at default."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    arm_mask = torch.tensor(
        [name.startswith(_ARM_JOINT_PREFIXES) for name in asset.data.joint_names],
        device=joint_pos.device,
    )
    noise = math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)
    joint_pos = torch.where(arm_mask, joint_pos + noise, joint_pos)

    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
