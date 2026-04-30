"""Reset events: per-joint randomization that skips gripper joints.

`isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events`'s
randomizer hard-codes "skip last 2 joints" to leave the Panda's two finger
joints alone. We filter by joint name list instead, supplied from the task
spec, so the same event works regardless of robot DOF or gripper joint
count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_arm_joints_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    arm_joint_names: Sequence[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Add gaussian noise to arm joints only; gripper joints stay at default.

    ``arm_joint_names`` is the explicit list of joints that count as part of
    the arm (i.e. should be jittered). Anything not in that list — including
    finger / gripper joints — is left at its default position.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    arm_set = set(arm_joint_names)
    arm_mask = torch.tensor(
        [name in arm_set for name in asset.data.joint_names],
        device=joint_pos.device,
    )
    noise = math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)
    joint_pos = torch.where(arm_mask, joint_pos + noise, joint_pos)

    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
