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


def randomize_dome_light_intensity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (1500.0, 4500.0),
    light_prim_path: str = "/World/light",
) -> None:
    """Randomize the dome light's intensity at each env reset.

    Visual DR for sim-to-real: the real workcell's lighting varies with
    time of day, ambient light, indicator LEDs, etc. Training the vision
    policy with varied dome intensity makes it less sensitive to absolute
    pixel brightness and forces it to use spatial/contrast cues that
    generalize to the real cameras.

    Note: the dome light is GLOBAL (one per scene, not per-env). So this
    event randomizes ALL envs to the same intensity on each call. With
    interval=0 reset mode, that means each env-reset batch shares one
    intensity sample. Not ideal but cheap and effective — full per-env
    light randomization would require per-env light prims.

    intensity_range is a (min, max) pair in nits. The default (1500,
    4500) is a ±50% range centered on the SeattleLabTable scene's
    default 3000 nits — wide enough to cover typical lab/factory
    ambient variation.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    import random

    import omni.usd
    from pxr import UsdLux

    # Sample one intensity per reset batch. random.uniform is fine here
    # because the dome light is global to the scene — varied per-env
    # lighting would require per-env light prims (more setup work).
    intensity = random.uniform(intensity_range[0], intensity_range[1])

    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath(light_prim_path)
    if not light_prim.IsValid():
        return
    light = UsdLux.DomeLight(light_prim)
    if light:
        light.GetIntensityAttr().Set(intensity)


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
