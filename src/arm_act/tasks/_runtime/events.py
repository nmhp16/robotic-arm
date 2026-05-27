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


def randomize_plant_and_vial_together(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict,
    plant_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    vial_cfg: SceneEntityCfg = SceneEntityCfg("vial"),
    plant_z: float = 0.0,
    vial_z: float = 0.035,
    intra_vial_xy: float = 0.005,
) -> None:
    """Place the vial per ``pose_range``; place the plant at the vial xy plus a
    small random INTRA-VIAL offset (±``intra_vial_xy`` m) + yaw, so the plant
    sits at a VARYING position within the well rather than pinned to the vial
    centre.

    This matches reality — a real plant leans/sits off-centre in the vial — and
    forces the policy (or a perception module) to localize the ACTUAL plant, not
    assume "grasp the vial centre" (a sim crutch). The offset is kept within the
    wide-vial well AND within the ~±1cm range the state policy was trained on, so
    the pose-conditioned state policy still handles it. ``pose_range`` x/y are
    absolute workspace coords. Set ``intra_vial_xy=0`` to restore the old
    co-located (centred) behaviour.
    """
    if env_ids is None:
        return
    rx = tuple(pose_range.get("x", (0.0, 0.0)))
    ry = tuple(pose_range.get("y", (0.0, 0.0)))
    ryaw = tuple(pose_range.get("yaw", (0.0, 0.0)))
    dev = env.device
    plant = env.scene[plant_cfg.name]
    vial = env.scene[vial_cfg.name]
    for cur in env_ids.tolist():
        x = float(torch.empty(1).uniform_(rx[0], rx[1]))
        y = float(torch.empty(1).uniform_(ry[0], ry[1]))
        yaw = float(torch.empty(1).uniform_(ryaw[0], ryaw[1]))
        dx = float(torch.empty(1).uniform_(-intra_vial_xy, intra_vial_xy))
        dy = float(torch.empty(1).uniform_(-intra_vial_xy, intra_vial_xy))
        ids = torch.tensor([cur], device=dev)
        origin = env.scene.env_origins[cur, 0:3]
        # (asset, z, yaw, x-offset, y-offset): the plant gets the intra-vial
        # offset so it varies within the well; the vial stays at the sampled xy.
        for asset, z, yw, ox, oy in (
            (plant, plant_z, yaw, dx, dy),
            (vial, vial_z, 0.0, 0.0, 0.0),
        ):
            pos = torch.tensor([[x + ox, y + oy, z]], device=dev) + origin
            quat = math_utils.quat_from_euler_xyz(
                torch.zeros(1, device=dev),
                torch.zeros(1, device=dev),
                torch.tensor([yw], device=dev),
            )
            asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
            asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev), env_ids=ids)
