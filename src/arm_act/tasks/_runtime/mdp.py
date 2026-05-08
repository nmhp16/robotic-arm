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


def kinematic_attach_payload(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.01,
    capture_distance: float = 0.10,
) -> None:
    """Snap the payload's pose to follow the gripper while the gripper is
    closed, ignoring dynamic-contact friction physics for the duration.

    Why: GPU PhysX is non-deterministic at the static-friction limit, which
    makes the cuboid micro-slip out of a parallel-jaw grip on replay. The
    contact-rich phases (CLOSE / LIFT / MOVE / PLACE) thus don't reproduce
    the recording, so imitation-learning data has trajectories whose
    actions don't actually achieve their recorded outcomes. This event term
    detaches the held-payload phase from physics: once the jaws close and
    the payload is within ``capture_distance`` of the TCP, the payload
    becomes a kinematic "child" of the gripper. Released when the jaws
    reopen. Trades realism (real grippers can drop things) for replayable
    deterministic data, which is what the imitation pipeline needs.

    Wire as ``mode="interval"`` with ``interval_range_s=(0.0, 0.0)`` so it
    fires every env step (after physics, before the next observation).
    """
    if env_ids is None or len(env_ids) == 0:
        return

    if not hasattr(env, "_kinematic_attach_offset"):
        env._kinematic_attach_offset = torch.zeros(env.num_envs, 3, device=env.device)
        env._kinematic_attach_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    payload: RigidObject = env.scene[payload_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene["robot"]
    try:
        grip_idx = robot.data.joint_names.index(driver_joint)
    except ValueError:
        return  # gripper missing — nothing to do

    grip_pos = robot.data.joint_pos[:, grip_idx]
    is_closed = grip_pos > closed_threshold
    tcp_w = ee_frame.data.target_pos_w[:, 0, :]
    payload_pos_w = payload.data.root_pos_w
    near_tcp = torch.linalg.vector_norm(tcp_w - payload_pos_w, dim=-1) < capture_distance

    upright = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    zero_vel = torch.zeros(1, 6, device=env.device)
    for env_id in env_ids.tolist():
        active = bool(env._kinematic_attach_active[env_id])
        if active and not is_closed[env_id]:
            # Release: leave payload where it is, return to dynamic physics.
            env._kinematic_attach_active[env_id] = False
            continue
        if active:
            target_pos = tcp_w[env_id] + env._kinematic_attach_offset[env_id]
            pose = torch.cat([target_pos.unsqueeze(0), upright.unsqueeze(0)], dim=-1)
            ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
            payload.write_root_pose_to_sim(pose, env_ids=ids)
            payload.write_root_velocity_to_sim(zero_vel, env_ids=ids)
            continue
        if is_closed[env_id] and near_tcp[env_id]:
            # First contact close: capture TCP→payload offset and lock.
            # Lateral (X/Y) is zeroed so the payload snaps to the TCP centerline
            # — otherwise the descend-time XY jitter from the oracle gets frozen
            # in for the rest of the episode, leaving the vial visibly off-axis
            # and dropping it off-target during PLACE.
            offset = payload_pos_w[env_id] - tcp_w[env_id]
            offset[0] = 0.0
            offset[1] = 0.0
            env._kinematic_attach_offset[env_id] = offset
            env._kinematic_attach_active[env_id] = True


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
