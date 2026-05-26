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
from isaaclab.envs.mdp import image as _image_hwc

# Note: image_chw is defined AFTER SceneEntityCfg import (further down)
# because its default args reference SceneEntityCfg.
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def image_chw(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Wrapper around ``isaaclab.envs.mdp.image`` that returns the image
    in channels-first ``(B, C, H, W)`` layout instead of Isaac Lab's
    default channels-last ``(B, H, W, C)``.

    rsl_rl's CNN model assumes channels-first input — it reads
    ``obs.shape[1]`` as channels and ``obs.shape[2:4]`` as (H, W). With
    channels-last from Isaac Lab the conv math goes negative on the
    width axis (``floor((3-8)/4 + 1) = -1``) and the MLP head crashes
    with ``Trying to create tensor with negative dimension``.

    Signature mirrors ``isaaclab.envs.mdp.image`` explicitly so Isaac
    Lab's ObservationManager parameter introspection sees the named
    kwargs (a ``*args, **kwargs`` wrapper appears as one parameter
    called ``kwargs`` and fails the ObsTerm contract).
    """
    img = _image_hwc(
        env,
        sensor_cfg=sensor_cfg,
        data_type=data_type,
        convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
        normalize=normalize,
    )
    # Isaac Lab returns (B, H, W, C); rsl_rl wants (B, C, H, W).
    return img.permute(0, 3, 1, 2).contiguous()


def image_rgbd_chw(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",            # ignored; signature matches image_chw
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """RGB + depth concatenated along channel dim → ``(B, 4, H, W)``.

    Depth is clamped to ``[0.02, 0.5]`` m and linearly normalized to
    ``[0, 1]``. That range covers the wrist-cam's reachable workspace
    (tool flange z ≈ 0.27, table at z ≈ 0.04, so depth from cam is
    roughly 0.05–0.25 m for objects on the table) with a margin on
    each side. inf-depth pixels (sky / open space) are mapped to 0 by
    Isaac Lab's normalization (see ``isaaclab.envs.mdp.image`` docstring),
    so after clamp+rescale they end up at 0 — the policy reads "very
    close" rather than "infinitely far". Acceptable: those pixels look
    away from the workspace anyway.

    The ``data_type`` arg is ignored. It's kept so this function is a
    drop-in replacement for ``image_chw`` wherever the ObsTerm is wired:
    the env_cfg builder swaps ``cam_obs.func = mdp.image_rgbd_chw`` and
    nothing else needs to change. The ``normalize`` flag is honored for
    the RGB half only; depth has its own fixed-range normalization.
    """
    rgb = image_chw(
        env,
        sensor_cfg=sensor_cfg,
        data_type="rgb",
        convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
        normalize=normalize,
    )
    depth_hwc = _image_hwc(
        env,
        sensor_cfg=sensor_cfg,
        data_type="distance_to_image_plane",
        convert_perspective_to_orthogonal=False,
        normalize=True,  # Isaac Lab maps inf → 0
    )
    # Squeeze the trailing channel dim (depth comes in as (B, H, W, 1) or
    # (B, H, W) depending on Isaac Lab version) and re-introduce as channel-0.
    if depth_hwc.dim() == 4 and depth_hwc.shape[-1] == 1:
        depth_hwc = depth_hwc.squeeze(-1)
    depth = depth_hwc.unsqueeze(1)  # (B, 1, H, W)
    depth = depth.clamp(min=0.02, max=0.5)
    depth = (depth - 0.02) / (0.5 - 0.02)
    return torch.cat([rgb, depth.to(rgb.dtype)], dim=1).contiguous()


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


# --- RL reward terms -------------------------------------------------------
# These are shaped rewards for closed-loop RL fine-tuning. They run on every
# step and return (num_envs,) tensors. Each returns a single scalar value
# per env that the RewardsCfg manager will sum (with per-term weights).

def reward_tcp_to_pickable(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    driver_joint: str = "finger_left_joint",  # kept for signature compat
    closed_threshold: float = 0.02,            # kept for signature compat
) -> torch.Tensor:
    """Negative xy distance between TCP and pickable, every step.

    Earlier versions gated this by an ``open_mask`` (only fire when the
    gripper is open), with the intent of not fighting the grasp phase.
    That gate turned out to be a degenerate exploit: PPO discovered it
    could close the gripper early to zero out the negative approach
    reward — observed on 2026-05-20 in v4 (approach dropped 100× in
    25 iters from -0.0229 to -0.0002 while action movement stayed
    constant; std + reward both collapsed toward the no-approach
    minimum). Removing the gate forces the policy to actually approach.

    With kinematic_attach enabled (the default), the plant welds to the
    TCP on grasp so plant_pos = TCP_pos and dist→0 naturally — no need
    for a manual mask. ``driver_joint`` and ``closed_threshold`` are
    kept in the signature so the env_cfg builder's params dict doesn't
    need a special case.

    The distance is computed in 3D, not just xy. The xy-only version
    plateaued the policy at TCP z = home pose z (0.22 m) with no
    gradient pushing it to descend toward the plant at z ≈ 0.04 m —
    150 iters of v5 training showed xy distance shrinking but no
    plant interaction. Including z gives the actor a clear "descend
    AND approach" signal. After kinematic_attach welds the plant to
    the TCP, dz → 0 naturally so the LIFT phase isn't fought."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    diff = obj.data.root_pos_w[:, :3] - ee_frame.data.target_pos_w[:, 0, :3]
    dist = torch.linalg.vector_norm(diff, dim=1)
    return -dist


def reward_pickable_to_target(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.02,
) -> torch.Tensor:
    """Negative xy distance between pickable and target. Active only when
    the gripper is CLOSED (i.e. the policy has the object and is transporting
    it). This drives the MOVE / ALIGN phases."""
    obj: RigidObject = env.scene[object_cfg.name]
    tgt: RigidObject = env.scene[target_cfg.name]
    diff = obj.data.root_pos_w[:, :2] - tgt.data.root_pos_w[:, :2]
    dist = torch.linalg.vector_norm(diff, dim=1)
    gripper_pos = _gripper_drive_pos(env, driver_joint)
    if gripper_pos is None:
        return -dist
    closed_mask = (gripper_pos >= closed_threshold).to(dist.dtype)
    return -dist * closed_mask


def reward_grasp_at_pickable(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    diff_threshold: float = 0.06,
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.02,
) -> torch.Tensor:
    """+1 every step the gripper is closed AND the TCP is within
    `diff_threshold` of the object. Encourages closing at the right moment."""
    return object_grasped(
        env,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        diff_threshold=diff_threshold,
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    ).to(torch.float32)


def reward_object_on_target(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.06,
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.02,
) -> torch.Tensor:
    """+1 every step task success holds (object placed at target, gripper
    open). Should be heavily weighted — this is the only signal that
    actually defines task success."""
    return object_on_target(
        env,
        object_cfg=object_cfg,
        target_cfg=target_cfg,
        xy_threshold=xy_threshold,
        height_threshold=height_threshold,
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    ).to(torch.float32)


def reward_action_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared L2 of the last action — regularizes against wild commands.
    Returned as POSITIVE; use a negative weight in the RewardsCfg."""
    a = env.action_manager.action
    return torch.sum(a * a, dim=-1)


def _pickable_initial_z(env: ManagerBasedRLEnv, object_name: str) -> torch.Tensor:
    """Lazily cache the at-reset z of the pickable so the lift reward can
    measure displacement relative to spawn height instead of an absolute
    threshold (works regardless of where the plant lands after settling)."""
    attr = f"_initial_pickable_z_{object_name}"
    obj: RigidObject = env.scene[object_name]
    cur = obj.data.root_pos_w[:, 2]
    if not hasattr(env, attr):
        setattr(env, attr, cur.clone())
    z0 = getattr(env, attr)
    # Re-capture per episode: for envs that just reset (episode start), reset
    # z0 to the current spawn z. The old once-ever cache froze z0 at the FIRST
    # episode's spawn for the whole run — wrong under per-episode spawn
    # randomization, and it captured pre-settle. Spawning at rest (yaml) plus
    # this per-episode refresh makes the lift baseline correct.
    just_reset = env.episode_length_buf <= 1
    z0 = torch.where(just_reset, cur, z0)
    setattr(env, attr, z0)
    return z0


def reward_lift_to_height(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    lift_height: float = 0.10,
    diff_threshold: float = 0.06,
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.02,
) -> torch.Tensor:
    """Continuous lift reward for factored pick-only RL.

    +1 per step while the gripper is closed near the pickable AND the
    pickable has been raised by ``lift_height`` metres relative to its
    spawn height. Use a large weight (~50) to make this the dominant
    success signal — paired with ``pickable_lifted`` termination, it
    forms a short-horizon "did the policy grab + lift?" reward that's
    much easier for RL to learn than the full pick-and-place sequence.

    For the full pick-place task use ``reward_object_on_target`` instead;
    they are complementary, not redundant."""
    grasped = object_grasped(
        env,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        diff_threshold=diff_threshold,
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    )
    obj: RigidObject = env.scene[object_cfg.name]
    z0 = _pickable_initial_z(env, object_cfg.name)
    lifted = (obj.data.root_pos_w[:, 2] - z0) >= lift_height
    return torch.logical_and(grasped, lifted).to(torch.float32)


def reward_lift_progress_dense(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    lift_height: float = 0.10,
    diff_threshold: float = 0.06,
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.02,
) -> torch.Tensor:
    """Dense partial-lift shaping reward, gated on a grasp.

    Returns ``clamp((pickable_z - spawn_z) / lift_height, 0, 1)`` on steps
    where the gripper is closed near the pickable, else 0. Unlike
    ``reward_lift_to_height`` (binary: fires only at the FULL lift_height)
    this gives a smooth gradient the instant the policy raises the object
    at all, bridging the grasp → partial-lift → full-lift gap.

    Essential for FRICTION GRIP, where lifts are partial/slipping and the
    binary lift reward never fires — leaving PPO with no signal that
    grasping leads anywhere (observed: zimmer_friction_v1 plateaued at 0%
    success with lift_bonus stuck at 0). Gating on the grasp prevents
    reward-hacking by merely knocking the object upward without holding it.
    """
    grasped = object_grasped(
        env,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        diff_threshold=diff_threshold,
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    )
    obj: RigidObject = env.scene[object_cfg.name]
    z0 = _pickable_initial_z(env, object_cfg.name)
    frac = torch.clamp((obj.data.root_pos_w[:, 2] - z0) / lift_height, 0.0, 1.0)
    return grasped.to(torch.float32) * frac


def pickable_lifted(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    lift_height: float = 0.10,
    diff_threshold: float = 0.06,
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.02,
    min_stable_steps: int = 10,
) -> torch.Tensor:
    """Termination: True when the policy has grasped + lifted the pickable
    by ``lift_height`` metres for ``min_stable_steps`` *consecutive* env
    steps. The stability counter blocks the 1-frame-overshoot exploit
    that a kinematic_attach env is vulnerable to — the policy can no
    longer trigger success by briefly teleporting the welded payload
    past the threshold; it has to hold it there.

    Reset behavior: counter is per-env and is zeroed on any step where
    the lift+grasp condition fails. So a momentary slip resets the
    countdown."""
    instantaneous = reward_lift_to_height(
        env,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        lift_height=lift_height,
        diff_threshold=diff_threshold,
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    ).bool()
    attr = "_stable_lift_counter"
    if not hasattr(env, attr) or getattr(env, attr).shape[0] != env.num_envs:
        setattr(env, attr, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
    counter: torch.Tensor = getattr(env, attr)
    counter.mul_(instantaneous.long())
    counter.add_(instantaneous.long())
    return counter >= min_stable_steps


def reward_lift_hold_progress(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    lift_height: float = 0.10,
    diff_threshold: float = 0.06,
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.02,
) -> torch.Tensor:
    """Per-step shaping reward that scales with how long the policy has
    been continuously holding the lift. Returns ``counter + 1`` on
    steps where the lift condition holds, where ``counter`` is the
    stable-lift counter maintained by ``pickable_lifted``.

    Critically, this *does not* reward firing the lift condition
    repeatedly — only holding it longer. A single 1-step lift gives 1
    reward total; five flicks give 1+1+1+1+1=5 total; a 5-step
    stable hold gives 1+2+3+4+5=15 total. Combined with the terminal
    success_bonus (which fires once at step 5 of a stable hold), the
    policy is incentivized to find AND maintain the hold.

    Reads the counter maintained by ``pickable_lifted``. Returns 0 if
    that counter doesn't exist yet (first step of training)."""
    instantaneous = reward_lift_to_height(
        env,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        lift_height=lift_height,
        diff_threshold=diff_threshold,
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    ).bool()
    counter = getattr(env, "_stable_lift_counter", None)
    if counter is None:
        return torch.zeros(env.num_envs, device=env.device)
    # counter holds the previous step's value at reward-eval time;
    # add 1 for the current step (matches what the terminator will
    # set the counter to right after).
    return ((counter + 1) * instantaneous.long()).to(torch.float32)


def reward_terminal_stable_lift(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pickable"),
    lift_height: float = 0.10,
    diff_threshold: float = 0.06,
    driver_joint: str = "finger_left_joint",
    closed_threshold: float = 0.02,
    min_stable_steps: int = 5,
) -> torch.Tensor:
    """+1 only on the step where ``pickable_lifted`` would terminate the
    episode (counter has just reached ``min_stable_steps`` for the first
    time). Use this as a one-shot terminal bonus paired with the
    ``pickable_lifted`` terminator — gives the policy a strong terminal
    incentive without per-step reward farming.

    Timing: Isaac Lab evaluates rewards BEFORE terminations, so the
    counter at reward-time still reflects the previous step's value.
    The success step is the one where counter is at
    ``min_stable_steps - 1`` AND the lift condition holds — that's the
    step the terminator will fire on. Returning 1 here gives 1 reward
    on the success step and 0 everywhere else."""
    instantaneous = reward_lift_to_height(
        env,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        lift_height=lift_height,
        diff_threshold=diff_threshold,
        driver_joint=driver_joint,
        closed_threshold=closed_threshold,
    ).bool()
    counter = getattr(env, "_stable_lift_counter", None)
    if counter is None:
        return torch.zeros(env.num_envs, device=env.device)
    return ((counter == min_stable_steps - 1) & instantaneous).to(torch.float32)
