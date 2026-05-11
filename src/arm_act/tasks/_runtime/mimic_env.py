"""Build a parametric mimic env class from a task spec.

Subclasses ``FrankaCubeStackIKRelMimicEnv`` and overrides the action↔pose
methods for the T3-401S SCARA's 4-D action `[dx, dy, dz, gripper]`
(position-only IK; J4 wrist yaw is pinned, so rotation carries no signal).

Franka's parent class assumes a 7-D action `[dxyz(3), axis_angle(3),
gripper(1)]`. Calling it as-is on our 4-D action slices `action[:, 3:6]`
into a 1-element "rotation" → quaternion math collapses to 2 elements →
`matrix_from_quat` raises "Expected 4 elements in a list but found 2".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import isaaclab.utils.math as PoseUtils
from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv


def build_mimic_env(spec: dict[str, Any], class_name: str) -> type:
    # Subtasks with a non-null signal need their signal returned to mimic;
    # the final subtask typically has signal=null and ends at episode end.
    signals = [st["signal"] for st in spec["mimic"]["subtasks"] if st.get("signal")]

    class _Env(FrankaCubeStackIKRelMimicEnv):
        def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
            if env_ids is None:
                env_ids = slice(None)
            subtask_terms = self.obs_buf["subtask_terms"]
            return {sig: subtask_terms[sig][env_ids] for sig in signals}

        def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
            eef_name = list(self.cfg.subtask_configs.keys())[0]
            delta_position = action[:, :3]
            curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)
            target_pos = curr_pos + delta_position
            # Rotation is mechanically fixed for SCARA (J4 yaw uncontrolled by IK).
            target_poses = PoseUtils.make_pose(target_pos, curr_rot).clone()
            return {eef_name: target_poses}

        def target_eef_pose_to_action(
            self,
            target_eef_pose_dict: dict,
            gripper_action_dict: dict,
            action_noise_dict: dict | None = None,
            env_id: int = 0,
        ) -> torch.Tensor:
            eef_name = list(self.cfg.subtask_configs.keys())[0]
            (target_eef_pose,) = target_eef_pose_dict.values()
            target_pos, _target_rot = PoseUtils.unmake_pose(target_eef_pose)
            curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
            curr_pos, _curr_rot = PoseUtils.unmake_pose(curr_pose)
            delta_position = target_pos - curr_pos
            (gripper_action,) = gripper_action_dict.values()
            pose_action = delta_position
            if action_noise_dict is not None:
                noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
                pose_action = torch.clamp(pose_action + noise, -1.0, 1.0)
            return torch.cat([pose_action, gripper_action], dim=0)

    _Env.__name__ = class_name
    _Env.__qualname__ = class_name
    return _Env
