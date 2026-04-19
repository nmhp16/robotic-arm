"""Runtime mimic env class for UR5e pick-and-place.

The delta-pose math for a 6-DoF IK-relative + 1-D gripper action space is
identical to Franka stack's, so we subclass the Isaac Lab Franka mimic env
and only override ``get_subtask_term_signals`` — our subtask names are
``grasp`` and ``place`` (Franka stack uses grasp_1 / stack_1 / grasp_2).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv


class UR5PickPlaceMimicEnv(FrankaCubeStackIKRelMimicEnv):
    """Mimic runtime env for UR5e pick-and-place."""

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        # Only ``grasp`` has a term signal — the final ``place`` subtask ends
        # at episode-end, no explicit edge detection needed.
        return {"grasp": subtask_terms["grasp"][env_ids]}
