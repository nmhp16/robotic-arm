"""Runtime mimic env class for UR10 pick-and-place.

The delta-pose math for a 6-DoF IK-relative + 1-D gripper action space is
identical to Franka stack's, so we subclass the Isaac Lab Franka mimic env
and only override ``get_subtask_term_signals`` — our subtask names differ
(``grasp``, ``place`` instead of ``grasp_1``, ``stack_1``, ``grasp_2``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv


class UR10PickPlaceMimicEnv(FrankaCubeStackIKRelMimicEnv):
    """Mimic runtime env for UR10 pick-and-place."""

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        # The final subtask (``place``) doesn't need a term signal — data
        # generation uses episode end as the stop — so we only expose ``grasp``.
        return {"grasp": subtask_terms["grasp"][env_ids]}
