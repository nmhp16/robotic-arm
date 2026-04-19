"""Runtime Mimic env class for UR10 pick-and-place."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv


class UR10PickPlaceMimicEnv(FrankaCubeStackIKRelMimicEnv):
    """Reuses the Franka stack IK-rel delta-pose math. Only subtask names differ."""

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        # The final `place` subtask ends at episode end; no edge signal needed.
        return {"grasp": subtask_terms["grasp"][env_ids]}
