"""Build a parametric mimic env class from a task spec.

Subclasses ``FrankaCubeStackIKRelMimicEnv`` (delta-pose math; same 6-D
pose + 1-D gripper action shape) and overrides ``get_subtask_term_signals``
to return whatever signals the YAML ``mimic.subtasks`` declares.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
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

    _Env.__name__ = class_name
    _Env.__qualname__ = class_name
    return _Env
