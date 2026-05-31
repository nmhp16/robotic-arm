"""Build a parametric mimic env cfg from a task spec.

Combines the task's env_cfg with ``MimicEnvCfg`` and declares subtasks
from the YAML ``mimic.subtasks`` list.
"""

from __future__ import annotations

from typing import Any

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils.configclass import configclass


def build_mimic_env_cfg(env_cfg_cls: type, spec: dict[str, Any], class_name: str) -> type:
    """Subclass ``env_cfg_cls`` + ``MimicEnvCfg``, populate datagen_config
    and subtask_configs from the YAML spec, return the new configclass."""

    mimic = spec["mimic"]

    class _Cfg(env_cfg_cls, MimicEnvCfg):
        def __post_init__(self):
            super().__post_init__()

            self.datagen_config.name = mimic["name"]
            self.datagen_config.generation_guarantee = True
            self.datagen_config.generation_keep_failed = False
            self.datagen_config.generation_num_trials = int(mimic.get("generation_num_trials", 10))
            self.datagen_config.generation_select_src_per_subtask = True
            self.datagen_config.generation_transform_first_robot_pose = False
            self.datagen_config.generation_interpolate_from_last_target_pose = True
            self.datagen_config.generation_relative = True
            self.datagen_config.max_num_failures = int(mimic.get("max_num_failures", 25))
            self.datagen_config.seed = int(mimic.get("seed", 1))

            subtasks = []
            for st in mimic["subtasks"]:
                subtasks.append(
                    SubTaskConfig(
                        object_ref=_resolve_object(st["object"], spec),
                        subtask_term_signal=st["signal"],
                        subtask_term_offset_range=tuple(st.get("offset_range", (0, 0))),
                        selection_strategy="nearest_neighbor_object",
                        selection_strategy_kwargs={"nn_k": 3},
                        action_noise=float(st.get("action_noise", 0.03)),
                        num_interpolation_steps=int(st.get("num_interpolation_steps", 5)),
                        num_fixed_steps=int(st.get("num_fixed_steps", 0)),
                        apply_noise_during_interpolation=False,
                        description=st.get("description", ""),
                        next_subtask_description=st.get("next_description", ""),
                    )
                )
            self.subtask_configs[spec["task"]["name"]] = subtasks

    _Cfg.__name__ = class_name
    _Cfg.__qualname__ = class_name
    return configclass(_Cfg)


def _resolve_object(name: str, spec: dict[str, Any]) -> str:
    """Map a YAML object name to the runtime scene-entity name.

    The runtime spawns objects under their *role* (pickable / target),
    not their human-readable YAML name — translate here.
    """
    obj = spec["objects"].get(name)
    if obj is None:
        raise KeyError(f"mimic.subtasks references unknown object {name!r}")
    role = obj.get("role")
    if role == "pickable":
        return "pickable"
    if role == "target":
        return "target"
    raise ValueError(f"object {name!r} has unsupported role {role!r}")
