"""Auto-register gym ids for every ``<name>.yaml`` directly under ``tasks/``.

For each ``tasks/<name>.yaml`` we build three classes from the spec:

* ``<Name>EnvCfg``        — env_cfg used by gym + rollout
* ``<Name>MimicEnvCfg``   — same env wrapped with isaaclab_mimic
* ``<Name>MimicEnv``      — mimic runtime class

Classes are stashed on this module so ``gym.register`` can resolve their
``module:Class`` entry-point strings (``arm_vla.tasks:<ClassName>``).

To add a new task variant: drop a ``tasks/<your_task>.yaml`` next to
``pick_place.yaml`` — the next process import re-scans and registers it.
No Python required for variants of an existing archetype.
"""

from __future__ import annotations

import gymnasium as gym

from arm_vla.config import list_tasks, load
from arm_vla.tasks._runtime.env_cfg import build_env_cfg
from arm_vla.tasks._runtime.mimic_env import build_mimic_env
from arm_vla.tasks._runtime.mimic_env_cfg import build_mimic_env_cfg


def _class_prefix(task_name: str) -> str:
    """`pick_place` -> `PickPlace`."""
    return "".join(part.capitalize() for part in task_name.split("_"))


def _register_all() -> None:
    for task_name in list_tasks():
        spec = load(task_name)
        prefix = _class_prefix(task_name)

        env_cfg_cls = build_env_cfg(spec, f"{prefix}EnvCfg")
        mimic_env_cls = build_mimic_env(spec, f"{prefix}MimicEnv")
        mimic_cfg_cls = build_mimic_env_cfg(env_cfg_cls, spec, f"{prefix}MimicEnvCfg")

        # Stash the classes on this module so the gym entry-point strings resolve.
        globals()[env_cfg_cls.__name__] = env_cfg_cls
        globals()[mimic_env_cls.__name__] = mimic_env_cls
        globals()[mimic_cfg_cls.__name__] = mimic_cfg_cls

        gym.register(
            id=spec["task"]["gym_id"],
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"arm_vla.tasks:{env_cfg_cls.__name__}"},
            disable_env_checker=True,
        )
        gym.register(
            id=spec["task"]["mimic_gym_id"],
            entry_point=f"arm_vla.tasks:{mimic_env_cls.__name__}",
            kwargs={"env_cfg_entry_point": f"arm_vla.tasks:{mimic_cfg_cls.__name__}"},
            disable_env_checker=True,
        )


_register_all()
