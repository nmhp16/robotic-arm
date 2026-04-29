"""Auto-register gym ids for every ``<name>.yaml`` directly under ``tasks/``.

For each ``tasks/<name>.yaml`` we build three classes from the spec:

* ``<Name>EnvCfg``        — env_cfg used by gym + rollout
* ``<Name>MimicEnvCfg``   — same env wrapped with isaaclab_mimic
* ``<Name>MimicEnv``      — mimic runtime class

Classes are stashed on this module so ``gym.register`` can resolve their
``module:Class`` entry-point strings (``arm_act.tasks:<ClassName>``).

To add a new task variant: drop a ``tasks/<your_task>.yaml`` next to
``pick_place.yaml`` — the next process import re-scans and registers it.
No Python required for variants of an existing archetype.

Multiple archetypes are supported via the optional ``template:`` key in
the task YAML. The default is ``_runtime``; sibling templates (e.g.,
``_runtime_jar``) live in their own subpackages alongside it and only
need to expose ``build_env_cfg``, ``build_mimic_env``, and
``build_mimic_env_cfg`` (re-exported from ``_runtime`` if unchanged).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import gymnasium as gym

from arm_act.config import list_tasks, load

DEFAULT_TEMPLATE = "_runtime"


def _class_prefix(task_name: str) -> str:
    """`pick_place` -> `PickPlace`."""
    return "".join(part.capitalize() for part in task_name.split("_"))


def _template_builders(template_name: str) -> tuple[Callable, Callable, Callable]:
    """Resolve the three builder functions for a given template subpackage.

    Each template package must expose modules ``env_cfg``, ``mimic_env``,
    and ``mimic_env_cfg`` (typically just re-exporting ``_runtime``'s
    versions when the env side is unchanged).
    """
    base = f"arm_act.tasks.{template_name}"
    env_cfg_mod = importlib.import_module(f"{base}.env_cfg")
    mimic_env_mod = importlib.import_module(f"{base}.mimic_env")
    mimic_cfg_mod = importlib.import_module(f"{base}.mimic_env_cfg")
    return (
        env_cfg_mod.build_env_cfg,
        mimic_env_mod.build_mimic_env,
        mimic_cfg_mod.build_mimic_env_cfg,
    )


def template_for(spec: dict[str, Any]) -> str:
    """Public helper used by CLI scripts to look up the runtime template."""
    return str(spec.get("template", DEFAULT_TEMPLATE))


def _register_all() -> None:
    for task_name in list_tasks():
        spec = load(task_name)
        prefix = _class_prefix(task_name)
        template = template_for(spec)
        build_env_cfg, build_mimic_env, build_mimic_env_cfg = _template_builders(template)

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
            kwargs={"env_cfg_entry_point": f"arm_act.tasks:{env_cfg_cls.__name__}"},
            disable_env_checker=True,
        )
        gym.register(
            id=spec["task"]["mimic_gym_id"],
            entry_point=f"arm_act.tasks:{mimic_env_cls.__name__}",
            kwargs={"env_cfg_entry_point": f"arm_act.tasks:{mimic_cfg_cls.__name__}"},
            disable_env_checker=True,
        )


_register_all()
