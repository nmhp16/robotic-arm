"""Lazy gym-id registration for every ``<name>.yaml`` directly under ``tasks/``.

For each ``tasks/<name>.yaml`` we build three classes from the spec:

* ``<Name>EnvCfg``        — env_cfg used by gym + rollout
* ``<Name>MimicEnvCfg``   — same env wrapped with isaaclab_mimic
* ``<Name>MimicEnv``      — mimic runtime class

Classes are stashed on this module so ``gym.register`` can resolve their
``module:Class`` entry-point strings (``arm_act.tasks:<ClassName>``).

To add a new task variant: drop a ``tasks/<your_task>.yaml`` next to the
existing YAMLs — the next process import re-scans and registers it.
No Python required for variants of an existing archetype.

Multiple archetypes are supported via the optional ``template:`` key in
the task YAML. The default is ``_runtime``; sibling templates (e.g.,
``_runtime_jar``) live in their own subpackages alongside it and only
need to expose ``build_env_cfg``, ``build_mimic_env``, and
``build_mimic_env_cfg`` (re-exported from ``_runtime`` if unchanged).

Registration is deferred (``register()`` must be called explicitly).
The builder pulls in ``isaaclab.sim`` which transitively requires the
Omniverse ``pxr`` Python module; ``pxr`` is only available after Isaac
Lab's ``AppLauncher`` has initialized Kit. CLI entry points and the
runtime modules call ``register()`` after launching the app.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

DEFAULT_TEMPLATE = "_runtime"

_registered = False


def _class_prefix(task_name: str) -> str:
    """`pick_plant_out` -> `PickPlantOut`."""
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


def register(force: bool = False) -> None:
    """Build env_cfg/mimic classes and call ``gym.register`` for every task.

    Idempotent: subsequent calls are no-ops unless ``force=True``. Must be
    invoked AFTER Isaac Lab's ``AppLauncher`` initializes (the env_cfg
    builder imports ``isaaclab.sim``, which in turn imports the Omniverse
    ``pxr`` python module).
    """
    global _registered
    if _registered and not force:
        return

    import gymnasium as gym

    from arm_act.config import list_tasks, load

    for task_name in list_tasks():
        spec = load(task_name)
        prefix = _class_prefix(task_name)
        template = template_for(spec)
        build_env_cfg, build_mimic_env, build_mimic_env_cfg = _template_builders(template)

        env_cfg_cls = build_env_cfg(spec, f"{prefix}EnvCfg")
        mimic_env_cls = build_mimic_env(spec, f"{prefix}MimicEnv")
        mimic_cfg_cls = build_mimic_env_cfg(env_cfg_cls, spec, f"{prefix}MimicEnvCfg")
        # RL variant: same scene/actions, with a shaped RewardsCfg attached.
        rl_env_cfg_cls = build_env_cfg(spec, f"{prefix}RLEnvCfg", enable_rewards=True)

        globals()[env_cfg_cls.__name__] = env_cfg_cls
        globals()[mimic_env_cls.__name__] = mimic_env_cls
        globals()[mimic_cfg_cls.__name__] = mimic_cfg_cls
        globals()[rl_env_cfg_cls.__name__] = rl_env_cfg_cls

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
        # RL gym id is derived from the IL gym id by replacing the trailing
        # version suffix with -RL-v0. Skip if the IL gym id doesn't end
        # cleanly (defensive — every current task ends with -v0).
        #
        # Also wires the rsl_rl agent config so Isaac Lab's bundled
        # ``rsl_rl/train.py`` can resolve the PPO hyperparams from the
        # gym registration. Per-task overrides should happen on the
        # train.py CLI; we don't fork the config per task.
        il_gym_id = spec["task"]["gym_id"]
        if il_gym_id.endswith("-v0"):
            rl_gym_id = il_gym_id[: -len("-v0")] + "-RL-v0"
            gym.register(
                id=rl_gym_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={
                    "env_cfg_entry_point": f"arm_act.tasks:{rl_env_cfg_cls.__name__}",
                    "rsl_rl_cfg_entry_point": "arm_act.training.ppo_cfg:DefaultPPORunnerCfg",
                },
                disable_env_checker=True,
            )

    _registered = True
