"""UR10 cube-stack variant of pick-and-place.

Registers gym id ``Isaac-Stack-UR10-IK-Rel-v0`` — blue cube must end up
on top of a red cube instead of on a flat target pad.
"""

import gymnasium as gym

gym.register(
    id="Isaac-Stack-UR10-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ur10_env_cfg:UR10StackEnvCfg",
    },
    disable_env_checker=True,
)
