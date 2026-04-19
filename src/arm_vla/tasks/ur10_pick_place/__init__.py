"""UR10 pick-and-place env.

Registers gym id ``Isaac-PickPlace-UR10-IK-Rel-v0``.
"""

import gymnasium as gym

gym.register(
    id="Isaac-PickPlace-UR10-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_ur10_env_cfg:UR10PickPlaceEnvCfg",
    },
    disable_env_checker=True,
)
