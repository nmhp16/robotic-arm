"""UR5 + Robotiq 2F-85 pick-and-place env.

Registers gym id ``Isaac-PickPlace-UR5-IK-Rel-v0``.
"""

import gymnasium as gym

gym.register(
    id="Isaac-PickPlace-UR5-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_ur5_env_cfg:UR5PickPlaceEnvCfg",
    },
    disable_env_checker=True,
)
