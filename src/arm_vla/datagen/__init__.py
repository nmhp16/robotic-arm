"""Mimic-based demo augmentation.

Registers gym id ``Isaac-PickPlace-UR10-IK-Rel-Mimic-v0``.
"""

import gymnasium as gym

gym.register(
    id="Isaac-PickPlace-UR10-IK-Rel-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:UR10PickPlaceMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:UR10PickPlaceMimicEnvCfg",
    },
    disable_env_checker=True,
)
