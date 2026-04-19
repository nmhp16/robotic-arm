"""Mimic-based demo augmentation — registers the Mimic gym id on import."""

import gymnasium as gym

gym.register(
    id="Isaac-PickPlace-UR5-IK-Rel-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:UR5PickPlaceMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:UR5PickPlaceMimicEnvCfg",
    },
    disable_env_checker=True,
)
