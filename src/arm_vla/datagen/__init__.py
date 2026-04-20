"""Mimic-based demo augmentation.

Registers gym ids:
* ``Isaac-PickPlace-UR5-IK-Rel-Mimic-v0`` (default for the pipeline)
* ``Isaac-PickPlace-UR10-IK-Rel-Mimic-v0`` (kept for the legacy UR10 path)
"""

import gymnasium as gym

gym.register(
    id="Isaac-PickPlace-UR5-IK-Rel-Mimic-v0",
    entry_point=f"{__name__}.mimic_env_ur5:UR5PickPlaceMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg_ur5:UR5PickPlaceMimicEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-UR10-IK-Rel-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:UR10PickPlaceMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:UR10PickPlaceMimicEnvCfg",
    },
    disable_env_checker=True,
)
