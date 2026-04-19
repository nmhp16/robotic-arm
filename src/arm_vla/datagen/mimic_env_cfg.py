"""Mimic cfg for UR5e pick-and-place.

Combines ``UR5PickPlaceEnvCfg`` with ``MimicEnvCfg`` and declares two
subtasks: grasp the cube, then place it on the target.
"""

from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from arm_vla.tasks.ur5_pick_place.pick_place_ur5_env_cfg import UR5PickPlaceEnvCfg


@configclass
class UR5PickPlaceMimicEnvCfg(UR5PickPlaceEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "ur5_pick_place_mimic_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        self.subtask_configs["ur5"] = [
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp the blue cube",
                next_subtask_description="Place the cube on the green target",
            ),
            SubTaskConfig(
                object_ref="target",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place the cube on the green target",
            ),
        ]
