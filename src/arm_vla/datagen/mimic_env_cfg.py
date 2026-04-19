"""Mimic cfg for UR10 pick-and-place.

Combines our ``UR10PickPlaceEnvCfg`` with Isaac Lab's ``MimicEnvCfg`` and wires
up two subtask configs (grasp cube → place on target). Data generation uses
curobo motion planning to transplant hand-recorded segments across randomized
scenes, so 15 teleop demos can yield 500+ synthetic ones.
"""

from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from arm_vla.tasks.ur10_pick_place.pick_place_ur10_env_cfg import UR10PickPlaceEnvCfg


@configclass
class UR10PickPlaceMimicEnvCfg(UR10PickPlaceEnvCfg, MimicEnvCfg):
    """Mimic-enabled cfg for UR10 pick-and-place.

    Two subtasks:
      1. ``grasp`` — approach and suction the cube
      2. final — carry the cube to the target and release (no term signal)
    """

    def __post_init__(self):
        super().__post_init__()

        # Data-generation knobs. ``generation_relative=True`` means target poses
        # are re-expressed relative to the object frame at the moment of segment
        # playback, which is what lets us transplant segments across randomized
        # scenes.
        self.datagen_config.name = "ur10_pick_place_mimic_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtasks = [
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="grasp",
                # Jitter the grasp/place boundary by 10–20 steps — helps the
                # augmented demos cover a wider distribution of transition
                # timings rather than collapsing to one cut point.
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
                # Final subtask: no term signal needed (ends at episode end).
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
        # The key here is the eef_name used by the runtime env (first key is
        # the one ``FrankaCubeStackIKRelMimicEnv.target_eef_pose_to_action``
        # looks up). We call it "ur10" to stay self-documenting.
        self.subtask_configs["ur10"] = subtasks
