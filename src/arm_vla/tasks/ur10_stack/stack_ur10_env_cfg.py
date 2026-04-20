"""UR10 stacking task: pick blue cube, stack on top of red cube.

Reuses the pick-and-place scene + action + camera setup. Two differences
from ``UR10PickPlaceEnvCfg``:

1. The "target" is a second dynamic cube (red) instead of a kinematic pad.
2. Success = blue cube resting on top of red cube in the xy plane
   (height ~= cube size) with suction released.
"""

from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from arm_vla.tasks.ur10_pick_place import mdp
from arm_vla.tasks.ur10_pick_place.pick_place_ur10_env_cfg import UR10PickPlaceEnvCfg


def _cube_on_cube(
    env,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    xy_threshold: float = 0.04,
    height_min: float = 0.025,
    height_max: float = 0.06,
):
    """Success: blue cube resting on top of red cube, suction released."""
    import torch
    from isaaclab.assets import RigidObject

    cube: RigidObject = env.scene[cube_cfg.name]
    base: RigidObject = env.scene[target_cfg.name]

    diff = cube.data.root_pos_w - base.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(diff[:, :2], dim=1)
    height_ok = (diff[:, 2] > height_min) & (diff[:, 2] < height_max)
    stacked = (xy_dist < xy_threshold) & height_ok

    state = mdp._suction_state(env)
    if state is not None:
        stacked = torch.logical_and(stacked, state == -1)
    return stacked


@configclass
class UR10StackEnvCfg(UR10PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Replace the flat green target pad with a second full cube (red).
        self.scene.target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BaseCube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0.1, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Randomize the base cube's xy (same z as pick cube — both rest on table).
        self.events.randomize_target.params["pose_range"] = {
            "x": (0.55, 0.75), "y": (-0.20, 0.20), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0),
        }
        self.events.randomize_target.params["min_separation"] = 0.12

        # Swap success termination for stacking criterion.
        self.terminations.success = DoneTerm(
            func=_cube_on_cube,
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )
