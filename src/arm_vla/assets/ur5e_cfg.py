"""UR5e ArticulationCfg with a surface-suction end effector.

Isaac Lab's ``isaaclab_assets.robots.universal_robots`` ships UR10/UR10e
only. This module mirrors the same pattern for the UR5e, pointing at the
UR5e USD on NVIDIA's Nucleus server. See ``README.md`` for the
local-URDF fallback.

We use a ``SurfaceGripper`` (suction) attached to ``ee_link`` rather than
a Robotiq 2F-85 variant. The 2F-85 variant in the UR5e USD spawns a
nested articulation that Isaac Lab rejects as "multiple articulations
under one prim"; untangling it requires USD surgery. Suction avoids the
problem entirely, trades parallel-jaw physics for suction physics, and
forces CPU sim (see PLAN note).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

_NUCLEUS_UR5E_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd"


UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_NUCLEUS_UR5E_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5707963267948966,
            "elbow_joint": 1.5707963267948966,
            "wrist_1_joint": -1.5707963267948966,
            "wrist_2_joint": -1.5707963267948966,
            "wrist_3_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=1320.0,
            damping=72.66,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=216.0,
            damping=29.39,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""UR5e arm, end-effector gripper attached by the env config (not the USD)."""
