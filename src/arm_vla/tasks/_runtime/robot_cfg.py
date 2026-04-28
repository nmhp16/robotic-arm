"""Robot ArticulationCfg lookup, keyed by ``task.robot.type`` in task.yaml.

Add a new robot variant by writing a builder here and registering it in
``ROBOT_BUILDERS`` — task.yaml then references it as ``robot.type: <key>``.
"""

from __future__ import annotations

import os
from typing import Callable

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _ur5_simple_gripper() -> ArticulationCfg:
    """UR5 arm + simple two-prismatic-finger parallel-jaw gripper."""
    usd_path = os.path.join(_REPO_ROOT, "assets", "ur5_simple_gripper", "ur5_simple_gripper.usd")
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
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
                "wrist_1_joint": 1.5707963267948966,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
                "finger_left_joint": 0.0,
                "finger_right_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*"],
                stiffness=1320.0,
                damping=72.6636085,
                friction=0.0,
                armature=0.0,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                stiffness=600.0,
                damping=34.64101615,
                friction=0.0,
                armature=0.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*"],
                stiffness=216.0,
                damping=29.39387691,
                friction=0.0,
                armature=0.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_.*_joint"],
                effort_limit_sim=50.0,
                velocity_limit_sim=0.5,
                stiffness=500.0,
                damping=30.0,
                friction=0.0,
                armature=0.0,
            ),
        },
    )


ROBOT_BUILDERS: dict[str, Callable[[], ArticulationCfg]] = {
    "ur5_simple_gripper": _ur5_simple_gripper,
}


def build_robot_cfg(robot_type: str) -> ArticulationCfg:
    if robot_type not in ROBOT_BUILDERS:
        available = ", ".join(sorted(ROBOT_BUILDERS)) or "(none)"
        raise KeyError(f"unknown robot type {robot_type!r}; available: {available}")
    return ROBOT_BUILDERS[robot_type]()
