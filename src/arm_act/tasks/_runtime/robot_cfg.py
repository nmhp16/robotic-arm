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


def _t3_401_simple_gripper() -> ArticulationCfg:
    """Epson T3-401 SCARA arm + simple two-prismatic-finger parallel-jaw gripper.

    Joint chain (matches assets/t3_401_simple_gripper/t3_401_simple_gripper.urdf):

        joint_1   shoulder, revolute about world Z
        joint_2   elbow, revolute about world Z
        joint_3   prismatic Z (positive = descend)
        joint_4   wrist yaw, revolute about world Z
        finger_left_joint / finger_right_joint   independent prismatic fingers
    """
    usd_path = os.path.join(_REPO_ROOT, "assets", "t3_401_simple_gripper", "t3_401_simple_gripper.usd")
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
            # Home pose: arm folded slightly, J3 retracted (shaft up), J4 zero.
            # joint_1 = -pi/2 swings the arm to the +X direction so the
            # workspace sits in front of the base in world coordinates.
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 0.0,
                "joint_4": 0.0,
                "finger_left_joint": 0.0,
                "finger_right_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            # J1 + J2 are large rotary actuators driving the planar arm.
            "shoulder_elbow": ImplicitActuatorCfg(
                joint_names_expr=["joint_1", "joint_2"],
                stiffness=1200.0,
                damping=70.0,
                friction=0.0,
                armature=0.0,
            ),
            # J3 prismatic Z drive: stiff position control to hold against gravity.
            "z_axis": ImplicitActuatorCfg(
                joint_names_expr=["joint_3"],
                stiffness=2000.0,
                damping=80.0,
                friction=0.0,
                armature=0.0,
            ),
            # J4 wrist: lighter rotary for yaw.
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint_4"],
                stiffness=200.0,
                damping=20.0,
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
    "t3_401_simple_gripper": _t3_401_simple_gripper,
}


def build_robot_cfg(robot_type: str) -> ArticulationCfg:
    if robot_type not in ROBOT_BUILDERS:
        available = ", ".join(sorted(ROBOT_BUILDERS)) or "(none)"
        raise KeyError(f"unknown robot type {robot_type!r}; available: {available}")
    return ROBOT_BUILDERS[robot_type]()
