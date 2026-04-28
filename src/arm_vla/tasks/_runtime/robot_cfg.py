"""ArticulationCfg for UR5 + simple two-prismatic-finger parallel-jaw gripper.

The USD is built from ``assets/ur5_simple_gripper/ur5_simple_gripper.urdf``
via ``scripts/convert_ur5_simple_gripper.py``. The gripper replaces the
troubled Robotiq 2F-85 model: two independent prismatic joints, no mimic,
no 4-bar linkage, symmetric closure out of the box. At the controller
level (binary open/close gripper action) this is identical to a real
2F-85, so policies fine-tuned here transfer to the physical robot.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_USD_PATH = os.path.join(_REPO_ROOT, "assets", "ur5_simple_gripper", "ur5_simple_gripper.usd")

_USD_PATH_2F85 = os.path.join(_REPO_ROOT, "assets", "ur5_2f85", "ur5_2f85.usd")

UR5_ROBOTIQ_2F_85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH_2F85,
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
            "robotiq_85_left_knuckle_joint": 0.0,
            "robotiq_85_right_knuckle_joint": 0.0,
            "robotiq_85_left_inner_knuckle_joint": 0.0,
            "robotiq_85_right_inner_knuckle_joint": 0.0,
            "robotiq_85_left_finger_tip_joint": 0.0,
            "robotiq_85_right_finger_tip_joint": 0.0,
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
            joint_names_expr=["robotiq_85_.*_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.0,
            stiffness=50.0,
            damping=2.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""UR5 arm + Robotiq 2F-85 parallel-jaw gripper. The same gripper used in OXE
datasets (berkeley_autolab_ur5, taco_play, bridge), so the OpenVLA-7B prior
sees a familiar end-effector. Pre-existed before commit b8ded24."""


UR5_SIMPLE_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
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
            # Gripper starts fully open (fingers at joint limit 0.0).
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
        # Both fingers driven by BinaryJointPositionActionCfg with identical
        # close/open targets — no mimic relationship needed because they're
        # each directly commanded.
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
"""UR5 arm + simple two-prismatic-finger parallel-jaw gripper."""
