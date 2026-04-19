"""UR5e + Robotiq 2F-85 ArticulationCfg.

Isaac Lab's ``isaaclab_assets.robots.universal_robots`` ships UR10/UR10e but
no UR5/UR5e. We mirror the ``UR10e_ROBOTIQ_2F_85_CFG`` pattern, pointing at
NVIDIA's Nucleus UR5e USD (which supports the same ``Robotiq_2f_85`` variant).

If the Nucleus path 404s for you, the fallback is to convert the URDF bundled
with Isaac Sim::

    # ur5e URDF is at
    #   ~/isaac/env_isaacsim/.../universal_robots/ur5e/ur5e.urdf
    ~/IsaacLab/isaaclab.sh -p ~/IsaacLab/scripts/tools/convert_urdf.py \\
        ~/isaac/env_isaacsim/lib/python3.12/site-packages/isaacsim/exts/\\
isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur5e/ur5e.urdf \\
        assets/ur5e/ur5e.usd

Then swap ``_NUCLEUS_URDF_USD`` below for a local path.
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
        # A fold that keeps the wrist above the table at the start of each
        # episode — roughly the "home" pose you'd use on a real UR5e.
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
"""Base UR5e arm, no gripper."""


# Robotiq 2F-85 variant. The USD on Nucleus exposes it as a spawn variant so
# we don't have to compose meshes manually — just toggle the variant, add
# gripper init joint states, and define gripper actuators.
UR5E_ROBOTIQ_2F_85_CFG = UR5E_CFG.copy()
UR5E_ROBOTIQ_2F_85_CFG.spawn.variants = {"Gripper": "Robotiq_2f_85"}
UR5E_ROBOTIQ_2F_85_CFG.init_state.joint_pos["finger_joint"] = 0.0
UR5E_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_inner_finger_joint"] = 0.0
UR5E_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_inner_finger_knuckle_joint"] = 0.0
UR5E_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_outer_.*_joint"] = 0.0

UR5E_ROBOTIQ_2F_85_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    # ``finger_joint`` is the single driving joint; the right-outer-knuckle
    # mimics it. PD gains copied from Isaac Lab's 2F-85 setup.
    joint_names_expr=["finger_joint"],
    effort_limit_sim=10.0,
    velocity_limit_sim=1.0,
    stiffness=11.25,
    damping=0.1,
    friction=0.0,
    armature=0.0,
)
UR5E_ROBOTIQ_2F_85_CFG.actuators["gripper_finger"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.2,
    damping=0.001,
    friction=0.0,
    armature=0.0,
)
UR5E_ROBOTIQ_2F_85_CFG.actuators["gripper_passive"] = ImplicitActuatorCfg(
    # Zero-PD passive joints closing the kinematic loop of the gripper linkage.
    joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
)
"""UR5e arm with Robotiq 2F-85 parallel-jaw gripper."""

# The `finger_joint` angle range: 0 rad = fully open, ~0.8 rad = fully closed.
# These are read by the env cfg to wire up the binary gripper action and the
# grasp-detection check.
UR5E_GRIPPER_OPEN_VAL: float = 0.0
UR5E_GRIPPER_CLOSE_VAL: float = 0.8
UR5E_GRIPPER_JOINT_NAME: str = "finger_joint"
