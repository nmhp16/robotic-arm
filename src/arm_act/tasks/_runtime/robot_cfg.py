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
                # explicit velocity_limit_sim overrides the URDF's 1 m/s cap
                # (which throttled descent to ~3 mm/env-step, stalling DESCEND
                # at TCP z~0.106 in pick_plant_out_of_vial); 2 m/s lets the IK
                # actually deliver the oracle's commanded max_dz=0.05 m/step.
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
                stiffness=4000.0,
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
                # Originally stiffness=500, effort=50 — too compliant for the
                # 2 cm cuboid grip. The recorded grip joint pos peaked at
                # 0.022 m (cuboid blocks the finger at its half-width), only
                # 2 mm above the closed_threshold; GPU-PhysX FP noise during
                # grasp would knock the cuboid loose ~80% of replay attempts,
                # making the recorded actions non-reproducible. Bumped to
                # tweezer-class stiffness so the gripper clamps the cuboid
                # firmly enough that small physics divergences can't unseat it.
                effort_limit_sim=200.0,
                velocity_limit_sim=0.5,
                stiffness=4000.0,
                damping=80.0,
                friction=0.0,
                armature=0.0,
            ),
        },
    )


def _t3_401_tweezer() -> ArticulationCfg:
    """Epson T3-401 SCARA arm + long pointed tweezer end-effector.

    Identical arm and J1..J4 actuators as ``_t3_401_simple_gripper``; the
    gripper actuator drives a pair of thin tweezer halves with a 5 mm
    pinching stroke (vs 40 mm for the paddle-finger variant). Lower
    stiffness/damping than the paddle gripper because the tweezer arms
    are 1/4 the mass and over-driving them oscillates the long, thin
    arms visibly.
    """
    usd_path = os.path.join(_REPO_ROOT, "assets", "t3_401_tweezer", "t3_401_tweezer.usd")
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
            "shoulder_elbow": ImplicitActuatorCfg(
                joint_names_expr=["joint_1", "joint_2"],
                stiffness=1200.0,
                damping=70.0,
                friction=0.0,
                armature=0.0,
            ),
            "z_axis": ImplicitActuatorCfg(
                joint_names_expr=["joint_3"],
                # explicit velocity_limit_sim overrides the URDF's 1 m/s cap
                # (which throttled descent to ~3 mm/env-step, stalling DESCEND
                # at TCP z~0.106 in pick_plant_out_of_vial); 2 m/s lets the IK
                # actually deliver the oracle's commanded max_dz=0.05 m/step.
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
                stiffness=4000.0,
                damping=80.0,
                friction=0.0,
                armature=0.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint_4"],
                stiffness=200.0,
                damping=20.0,
                friction=0.0,
                armature=0.0,
            ),
            "tweezer": ImplicitActuatorCfg(
                joint_names_expr=["finger_.*_joint"],
                effort_limit_sim=200.0,       # large clamping force; thin tweezer relies
                velocity_limit_sim=0.2,        # heavily on stiffness*error to hold
                stiffness=8000.0,             # very stiff close — keeps fingers locked
                damping=80.0,                  # against the stem under lift acceleration
                friction=0.0,
                armature=0.0,
            ),
        },
    )


def _t3_401_zimmer() -> ArticulationCfg:
    """Epson T3-401 SCARA arm + Zimmer GEP2010IL-00-B parallel-jaw gripper.

    Same arm kinematics + actuators as the simple_gripper / tweezer variants.
    The gripper section uses a REAL industrial parallel-jaw mechanism (Zimmer
    GEP2010IL) extracted from manufacturer CAD, with custom long-thin
    finger blades (5×3×50 mm) for vial-mouth clearance + plant-stem reach.

    Designed for friction grip without kinematic_attach — the long inertial
    body of the Zimmer + 200 N grip force should clamp the plant stem
    reliably (vs the tweezer's <5% friction success documented in
    friction_grip_attempted.md).
    """
    usd_path = os.path.join(_REPO_ROOT, "assets", "t3_401_zimmer", "t3_401_zimmer.usd")
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
            "shoulder_elbow": ImplicitActuatorCfg(
                joint_names_expr=["joint_1", "joint_2"],
                stiffness=1200.0,
                damping=70.0,
                friction=0.0,
                armature=0.0,
            ),
            "z_axis": ImplicitActuatorCfg(
                joint_names_expr=["joint_3"],
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
                stiffness=4000.0,
                damping=80.0,
                friction=0.0,
                armature=0.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint_4"],
                stiffness=200.0,
                damping=20.0,
                friction=0.0,
                armature=0.0,
            ),
            # Zimmer GEP2010IL grip force is 50-200 N. The PD controller
            # delivers approximate force = stiffness × position-error. With
            # stiffness=12000 and a typical 1 mm overshoot past contact, force
            # ≈ 12 N — too light for a 200 N industrial gripper. Bumping
            # stiffness to 80000 puts the realistic per-mm force at ~80 N,
            # matching the lower end of Zimmer's spec. damping=200 keeps
            # the closing motion stable without oscillation.
            "zimmer_fingers": ImplicitActuatorCfg(
                joint_names_expr=["finger_.*_joint"],
                effort_limit_sim=500.0,
                velocity_limit_sim=0.1,
                stiffness=80000.0,
                damping=200.0,
                friction=0.0,
                armature=0.0,
            ),
        },
    )


def _ur5_omnipicker() -> ArticulationCfg:
    """UR5 6-axis arm + AgiBot OmniPicker parallel-jaw gripper.

    Joint chain (matches assets/ur5_omnipicker/ur5_omnipicker.urdf):

        shoulder_pan_joint   revolute about base Z
        shoulder_lift_joint  revolute (perpendicular)
        elbow_joint          revolute
        wrist_1_joint        revolute
        wrist_2_joint        revolute
        wrist_3_joint        revolute (tool roll)
        finger_left_joint /  finger_right_joint   independent prismatic fingers,
                             driven in parallel by the gripper actuator (treat
                             as a 1-DOF mimic-style gripper — both joints get
                             the same command from BinaryJointPositionActionCfg)

    Reach: ~850 mm (vs T3-401's 400 mm). Use task spawn ranges accordingly.
    Home pose places the wrist roughly above the working volume in front of
    the base — set ``ik_command_type: pose`` in the task YAML to use full
    6-DOF pose targets instead of position-only IK.
    """
    usd_path = os.path.join(_REPO_ROOT, "assets", "ur5_omnipicker", "ur5_omnipicker.usd")
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
            # "Vertical ready" home: shoulder up, elbow forward, wrist down so
            # tool0 +Z points roughly toward -Z (descending toward the table).
            # EE ends up around (0.5, 0.0, 0.4) above the table — well within
            # the workspace for the holder + tray task.
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.0,
                "elbow_joint": 1.5,
                "wrist_1_joint": -1.5,
                "wrist_2_joint": -1.5708,
                "wrist_3_joint": 0.0,
                "finger_left_joint": 0.0,
                "finger_right_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            # Shoulder + elbow: large rotary actuators carrying the bulk of
            # the arm load. Stiffness sized for IK position-tracking with
            # damping-dominated transient response.
            "shoulder_elbow": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
                stiffness=4000.0,
                damping=200.0,
                friction=0.0,
                armature=0.0,
            ),
            # Wrists: lighter actuators, lower stiffness OK because they
            # carry only the gripper + payload.
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_[1-3]_joint"],
                effort_limit_sim=28.0,
                velocity_limit_sim=3.14,
                stiffness=1500.0,
                damping=80.0,
                friction=0.0,
                armature=0.0,
            ),
            # OmniPicker fingers: same firm-clamp profile as the T3-401
            # simple gripper. The 24 mm-per-finger stroke + 5 mm closed gap
            # means the fingers must hold against PhysX FP noise once the
            # vial is captured. velocity_limit_sim is intentionally low
            # (0.1 m/s) so the gripper takes ~3 env.steps to fully close
            # rather than ~1 — this gives the kinematic_attach event term
            # (which fires once per env.step, post-physics) time to snap
            # the vial to the TCP while the fingers are still in
            # non-contact territory. With the previous 0.5 m/s, the
            # fingers slammed shut within a single env.step (5 physics
            # sub-steps × 0.01 s × 0.5 m/s = 25 mm of finger motion,
            # enough to traverse the full 14 mm vial-finger gap and
            # knock the vial 5-8 cm sideways before kinematic_attach
            # could capture it.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_.*_joint"],
                # Increased from (eff=200, k=4000) to support 400g holder
                # rack grasp via friction-only contact. The grip force
                # equation is roughly stiffness × (target − actual). At
                # k=12000 with the gripper hitting a hard stop ~3mm before
                # the closed target, normal force ≈ 36N per finger; with
                # μ=0.9 that yields friction ≈ 32N, ample for the 4N
                # weight of the rack. The vial-pick task (18g vial)
                # doesn't notice the change.
                effort_limit_sim=500.0,
                velocity_limit_sim=0.1,
                stiffness=12000.0,
                damping=120.0,
                friction=0.0,
                armature=0.0,
            ),
        },
    )


ROBOT_BUILDERS: dict[str, Callable[[], ArticulationCfg]] = {
    "t3_401_simple_gripper": _t3_401_simple_gripper,
    "t3_401_tweezer": _t3_401_tweezer,
    "t3_401_zimmer": _t3_401_zimmer,
    "ur5_omnipicker": _ur5_omnipicker,
}


def build_robot_cfg(robot_type: str) -> ArticulationCfg:
    if robot_type not in ROBOT_BUILDERS:
        available = ", ".join(sorted(ROBOT_BUILDERS)) or "(none)"
        raise KeyError(f"unknown robot type {robot_type!r}; available: {available}")
    return ROBOT_BUILDERS[robot_type]()
