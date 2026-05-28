"""Custom action terms.

`BinaryJointEffortAction` is the force-control analogue of Isaac Lab's
`BinaryJointPositionAction`: on "close" / "open" it sets a joint *effort*
target (torque/force) rather than a position target. Used for gripper joints
whose actuator has stiffness=0 + low damping, so the joint position settles
at whatever the world allows under the commanded force — i.e., true
force-controlled grip (jaws squeeze with a constant N regardless of the
stem's thickness).
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class BinaryJointEffortAction(BinaryJointAction):
    """Binary joint action that sets the binary action into joint effort targets."""

    def apply_actions(self):
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class BinaryJointEffortActionCfg(BinaryJointActionCfg):
    """Force-control gripper: `open_command_expr` / `close_command_expr` are
    effort (N for prismatic, Nm for revolute) instead of position."""

    class_type: type[ActionTerm] = BinaryJointEffortAction
