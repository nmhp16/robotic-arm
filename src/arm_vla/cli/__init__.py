"""Thin CLI entry points used by the shell scripts in ``scripts/``.

Each module here resolves ``--task <name>`` against
``arm_vla.tasks.<task>/task.yaml``, then dispatches to the right Isaac Lab
tool or task module. Shell scripts stay one-liners; the routing logic
lives here so it can be tested.
"""
