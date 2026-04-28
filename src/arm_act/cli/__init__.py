"""Thin CLI entry points used by the shell scripts in ``scripts/``.

Each module here resolves ``--task <name>`` to ``arm_act/tasks/<name>.yaml``
via :func:`arm_act.config.load`, then dispatches to the right Isaac Lab
tool or runtime template. Shell scripts stay one-liners; the routing
logic lives here so it can be tested.
"""

from __future__ import annotations

import os
import pathlib
import sys


def isaaclab_script(rel_path: str) -> pathlib.Path:
    """Resolve a path relative to ``$ISAACLAB`` and verify it exists.

    Args:
        rel_path: Path relative to the Isaac Lab install root,
            e.g. ``"scripts/tools/record_demos.py"``.

    Returns:
        The absolute path.

    Raises:
        SystemExit: With status 2 if the file does not exist.
    """
    isaaclab = pathlib.Path(os.environ.get("ISAACLAB", os.path.expanduser("~/IsaacLab")))
    path = isaaclab / rel_path
    if not path.is_file():
        print(f"could not find {path} — set ISAACLAB=...", file=sys.stderr)
        raise SystemExit(2)
    return path


def register_tasks() -> None:
    """Trigger gym registration of every ``tasks/<name>.yaml``.

    Imports :mod:`arm_act.tasks` for its side effect.
    """
    __import__("arm_act.tasks")
