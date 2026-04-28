"""Shared helpers for sim rollout evaluations.

Both ``rollout.py`` (fine-tuned) and ``zeroshot.py`` (pretrained baseline)
share a task registry, video writer, and JSON summary writer. Kept here
so the two CLIs stay in sync when a new task is added.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


TASK_REGISTRY: dict[str, dict[str, str]] = {
    "pick_place": {
        "gym_id": "Isaac-PickPlace-UR5-IK-Rel-v0",
        "module": "arm_vla.tasks.ur5_pick_place",
        "cfg_path": "arm_vla.tasks.ur5_pick_place.pick_place_ur5_env_cfg:UR5PickPlaceEnvCfg",
        # Match the instruction baked into the RLDS training data
        # (data/rlds/ur5_pick_place/1.0.0 episodes set this string).
        "instruction": "put the blue cube on the green target",
        "unnorm_key": "ur5_pick_place",
    },
    "pick_place_ur10": {
        "gym_id": "Isaac-PickPlace-UR10-IK-Rel-v0",
        "module": "arm_vla.tasks.ur10_pick_place",
        "cfg_path": "arm_vla.tasks.ur10_pick_place.pick_place_ur10_env_cfg:UR10PickPlaceEnvCfg",
        "instruction": "pick up the blue block and place it on the green pad",
        "unnorm_key": "ur10_pick_place",
    },
    "stack": {
        "gym_id": "Isaac-Stack-UR10-IK-Rel-v0",
        "module": "arm_vla.tasks.ur10_stack",
        "cfg_path": "arm_vla.tasks.ur10_stack.stack_ur10_env_cfg:UR10StackEnvCfg",
        "instruction": "stack the blue block on top of the red block",
        "unnorm_key": "ur10_pick_place",
    },
}


def save_video(path: pathlib.Path, frames: list, fps: int = 15) -> None:
    """Write ``frames`` as an mp4 to ``path``.

    No-op if ``frames`` is empty. Logs (rather than raises) when ``imageio``
    is unavailable — videos are an artifact, not a correctness requirement.
    """
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
    except ImportError:
        logger.warning("imageio not installed; skipping video %s", path)
        return
    imageio.mimsave(str(path), frames, fps=fps)
    logger.info("wrote %s (%d frames @ %d fps)", path, len(frames), fps)


def save_summary(path: pathlib.Path, data: dict[str, Any]) -> None:
    """Pretty-print ``data`` as JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("wrote %s", path)


def setup_logging(level: int = logging.INFO) -> None:
    """Install a basic root-logger handler, idempotent across calls."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
