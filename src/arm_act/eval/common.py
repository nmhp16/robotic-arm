"""Shared helpers for sim rollout evaluations: video + summary writers + logging."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


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
