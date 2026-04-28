"""Shared helpers for sim rollout evaluations: video + summary I/O + logging."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def save_video(path: pathlib.Path, frames: Sequence[Any], fps: int = 15) -> None:
    """Write ``frames`` to an mp4 at ``path``.

    No-op if ``frames`` is empty. Logs and returns (rather than raises)
    when ``imageio`` is unavailable — videos are an artifact, not a
    correctness requirement.

    Args:
        path: Output mp4 path. Parent directories are created if needed.
        frames: Sequence of HxWx3 uint8 arrays (one per frame).
        fps: Frames per second to encode.
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
    """Pretty-print ``data`` as JSON to ``path`` (creating dirs as needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("wrote %s", path)


def setup_logging(level: int = logging.INFO) -> None:
    """Install a basic root-logger handler. Idempotent across calls."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
