"""Unit tests for ``arm_vla.eval.common`` helpers."""

from __future__ import annotations

import json

import pytest


def test_save_summary_writes_json(tmp_path) -> None:
    from arm_vla.eval.common import save_summary

    path = tmp_path / "nested" / "summary.json"
    data = {"task": "pick_place", "success_rate": 0.0, "episodes": [1, 2, 3]}
    save_summary(path, data)

    assert path.exists()
    assert json.loads(path.read_text()) == data


def test_save_video_empty_is_noop(tmp_path) -> None:
    from arm_vla.eval.common import save_video

    path = tmp_path / "empty.mp4"
    save_video(path, [], fps=15)

    assert not path.exists()


def test_save_video_writes_when_imageio_available(tmp_path) -> None:
    imageio = pytest.importorskip("imageio")  # noqa: F841 — import guard only
    np = pytest.importorskip("numpy")
    from arm_vla.eval.common import save_video

    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)]
    path = tmp_path / "out.mp4"
    save_video(path, frames, fps=10)

    assert path.exists() and path.stat().st_size > 0


def test_setup_logging_is_idempotent() -> None:
    from arm_vla.eval.common import setup_logging

    # basicConfig is a no-op when handlers already exist (e.g. pytest's
    # capture handler). We just assert it runs twice without raising —
    # CLI entry points call it once per process, so level-setting is
    # verified there, not here.
    setup_logging()
    setup_logging()
