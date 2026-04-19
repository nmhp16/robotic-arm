"""Classify failed eval episodes using Claude Code (CLI subprocess).

Uses the local ``claude`` CLI, so it inherits Claude Code's auth (via
``claude login``) — no ``ANTHROPIC_API_KEY`` required.

Given an image path and an instruction, asks the model to pick one of
six failure labels and return JSON. Unparseable responses fall back to
``{"failure": "other", "reason": "<raw>"}``.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
from collections import Counter
from typing import Iterable

FAILURE_LABELS = (
    "collision",
    "grasp_slip",
    "drop",
    "misplacement",
    "drift",
    "other",
)


def _build_prompt(frame_path: pathlib.Path, instruction: str) -> str:
    return (
        f"A robot was commanded to: {instruction}\n\n"
        f"Open and look at the image at this absolute path using the Read tool:\n"
        f"  {frame_path}\n\n"
        "This image is the final frame of a FAILED episode. Classify the "
        "failure as exactly one of:\n"
        "  collision     — robot or arm hit something it shouldn't have\n"
        "  grasp_slip    — tried to grasp but cube slipped out\n"
        "  drop          — cube was held then dropped mid-trajectory\n"
        "  misplacement  — cube was released but not on the target\n"
        "  drift         — arm drifted or never approached the cube/target\n"
        "  other         — none of the above\n\n"
        'Respond with a single JSON object ON ONE LINE only: '
        '{"failure": "<label>", "reason": "<one short sentence>"}'
    )


def classify(frame_path: pathlib.Path, instruction: str, timeout: float = 60.0) -> dict:
    """Run one claude CLI call to classify a failure. Never raises."""
    if shutil.which("claude") is None:
        return {"failure": "other", "reason": "claude CLI not on PATH"}

    try:
        result = subprocess.run(
            [
                "claude",
                "-p",
                _build_prompt(frame_path, instruction),
                "--output-format",
                "json",
                "--allowedTools",
                "Read",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"failure": "other", "reason": f"timeout after {timeout:.0f}s"}

    if result.returncode != 0:
        return {"failure": "other", "reason": f"claude exit {result.returncode}"}

    return _extract_json(result.stdout)


def _extract_json(stdout: str) -> dict:
    """Parse the claude --output-format json envelope and recover the
    inner classification JSON from the assistant's final text."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        return {"failure": "other", "reason": "non-JSON CLI output"}

    text = envelope.get("result") or envelope.get("response") or ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"failure": "other", "reason": text.strip()[:120] or "empty response"}

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"failure": "other", "reason": text[start : end + 1][:120]}

    label = parsed.get("failure", "other")
    if label not in FAILURE_LABELS:
        label = "other"
    return {
        "failure": label,
        "reason": str(parsed.get("reason", ""))[:200],
    }


def summarize(records: Iterable[dict]) -> dict:
    """Aggregate per-episode labels into a histogram."""
    counts = Counter(r.get("failure", "other") for r in records)
    return {label: int(counts.get(label, 0)) for label in FAILURE_LABELS}
