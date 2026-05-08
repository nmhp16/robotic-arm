"""Client wrapper that mimics the ACT policy interface but forwards
inference to ``arm_act.eval.smolvla_server`` running in ``arm-act-venv``.

This file runs in Isaac Lab's bundled python; the server runs in
arm-act-venv. Communication is 4-byte length-prefixed pickle over stdin/
stdout pipes. See ``smolvla_server.py`` for the protocol.
"""

from __future__ import annotations

import logging
import os
import pickle
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RemotePolicy:
    """Drop-in for ACTPolicy so rollout.py doesn't have to branch."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        server_python: Path,
        task_instruction: str,
        camera_keys: list[str],
        chunk_size: int = 50,
        action_horizon: int | None = None,
    ) -> None:
        cmd = [
            str(server_python),
            "-m",
            "arm_act.eval.smolvla_server",
            "--checkpoint",
            str(checkpoint),
        ]
        env = os.environ.copy()
        # Isaac Lab's PYTHONPATH points at its own site-packages; that would
        # poison the server's import resolution. Drop it.
        env.pop("PYTHONPATH", None)
        # Force unbuffered stdio in the child so length-prefixed reads on our
        # side don't block waiting for a kernel buffer flush.
        env["PYTHONUNBUFFERED"] = "1"
        logger.info("spawning smolvla_server: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # let server logs flow to terminal
            env=env,
        )
        ready = self._read_msg()
        if not ready or not ready.get("ready"):
            raise RuntimeError(f"smolvla_server failed to start: ready={ready!r}")

        self._task_instruction = task_instruction
        self.action_horizon = action_horizon if action_horizon is not None else chunk_size
        # rollout.py reads policy.model.camera_keys + policy.model.cfg.chunk_size
        self.model = SimpleNamespace(
            camera_keys=camera_keys,
            cfg=SimpleNamespace(chunk_size=chunk_size),
        )

    # Numpy-version-agnostic wire format. The server runs numpy 2.x; we run
    # numpy 1.26 (Isaac Lab pin). Pickling ndarrays directly across that gap
    # raises "No module named 'numpy._core'", so we tunnel arrays as
    # ``(_ARR_TAG, dtype_str, shape, bytes)`` tuples and reconstruct with
    # whichever numpy is local. Mirror in ``smolvla_server.py``.
    _ARR_TAG = "__np__"

    @classmethod
    def _pack_arr(cls, arr: np.ndarray) -> tuple:
        return (cls._ARR_TAG, str(arr.dtype), tuple(arr.shape), arr.tobytes())

    @classmethod
    def _unpack_arrs(cls, obj):
        if isinstance(obj, tuple) and len(obj) == 4 and obj[0] == cls._ARR_TAG:
            _, dtype_str, shape, raw = obj
            return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape).copy()
        if isinstance(obj, dict):
            return {k: cls._unpack_arrs(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._unpack_arrs(v) for v in obj]
        return obj

    def _read_msg(self) -> dict | None:
        hdr = self._proc.stdout.read(4)
        if len(hdr) < 4:
            return None
        (n,) = struct.unpack(">I", hdr)
        payload = self._proc.stdout.read(n)
        if len(payload) < n:
            return None
        return self._unpack_arrs(pickle.loads(payload))

    def _write_msg(self, msg: dict) -> None:
        payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        self._proc.stdin.write(struct.pack(">I", len(payload)))
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()

    def reset(self) -> None:
        self._write_msg({"kind": "reset"})
        ack = self._read_msg()
        if not ack or not ack.get("ok"):
            raise RuntimeError(f"reset failed: {ack!r}")

    def select_action(self, cam_imgs: dict[str, torch.Tensor], state: torch.Tensor) -> np.ndarray:
        images = {k: self._pack_arr(img.cpu().numpy()) for k, img in cam_imgs.items()}
        self._write_msg(
            {
                "kind": "act",
                "state": self._pack_arr(state.cpu().numpy()),
                "images": images,
                "task": self._task_instruction,
            }
        )
        resp = self._read_msg()
        if resp is None or "action" not in resp:
            raise RuntimeError(f"act failed: resp={resp!r}")
        return resp["action"]

    def close(self) -> None:
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
