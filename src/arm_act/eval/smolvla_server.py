"""Subprocess server that hosts a SmolVLA policy for cross-venv rollout.

Spawned by ``arm_act.eval.rollout`` when ``--policy-type=smolvla``. Lives
in ``arm-act-venv`` because lerobot can't be installed into Isaac Lab's
bundled python — Isaac Lab pins ``numpy<2`` and ``transformers==4.57.6``,
lerobot 0.5 needs ``numpy>=2`` and ``transformers==5.3.0``.

Wire format on stdin/stdout: 4-byte big-endian length + pickled dict.

Requests
--------
``{"kind": "act", "state": np.ndarray(8,), "images": {cam: np.ndarray(3,H,W) uint8}, "task": str}``
  -> ``{"action": np.ndarray(action_dim,)}``

``{"kind": "reset"}``
  -> ``{"ok": True}``

EOF on stdin terminates the loop.
"""

from __future__ import annotations

import os
import sys

# CRITICAL: do this BEFORE importing torch/lerobot. SmolVLA's model loader
# emits bare ``print(...)`` calls to stdout ("Reducing the number of VLM
# layers to 16 ...", "Loading weights from local directory") and HF Hub
# may also write progress bars there. Any of that corrupts the first
# pickle length-prefix the parent reads, deadlocking the rollout. Move
# the protocol channel to a duped FD and redirect FD 1 to FD 2.
_PROTOCOL_FD = os.dup(1)
os.dup2(2, 1)
_PROTOCOL_OUT = os.fdopen(_PROTOCOL_FD, "wb", buffering=0)
sys.stdout = sys.stderr  # also catch python-level sys.stdout.write

import argparse  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import struct  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

logger = logging.getLogger("smolvla_server")


# Numpy-version-agnostic wire format for arrays. Server runs numpy 2.x;
# client (Isaac Lab venv) runs numpy 1.26 — pickled ndarrays cross-version
# break with "No module named 'numpy._core'". Convert at the boundary.
_ARR_TAG = "__np__"


def _pack_arr(arr: np.ndarray) -> tuple:
    return (_ARR_TAG, str(arr.dtype), tuple(arr.shape), arr.tobytes())


def _unpack_arrs(obj):
    if isinstance(obj, tuple) and len(obj) == 4 and obj[0] == _ARR_TAG:
        _, dtype_str, shape, raw = obj
        return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    if isinstance(obj, dict):
        return {k: _unpack_arrs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack_arrs(v) for v in obj]
    return obj


def _read_msg(stream) -> dict | None:
    hdr = stream.read(4)
    if len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    payload = stream.read(n)
    if len(payload) < n:
        return None
    return _unpack_arrs(pickle.loads(payload))


def _write_msg(stream, msg: dict) -> None:
    payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s smolvla_server %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    logger.info("loading checkpoint %s", args.checkpoint)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    pre = DataProcessorPipeline.from_pretrained(
        args.checkpoint, config_filename="policy_preprocessor.json"
    )
    post = DataProcessorPipeline.from_pretrained(
        args.checkpoint, config_filename="policy_postprocessor.json"
    )

    _write_msg(_PROTOCOL_OUT, {"ready": True})
    logger.info("ready; awaiting requests on stdin")

    n_acts = 0
    with torch.inference_mode():
        while True:
            req = _read_msg(sys.stdin.buffer)
            if req is None:
                break
            kind = req.get("kind")
            if kind == "reset":
                policy.reset()
                _write_msg(_PROTOCOL_OUT, {"ok": True})
            elif kind == "act":
                state = torch.from_numpy(np.asarray(req["state"], dtype=np.float32))
                sample: dict = {
                    "observation.state": state,
                    "task": req["task"],
                }
                for cam, img in req["images"].items():
                    # The dataset's video decoder yields float32 in [0,1] (CHW).
                    # rollout.py sends uint8 (CHW) — same as you'd get straight
                    # from the camera — so divide here.
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    sample[f"observation.images.{cam}"] = torch.from_numpy(arr)
                batch = pre(sample)
                pred = policy.select_action(batch)
                pred_unnorm = post({"action": pred})["action"]
                action = pred_unnorm.squeeze(0).cpu().numpy().astype(np.float32)
                _write_msg(_PROTOCOL_OUT, {"action": _pack_arr(action)})
                n_acts += 1
            else:
                _write_msg(_PROTOCOL_OUT, {"error": f"unknown kind: {kind}"})

    logger.info("exiting after %d act requests", n_acts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
