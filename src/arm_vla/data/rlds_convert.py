"""HDF5 → RLDS (TFDS) conversion for OpenVLA fine-tuning.

Reads the mimic-augmented HDF5 at ``data/augmented/demos.hdf5`` and writes a
TFDS dataset matching the Open X-Embodiment (OXE) feature schema that
OpenVLA's fine-tune script consumes.

Per-step features written:

  observation.image         (224, 224, 3) uint8    — third-person (table_cam)
  observation.wrist_image   (224, 224, 3) uint8    — wrist cam
  observation.state         (8,) float32           — eef_pos(3) + eef_quat(4) + gripper(1)
  action                    (7,) float32           — Δxyz(3) + Δrpy(3) + gripper(1)
  language_instruction      string
  is_first / is_last / is_terminal  bool

A train/val split is made by demo id: first 90% to train, last 10% to val.
Language is a single fixed sentence — this is a single-task dataset, the VLA
just needs to condition on text even though all episodes share it.

Run with::

    python -m arm_vla.data.rlds_convert \\
        --input data/augmented/demos.hdf5 \\
        --output data/rlds \\
        --instruction "put the blue cube on the green target"
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Iterator

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_DEFAULT_INSTRUCTION = "put the blue cube on the green target"
_IMG_H, _IMG_W = 224, 224
_STATE_DIM = 8  # eef_pos(3) + eef_quat(4) + gripper(1)
_ACTION_DIM = 7  # 6-DoF IK-rel + binary gripper


class UR5PickPlace(tfds.core.GeneratorBasedBuilder):
    """OXE-compatible TFDS builder for our UR5e pick-and-place demos."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial."}

    # These are populated by the CLI and baked into the builder at build time —
    # TFDS requires them before _info() is called.
    _input_path: str = ""
    _instruction: str = _DEFAULT_INSTRUCTION

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="UR5e pick-and-place demos (sim, teleop + mimic-augmented).",
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(_IMG_H, _IMG_W, 3),
                            dtype=tf.uint8,
                            encoding_format="jpeg",
                            doc="Third-person RGB (table_cam).",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(_IMG_H, _IMG_W, 3),
                            dtype=tf.uint8,
                            encoding_format="jpeg",
                            doc="Wrist-mounted RGB.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(_STATE_DIM,),
                            dtype=tf.float32,
                            doc="[eef_x, eef_y, eef_z, quat_w, quat_x, quat_y, quat_z, gripper].",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(_ACTION_DIM,),
                        dtype=tf.float32,
                        doc="[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd].",
                    ),
                    "discount": tfds.features.Scalar(dtype=tf.float32),
                    "reward": tfds.features.Scalar(dtype=tf.float32),
                    "is_first": tfds.features.Scalar(dtype=tf.bool),
                    "is_last": tfds.features.Scalar(dtype=tf.bool),
                    "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "demo_id": tfds.features.Text(),
                }),
            }),
        )

    def _split_generators(self, dl_manager):  # noqa: ARG002
        with h5py.File(self._input_path, "r") as f:
            demo_ids = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k else 0)
        n_train = max(1, int(0.9 * len(demo_ids)))
        return {
            "train": self._generate_examples(demo_ids[:n_train]),
            "val": self._generate_examples(demo_ids[n_train:]) if len(demo_ids) > n_train else self._generate_examples([]),
        }

    def _generate_examples(self, demo_ids: list[str]) -> Iterator[tuple[str, dict[str, Any]]]:
        if not demo_ids:
            return
        with h5py.File(self._input_path, "r") as f:
            for demo_id in demo_ids:
                demo = f["data"][demo_id]
                yield demo_id, _episode_from_h5(demo, demo_id, self._input_path, self._instruction)


def _episode_from_h5(demo: h5py.Group, demo_id: str, path: str, instruction: str) -> dict[str, Any]:
    actions = np.asarray(demo["actions"], dtype=np.float32)
    T = actions.shape[0]
    obs = demo["obs"]["policy"]
    table_cam = np.asarray(obs["table_cam"], dtype=np.uint8)
    wrist_cam = np.asarray(obs["wrist_cam"], dtype=np.uint8)
    eef_pos = np.asarray(obs["eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(obs["eef_quat"], dtype=np.float32)
    gripper = np.asarray(obs["gripper_pos"], dtype=np.float32).reshape(T, -1)[:, :1]
    state = np.concatenate([eef_pos, eef_quat, gripper], axis=1)

    steps = []
    for t in range(T):
        steps.append({
            "observation": {
                "image": table_cam[t],
                "wrist_image": wrist_cam[t],
                "state": state[t],
            },
            "action": actions[t],
            "discount": np.float32(1.0),
            "reward": np.float32(1.0 if t == T - 1 else 0.0),
            "is_first": t == 0,
            "is_last": t == T - 1,
            "is_terminal": t == T - 1,
            "language_instruction": instruction,
        })
    return {
        "steps": steps,
        "episode_metadata": {"file_path": path, "demo_id": demo_id},
    }


def _build_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=pathlib.Path, default=pathlib.Path("data/augmented/demos.hdf5"))
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/rlds"))
    p.add_argument("--instruction", type=str, default=_DEFAULT_INSTRUCTION)
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"missing input dataset: {args.input}")

    UR5PickPlace._input_path = str(args.input.resolve())
    UR5PickPlace._instruction = args.instruction

    builder = UR5PickPlace(data_dir=str(args.output))
    builder.download_and_prepare()
    info = builder.info
    print(f"built TFDS dataset: {info.full_name} → {args.output}")
    print(f"  train: {info.splits['train'].num_examples} episodes")
    if "val" in info.splits:
        print(f"  val:   {info.splits['val'].num_examples} episodes")


if __name__ == "__main__":
    _build_cli()
