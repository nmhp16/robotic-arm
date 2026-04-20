"""HDF5 → RLDS (TFDS) conversion for OpenVLA fine-tuning.

Reads Mimic-augmented HDF5 episodes and emits a TFDS dataset matching the
Open X-Embodiment feature schema that OpenVLA expects.

Per-step features::

    observation.image         (224, 224, 3) uint8   third-person RGB
    observation.wrist_image   (224, 224, 3) uint8   wrist RGB
    observation.state         (8,) float32          eef_pos(3) + eef_quat(4) + gripper(1)
    action                    (7,) float32          Delta pose(6) + gripper(1)
    language_instruction      string
    is_first / is_last / is_terminal

Splits: 90% of demos to train, 10% to val, by demo id.

Usage::

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
_STATE_DIM = 8
_ACTION_DIM = 7


class Ur5PickPlace(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for UR5 + 2F-85 pick-and-place demos."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial."}

    # Set by the CLI before the builder constructs.
    _input_path: str = ""
    _instruction: str = _DEFAULT_INSTRUCTION

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="UR5 + 2F-85 pick-and-place demos (sim, teleop + mimic-augmented).",
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(_IMG_H, _IMG_W, 3),
                            dtype=tf.uint8,
                            encoding_format="jpeg",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(_IMG_H, _IMG_W, 3),
                            dtype=tf.uint8,
                            encoding_format="jpeg",
                        ),
                        "state": tfds.features.Tensor(shape=(_STATE_DIM,), dtype=tf.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(_ACTION_DIM,), dtype=tf.float32),
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
            demo_ids = sorted(
                f["data"].keys(),
                key=lambda k: int(k.split("_")[-1]) if "_" in k else 0,
            )
        n_train = max(1, int(0.9 * len(demo_ids)))
        val_ids = demo_ids[n_train:] if len(demo_ids) > n_train else []
        return {
            "train": self._generate_examples(demo_ids[:n_train]),
            "val": self._generate_examples(val_ids),
        }

    def _generate_examples(self, demo_ids: list[str]) -> Iterator[tuple[str, dict[str, Any]]]:
        if not demo_ids:
            return
        with h5py.File(self._input_path, "r") as f:
            for demo_id in demo_ids:
                yield demo_id, _episode_from_h5(
                    f["data"][demo_id], demo_id, self._input_path, self._instruction
                )


def _episode_from_h5(
    demo: h5py.Group, demo_id: str, path: str, instruction: str
) -> dict[str, Any]:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=pathlib.Path, default=pathlib.Path("data/augmented/demos.hdf5"))
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/rlds"))
    p.add_argument("--instruction", type=str, default=_DEFAULT_INSTRUCTION)
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    Ur5PickPlace._input_path = str(args.input.resolve())
    Ur5PickPlace._instruction = args.instruction

    builder = Ur5PickPlace(data_dir=str(args.output))
    builder.download_and_prepare()
    info = builder.info
    print(f"built {info.full_name} at {args.output}")
    print(f"  train: {info.splits['train'].num_examples} episodes")
    if "val" in info.splits:
        print(f"  val:   {info.splits['val'].num_examples} episodes")


if __name__ == "__main__":
    main()
