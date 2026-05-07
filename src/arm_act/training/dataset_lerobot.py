"""Convert Isaac Lab oracle hdf5 demos to LeRobotDataset format.

Why: SmolVLA / pi0 / pi0-fast are fine-tuned via LeRobot's training CLI,
which only ingests the LeRobotDataset on-disk format (Parquet metadata
+ MP4 video tracks per camera). The oracle pipeline writes
Robomimic-style hdf5 (one dataset per task, one Group per demo, dense
arrays per observation key). This module translates the second to the
first.

Schema mapping for a UR5 + OmniPicker delta-IK task:

  hdf5                                        LeRobotDataset
  ──────────────────────────────────────────  ──────────────────────────────────
  obs/eef_pos       (T, 3)   float32          ─┐
  obs/eef_quat      (T, 4)   float32          ─┼─→ observation.state    (T, 8)
  obs/gripper_pos   (T, 1)   float32          ─┘
  obs/table_cam     (T, H, W, 3)  uint8        →  observation.images.table_cam
  obs/wrist_cam     (T, H, W, 3)  uint8        →  observation.images.wrist_cam
  actions           (T, 7)   float32           →  action
  attrs.success     bool                       →  per-episode metadata

The task instruction (constant per demo, drawn from the YAML's
``task.instruction``) is attached as the ``task`` field of every frame
— SmolVLA's text encoder uses it as the language conditioning.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Default state composition: TCP pose + gripper position. 3 + 4 + 1 = 8.
# This matches what the IK action commands operate against and is what
# SmolVLA / pi0 typically expect for arm + parallel-jaw setups (joint
# positions are also valid; switch via the ``state_keys`` arg).
DEFAULT_STATE_KEYS: tuple[str, ...] = ("eef_pos", "eef_quat", "gripper_pos")
DEFAULT_CAMERAS: tuple[str, ...] = ("table_cam", "wrist_cam")


def _build_state(obs: h5py.Group, state_keys: list[str]) -> np.ndarray:
    """Concatenate the configured obs keys into a (T, state_dim) float32 array."""
    arrays = [np.asarray(obs[k], dtype=np.float32) for k in state_keys]
    return np.concatenate(arrays, axis=1)


def convert_hdf5_to_lerobot(
    *,
    hdf5_path: Path,
    output_root: Path,
    repo_id: str,
    instruction: str,
    fps: int = 20,
    cameras: list[str] | None = None,
    state_keys: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert a Robomimic hdf5 demo file to a LeRobotDataset on disk.

    Parameters
    ----------
    hdf5_path
        Path to ``demos.hdf5`` written by the oracle / mimic pipeline.
    output_root
        Root directory under which the LeRobotDataset will be created;
        the actual dataset lives at ``output_root / repo_id``.
    repo_id
        LeRobot's identifier for the dataset (e.g., ``local/pick_vial``).
        Forms the relative directory layout under ``output_root``.
    instruction
        Language string attached to every frame as the ``task`` column.
        SmolVLA's text encoder consumes this verbatim.
    fps
        Effective control frequency. For Isaac Lab with ``sim.dt=0.01`` and
        ``decimation=5`` this is 20 Hz.
    cameras
        Which obs.* image keys to include as separate camera tracks.
        Each becomes ``observation.images.<key>`` in the dataset.
    state_keys
        Which obs.* keys to concatenate into ``observation.state``.
        Default: end-effector pose + gripper position (8 D).
    overwrite
        If True, delete an existing dataset at the target path first.

    Returns
    -------
    Path
        The directory the LeRobotDataset was written to.
    """
    # Imported here so the module is importable even when LeRobot isn't installed
    # (e.g., when only ACT training is needed). The user runs this only after
    # `pip install lerobot` in their training venv.
    # Path was `lerobot.common.datasets.lerobot_dataset` pre-0.5; the `common.`
    # namespace was removed in 0.5.0.
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:  # fallback for older lerobot installs
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    cams = list(cameras or DEFAULT_CAMERAS)
    sk = list(state_keys or DEFAULT_STATE_KEYS)

    dataset_dir = output_root / repo_id
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dataset_dir} already exists; pass overwrite=True to replace."
            )
        logger.info("removing existing dataset at %s", dataset_dir)
        shutil.rmtree(dataset_dir)

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.replace("demo_", "")))
        if not demos:
            raise ValueError(f"hdf5 has no demos: {hdf5_path}")

        # Probe the first demo to fix the feature spec before creating the dataset.
        first = f["data"][demos[0]]
        state0 = _build_state(first["obs"], sk)
        action0 = np.asarray(first["actions"], dtype=np.float32)

        features: dict[str, dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state0.shape[1],),
                "names": _state_names(sk),
            },
            "action": {
                "dtype": "float32",
                "shape": (action0.shape[1],),
                "names": _action_names(action0.shape[1]),
            },
        }
        for cam in cams:
            ref = first["obs"][cam]
            if ref.ndim != 4 or ref.shape[-1] != 3:
                raise ValueError(
                    f"camera {cam!r} has unexpected shape {ref.shape}; "
                    f"expected (T, H, W, 3)."
                )
            # LeRobotDataset wants channel-first images (C, H, W).
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": (3, ref.shape[1], ref.shape[2]),
                "names": ["channels", "height", "width"],
            }

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=dataset_dir,
            use_videos=True,
        )

        total_frames = 0
        kept = 0
        for demo_key in demos:
            ep = f["data"][demo_key]
            num_samples = int(ep.attrs.get("num_samples", 0))
            success = bool(ep.attrs.get("success", True))
            if not success:
                # The oracle's recorder is configured EXPORT_SUCCEEDED_ONLY so
                # this should be no-op, but defensive: skip failed demos.
                logger.info("skipping %s (success=False)", demo_key)
                continue
            if num_samples == 0 or "actions" not in ep:
                # Isaac Lab's ActionStateRecorder writes a placeholder demo
                # group containing only ``initial_state`` whenever an env
                # reset fires without producing a step. After a real success
                # is exported, the next reset-without-step adds one of these
                # stubs. Harmless but useless — skip.
                logger.info("skipping %s (empty placeholder)", demo_key)
                continue

            state = _build_state(ep["obs"], sk)
            action = np.asarray(ep["actions"], dtype=np.float32)
            cam_arrays = {
                cam: np.asarray(ep["obs"][cam], dtype=np.uint8)  # (T, H, W, 3)
                for cam in cams
            }

            for t in range(num_samples):
                frame: dict[str, Any] = {
                    "observation.state": state[t],
                    "action": action[t],
                    # LeRobot 0.5+ requires `task` as a frame field, not an
                    # add_frame kwarg. Constant per demo for single-task
                    # datasets; varies per-demo for multi-task datasets.
                    "task": instruction,
                }
                for cam, arr in cam_arrays.items():
                    # uint8 HWC -> float32 CHW in [0, 1] is what LeRobot's
                    # video writer + dataloader expect.
                    img = arr[t].astype(np.float32) / 255.0
                    frame[f"observation.images.{cam}"] = np.transpose(img, (2, 0, 1))
                dataset.add_frame(frame)

            dataset.save_episode()
            total_frames += num_samples
            kept += 1
            logger.info("episode %s: %d frames", demo_key, num_samples)

    logger.info(
        "wrote %d episodes / %d frames to %s (skipped %d placeholders)",
        kept,
        total_frames,
        dataset_dir,
        len(demos) - kept,
    )
    return dataset_dir


def _state_names(keys: list[str]) -> list[str]:
    """Human-readable per-dimension labels for the state vector."""
    names: list[str] = []
    for k in keys:
        if k == "eef_pos":
            names.extend(["x", "y", "z"])
        elif k == "eef_quat":
            names.extend(["qw", "qx", "qy", "qz"])
        elif k == "gripper_pos":
            names.append("gripper")
        elif k == "joint_pos":
            names.extend([f"joint_{i}" for i in range(8)])
        else:
            names.append(k)
    return names


def _action_names(dim: int) -> list[str]:
    if dim == 7:
        return ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    if dim == 4:
        return ["dx", "dy", "dz", "gripper"]
    return [f"a{i}" for i in range(dim)]
