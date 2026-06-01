"""HDF5-backed PyTorch dataset for action-chunking imitation learning.

Reads the Mimic-augmented HDF5 (``data/augmented/demos.hdf5`` by default)
directly — no intermediate format conversion. Each ``__getitem__`` returns
one window:

    images:      dict[str, Tensor(3, H, W) uint8]   one per camera
    state:       Tensor(state_dim,) float32         eef_pos(3) + eef_quat(4) + gripper(1)
    action:      Tensor(chunk, action_dim) float32  delta_pose(6) + gripper(1)
    action_pad:  Tensor(chunk,) bool                True where padded past episode end

The dataset also computes per-dim action/state stats once at init so the
trainer can save them next to the checkpoint for inference-time
unnormalization.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class NormStats:
    """Per-dim min/max/mean/std for a tensor stream."""

    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> NormStats:
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float32),
            std=np.asarray(d["std"], dtype=np.float32),
            min=np.asarray(d["min"], dtype=np.float32),
            max=np.asarray(d["max"], dtype=np.float32),
        )


class HDF5DemoDataset(Dataset):
    """Action-chunked windows from an Isaac Lab Mimic HDF5 demo file."""

    # Default proprio-only state (8D): eef_pos(3) + eef_quat(4) + gripper(1).
    DEFAULT_STATE_KEYS = ("eef_pos", "eef_quat", "gripper_pos")

    def __init__(
        self,
        hdf5_path: str | pathlib.Path,
        camera_keys: tuple[str, ...] = ("table_cam", "wrist_cam"),
        chunk_size: int = 50,
        state_keys: tuple[str, ...] | None = None,
        holdout_last: int = 0,
    ) -> None:
        self.hdf5_path = str(hdf5_path)
        self.camera_keys = tuple(camera_keys)
        self.chunk_size = int(chunk_size)
        # Reserve the last N demos (by id order) as a held-out eval set — they are
        # excluded from the index AND the norm stats. self.heldout_ids records them
        # so an offline eval can target exactly the unseen demos.
        self.holdout_last = int(holdout_last)
        self.heldout_ids: list[str] = []
        # Which obs/* arrays concatenate into the policy state vector. Defaults to
        # proprio-only (8D) for the camera ACT; a state-only (camera-less) policy
        # should add object state (e.g. pickable_pos, target_pos) so it can perceive
        # the pick location — at deploy time supplied by a detector (see memory
        # detector->state-policy). Each key's obs array must be (T, d).
        self.state_keys = tuple(state_keys) if state_keys else self.DEFAULT_STATE_KEYS
        self._h5: h5py.File | None = None  # opened lazily per DataLoader worker

        # Build the (demo_id, t) index and gather stats by scanning once.
        actions_all: list[np.ndarray] = []
        states_all: list[np.ndarray] = []
        index: list[tuple[str, int, int]] = []  # (demo_id, t, ep_len)

        with h5py.File(self.hdf5_path, "r") as f:
            demo_ids = sorted(
                f["data"].keys(),
                key=lambda k: int(k.split("_")[-1]) if "_" in k else 0,
            )
            if self.holdout_last > 0:
                self.heldout_ids = demo_ids[-self.holdout_last:]
                demo_ids = demo_ids[: -self.holdout_last]
            skipped = 0
            kept_ids: list[str] = []
            for did in demo_ids:
                d = f["data"][did]
                # Mimic + teleop sometimes write episodes that only contain
                # `initial_state` (aborted reset before any action was sent).
                # Skip them — they have no demo data to learn from.
                if "actions" not in d or "obs" not in d:
                    skipped += 1
                    continue
                kept_ids.append(did)
                actions = np.asarray(d["actions"], dtype=np.float32)
                if actions.shape[0] == 0:
                    skipped += 1
                    continue
                obs = d["obs"]
                parts = [np.asarray(obs[k], dtype=np.float32).reshape(actions.shape[0], -1)
                         for k in self.state_keys]
                state = np.concatenate(parts, axis=1)

                actions_all.append(actions)
                states_all.append(state)
                ep_len = actions.shape[0]
                # Sample any starting timestep; chunks past the end are
                # padded with the final action and masked out of the loss.
                for t in range(ep_len):
                    index.append((did, t, ep_len))

        self.index = index
        if not actions_all:
            raise RuntimeError(
                f"no usable demos in {self.hdf5_path} (skipped {skipped} incomplete)"
            )
        actions_cat = np.concatenate(actions_all, axis=0)
        states_cat = np.concatenate(states_all, axis=0)

        # Last action dim is the binary gripper command — mean/std on a
        # bimodal signal is meaningless, so we'll mask it out at norm time.
        self.action_stats = NormStats(
            mean=actions_cat.mean(0).astype(np.float32),
            std=(actions_cat.std(0) + 1e-6).astype(np.float32),
            min=actions_cat.min(0).astype(np.float32),
            max=actions_cat.max(0).astype(np.float32),
        )
        self.state_stats = NormStats(
            mean=states_cat.mean(0).astype(np.float32),
            std=(states_cat.std(0) + 1e-6).astype(np.float32),
            min=states_cat.min(0).astype(np.float32),
            max=states_cat.max(0).astype(np.float32),
        )
        self.state_dim = states_cat.shape[1]
        self.action_dim = actions_cat.shape[1]

        logger.info(
            "HDF5DemoDataset: %d windows from %d demos (skipped %d), chunk=%d, state_dim=%d, action_dim=%d",
            len(self.index),
            len(set(d for d, _, _ in self.index)),
            skipped,
            self.chunk_size,
            self.state_dim,
            self.action_dim,
        )

    def _open(self) -> h5py.File:
        # h5py file handles aren't safe to share across worker processes,
        # so each worker opens its own on first access.
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        demo_id, t, ep_len = self.index[idx]
        f = self._open()
        d = f["data"][demo_id]
        obs = d["obs"]

        # Images: (T, H, W, 3) uint8 -> (3, H, W) uint8 for one frame.
        images: dict[str, torch.Tensor] = {}
        for cam in self.camera_keys:
            img = np.asarray(obs[cam][t], dtype=np.uint8)  # (H, W, 3)
            images[cam] = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        # State at time t (concatenation of the configured obs keys).
        state = np.concatenate(
            [np.asarray(obs[k][t], dtype=np.float32).reshape(-1) for k in self.state_keys],
            axis=0,
        )

        # Action chunk [t, t+chunk_size). Pad past episode end with last action.
        end = min(t + self.chunk_size, ep_len)
        actions = np.asarray(d["actions"][t:end], dtype=np.float32)
        pad_n = self.chunk_size - actions.shape[0]
        if pad_n > 0:
            last = actions[-1:] if actions.shape[0] > 0 else np.zeros((1, self.action_dim), dtype=np.float32)
            actions = np.concatenate([actions, np.repeat(last, pad_n, axis=0)], axis=0)
        action_pad = np.zeros(self.chunk_size, dtype=bool)
        action_pad[self.chunk_size - pad_n :] = True

        return {
            "images": images,
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(actions),
            "action_pad": torch.from_numpy(action_pad),
        }


def collate(batch: list[dict]) -> dict:
    """Stack a list of __getitem__ outputs into a batch."""
    images = {k: torch.stack([b["images"][k] for b in batch]) for k in batch[0]["images"]}
    return {
        "images": images,
        "state": torch.stack([b["state"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "action_pad": torch.stack([b["action_pad"] for b in batch]),
    }
