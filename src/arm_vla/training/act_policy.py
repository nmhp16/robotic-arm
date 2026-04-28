"""Vendored Action Chunking Transformer (ACT) for single-task imitation.

Faithful to the original ACT architecture (Zhao et al., 2023) minus the
CVAE encoder — empirically the latent z is optional for short-horizon
single-task BC and adds significant code/complexity. What's kept:

  - Per-camera ResNet18 feature extractor (ImageNet-init).
  - 1x1 projection to hidden_dim, sinusoidal 2D positional embedding.
  - State token via Linear projection.
  - Transformer encoder over (image tokens... + state token).
  - Transformer decoder with learned action queries cross-attending the
    encoder, producing a chunk of action_dim-vectors.

Model size on the default config (256 hidden, 4 enc / 1 dec layer, 2 cams):
roughly 30 M params — small enough to train end-to-end in <1 h on a
single GPU at batch 64, and to do inference in ms per query.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from arm_vla.training.dataset import NormStats

# ImageNet stats for ResNet18 inputs.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class ACTConfig:
    """All policy hyperparameters. Mirrors a section of config.yaml."""

    camera_keys: tuple[str, ...] = ("table_cam", "wrist_cam")
    state_dim: int = 8
    action_dim: int = 7
    chunk_size: int = 50

    hidden_dim: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    dim_feedforward: int = 1024
    dropout: float = 0.1
    pretrained_backbone: bool = True

    # Action min-max from training data (filled in by the trainer before save).
    action_min: list[float] = field(default_factory=list)
    action_max: list[float] = field(default_factory=list)
    state_mean: list[float] = field(default_factory=list)
    state_std: list[float] = field(default_factory=list)


def _sinusoidal_2d_pos_embed(h: int, w: int, dim: int) -> torch.Tensor:
    """Standard 2D sinusoidal positional embedding, (h*w, dim)."""
    assert dim % 4 == 0, "dim must be divisible by 4"
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    quarter = dim // 4
    omega = torch.arange(quarter, dtype=torch.float32) / quarter
    omega = 1.0 / (10000**omega)  # (quarter,)

    out_x = x.flatten()[:, None] * omega[None, :]  # (h*w, quarter)
    out_y = y.flatten()[:, None] * omega[None, :]
    pe = torch.cat(
        [out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1
    )  # (h*w, dim)
    return pe


class ImageBackbone(nn.Module):
    """ResNet18 truncated at the last conv block."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        # layer4 ends with 512-channel feature map at 1/32 resolution.
        # For 224x224 input that's (B, 512, 7, 7).
        self.extractor = create_feature_extractor(net, return_nodes={"layer4": "feat"})
        self.feat_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)["feat"]


class ACTModel(nn.Module):
    """Transformer encoder-decoder over multi-cam images + proprio state."""

    def __init__(self, cfg: ACTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.camera_keys = tuple(cfg.camera_keys)

        # Separate backbone per camera — table_cam and wrist_cam see very
        # different distributions, and shared weights underperformed in
        # the ACT paper's ablations.
        self.backbones = nn.ModuleDict(
            {k: ImageBackbone(pretrained=cfg.pretrained_backbone) for k in self.camera_keys}
        )
        # 1x1 conv projection 512 -> hidden_dim.
        self.input_proj = nn.ModuleDict(
            {k: nn.Conv2d(512, cfg.hidden_dim, kernel_size=1) for k in self.camera_keys}
        )
        self.state_proj = nn.Linear(cfg.state_dim, cfg.hidden_dim)

        self.action_queries = nn.Embedding(cfg.chunk_size, cfg.hidden_dim)
        self.action_head = nn.Linear(cfg.hidden_dim, cfg.action_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_decoder_layers)

        # 7x7 image features for 224x224 input through ResNet18.
        self._img_h = self._img_w = 7
        self.register_buffer(
            "_img_pos_embed",
            _sinusoidal_2d_pos_embed(self._img_h, self._img_w, cfg.hidden_dim),
            persistent=False,
        )
        # One extra learnable embedding for the state token (one per cam set).
        self.state_pos_embed = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        nn.init.normal_(self.state_pos_embed, std=0.02)

        self.register_buffer("_imagenet_mean", _IMAGENET_MEAN, persistent=False)
        self.register_buffer("_imagenet_std", _IMAGENET_STD, persistent=False)

    def _normalize_image(self, img_uint8: torch.Tensor) -> torch.Tensor:
        # uint8 -> float [0,1] -> ImageNet z-score, in the model so the
        # dataset stays in raw bytes (saves memory + worker time).
        img = img_uint8.to(self._imagenet_mean.dtype) / 255.0
        return (img - self._imagenet_mean) / self._imagenet_std

    def _encode_images(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        # Returns (B, n_cams * 49, hidden_dim) with positional embedding added.
        feats = []
        for k in self.camera_keys:
            x = self._normalize_image(images[k])  # (B, 3, 224, 224)
            f = self.backbones[k](x)  # (B, 512, 7, 7)
            f = self.input_proj[k](f)  # (B, hidden, 7, 7)
            f = f.flatten(2).transpose(1, 2)  # (B, 49, hidden)
            f = f + self._img_pos_embed.unsqueeze(0)
            feats.append(f)
        return torch.cat(feats, dim=1)

    def forward(self, images: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """Return (B, chunk_size, action_dim) action prediction."""
        B = state.shape[0]
        img_tokens = self._encode_images(images)  # (B, n_cams * 49, hidden)

        state_token = self.state_proj(state).unsqueeze(1)  # (B, 1, hidden)
        state_token = state_token + self.state_pos_embed
        memory_in = torch.cat([img_tokens, state_token], dim=1)  # (B, n_cams*49 + 1, hidden)

        memory = self.encoder(memory_in)
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, chunk, hidden)
        decoded = self.decoder(queries, memory)  # (B, chunk, hidden)
        return self.action_head(decoded)


class ACTPolicy:
    """Inference-time wrapper: load + select_action with chunk caching.

    During eval we predict a chunk of `chunk_size` actions every
    `action_horizon` steps, then advance the cursor through the chunk.
    Default `action_horizon = chunk_size`, i.e. open-loop chunk replay
    (matches the original ACT paper's "no temporal aggregation" mode).
    """

    def __init__(
        self,
        model: ACTModel,
        action_stats: NormStats,
        state_stats: NormStats,
        device: str = "cuda",
        action_horizon: int | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.action_stats = action_stats
        self.state_stats = state_stats
        self.device = device
        self.action_horizon = action_horizon or self.model.cfg.chunk_size

        self._chunk: torch.Tensor | None = None  # (chunk_size, action_dim) cached prediction
        self._cursor: int = 0

    def reset(self) -> None:
        self._chunk = None
        self._cursor = 0

    @torch.inference_mode()
    def select_action(self, images: dict[str, torch.Tensor], state: torch.Tensor) -> np.ndarray:
        """Predict (or replay from cache) the next action.

        ``images``: dict[cam_name -> Tensor(3, H, W) uint8 on any device]
        ``state``:  Tensor(state_dim,) float32
        Returns a numpy array of shape (action_dim,).
        """
        if self._chunk is None or self._cursor >= self.action_horizon:
            # Re-plan: run the model to get a fresh chunk.
            batched = {k: v.to(self.device).unsqueeze(0) for k, v in images.items()}
            state_norm = self._normalize_state(state).to(self.device).unsqueeze(0)
            pred_norm = self.model(batched, state_norm)[0]  # (chunk, action_dim)
            self._chunk = self._unnormalize_action(pred_norm).cpu()
            self._cursor = 0

        a = self._chunk[self._cursor].numpy().astype(np.float32)
        self._cursor += 1
        return a

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.state_stats.mean, dtype=state.dtype, device=state.device)
        std = torch.as_tensor(self.state_stats.std, dtype=state.dtype, device=state.device)
        return (state - mean) / std

    def _unnormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        # Inverse of normalize_actions(): action_norm in [-1, 1] -> raw range.
        a_min = torch.as_tensor(self.action_stats.min, dtype=action_norm.dtype, device=action_norm.device)
        a_max = torch.as_tensor(self.action_stats.max, dtype=action_norm.dtype, device=action_norm.device)
        span = (a_max - a_min).clamp_min(1e-6)
        return (action_norm + 1.0) * 0.5 * span + a_min


def normalize_actions(actions: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """Map raw actions into [-1, 1] using the dataset min/max."""
    a_min = torch.as_tensor(stats.min, dtype=actions.dtype, device=actions.device)
    a_max = torch.as_tensor(stats.max, dtype=actions.dtype, device=actions.device)
    span = (a_max - a_min).clamp_min(1e-6)
    return (actions - a_min) / span * 2.0 - 1.0


def normalize_states(states: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """Z-score state for the proprio token."""
    mean = torch.as_tensor(stats.mean, dtype=states.dtype, device=states.device)
    std = torch.as_tensor(stats.std, dtype=states.dtype, device=states.device)
    return (states - mean) / std


def save_policy(
    out_dir: pathlib.Path,
    model: ACTModel,
    action_stats: NormStats,
    state_stats: NormStats,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    cfg_dict = asdict(model.cfg)
    cfg_dict["camera_keys"] = list(cfg_dict["camera_keys"])
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(
            {"action": action_stats.to_dict(), "state": state_stats.to_dict()},
            f,
            indent=2,
        )


def load_policy(ckpt_dir: pathlib.Path, device: str = "cuda") -> ACTPolicy:
    with open(ckpt_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["camera_keys"] = tuple(cfg_dict["camera_keys"])
    cfg = ACTConfig(**cfg_dict)
    model = ACTModel(cfg)
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    with open(ckpt_dir / "norm_stats.json") as f:
        n = json.load(f)
    action_stats = NormStats.from_dict(n["action"])
    state_stats = NormStats.from_dict(n["state"])
    return ACTPolicy(model, action_stats, state_stats, device=device)
