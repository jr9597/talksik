"""Small pre-norm transformer encoder over log-mel frames."""

from __future__ import annotations

import torch
from torch import nn

from audio import N_MELS, n_frames


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.ff(self.norm2(x))
        return x


class VoiceEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        frames: int | None = None,
        dim: int = 96,
        layers: int = 4,
        heads: int = 4,
        n_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        frames = n_frames() if frames is None else frames
        self.proj = nn.Linear(n_mels, dim)
        self.pos = nn.Parameter(torch.zeros(1, frames, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(EncoderBlock(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (batch, frames, n_mels)
        x = self.proj(mel) + self.pos[:, : mel.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x.mean(dim=1)))
