"""LCNN backend for MFCC/LFCC anti-spoofing features.

Adapted from piotrkawa/deepfake-whisper-features at
5829735a54aca3b37b48ceb59bb206cbb41560a1.
"""

from __future__ import annotations

import torch
from torch import nn


DEFAULT_NUM_COEFFICIENTS = 384


class BLSTMLayer(nn.Module):
    """Bidirectional LSTM layer that preserves sequence length."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if output_dim % 2 != 0:
            raise ValueError("BLSTMLayer output_dim must be even")
        self.blstm = nn.LSTM(input_dim, output_dim // 2, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.blstm(x.permute(1, 0, 2))
        return y.permute(1, 0, 2)


class MaxFeatureMap2D(nn.Module):
    """Max feature map activation over channel pairs."""

    def __init__(self, max_dim: int = 1) -> None:
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = list(inputs.size())
        if self.max_dim >= len(shape):
            raise ValueError(f"max_dim={self.max_dim} invalid for shape={shape}")
        if shape[self.max_dim] % 2 != 0:
            raise ValueError(f"dimension {self.max_dim} must be even for max feature map")

        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)
        return inputs.view(*shape).max(self.max_dim).values


class LCNN(nn.Module):
    """LCNN classifier for feature tensors shaped `[B, C, F, T]`.

    `F` must match `num_coefficients`; for MFCC/LFCC double-delta features this
    is normally `128 * 3 = 384`.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_coefficients: int = DEFAULT_NUM_COEFFICIENTS,
        output_dim: int = 2,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if num_coefficients % 16 != 0:
            raise ValueError("num_coefficients must be divisible by 16")

        self.num_coefficients = num_coefficients
        hidden_dim = (num_coefficients // 16) * 32

        self.transform = nn.Sequential(
            nn.Conv2d(input_channels, 64, (5, 5), 1, padding=(2, 2)),
            MaxFeatureMap2D(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(32, 64, (1, 1), 1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.BatchNorm2d(48, affine=False),
            nn.Conv2d(48, 96, (1, 1), 1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(48, affine=False),
            nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, (1, 1), 1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, (1, 1), 1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Dropout(dropout),
        )

        self.before_pooling = nn.Sequential(
            BLSTMLayer(hidden_dim, hidden_dim),
            BLSTMLayer(hidden_dim, hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"LCNN expects [B, C, F, T] or [B, F, T], got {tuple(x.shape)}")
        if x.shape[2] != self.num_coefficients:
            raise ValueError(
                f"feature dim mismatch: expected {self.num_coefficients}, got {x.shape[2]}"
            )

        batch_size = x.shape[0]
        x = x.permute(0, 1, 3, 2)
        hidden = self.transform(x)
        hidden = hidden.permute(0, 2, 1, 3).contiguous()
        frame_num = hidden.shape[1]
        hidden = hidden.view(batch_size, frame_num, -1)
        hidden_lstm = self.before_pooling(hidden)
        return self.output((hidden_lstm + hidden).mean(1))

