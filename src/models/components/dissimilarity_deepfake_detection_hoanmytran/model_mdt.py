from __future__ import annotations

import fairseq
import torch
from torch import nn

from src.models.components.dissimilarity_deepfake_detection_hoanmytran.multiconv_cgmlp import (
    MultiConvolutionalGatingMLP,
)
from src.models.components.dissimilarity_deepfake_detection_hoanmytran.pooling import (
    MultiHeadAttentionPooling,
)


class SSLModel(nn.Module):
    def __init__(self, cp_path: str):
        super().__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.out_dim = 1024

    def extract_feat(self, input_data: torch.Tensor) -> torch.Tensor:
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        return self.model(input_data, mask=False, features_only=True)["x"]


class SwiGLU(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear_1 = nn.Linear(dimension, dimension)
        self.linear_2 = nn.Linear(dimension, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        return swish * self.linear_2(x)


class Model(nn.Module):
    """MDT-compatible adaptation of Hoan My Tran dissimilarity architecture."""

    def __init__(self, cp_path: str, num_blocks: int = 4):
        super().__init__()
        self.ssl_model = SSLModel(cp_path)
        self.feature_projection = nn.Linear(self.ssl_model.out_dim, 128)
        self.silu = SwiGLU(128)
        self.blocks = nn.ModuleList(
            [
                MultiConvolutionalGatingMLP(
                    size=128,
                    linear_units=1024,
                    arch_type="concat_fusion",
                    kernel_sizes="3,7,11,15",
                    merge_conv_kernel=15,
                    use_non_linear=True,
                    dropout_rate=0.1,
                    use_linear_after_conv=True,
                    activation="silu",
                    gate_activation="silu",
                )
                for _ in range(num_blocks)
            ]
        )
        self.pooling = MultiHeadAttentionPooling(512)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SELU(inplace=True),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        hidden_states = self.feature_projection(x_ssl_feat)
        hidden_states = self.silu(hidden_states)

        x = hidden_states
        hidden_states_processed = []
        for block in self.blocks:
            x = block(x)
            hidden_states_processed.append(x)

        hidden_states_processed = torch.stack(hidden_states_processed, dim=1)
        hidden_states_processed = hidden_states_processed.view(
            hidden_states_processed.shape[0], hidden_states_processed.shape[2], -1
        )
        pooled = self.pooling(hidden_states_processed.permute(0, 2, 1)).permute(0, 2, 1)
        logits = self.classifier(pooled.squeeze(1))
        return logits

