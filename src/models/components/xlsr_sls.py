from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class XLSRLayerFrontend(nn.Module):
    def __init__(self, ssl_pretrained_path: str) -> None:
        super().__init__()
        if ssl_pretrained_path is None:
            raise ValueError("ssl_pretrained_path must be provided for xlsr_sls")

        import fairseq

        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ssl_pretrained_path]
        )
        self.model = model[0]
        self.out_dim = int(getattr(self.model.cfg, "encoder_embed_dim", 1024))

    def extract_feat(self, input_data: torch.Tensor) -> tuple[torch.Tensor, Sequence[Any]]:
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        output = self.model(input_tmp, mask=False, features_only=True)
        return output["x"], output["layer_results"]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        emb, _ = self.extract_feat(input_data)
        return emb


def _layer_to_batch_time_feature(layer: Any, batch_size: int) -> torch.Tensor:
    layer_tensor = layer[0] if isinstance(layer, (tuple, list)) else layer
    if layer_tensor.ndim != 3:
        raise ValueError(f"Expected layer tensor with 3 dims, got {tuple(layer_tensor.shape)}")

    if layer_tensor.shape[0] == batch_size:
        return layer_tensor
    if layer_tensor.shape[1] == batch_size:
        return layer_tensor.transpose(0, 1)

    raise ValueError(
        "Cannot infer batch dimension from layer tensor "
        f"{tuple(layer_tensor.shape)} and batch_size={batch_size}"
    )


def get_sls_layer_features(layer_results: Iterable[Any], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    pooled_layers = []
    full_layers = []

    for layer in layer_results:
        layer_btd = _layer_to_batch_time_feature(layer, batch_size)
        pooled = F.adaptive_avg_pool1d(layer_btd.transpose(1, 2), 1).transpose(1, 2)
        pooled_layers.append(pooled)
        full_layers.append(layer_btd.unsqueeze(1))

    if not pooled_layers:
        raise ValueError("layer_results must contain at least one layer")

    return torch.cat(pooled_layers, dim=1), torch.cat(full_layers, dim=1)


class Model(nn.Module):
    def __init__(
        self,
        args: dict[str, Any] | None = None,
        ssl_pretrained_path: str | None = None,
        front_end: nn.Module | None = None,
    ) -> None:
        super().__init__()
        args = args or {}
        self.front_end = front_end or XLSRLayerFrontend(ssl_pretrained_path)

        ssl_dim = int(args.get("ssl_dim", getattr(self.front_end, "out_dim", 1024)))
        hidden_dim = int(args.get("hidden_dim", 1024))
        output_dim = int(args.get("output_dim", 2))
        pool_time = int(args.get("pool_time", 67))
        pool_feature = int(args.get("pool_feature", max(1, ssl_dim // 3)))

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.layer_gate = nn.Linear(ssl_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveMaxPool2d((pool_time, pool_feature))
        self.fc1 = nn.Linear(pool_time * pool_feature, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.apply_output_selu = bool(args.get("apply_output_selu", False))
        self.return_log_probs = bool(args.get("return_log_probs", False))

    def forward(self, x: torch.Tensor, last_emb: bool = False) -> torch.Tensor:
        _, layer_results = self.front_end.extract_feat(x.squeeze(-1))
        layer_summary, full_features = get_sls_layer_features(layer_results, x.shape[0])

        layer_weights = self.sigmoid(self.layer_gate(layer_summary)).unsqueeze(-1)
        x = torch.sum(full_features * layer_weights, dim=1).unsqueeze(1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.selu(self.fc1(x))
        if last_emb:
            return x
        x = self.fc_out(x)
        if self.apply_output_selu:
            x = self.selu(x)
        if self.return_log_probs:
            x = F.log_softmax(x, dim=1)
        return x
