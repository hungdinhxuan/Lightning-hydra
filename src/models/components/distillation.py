from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LogitKDLoss(nn.Module):
    """KL-based teacher-student logit distillation loss."""

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float | None = None,
    ) -> torch.Tensor:
        if student_logits.numel() == 0:
            return student_logits.new_zeros(())

        temp = self.temperature if temperature is None else float(temperature)
        if temp <= 0:
            raise ValueError("temperature must be > 0")

        student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp**2)


class FeatureKDLoss(nn.Module):
    """Optional feature distillation loss for matching same-shaped tensors."""

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        if student_features.numel() == 0:
            return student_features.new_zeros(())
        return F.mse_loss(student_features, teacher_features)
