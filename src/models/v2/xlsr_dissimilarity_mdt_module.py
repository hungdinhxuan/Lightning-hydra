from typing import Any, Dict, Union

import torch

from src.models.base.mdt_module import MDTLitModule
from src.models.components.dissimilarity_deepfake_detection_hoanmytran.model_mdt import (
    Model as XLSRDissimilarityModel,
)


class XLSRDissimilarityMDTLitModule(MDTLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.net = self.init_model(**kwargs)
        self.init_adapter()

    def forward(self, x: torch.Tensor, inference_mode: bool = False) -> torch.Tensor:
        return self.net(x)

    def init_model(self, **kwargs) -> torch.nn.Module:
        ssl_pretrained_path = kwargs.get("ssl_pretrained_path", None)
        if ssl_pretrained_path is None:
            raise ValueError(
                "ssl_pretrained_path is required for XLSRDissimilarityMDTLitModule"
            )
        return XLSRDissimilarityModel(cp_path=ssl_pretrained_path)

