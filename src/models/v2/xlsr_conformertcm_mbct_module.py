import torch
from typing import Any, Dict, Union

from src.models.base.mbct_module import MBCTLitModule
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM


class XLSRConformertcmMBCTLitModule(MBCTLitModule):
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

    def forward(self, x: torch.Tensor, inference_mode=False) -> torch.Tensor:
        return self.net(x)

    def init_model(self, **kwargs) -> torch.nn.Module:
        ssl_pretrained_path = kwargs.get("ssl_pretrained_path", None)
        if ssl_pretrained_path is None:
            raise ValueError("ssl_pretrained_path is required for XLSRConformertcmMBCTLitModule")
        return XLSRConformerTCM(
            self.args["conformer"], ssl_pretrained_path
        )
