import torch
from src.models.components.distil.distil_xlsr_n_trans_layers_vib import Model as Distil_XLSR_N_Trans_Layer_VIB
from src.models.base.normal_module import NormalLitModule
from typing import Any, Dict, Union
from torch import nn

class Distil_XLSR_N_Trans_Layer_VIBNormalLitModule(NormalLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        self.args = args
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.net = self.init_model(**kwargs)
        self.init_adapter()

    def init_model(self, **kwargs) -> nn.Module:
        _kwargs = self.args.get("kwargs", {})
        return Distil_XLSR_N_Trans_Layer_VIB(**_kwargs)
    
    def forward(self, x: torch.Tensor, inference_mode=False) -> torch.Tensor:
        return self.net(x)