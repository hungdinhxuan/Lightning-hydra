import torch
from src.models.components.distil.distil_wav2vec2_n_trans_layers_conformertcm import Model as Distil_Wav2vec2_N_Trans_Layer_ConformerTCM
from src.models.base.normal_module import NormalLitModule
from typing import Any, Dict, Tuple, Union
from torch import nn

class Distil_Wav2vec2_N_Trans_Layer_ConformerTCMNormalLitModule(NormalLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs  # 👈 Store kwargs for later use
        self.args = args
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.net = self.init_model(**kwargs)
        self.init_adapter()

    def init_model(self, **kwargs) -> nn.Module:
        _args = self.args.get("conformer", {})
        _kwargs = self.args.get("kwargs", {})
        print(_args)
        print(_kwargs)
        return Distil_Wav2vec2_N_Trans_Layer_ConformerTCM(_args, **_kwargs)
    
    def forward(self, x: torch.Tensor, inference_mode=False) -> torch.Tensor:
        return self.net(x)