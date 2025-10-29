import torch
from src.models.components.xlsr_vib import Model as XLSRVIB
from src.models.base.normal_module import NormalLitModule
from typing import Any, Dict, Tuple, Union
from torch import nn
import torch.nn.functional as F
from src.models.components.wavlm_layerwise_conformertcm import Model as WavlmConformerTCM

class WavLMConformertcmNormalLitModule(NormalLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs  # 👈 Store kwargs for later use
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.net = self.init_model(**kwargs)
        self.init_adapter()

        
    def init_model(self, **kwargs) -> nn.Module:
        ssl_pretrained_path = kwargs.get("ssl_pretrained_path", None)
        if ssl_pretrained_path is None:
            raise ValueError("ssl_pretrained_path is required for XLSRVIBNormalLitModule")
        return WavlmConformerTCM(self.args, ssl_pretrained_path, n_layers=self.args.get("n_layers", None))
    
    