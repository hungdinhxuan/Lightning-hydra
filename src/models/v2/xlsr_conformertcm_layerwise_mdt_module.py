import torch
from typing import Any, Dict, Tuple, Union
from src.models.base.mdt_module import MDTLitModule
from src.models.components.xlsr_conformertcm_baseline_layerwise import Model as XLSRConformerTCMLayerwise

class XLSRConformertcmLayerwiseMDTLitModule(MDTLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.net = self.init_model(**kwargs)
        self.init_adapter()
        
    def forward(self, x: torch.Tensor, inference_mode=False) -> torch.Tensor:
        return self.net(x)
    
    def init_model(self, **kwargs) -> torch.nn.Module:
        ssl_pretrained_path = kwargs.get("ssl_pretrained_path", None)
        n_layers = kwargs.get("n_layers", None)
        extractor_type = kwargs.get("extractor_type", 'default')
        if ssl_pretrained_path is None:
            raise ValueError("ssl_pretrained_path is required for XLSRConformertcmLayerwiseMDTLitModule")
        return XLSRConformerTCMLayerwise(
            self.args['conformer'], ssl_pretrained_path, n_layers, extractor_type
        )