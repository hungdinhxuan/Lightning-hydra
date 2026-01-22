import torch
from src.models.components.xlsr_beats_conformertcm import Model as XLSRBeatsConformerTCM
from src.models.base.normal_module import NormalLitModule
from typing import Any, Dict, Tuple, Union
from torch import nn

class XLSRBeatsConformertcmNormalLitModule(NormalLitModule):
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
        
    def init_model(self, **kwargs) -> nn.Module:
        ssl_pretrained_path = kwargs.get("ssl_pretrained_path", None)
        beats_pretrained_path = kwargs.get("beats_pretrained_path", None)
        if ssl_pretrained_path is None or beats_pretrained_path is None:
            raise ValueError("ssl_pretrained_path and beats_pretrained_path are required for XLSRBeatsConformertcmNormalLitModule")
        return XLSRBeatsConformerTCM(
            self.args['conformer'], ssl_pretrained_path, beats_pretrained_path
        )