import torch
from typing import Any, Dict, Tuple, Union
from src.models.base.mdt_module import MDTLitModule
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM

class XLSRConformertcmMDTLitModule(MDTLitModule):
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
        
    def init_model(self, **kwargs) -> nn.Module:
        return XLSRConformerTCM(**kwargs)