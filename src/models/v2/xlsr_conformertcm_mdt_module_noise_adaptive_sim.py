import torch
from typing import Any, Dict, Tuple, Union
from src.models.base.mdt_module import MDTLitModule
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM

class XLSRConformertcmMDTNoiseAdaptiveSimLitModule(MDTLitModule):
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
        
        self.class_labels_to_lora_groups = {
            "Background Noise": "g1",
            "Background Music": "g1",
            "Gaussian Noise": "g2",
            "Bandpass Filter": "g3",
            "Time-Pitch Modulation": "g5",
            "Autotune": "g6",
            "Echo": "g6",
            "Reverberation": "g7",
            "Clean": "g0", # g0 here is not used, it is just for padding
        }
    
    def route_decision(self, noise_type: str):
        """
        Route decision based on noise type to determine LoRA group
        Returns the LoRA group string for the given noise type
        """
        if noise_type not in self.class_labels_to_lora_groups:
            return "g0"
        
        lora_group = self.class_labels_to_lora_groups[noise_type]
        return lora_group
        
    def forward(self, x: torch.Tensor, noise_labels: List[str], inference_mode=False) -> torch.Tensor:
        lora_groups = [self.route_decision(noise_label) for noise_label in noise_labels]
        self.adapter.set_lora_adapter(lora_groups)
        return self.net(x)
    
    def init_model(self, **kwargs) -> torch.nn.Module:
        ssl_pretrained_path = kwargs.get("ssl_pretrained_path", None)
        if ssl_pretrained_path is None:
            raise ValueError("ssl_pretrained_path is required for XLSRConformertcmMDTLitModule")
        return XLSRConformerTCM(
            self.args['conformer'], ssl_pretrained_path
        )