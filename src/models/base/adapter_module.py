from typing import Any, Dict, Tuple, Optional, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy

from typing import Union

import torch
from src.utils import load_ln_model_weights
from peft import LoraConfig, TaskType
import peft
from peft import PeftModel
from src.models.base.base_module import BaseLitModule
from src.utils import load_ln_model_weights

class AdapterLitModule(BaseLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Optional[Dict[str, Any]] = None,
        adapter_type: Optional[str] = None,
        base_model_path: Optional[str] = None,
        adapter_paths: Optional[Union[str, List[str]]] = None,
        adapter_weights: Optional[List[float]] = None,
        merge_adapters: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(optimizer, scheduler, args, **kwargs)

        self.adapter_type = adapter_type.lower()  # Normalize input
        self.use_adapter = self.adapter_type is not None  # Generalized flag
        self.merge_adapters = merge_adapters

        # Load base model if provided
        if base_model_path:
            ckpt = torch.load(base_model_path, weights_only=False)
            self.net = load_ln_model_weights(self.net, ckpt['state_dict'])
            print("Loaded baseline model from:", base_model_path)

        # Apply adapter method
        if self.use_adapter:
            self.apply_adapter()

        # Load adapters if provided
        if adapter_paths:
            self.load_adapters(adapter_paths, adapter_weights)

    def apply_adapter(self):
        """Generalized method to apply adapters based on the chosen type."""
        if self.adapter_type == "lora":
            self.apply_lora()
        elif self.adapter_type == "prefix":
            self.apply_prefix_tuning()
        elif self.adapter_type == "adapterfusion":
            self.apply_adapter_fusion()
        else:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")

    def apply_lora(self):
        """Applies LoRA adaptation."""
        from peft import LoraConfig, get_peft_model

        print("Applying LoRA adapter...")
        lora_config = LoraConfig(
            r=self.args['adapter']['r'],
            target_modules=list(self.args['adapter']['target_modules']),
            modules_to_save=list(self.args['adapter']['modules_to_save']),
            lora_dropout=self.args['adapter'].get('lora_dropout', 0.0),
            lora_alpha=self.args['adapter'].get('lora_alpha', 8),
        )
        self.net = get_peft_model(self.net, lora_config)
        self.net.print_trainable_parameters()

    def apply_prefix_tuning(self):
        """Placeholder for Prefix-Tuning (to be implemented)."""
        print("Applying Prefix-Tuning (not implemented yet).")

    def apply_adapter_fusion(self):
        """Placeholder for AdapterFusion (to be implemented)."""
        print("Applying AdapterFusion (not implemented yet).")

    def load_adapters(self, adapter_paths: Union[str, List[str]], adapter_weights: Optional[List[float]] = None):
        """Loads and optionally merges multiple adapters."""
        if isinstance(adapter_paths, str):
            adapter_paths = [adapter_paths]

        if self.merge_adapters and len(adapter_paths) > 1:
            if adapter_weights is None:
                adapter_weights = [1.0 / len(adapter_paths)] * len(adapter_paths)
            self.load_and_merge_adapters(adapter_paths, adapter_weights)
        else:
            self.load_separate_adapters(adapter_paths)

    def load_and_merge_adapters(self, adapter_paths: List[str], adapter_weights: List[float]):
        """Handles merging multiple adapters."""
        print(f"Merging adapters: {adapter_paths} with weights {adapter_weights}")
        # Implement merging logic here

    def load_separate_adapters(self, adapter_paths: List[str]):
        """Loads adapters separately."""
        for path in adapter_paths:
            print(f"Loading adapter from {path}")
            # Implement separate loading logic here