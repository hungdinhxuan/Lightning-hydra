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

class AdapterLitModule(BaseLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Optional[Dict[str, Any]] = None,
        adapter_type: Optional[str] = None,
        base_model_path: Optional[str] = None,
        adapter_paths: str = None,
        adapter_weights: str = None,
        merge_adapters: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(optimizer, scheduler, args, **kwargs)

        self.adapter_type = adapter_type.lower()  # Normalize input
        self.use_adapter = self.adapter_type is not None  # Generalized flag
        self.merge_adapters = merge_adapters
        self.base_model_path = base_model_path
        self.adapter_paths = adapter_paths
        self.adapter_weights = adapter_weights

        
    def init_adapter(self):
        """Initializes the adapter type.
            This method should be called after the model is initialized.
        """
        is_base_model_path_ln = self.kwargs.get("is_base_model_path_ln", True)
        # Load base model if provided
        if self.base_model_path:
            ckpt = torch.load(self.base_model_path, weights_only=False)
            #print(ckpt)
            #print("is_base_model_path_ln", is_base_model_path_ln)
            if is_base_model_path_ln:
                self.net = load_ln_model_weights(self.net, ckpt['state_dict'])  
            else:
                self.net.load_state_dict(ckpt)
            print("Loaded baseline model from:", self.base_model_path)

        # Apply adapter method
        if self.use_adapter:
            self.apply_adapter()

        # Load adapters if provided
        if self.adapter_paths:
            # parse adapter_paths 
            print("Loading adapters from:", self.adapter_paths)
            self.adapter_paths = self.adapter_paths.split(",")
            
            if self.adapter_weights is not None:
                self.adapter_weights = self.adapter_weights.split(",")
                self.adapter_weights = [float(w) for w in self.adapter_weights if w.strip() != ""]
            
            # if self.adapter_weights is None:
            #     self.adapter_weights = [1.0] * len(self.adapter_paths) # Default to equal weights
            
            if len(self.adapter_paths) > 1:
                print("Loading multiple adapters...")
                self.load_adapters(self.adapter_paths, self.adapter_weights)
            else:
                self.load_single_lora_adapter(self.adapter_paths[0])
                
    def load_single_lora_adapter(self, checkpoint_path: str, adapter_name: str = "default"):
        print(f"Loading LoRA adapter from {checkpoint_path}")
        """Specialized method for loading LoRA adapters"""
        if hasattr(self.net, 'load_adapter'):
            self.net.load_adapter(checkpoint_path, adapter_name=adapter_name)
            self.net.set_adapter(adapter_name)
        else:
            self.net = PeftModel.from_pretrained(self.net, checkpoint_path)
            self.net.merge_and_unload()
        
        print(f"Loaded LoRA adapter from {checkpoint_path}")
    
    def apply_adapter(self):
        """Generalized method to apply adapters based on the chosen type."""
        if self.adapter_type == "lora":
            self.apply_lora()
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

    def load_adapters(self, adapter_paths: Union[str, List[str]], adapter_weights: Optional[List[float]] = None):
        """Loads and merges multiple adapters with specified weights handling potential key mismatches.
        
        Args:
            adapter_paths: List of file paths to adapter checkpoints or directories containing adapter files
            adapter_weights: List of weights for each adapter (defaults to equal weights)
        """
        from peft import PeftModel
        import os
        import warnings
        
        # Temporarily suppress specific warnings about missing keys
        warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint")
        
        if isinstance(adapter_paths, str):
            adapter_paths = [adapter_paths]
            
        # Set default weights if not provided
        if adapter_weights is None:
            adapter_weights = [1.0 / len(adapter_paths)] * len(adapter_paths)
        
        assert len(adapter_paths) == len(adapter_weights), "Number of adapter paths must match number of weights"
        assert all(os.path.exists(path) for path in adapter_paths), "One or more adapter paths do not exist"
        
        print(f"Loading adapters: {adapter_paths} with weights {adapter_weights}")
        
        # If there's only one adapter, just load it without merging
        if len(adapter_paths) == 1:
            path = adapter_paths[0]
            print(f"Loading single adapter from {path}")
            
            self.net = PeftModel.from_pretrained(
                self.net,
                path,
                is_trainable=True
            )
            return
        
        # For multiple adapters, we'll load them individually and merge manually
        print(f"Merging {len(adapter_paths)} adapters with weights {adapter_weights}")
        
        # Load the first adapter
        first_adapter_path = adapter_paths[0]
        first_adapter_weight = adapter_weights[0]
        
        print(f"Loading first adapter from {first_adapter_path}")
        model = PeftModel.from_pretrained(
            self.net,
            first_adapter_path,
            adapter_name="adapter_0"
        )
        
        weighted_adapter_name = "weighted_merged_adapters"
        # Then load a different adapter and merge it with the first one:
        for i, (adapter_path, adapter_weight) in enumerate(zip(adapter_paths[1:], adapter_weights[1:]), 1):
            print(f"Loading additional adapter from {adapter_path}")
            model.load_adapter(adapter_path, adapter_name=f"adapter_{i}")
        
        print("Adding weighted adapter...")
        model.add_weighted_adapter(
            adapters=[
                f"adapter_{i}" for i in range(len(adapter_paths))
            ],
            weights=adapter_weights,
            adapter_name=weighted_adapter_name,
            combination_type="linear"
        )
        
        model.set_adapter(weighted_adapter_name)            
        
        
        # Optionally merge weights into the base model for inference
        if not self.args.get('keep_adapters_separate', False):
            print("Merging adapters into base model...")
            model = model.merge_and_unload()
        
        self.net = model
        print("Successfully merged all adapters")
        # import sys
        # sys.exit(1)    

    def load_separate_adapters(self, adapter_paths: List[str]):
        """Loads adapters separately without merging.
        
        This is useful when you want to switch between different adapters
        during inference or fine-tuning.
        
        Args:
            adapter_paths: List of file paths to adapter checkpoints
        """
        from peft import PeftModel
        import os
        
        assert all(os.path.exists(path) for path in adapter_paths), "One or more adapter paths do not exist"
        
        # Load the first adapter to initialize the PEFT model
        first_path = adapter_paths[0]
        print(f"Loading primary adapter from {first_path}")
        
        self.net = PeftModel.from_pretrained(
            self.net,
            first_path,
            is_trainable=True
        )
        
        # Get default adapter name
        default_adapter = self.net.active_adapter
        
        # Load additional adapters with unique names
        for i, path in enumerate(adapter_paths[1:], 1):
            adapter_name = f"adapter_{i}"
            print(f"Loading additional adapter from {path} as '{adapter_name}'")
            
            # Load the adapter state dict
            adapter_state_dict = torch.load(path)
            
            # If it's a checkpoint with 'state_dict' key, extract just the adapter weights
            if "state_dict" in adapter_state_dict:
                adapter_state_dict = adapter_state_dict["state_dict"]
            
            # Extract adapter-specific weights and rename them with the new adapter name
            adapter_weights = {}
            for key, value in adapter_state_dict.items():
                if default_adapter in key:
                    # Replace the default adapter name with the new name
                    new_key = key.replace(default_adapter, adapter_name)
                    adapter_weights[new_key] = value
            
            # Add the adapter to the model
            self.net.add_adapter(adapter_name, adapter_weights)
        
        # Set the active adapter to the first one by default
        self.net.set_adapter(default_adapter)
        
        print(f"Successfully loaded {len(adapter_paths)} adapters. Available adapters: {self.net.peft_config.keys()}")
        print(f"Active adapter: {self.net.active_adapter}")
        
        # Provide info on how to switch adapters
        print("To switch adapters, use: model.set_adapter(adapter_name)")
    
    # def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
    #     """Overridden test step to handle adapter-specific logic."""
    #     if self.use_adapter:
    #         self.net = self.net.merge_and_unload()
    #     super().test_step(batch, batch_idx)