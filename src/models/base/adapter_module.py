from typing import Any, Dict, Tuple, Optional, List, Union
import os
import warnings

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy

import peft
from peft import PeftModel
from src.utils import load_ln_model_weights
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

        self.adapter_type = adapter_type
        # More robust check that handles both None and string "None"/"none"
        self.use_adapter = (
            self.adapter_type is not None 
            and str(self.adapter_type).lower() not in ["none", ""]
        )
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
            self._configure_and_load_adapters()

    def _load_base_model(self):
        """Load the base model from checkpoint."""
        is_base_model_path_ln = self.kwargs.get("is_base_model_path_ln", True)
        
        print(f"Loading baseline model from: {self.base_model_path}")
        ckpt = torch.load(self.base_model_path, weights_only=False)
        
        if is_base_model_path_ln:
            self.net = load_ln_model_weights(self.net, ckpt['state_dict'])  
        else:
            self.net.load_state_dict(ckpt)
            
        print("Successfully loaded baseline model")

    def _configure_and_load_adapters(self):
        """Configure and load adapter paths and weights."""
        print(f"Loading adapters from: {self.adapter_paths}")
        
        # Parse adapter paths
        adapter_paths = self.adapter_paths.split(",")
        
        # Parse adapter weights if provided
        adapter_weights = None
        if self.adapter_weights is not None:
            adapter_weights = [
                float(w) for w in self.adapter_weights.split(",") 
                if w.strip() != ""
            ]
        
        # Load adapters
        if len(adapter_paths) > 1:
            print("Loading multiple adapters...")
            self.load_adapters(adapter_paths, adapter_weights)
        else:
            self.load_single_lora_adapter(adapter_paths[0])

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
        try:
            adapter_config = self._create_adapter_config()
            self.net = self._apply_peft_adapter(adapter_config)
            self.net.print_trainable_parameters()
            print(f"Successfully applied {self.adapter_type.upper()} adapter")
        except Exception as e:
            raise ValueError(f"Failed to apply {self.adapter_type} adapter: {str(e)}")

    def _create_adapter_config(self):
        """Creates the appropriate adapter configuration based on adapter type."""
        if not self.args or 'adapter' not in self.args:
            raise ValueError("Adapter configuration not found in args")
        
        adapter_args = self.args['adapter']
        
        # Common parameters across all adapter types
        common_params = {
            'target_modules': list(adapter_args['target_modules']),
            'modules_to_save': list(adapter_args.get('modules_to_save', [])),
        }
        
        # Adapter-specific configurations
        adapter_configs = {
            'lora': self._get_lora_config,
            'loha': self._get_loha_config,
            'adalora': self._get_adalora_config,
            'lokr': self._get_lokr_config,
            'vera': self._get_vera_config,
            'randlora': self._get_randlora_config,
        }
        
        if self.adapter_type not in adapter_configs:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")
        
        return adapter_configs[self.adapter_type](adapter_args, common_params)

    def _get_lora_config(self, adapter_args, common_params):
        """Get LoRA configuration."""
        from peft import LoraConfig
        
        return LoraConfig(
            r=adapter_args['r'],
            lora_dropout=adapter_args.get('lora_dropout', 0.0),
            lora_alpha=adapter_args.get('lora_alpha', 8),
            **common_params
        )
    def _get_loha_config(self, adapter_args, common_params):
        """Get LoHA configuration."""
        from peft import LoHaConfig
        
        return LoHaConfig(
            r=adapter_args['r'],
            alpha=adapter_args.get('alpha', 8),
            rank_dropout=adapter_args.get('rank_dropout', 0.0),
            module_dropout=adapter_args.get('module_dropout', 0.0),
            init_weights=True,
            **common_params
        )
    def _get_adalora_config(self, adapter_args, common_params):
        """Get AdaLoRA configuration."""
        from peft import AdaLoraConfig
        
        return AdaLoraConfig(
            init_r=adapter_args['init_r'],
            lora_dropout=adapter_args.get('lora_dropout', 0.0),
            lora_alpha=adapter_args.get('lora_alpha', 8),
            # tinit=adapter_args.get('tinit', 1),
            # tfinal=adapter_args.get('tfinal', 10000),
            total_step=adapter_args.get('total_step', 10000),
            **common_params
        )

    def _get_lokr_config(self, adapter_args, common_params):
        """Get LoKR configuration."""
        from peft import LoKrConfig
        
        return LoKrConfig(
            r=adapter_args['r'],
            rank_dropout=adapter_args.get('rank_dropout', 0.0),
            module_dropout=adapter_args.get('module_dropout', 0.0),
            alpha =adapter_args.get('alpha', 8),
            init_weights=True,
            **common_params
        )

    def _get_vera_config(self, adapter_args, common_params):
        """Get VERA configuration.
        
        Note: VERA ranking is much larger than LoRA, recommended to use r=256 or higher.
        """
        from peft import VeraConfig
        
        return VeraConfig(
            r=adapter_args['r'],
            vera_dropout=adapter_args.get('vera_dropout', 0.0),
            **common_params
        )

    def _get_randlora_config(self, adapter_args, common_params):
        """Get RandLoRA configuration."""
        from peft import RandLoRAConfig
        
        return RandLoRAConfig(
            r=adapter_args['r'],
            sparse=adapter_args.get('sparse', True),
            randlora_dropout=adapter_args.get('randlora_dropout', 0.0),
            randlora_alpha=adapter_args.get('randlora_alpha', 8),
            **common_params
        )

    def _apply_peft_adapter(self, config):
        """Apply PEFT adapter to the model."""
        from peft import get_peft_model
        
        print(f"Applying {self.adapter_type.upper()} adapter...")
        return get_peft_model(self.net, config)

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
        if not adapter_paths:
            raise ValueError("No adapter paths provided")
        
        self._validate_adapter_paths(adapter_paths)
        
        # Load the first adapter to initialize the PEFT model
        self._load_primary_adapter(adapter_paths[0])
        
        # Load additional adapters with unique names
        if len(adapter_paths) > 1:
            self._load_additional_adapters(adapter_paths[1:])
        
        self._print_adapter_info()

    def _validate_adapter_paths(self, adapter_paths: List[str]):
        """Validate that all adapter paths exist."""
        for path in adapter_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Adapter path does not exist: {path}")

    def _load_primary_adapter(self, adapter_path: str):
        """Load the primary adapter to initialize the PEFT model."""
        print(f"Loading primary adapter from {adapter_path}")
        
        self.net = PeftModel.from_pretrained(
            self.net,
            adapter_path,
            is_trainable=True
        )
        
        # Store the default adapter name for reference
        self.default_adapter = self.net.active_adapter

    def _load_additional_adapters(self, adapter_paths: List[str]):
        """Load additional adapters with unique names."""
        for i, path in enumerate(adapter_paths, 1):
            adapter_name = f"adapter_{i}"
            print(f"Loading additional adapter from {path} as '{adapter_name}'")
            
            try:
                # Load the adapter state dict
                adapter_state_dict = torch.load(path)
                
                # Extract adapter weights if it's a checkpoint
                if "state_dict" in adapter_state_dict:
                    adapter_state_dict = adapter_state_dict["state_dict"]
                
                # Extract and rename adapter-specific weights
                adapter_weights = self._extract_adapter_weights(adapter_state_dict, adapter_name)
                
                # Add the adapter to the model
                self.net.add_adapter(adapter_name, adapter_weights)
                
            except Exception as e:
                print(f"Warning: Failed to load adapter from {path}: {str(e)}")
                continue

    def _extract_adapter_weights(self, adapter_state_dict: Dict, new_adapter_name: str) -> Dict:
        """Extract and rename adapter weights for the new adapter."""
        adapter_weights = {}
        
        for key, value in adapter_state_dict.items():
            if self.default_adapter in key:
                # Replace the default adapter name with the new name
                new_key = key.replace(self.default_adapter, new_adapter_name)
                adapter_weights[new_key] = value
        
        return adapter_weights
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        if self.adapter_type == 'adalora':
            i_step = int(self.global_step)
            print(f"Updating and allocating base model with global step: {i_step}")
            self.net.base_model.update_and_allocate(i_step)
        optimizer.zero_grad(set_to_none=True)
    def _print_adapter_info(self):
        """Print information about loaded adapters."""
        available_adapters = list(self.net.peft_config.keys())
        print(f"Successfully loaded {len(available_adapters)} adapters.")
        print(f"Available adapters: {available_adapters}")
        print(f"Active adapter: {self.net.active_adapter}")
        print("To switch adapters, use: model.set_adapter(adapter_name)")
