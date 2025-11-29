"""
Model Registry System for Hugging Face Hub Integration

This module provides a flexible way to register and load models without hardcoding
model configurations in the main upload script.
"""

import os
import importlib.util
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Type
from abc import ABC, abstractmethod


class ModelConfig(ABC):
    """Abstract base class for model configurations."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name identifier."""
        pass
    
    @property
    @abstractmethod
    def import_path(self) -> str:
        """Return the import path for the model module."""
        pass
    
    @property
    @abstractmethod
    def class_name(self) -> str:
        """Return the name of the model class."""
        pass
    
    @abstractmethod
    def get_init_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return initialization arguments for the model."""
        pass
    
    def load_model_class(self) -> Type[torch.nn.Module]:
        """Load and return the model class."""
        spec = importlib.util.spec_from_file_location(
            self.class_name,
            self.import_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, self.class_name)


class ConformerTCMConfig(ModelConfig):
    """Configuration for XLSR Conformer TCM model."""
    
    @property
    def model_name(self) -> str:
        return "xlsr_conformer_tcm"
    
    @property
    def import_path(self) -> str:
        return "/home/hungdx/code/Lightning-hydra/scripts/huggingface/conformer_tcm.py"
    
    @property
    def class_name(self) -> str:
        return "Model"
    
    def get_init_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "args": config.get('model', {}).get('args', {}).get('conformer', {}),
            "ssl_pretrained_path": os.getenv("XLSR_PRETRAINED_MODEL_PATH", None)
        }


class ModelRegistry:
    """Registry for managing model configurations."""
    
    def __init__(self):
        self._configs: Dict[str, ModelConfig] = {}
        self._register_default_configs()
    
    def _register_default_configs(self):
        """Register default model configurations."""
        self.register(ConformerTCMConfig())
    
    def register(self, config: ModelConfig):
        """Register a new model configuration."""
        self._configs[config.model_name] = config
    
    def get_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self._configs.get(model_name)
    
    def list_models(self) -> list:
        """List all registered model names."""
        return list(self._configs.keys())
    
    def load_model(self, model_name: str, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> torch.nn.Module:
        """Load a model using the registered configuration."""
        model_config = self.get_config(model_name)
        
        if model_config is None:
            # Try to load from file system as fallback
            return self._load_model_from_file(model_name, config, checkpoint)
        
        # Load model class
        model_class = model_config.load_model_class()
        
        # Initialize with configuration
        init_kwargs = model_config.get_init_kwargs(config)
        model = model_class(**init_kwargs)
        
        # Load checkpoint weights
        self._load_checkpoint_weights(model, checkpoint)
        
        return model
    
    def _load_model_from_file(self, model_name: str, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> torch.nn.Module:
        """Fallback method to load model from file system."""
        try:
            spec = importlib.util.spec_from_file_location(
                model_name, 
                f"/home/hungdx/code/Lightning-hydra/scripts/huggingface/{model_name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the model class
            model_class = getattr(module, 'Model', None)
            if model_class is None:
                model_class = getattr(module, model_name.title(), None)
            if model_class is None:
                # Try to find any class that inherits from nn.Module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, torch.nn.Module) and 
                        attr != torch.nn.Module):
                        model_class = attr
                        break
            
            if model_class is None:
                raise ImportError(f"Could not find model class in {model_name}.py")
            
            # Initialize model with config
            model = model_class(**config.get('model', {}).get('args', {}))
            
            # Load checkpoint weights
            self._load_checkpoint_weights(model, checkpoint)
            
            return model
            
        except Exception as e:
            raise ImportError(f"Could not import model from {model_name}.py: {str(e)}")
    
    def _load_checkpoint_weights(self, model: torch.nn.Module, checkpoint: Dict[str, Any]):
        """Load checkpoint weights into the model."""
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Handle Lightning checkpoint format
            model.load_state_dict(state_dict, strict=False)
        else:
            # Assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint, strict=False)


# Global registry instance
model_registry = ModelRegistry()


def register_model_config(config: ModelConfig):
    """Convenience function to register a model configuration."""
    model_registry.register(config)


def get_model_registry() -> ModelRegistry:
    """Get the global model registry."""
    return model_registry
