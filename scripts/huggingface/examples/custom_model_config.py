"""
Example: Custom Model Configuration

This file shows how to create a custom model configuration for your own models.
You can use this as a template to register your models with the Hugging Face Hub integration.
"""

import os
from typing import Dict, Any
from model_registry import ModelConfig


class MyCustomModelConfig(ModelConfig):
    """Example configuration for a custom model."""
    
    @property
    def model_name(self) -> str:
        return "my_custom_model"
    
    @property
    def import_path(self) -> str:
        # Path to your model file
        return "/path/to/your/model_file.py"
    
    @property
    def class_name(self) -> str:
        # Name of your model class
        return "MyCustomModel"
    
    def get_init_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Define how to initialize your model from the config."""
        return {
            # Extract relevant parameters from config
            "hidden_size": config.get('model', {}).get('args', {}).get('hidden_size', 768),
            "num_layers": config.get('model', {}).get('args', {}).get('num_layers', 12),
            "pretrained_path": os.getenv("MY_MODEL_PRETRAINED_PATH", None)
        }


class AnotherModelConfig(ModelConfig):
    """Another example configuration."""
    
    @property
    def model_name(self) -> str:
        return "another_model"
    
    @property
    def import_path(self) -> str:
        return "/path/to/another_model.py"
    
    @property
    def class_name(self) -> str:
        return "AnotherModel"
    
    def get_init_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "config_dict": config.get('model', {}).get('args', {}),
            "device": "cuda" if config.get('model', {}).get('use_cuda', False) else "cpu"
        }


# Example of how to register your custom models
def register_custom_models():
    """Register your custom model configurations."""
    from model_registry import register_model_config
    
    # Register your custom models
    register_model_config(MyCustomModelConfig())
    register_model_config(AnotherModelConfig())


if __name__ == "__main__":
    # Example usage
    register_custom_models()
    print("Custom models registered successfully!")
