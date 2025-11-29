"""
Example: How to use the optimized Hugging Face Hub integration

This example shows how to use the new decoupled model registry system.
"""

import os
from pathlib import Path
from push_to_hf import push_model_to_hf
from model_registry import get_model_registry, register_model_config
from examples.custom_model_config import MyCustomModelConfig, AnotherModelConfig


def example_usage():
    """Example of how to use the system."""
    
    # 1. Register custom models (if you have any)
    register_model_config(MyCustomModelConfig())
    register_model_config(AnotherModelConfig())
    
    # 2. List available models
    registry = get_model_registry()
    print("Available models:", registry.list_models())
    
    # 3. Upload a model using the existing registry
    success = push_model_to_hf(
        model_name="xlsr_conformer_tcm",  # This is already registered
        config_path="/path/to/your/config.yaml",
        checkpoint_path="/path/to/your/checkpoint.pth",
        repo_name="your-username/your-model-name",
        private=False,
        commit_message="Upload my model"
    )
    
    if success:
        print("Model uploaded successfully!")
    else:
        print("Model upload failed!")


def example_with_custom_model():
    """Example using a custom model."""
    
    # Register your custom model first
    register_model_config(MyCustomModelConfig())
    
    # Now you can use it
    success = push_model_to_hf(
        model_name="my_custom_model",  # Your custom model name
        config_path="/path/to/your/config.yaml",
        checkpoint_path="/path/to/your/checkpoint.pth",
        repo_name="your-username/my-custom-model",
        private=False,
        commit_message="Upload my custom model"
    )
    
    return success


if __name__ == "__main__":
    print("=== Example Usage ===")
    example_usage()
    
    print("\n=== Custom Model Example ===")
    example_with_custom_model()
