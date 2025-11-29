#!/usr/bin/env python3
"""
Example usage of the push_model_to_hf function.

This script demonstrates how to use the push_model_to_hf function
to upload models to Hugging Face Hub.
"""

import os
import sys
from pathlib import Path

# Add the huggingface scripts directory to path
sys.path.append(str(Path(__file__).parent))

from push_to_hf import push_model_to_hf

def main():
    """
    Example usage of the push_model_to_hf function.
    """
    
    # Example 1: Upload a conformer_tcm model
    print("=== Example 1: Uploading conformer_tcm model ===")
    
    success = push_model_to_hf(
        model_name="conformer_tcm",
        config_path="/path/to/your/config.yaml",
        checkpoint_path="/path/to/your/checkpoint.pth",
        repo_name="your-username/conformer-tcm-model",
        private=False,
        commit_message="Initial conformer_tcm model upload"
    )
    
    if success:
        print("✅ Conformer TCM model uploaded successfully!")
    else:
        print("❌ Conformer TCM model upload failed!")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Upload a custom model from huggingface directory
    print("=== Example 2: Uploading custom model ===")
    
    success = push_model_to_hf(
        model_name="your_custom_model",  # This should match a .py file in the huggingface directory
        config_path="/path/to/your/config.yaml",
        checkpoint_path="/path/to/your/checkpoint.pth",
        repo_name="your-username/custom-model",
        private=True,  # Make it private
        commit_message="Custom model upload with private repository"
    )
    
    if success:
        print("✅ Custom model uploaded successfully!")
    else:
        print("❌ Custom model upload failed!")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Upload with authentication token
    print("=== Example 3: Upload with authentication token ===")
    
    # You can get your token from https://huggingface.co/settings/tokens
    hf_token = os.getenv("HUGGINGFACE_TOKEN")  # Set this environment variable
    
    if not hf_token:
        print("⚠️  Warning: HUGGINGFACE_TOKEN environment variable not set")
        print("   You can set it with: export HUGGINGFACE_TOKEN=your_token_here")
        hf_token = None
    
    success = push_model_to_hf(
        model_name="xlsr_conformertcm_baseline",
        config_path="/path/to/your/config.yaml",
        checkpoint_path="/path/to/your/checkpoint.pth",
        repo_name="your-username/xlsr-conformer-model",
        private=False,
        commit_message="XLSR Conformer model upload",
        token=hf_token
    )
    
    if success:
        print("✅ XLSR Conformer model uploaded successfully!")
    else:
        print("❌ XLSR Conformer model upload failed!")


def validate_paths_example():
    """
    Example of how to validate paths before uploading.
    """
    print("=== Path Validation Example ===")
    
    # Define your paths
    config_path = "/path/to/your/config.yaml"
    checkpoint_path = "/path/to/your/checkpoint.pth"
    
    # Validate paths exist
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return False
    
    print("✅ All paths validated successfully!")
    return True


if __name__ == "__main__":
    print("Hugging Face Model Upload Examples")
    print("=" * 40)
    
    # Run the main examples
    main()
    
    print("\n" + "=" * 40)
    print("Path validation example:")
    validate_paths_example()
    
    print("\n" + "=" * 40)
    print("📝 Notes:")
    print("1. Make sure to replace the example paths with your actual file paths")
    print("2. Set your HUGGINGFACE_TOKEN environment variable for authentication")
    print("3. The model_name should match the filename (without .py) in the huggingface directory")
    print("4. Repository names should be in the format 'username/model-name'")
    print("5. Make sure your model classes inherit from PyTorchModelHubMixin for proper HF integration")
