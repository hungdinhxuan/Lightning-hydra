import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from huggingface_hub import HfApi, Repository, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import importlib.util
import dotenv

# Import the model registry
from model_registry import get_model_registry, register_model_config

dotenv.load_dotenv()

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def push_model_to_hf(
    model_name: str,
    config_path: str,
    checkpoint_path: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload model",
    token: Optional[str] = os.getenv("HF_TOKEN", None)
) -> bool:
    """
    Push a model to Hugging Face Hub with proper configuration and checkpoint loading.
    
    Args:
        model_name (str): Name of the model class (should match filename without .py)
        config_path (str): Path to the YAML configuration file
        checkpoint_path (str): Path to the model checkpoint file
        repo_name (str): Name of the Hugging Face repository (e.g., "username/model-name")
        private (bool): Whether to create a private repository (default: False)
        commit_message (str): Commit message for the upload (default: "Upload model")
        token (Optional[str]): Hugging Face token (if None, will use cached token)
    
    Returns:
        bool: True if successful, False otherwise
    
    Raises:
        FileNotFoundError: If config or checkpoint files don't exist
        ImportError: If model class cannot be imported
        ValueError: If configuration is invalid
        RuntimeError: If model loading or upload fails
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Token: {token}")
    
    try:
        # Validate inputs
        logger.info(f"Starting model upload process for {model_name}")
        logger.info(f"Config path: {config_path}")
        logger.info(f"Checkpoint path: {checkpoint_path}")
        logger.info(f"Repository: {repo_name}")
        
        # Validate file paths
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Load configuration
        logger.info("Loading configuration...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            raise ValueError("Configuration file is empty or invalid")
        
        # Load checkpoint
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        
        # Import and instantiate model using registry
        logger.info(f"Loading model: {model_name}")
        registry = get_model_registry()
        model = registry.load_model(model_name, config, checkpoint)
        
        # Create or get repository
        logger.info(f"Setting up repository: {repo_name}")
        api = HfApi(token=token)
        
        try:
            # Try to get existing repository
            repo_info = api.repo_info(repo_id=repo_name, repo_type="model")
            logger.info(f"Using existing repository: {repo_name}")
        except RepositoryNotFoundError:
            # Create new repository
            logger.info(f"Creating new repository: {repo_name}")
            create_repo(
                repo_id=repo_name,
                token=token,
                private=private,
                repo_type="model"
            )
        
        # Save model to temporary directory
        temp_dir = Path(f"temp_{model_name}_upload")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            logger.info("Saving model...")
            model.save_pretrained(str(temp_dir))
            
            # Create model card
            model_card_content = _generate_model_card(model_name, config, checkpoint_path)
            with open(temp_dir / "README.md", "w") as f:
                f.write(model_card_content)
            
            # Upload to Hugging Face Hub
            logger.info("Uploading to Hugging Face Hub...")
            api.upload_folder(
                folder_path=str(temp_dir),
                repo_id=repo_name,
                token=token,
                commit_message=commit_message,
                repo_type="model"
            )
            
            logger.info(f"Successfully uploaded model to {repo_name}")
            return True
            
        finally:
            # Clean up temporary directory
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary files")
    
    except Exception as e:
        logger.error(f"Error during model upload: {str(e)}")
        return False




def _generate_model_card(model_name: str, config: Dict[str, Any], checkpoint_path: str) -> str:
    """
    Generate a model card for the uploaded model.
    
    Args:
        model_name (str): Name of the model
        config (Dict[str, Any]): Configuration dictionary
        checkpoint_path (str): Path to the checkpoint file
    
    Returns:
        str: Model card content
    """
    
    model_card = f"""---
license: apache-2.0
tags:
- pytorch
- audio
- deepfake-detection
- conformer
---

# {model_name}

This model was uploaded using the Lightning-hydra project's Hugging Face integration.

## Model Details

- **Model Name**: {model_name}
- **Architecture**: Conformer-based audio deepfake detection
- **Checkpoint**: {os.path.basename(checkpoint_path)}

## Configuration

```yaml
{_format_config_for_card(config)}
```

## Usage

```python
from transformers import AutoModel
import torch

# Load the model
model = AutoModel.from_pretrained("{model_name}")

# Example usage
input_tensor = torch.randn(1, 16000)  # Example audio input
output = model(input_tensor)
```

## Training Details

This model was trained using the Lightning-hydra framework with the configuration shown above.

## Citation

If you use this model, please cite the original paper and the Lightning-hydra framework.
"""
    
    return model_card


def _format_config_for_card(config: Dict[str, Any], indent: int = 0) -> str:
    """
    Format configuration dictionary for the model card.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        indent (int): Indentation level
    
    Returns:
        str: Formatted configuration string
    """
    
    lines = []
    spaces = "  " * indent
    
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{spaces}{key}:")
            lines.append(_format_config_for_card(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{spaces}{key}: {value}")
        else:
            lines.append(f"{spaces}{key}: {value}")
    
    return "\n".join(lines)


def list_available_models() -> list:
    """
    List all available model names in the registry.
    
    Returns:
        list: List of available model names
    """
    registry = get_model_registry()
    return registry.list_models()


def register_custom_model(model_config):
    """
    Register a custom model configuration.
    
    Args:
        model_config: ModelConfig instance to register
    """
    register_model_config(model_config)


# Example usage
if __name__ == "__main__":
    # List available models
    print("Available models:", list_available_models())
    
    # Example usage of the function
    success = push_model_to_hf(
        model_name="xlsr_conformer_tcm",  # Use the correct model name
        config_path="/path/to/config.yaml",
        checkpoint_path="/path/to/checkpoint.pth",
        repo_name="username/model-name",
        private=False,
        commit_message="Initial model upload"
    )
    
    if success:
        print("Model uploaded successfully!")
    else:
        print("Model upload failed!")