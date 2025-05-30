import torch
import sys
import argparse
from pathlib import Path
import os
import yaml

# Get the absolute path to the project root directory
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM
from collections import OrderedDict


def load_model_weights(model, checkpoint, required_prefix='net'):
    """
    Load model weights from checkpoint with specific key prefix requirements.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to the checkpoint file
        required_prefix: Required prefix for state dict keys (default: 'net')
        
    Returns:
        model: Model with loaded weights
    """
  
    state_dict = checkpoint if isinstance(
        checkpoint, dict) else checkpoint.state_dict()
    
    print(f"Loading {len(state_dict)} parameters from checkpoint")

    # Create new state dict with processed keys
    new_state_dict = OrderedDict()

    # Process the state dict keys
    for key, value in state_dict.items():
        # Skip keys that don't start with the required prefix
        if not key.startswith(required_prefix):
            continue

        # Remove the 'net.' prefix to match model's state dict keys
        new_key = key[len(required_prefix) + 1:]  # +1 for the dot after prefix
        new_state_dict[new_key] = value

    # Load the processed state dict
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(f"Successfully loaded {len(new_state_dict)} parameters")
    except RuntimeError as e:
        print(f"Error loading state dict: {str(e)}")
        raise

    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Merge LoRA weights to base model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the checkpoint file')
    parser.add_argument('--ssl_pretrained_path', type=str, 
                      default='/nvme1/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/pretrained/xlsr2_300m.pt',
                      help='Path to SSL pretrained model')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to the config file')
    
    return parser.parse_args()

def main():
    args = parse_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint_path, weights_only=False)
    # remove "base_model." prefix from ckpt
    ckpt = {k.replace("base_model.model.", ""): v for k, v in ckpt.items()}
    print(config['model']['args'])
    
    model_args = config['model']['args']['conformer']
    
    # Initialize model
    print("Initializing model...")
    model = XLSRConformerTCM(args=model_args,
                            ssl_pretrained_path=args.ssl_pretrained_path)
    print("Model initialized")
    # Load model weights
    print("Loading model weights...")
    
    model.load_state_dict(ckpt)
    print("Model weights loaded")

if __name__ == '__main__':
    main()