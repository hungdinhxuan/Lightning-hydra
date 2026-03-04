import torch
import sys
import os
import argparse

sys.path.append('../../')
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM

parser = argparse.ArgumentParser(description='Convert checkpoint to plain state_dict with normalized keys')
parser.add_argument('ckpt_path', type=str, help='Path to the input checkpoint (.pt)')
args_cli = parser.parse_args()

input_ckpt_path = args_cli.ckpt_path
ckpt = torch.load(input_ckpt_path, weights_only=False)

import yaml
with open('/nvme2/hungdx/Lightning-hydra/configs/experiment/cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs_july4.yaml', 'r') as f:
    config = yaml.safe_load(f)

args = config['model']['args']['conformer']


model = XLSRConformerTCM(args=args,
                         ssl_pretrained_path='/nvme2/hungdx/Lightning-hydra/pretrained/xlsr2_300m.pt')

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
  
    # If this is a full checkpoint (e.g., from PyTorch Lightning), extract the state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
    
    print(f"Loading {len(state_dict)} parameters from checkpoint")
    #print(state_dict.keys())
    # Create new state dict with processed keys
    
    # Keep only the keys that start with 'net.' and remove the 'net.' prefix
    if any(k.startswith(f'{required_prefix}.') for k in state_dict.keys()):
        new_state_dict = {k[len(required_prefix) + 1:]: v for k, v in state_dict.items() if k.startswith(f'{required_prefix}.')}
    else:
        new_state_dict = state_dict
    

    # Load the processed state dict
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(f"Successfully loaded {len(new_state_dict)} parameters")
    except RuntimeError as e:
        print(f"Error loading state dict: {str(e)}")
        raise

    return model

# Prefer loading via processed keys (handles Lightning checkpoints and 'net.' prefix)
model = load_model_weights(model, ckpt)

# Build output path: <input_path>_converted.pt (insert before extension if present)
root, ext = os.path.splitext(input_ckpt_path)
out_path = f"{root}_converted.pt" if ext else f"{input_ckpt_path}_converted.pt"

torch.save(model.state_dict(), out_path)
print(f"Saved converted state_dict to: {out_path}")
