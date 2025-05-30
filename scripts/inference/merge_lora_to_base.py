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
    parser.add_argument('--lora_path', type=str, required=True,
                      help='Path to the lora checkpoint file')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save the merged model')
    return parser.parse_args()

def main():
    args = parse_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    from peft import LoraConfig, get_peft_model

    print("Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=config['model']['args']['adapter']['r'],
        target_modules=list(config['model']['args']['adapter']['target_modules']),
        modules_to_save=list(config['model']['args']['adapter']['modules_to_save']),
        lora_dropout=config['model']['args']['adapter'].get('lora_dropout', 0.0),
        lora_alpha=config['model']['args']['adapter'].get('lora_alpha', 8),
    )
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint_path, weights_only=False)
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
    
    # Load LoRA weights
    model = get_peft_model(model, lora_config)
    
    if hasattr(model, 'load_adapter'):
        print("Loading LoRA weights (1)...")
        adapter_name = "default"
        model.load_adapter(args.lora_path, adapter_name=adapter_name)
        model.set_adapter(adapter_name)
    else:
        print("Loading LoRA weights (2)...")
        model = PeftModel.from_pretrained(model, args.lora_path)
        
    print("Merging LoRA weights...")
    model.merge_and_unload()
    #print(model)
    
    # Save the merged model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # remove "base_model." prefix from model.state_dict()
    model_state_dict = {k.replace("base_model.model.", ""): v for k, v in model.state_dict().items()}
    torch.save(model_state_dict, output_path)
    print(f"Model saved to {output_path}")

if __name__ == '__main__':
    main()