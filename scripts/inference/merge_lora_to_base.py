import torch
import sys
import argparse
from pathlib import Path
import os
import yaml
from collections import OrderedDict

# Get the absolute path to the project root directory
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM
from peft import LoraConfig, get_peft_model, PeftModel

def load_checkpoint(checkpoint_path):
    """Load checkpoint with error handling"""
    try:
        return torch.load(checkpoint_path, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {str(e)}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config

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
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (cuda/cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config_path)
    
    print("Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=config['model']['args']['adapter']['r'],
        target_modules=list(config['model']['args']['adapter']['target_modules']),
        modules_to_save=list(config['model']['args']['adapter']['modules_to_save']),
        lora_dropout=config['model']['args']['adapter'].get('lora_dropout', 0.0),
        lora_alpha=config['model']['args']['adapter'].get('lora_alpha', 8),
    )
    
    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint_path)
    print(f"Loaded checkpoint with {len(ckpt)} parameters")
    
    model_args = config['model']['args']['conformer']
    
    # Initialize model
    print("Initializing model...")
    model = XLSRConformerTCM(args=model_args,
                            ssl_pretrained_path=args.ssl_pretrained_path)
    model = model.to(device)
    print("Model initialized")
    
    # Load model weights
    print("Loading model weights...")
    try:
        model.load_state_dict(ckpt)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {str(e)}")
        raise
    
    # Load LoRA weights
    model = get_peft_model(model, lora_config)
    
    try:
        if hasattr(model, 'load_adapter'):
            print("Loading LoRA weights (method 1)...")
            adapter_name = "default"
            model.load_adapter(args.lora_path, adapter_name=adapter_name)
            model.set_adapter(adapter_name)
        else:
            print("Loading LoRA weights (method 2)...")
            model = PeftModel.from_pretrained(model, args.lora_path)
        print("LoRA weights loaded successfully")
    except Exception as e:
        print(f"Error loading LoRA weights: {str(e)}")
        raise
        
    print("Merging LoRA weights...")
    try:
        model = model.merge_and_unload()
        print("LoRA weights merged successfully")
    except Exception as e:
        print(f"Error merging LoRA weights: {str(e)}")
        raise
    
    # Save the merged model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove "base_model." prefix from model.state_dict()
    model_state_dict = {k.replace("base_model.model.", ""): v for k, v in model.state_dict().items()}
    
    try:
        torch.save(model_state_dict, output_path)
        print(f"Model saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

if __name__ == '__main__':
    main()