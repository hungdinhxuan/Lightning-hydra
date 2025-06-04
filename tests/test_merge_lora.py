import torch
import sys
from pathlib import Path
import argparse
import yaml
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
import os

# Get the absolute path to the project root directory
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_test_input(seed=42):
    """Create deterministic test input tensor"""
    set_seed(seed)
    # Create a fixed-size input tensor (adjust size according to your model's requirements)
    test_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz
    return test_input

def parse_args():
    parser = argparse.ArgumentParser(description='Test LoRA merging process')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the checkpoint file')
    parser.add_argument('--ssl_pretrained_path', type=str, 
                      default='/nvme1/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/pretrained/xlsr2_300m.pt',
                      help='Path to SSL pretrained model')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to the config file')
    parser.add_argument('--lora_path', type=str, required=True,
                      help='Path to the lora checkpoint file')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()

def save_output(output, path):
    """Save output tensor to file"""
    torch.save(output, path)

def test_model_outputs(model, test_input, stage_name):
    """Test model outputs and save results"""
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        save_output(output, f"test_output_{stage_name}.pt")
        print(f"Saved {stage_name} output to test_output_{stage_name}.pt")
    return output

def main():
    args = parse_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create test input
    test_input = create_test_input(args.seed)
    print(f"Created test input with shape: {test_input.shape}")
    print(f"Test input mean: {test_input.mean():.6f}, std: {test_input.std():.6f}")

    # Initialize base model
    print("\n1. Initializing base model...")
    model_args = config['model']['args']['conformer']
    base_model = XLSRConformerTCM(args=model_args,
                                 ssl_pretrained_path=args.ssl_pretrained_path)
    
    # Load base model weights
    ckpt = torch.load(args.checkpoint_path, weights_only=False)
    base_model.load_state_dict(ckpt)
    print("Base model initialized and weights loaded")

    # Test base model output
    print("\n2. Testing base model output...")
    base_output = test_model_outputs(base_model, test_input, "base")

    # Apply LoRA
    print("\n3. Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=config['model']['args']['adapter']['r'],
        target_modules=list(config['model']['args']['adapter']['target_modules']),
        modules_to_save=list(config['model']['args']['adapter']['modules_to_save']),
        lora_dropout=config['model']['args']['adapter'].get('lora_dropout', 0.0),
        lora_alpha=config['model']['args']['adapter'].get('lora_alpha', 8),
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    
    # Load LoRA weights
    if hasattr(lora_model, 'load_adapter'):
        adapter_name = "default"
        lora_model.load_adapter(args.lora_path, adapter_name=adapter_name)
        lora_model.set_adapter(adapter_name)
    else:
        lora_model = PeftModel.from_pretrained(lora_model, args.lora_path)
    
    # Test LoRA model output
    print("\n4. Testing LoRA model output...")
    lora_output = test_model_outputs(lora_model, test_input, "lora")

    # Merge LoRA weights
    print("\n5. Merging LoRA weights...")
    merged_model = lora_model.merge_and_unload()
    
    # Test merged model output
    print("\n6. Testing merged model output...")
    merged_output = test_model_outputs(merged_model, test_input, "merged")

    # Remove prefix and test
    print("\n7. Testing model after removing prefix...")
    model_state_dict = {k.replace("base_model.model.", ""): v for k, v in merged_model.state_dict().items()}
    final_model = XLSRConformerTCM(args=model_args,
                                  ssl_pretrained_path=args.ssl_pretrained_path)
    final_model.load_state_dict(model_state_dict)
    final_output = test_model_outputs(final_model, test_input, "final")

    # Compare outputs
    print("\n8. Comparing outputs...")
    print(f"Base vs LoRA output difference: {torch.mean(torch.abs(base_output - lora_output)):.6f}")
    print(f"LoRA vs Merged output difference: {torch.mean(torch.abs(lora_output - merged_output)):.6f}")
    print(f"Merged vs Final output difference: {torch.mean(torch.abs(merged_output - final_output)):.6f}")

if __name__ == '__main__':
    main() 