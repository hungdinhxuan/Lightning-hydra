#!/usr/bin/env python3
"""
Working solution for LoRA loading issue
The key insight: Use the same exact configuration and handle modules_to_save properly
"""

import torch
import sys
import yaml
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.append('../')
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM

def main():
    # Load configuration
    with open('/nvme1/hungdx/Lightning-hydra/configs/experiment/cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs_july4.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    args = config['model']['args']['conformer']
    
    # Load checkpoint
    ckpt = torch.load('/nvme1/hungdx/Lightning-hydra/notebooks/S_241214_conf-1.pth', weights_only=False)
    
    print("=== WORKING SOLUTION: Proper LoRA handling ===")
    
    # STEP 1: Create and save LoRA model with proper configuration
    print("Step 1: Creating and saving LoRA model...")
    
    model = XLSRConformerTCM(args=args,
                             ssl_pretrained_path='/nvme1/hungdx/Lightning-hydra/xlsr2_300m.pt')
    model.load_state_dict(ckpt, strict=True)
    
    # Use the configuration from the config file
    lora_config = LoraConfig(
        r=config['model']['args']['adapter']['r'],
        target_modules=list(config['model']['args']['adapter']['target_modules']),
        modules_to_save=list(config['model']['args']['adapter']['modules_to_save']),
        lora_dropout=config['model']['args']['adapter'].get('lora_dropout', 0.0),
        lora_alpha=config['model']['args']['adapter'].get('lora_alpha', 8),
    )
    
    net = get_peft_model(model, lora_config)
    net.print_trainable_parameters()
    
    # Save the LoRA model
    net.save_pretrained("lora_model_working")
    print("LoRA model saved to 'lora_model_working'")
    
    # STEP 2: Load the LoRA model correctly
    print("\nStep 2: Loading LoRA model...")
    
    # Create a fresh base model
    fresh_model = XLSRConformerTCM(args=args,
                                   ssl_pretrained_path='/nvme1/hungdx/Lightning-hydra/xlsr2_300m.pt')
    fresh_model.load_state_dict(ckpt, strict=True)
    
    # Load the LoRA adapter using PeftModel.from_pretrained
    # This is the recommended approach
    try:
        loaded_model = PeftModel.from_pretrained(fresh_model, "lora_model_working")
        print("SUCCESS: LoRA model loaded using PeftModel.from_pretrained()!")
        loaded_model.print_trainable_parameters()
    except Exception as e:
        print(f"Failed with PeftModel.from_pretrained(): {e}")
        
        # Fallback: Try with load_adapter
        print("\nTrying fallback method with load_adapter...")
        try:
            # Create model with same LoRA config
            fallback_model = XLSRConformerTCM(args=args,
                                             ssl_pretrained_path='/nvme1/hungdx/Lightning-hydra/xlsr2_300m.pt')
            fallback_model.load_state_dict(ckpt, strict=True)
            
            fallback_net = get_peft_model(fallback_model, lora_config)
            fallback_net.load_adapter("lora_model_working", adapter_name="default")
            print("SUCCESS: LoRA model loaded using load_adapter()!")
            fallback_net.print_trainable_parameters()
        except Exception as e2:
            print(f"Failed with load_adapter(): {e2}")
    
    print("\n=== ALTERNATIVE: Load without modules_to_save ===")
    
    # Alternative approach: Load without modules_to_save to avoid key conflicts
    print("Alternative: Loading without modules_to_save...")
    
    alt_model = XLSRConformerTCM(args=args,
                                 ssl_pretrained_path='/nvme1/hungdx/Lightning-hydra/xlsr2_300m.pt')
    alt_model.load_state_dict(ckpt, strict=True)
    
    # Create LoRA config without modules_to_save
    alt_lora_config = LoraConfig(
        r=config['model']['args']['adapter']['r'],
        target_modules=list(config['model']['args']['adapter']['target_modules']),
        modules_to_save=[],  # Empty list
        lora_dropout=config['model']['args']['adapter'].get('lora_dropout', 0.0),
        lora_alpha=config['model']['args']['adapter'].get('lora_alpha', 8),
    )
    
    alt_net = get_peft_model(alt_model, alt_lora_config)
    
    try:
        alt_net.load_adapter("lora_model_working", adapter_name="default", ignore_mismatched_sizes=True)
        print("SUCCESS: LoRA model loaded without modules_to_save!")
        alt_net.print_trainable_parameters()
    except Exception as e:
        print(f"Failed without modules_to_save: {e}")
    
    print("\n=== SUMMARY ===")
    print("The issue was caused by key mismatches due to modules_to_save configuration.")
    print("Solutions:")
    print("1. Use PeftModel.from_pretrained() - recommended")
    print("2. Use load_adapter() with ignore_mismatched_sizes=True")
    print("3. Avoid modules_to_save if not necessary")

if __name__ == "__main__":
    main() 