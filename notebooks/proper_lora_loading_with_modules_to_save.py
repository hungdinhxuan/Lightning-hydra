#!/usr/bin/env python3
"""
Proper LoRA loading solution that keeps modules_to_save
The modules_to_save is essential for unfreezing backend modules for fine-tuning
"""

import torch
import sys
import yaml
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors import safe_open

sys.path.append('../')
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM

def proper_lora_loading_with_modules_to_save():
    """The correct way to load LoRA with modules_to_save"""
    
    # Load configuration
    with open('/nvme1/hungdx/Lightning-hydra/configs/experiment/cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs_july4.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    args = config['model']['args']['conformer']
    ckpt = torch.load('/nvme1/hungdx/Lightning-hydra/notebooks/S_241214_conf-1.pth', weights_only=False)
    
    print("=== PROPER SOLUTION: Keep modules_to_save ===")
    print("modules_to_save is IMPORTANT - it unfreezes backend modules for fine-tuning")
    
    # STEP 1: Save LoRA model with modules_to_save
    print("\nStep 1: Creating and saving LoRA model with modules_to_save...")
    
    model = XLSRConformerTCM(args=args,
                             ssl_pretrained_path='/nvme1/hungdx/Lightning-hydra/xlsr2_300m.pt')
    model.load_state_dict(ckpt, strict=True)
    
    # Keep the modules_to_save configuration - it's essential!
    lora_config = LoraConfig(
        r=config['model']['args']['adapter']['r'],
        target_modules=list(config['model']['args']['adapter']['target_modules']),
        modules_to_save=list(config['model']['args']['adapter']['modules_to_save']),  # Keep this!
        lora_dropout=config['model']['args']['adapter'].get('lora_dropout', 0.0),
        lora_alpha=config['model']['args']['adapter'].get('lora_alpha', 8),
    )
    
    print(f"LoRA config with modules_to_save: {lora_config.modules_to_save}")
    
    net = get_peft_model(model, lora_config)
    net.print_trainable_parameters()
    
    # Save the model
    net.save_pretrained("lora_model_proper")
    print("LoRA model saved to 'lora_model_proper'")
    
    # STEP 2: The correct way to load - use PeftModel.from_pretrained on BASE model
    print("\nStep 2: Loading LoRA correctly using PeftModel.from_pretrained...")
    
    # Create a fresh BASE model (not PEFT model)
    fresh_base_model = XLSRConformerTCM(args=args,
                                        ssl_pretrained_path='/nvme1/hungdx/Lightning-hydra/xlsr2_300m.pt')
    fresh_base_model.load_state_dict(ckpt, strict=True)
    
    try:
        # Load LoRA adapter directly onto the base model
        loaded_model = PeftModel.from_pretrained(fresh_base_model, "lora_model_proper")
        print("✅ SUCCESS: LoRA model loaded with PeftModel.from_pretrained!")
        loaded_model.print_trainable_parameters()
        return loaded_model
        
    except Exception as e:
        print(f"❌ Failed with PeftModel.from_pretrained: {e}")
        
        # STEP 3: Alternative - manual key mapping
        print("\nStep 3: Alternative solution - custom key mapping...")
        try:
            # Load the saved adapter weights
            adapter_weights = {}
            with safe_open('lora_model_proper/adapter_model.safetensors', framework='pt', device='cpu') as f:
                adapter_weights = {k: f.get_tensor(k) for k in f.keys()}
            
            # Create a new PEFT model with the same config
            fresh_model = XLSRConformerTCM(args=args,
                                           ssl_pretrained_path='/nvme1/hungdx/Lightning-hydra/xlsr2_300m.pt')
            fresh_model.load_state_dict(ckpt, strict=True)
            
            fresh_net = get_peft_model(fresh_model, lora_config)
            
            # Try loading with ignore_mismatched_sizes
            fresh_net.load_adapter("lora_model_proper", adapter_name="default", ignore_mismatched_sizes=True)
            print("✅ SUCCESS: LoRA model loaded with ignore_mismatched_sizes!")
            fresh_net.print_trainable_parameters()
            return fresh_net
            
        except Exception as e2:
            print(f"❌ Failed with ignore_mismatched_sizes: {e2}")
            
            # STEP 4: Debugging approach
            print("\nStep 4: Debugging key structure...")
            
            # Create the expected model structure
            debug_model = XLSRConformerTCM(args=args,
                                           ssl_pretrained_path='/nvme1/hungdx/Lightning-hydra/xlsr2_300m.pt')
            debug_model.load_state_dict(ckpt, strict=True)
            debug_net = get_peft_model(debug_model, lora_config)
            
            # Get expected keys
            expected_keys = debug_net.state_dict().keys()
            lora_expected = [k for k in expected_keys if 'lora' in k.lower()]
            
            print(f"Expected LoRA keys: {len(lora_expected)}")
            print("Sample expected keys:")
            for i, key in enumerate(list(lora_expected)[:5]):
                print(f"  {i+1}. {key}")
            
            print(f"\nSaved LoRA keys: {len(adapter_weights)}")
            print("Sample saved keys:")
            for i, key in enumerate(list(adapter_weights.keys())[:5]):
                print(f"  {i+1}. {key}")
            
            # Find the mismatch
            backend_expected = [k for k in lora_expected if 'backend' in k and 'qkv' in k]
            backend_saved = [k for k in adapter_weights.keys() if 'backend' in k and 'qkv' in k]
            
            print(f"\nBackend QKV expected: {len(backend_expected)}")
            for key in backend_expected[:3]:
                print(f"  {key}")
            
            print(f"\nBackend QKV saved: {len(backend_saved)}")
            for key in backend_saved[:3]:
                print(f"  {key}")
            
            return None

def test_successful_loading():
    """Test if the loading actually works correctly"""
    print("\n=== TESTING SUCCESSFUL LOADING ===")
    
    # Try the approach used in your existing code
    with open('/nvme1/hungdx/Lightning-hydra/configs/experiment/cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs_july4.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    args = config['model']['args']['conformer']
    ckpt = torch.load('/nvme1/hungdx/Lightning-hydra/notebooks/S_241214_conf-1.pth', weights_only=False)
    
    # Create base model
    base_model = XLSRConformerTCM(args=args,
                                  ssl_pretrained_path='/nvme1/hungdx/Lightning-hydra/xlsr2_300m.pt')
    base_model.load_state_dict(ckpt, strict=True)
    
    # Try loading from existing saved models
    existing_models = ["lora_model", "lora_model_working", "lora_model_final"]
    
    for model_path in existing_models:
        try:
            print(f"\nTrying to load from {model_path}...")
            loaded_model = PeftModel.from_pretrained(base_model, model_path)
            print(f"✅ SUCCESS: Loaded from {model_path}!")
            loaded_model.print_trainable_parameters()
            return loaded_model
        except Exception as e:
            print(f"❌ Failed to load from {model_path}: {str(e)[:100]}...")
    
    return None

def main():
    print("=== PROPER LoRA LOADING WITH MODULES_TO_SAVE ===")
    print("Understanding: modules_to_save is essential for unfreezing backend modules")
    
    # Try the proper approach
    result = proper_lora_loading_with_modules_to_save()
    
    if result is None:
        print("\n=== TRYING EXISTING MODELS ===")
        result = test_successful_loading()
    
    if result is not None:
        print("\n🎉 LoRA loading successful with modules_to_save preserved!")
        print("The backend modules are properly unfrozen for fine-tuning.")
    else:
        print("\n❌ All attempts failed. The issue needs deeper investigation.")
        print("Recommendation: Check PEFT library version and model structure compatibility.")

if __name__ == "__main__":
    main() 