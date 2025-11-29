#!/usr/bin/env python3
"""
Command-line interface for uploading models to Hugging Face Hub.

Usage:
    python upload_model.py --model_name conformer_tcm --config_path config.yaml --checkpoint_path model.pth --repo_name username/model-name

Or with all options:
    python upload_model.py --model_name conformer_tcm --config_path config.yaml --checkpoint_path model.pth --repo_name username/model-name --private --token your_token_here --commit_message "My model upload"
"""

import argparse
import os
import sys
from pathlib import Path

# Add the huggingface scripts directory to path
sys.path.append(str(Path(__file__).parent))

from push_to_hf import push_model_to_hf

def main():
    parser = argparse.ArgumentParser(
        description="Upload a model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic upload
  python upload_model.py --model_name conformer_tcm --config_path config.yaml --checkpoint_path model.pth --repo_name username/model-name
  
  # Private repository with token
  python upload_model.py --model_name conformer_tcm --config_path config.yaml --checkpoint_path model.pth --repo_name username/private-model --private --token hf_xxxxxxxxxxxx
  
  # Using environment variable for token
  export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx
  python upload_model.py --model_name conformer_tcm --config_path config.yaml --checkpoint_path model.pth --repo_name username/model-name
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model class (should match filename without .py)"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file"
    )
    
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Name of the Hugging Face repository (e.g., 'username/model-name')"
    )
    
    # Optional arguments
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default: False)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (if not provided, will use cached token or HUGGINGFACE_TOKEN env var)"
    )
    
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload model",
        help="Commit message for the upload (default: 'Upload model')"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.config_path):
        print(f"❌ Error: Config file not found: {args.config_path}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Get token from environment if not provided
    token = args.token
    if not token:
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("⚠️  Warning: No Hugging Face token provided")
            print("   You can set HUGGINGFACE_TOKEN environment variable or use --token argument")
    
    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    print(f"🚀 Starting model upload...")
    print(f"   Model: {args.model_name}")
    print(f"   Config: {args.config_path}")
    print(f"   Checkpoint: {args.checkpoint_path}")
    print(f"   Repository: {args.repo_name}")
    print(f"   Private: {args.private}")
    print(f"   Commit message: {args.commit_message}")
    print()
    
    # Upload the model
    try:
        success = push_model_to_hf(
            model_name=args.model_name,
            config_path=args.config_path,
            checkpoint_path=args.checkpoint_path,
            repo_name=args.repo_name,
            private=args.private,
            commit_message=args.commit_message,
            token=token
        )
        
        if success:
            print(f"✅ Successfully uploaded model to {args.repo_name}")
            print(f"   View at: https://huggingface.co/{args.repo_name}")
        else:
            print("❌ Model upload failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Upload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
