import argparse
import json
from src.training.trainer import ContinualLearningTrainer
from src.configs.training_config import TrainingConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train continual learning model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume training")
    return parser.parse_args()

def load_config(config_path: str) -> TrainingConfig:
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return TrainingConfig(**config_dict)

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = ContinualLearningTrainer(config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    trainer.train()
    
    # Final evaluation
    results = trainer.evaluate()
    print("Final evaluation results:", results)

if __name__ == "__main__":
    main() 