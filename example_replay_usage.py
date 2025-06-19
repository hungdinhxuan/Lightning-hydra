#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of ReplayDataModule for continual learning with replay buffer.

This example shows how to:
1. Set up the replay datamodule with novel and replay sets
2. Configure the ratio of novel vs replay samples in each batch
3. Use separate protocol files for novel and replay data
"""

import os
from src.data.replay_multiview_datamodule import ReplayDataModule

def main():
    # Configuration for the replay datamodule
    args = {
        'views': 4,                    # Number of views for multi-view processing
        'wav_samp_rate': 16000,       # Audio sampling rate
        'padding_type': 'repeat',      # Padding type: 'repeat' or 'zero'
        'random_start': True,          # Whether to use random start for audio segments
        'view_padding_configs': None,  # Custom padding configs for each view
        'trim_length': 66800,          # Length to trim/pad audio to (~4.175 seconds at 16kHz)
        'augmentation_methods': [],    # List of augmentation methods to apply
        'eval_augment': None,          # Augmentation for evaluation
        'num_additional_real': 2,      # Additional real samples for augmentation
        'num_additional_spoof': 2,     # Additional spoof samples for augmentation
        'noise_path': None,            # Path to noise files for augmentation
        'rir_path': None,              # Path to RIR files for augmentation
        'aug_dir': None,               # Directory for augmented files
        'online_aug': True,            # Whether to apply augmentation online
        'repeat_pad': True,            # Whether to use repeat padding
        'algo': 5,                     # Algorithm version
        'vocoders': [],                # List of vocoders
        'enable_cache': False,         # Whether to enable caching
        'cache_dir': None,             # Cache directory
    }
    
    # Create the replay datamodule
    datamodule = ReplayDataModule(
        data_dir="data/",                                    # Base data directory
        batch_size=32,                                       # Batch size
        num_workers=4,                                       # Number of data loading workers
        pin_memory=True,                                     # Whether to pin memory
        args=args,                                           # Additional arguments
        novel_ratio=0.7,                                     # 70% novel samples per batch
        replay_ratio=0.3,                                    # 30% replay samples per batch
        novel_protocol_path="data/novel_protocol.txt",       # Protocol file for novel set
        replay_protocol_path="data/replay_protocol.txt",     # Protocol file for replay set
        chunking_eval=False,                                 # Whether to use chunking for evaluation
        enable_cache=False,                                  # Whether to enable caching
    )
    
    # Setup the datamodule (this would normally be called by Lightning trainer)
    try:
        datamodule.setup(stage="fit")
        
        # Get the training dataloader
        train_loader = datamodule.train_dataloader()
        
        print(f"Training dataloader created successfully!")
        print(f"Number of batches per epoch: {len(train_loader)}")
        
        # Example: iterate through a few batches
        print("\nExample batch information:")
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # Only show first 3 batches
                break
                
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                features, targets = batch[0], batch[1]
                print(f"Batch {batch_idx}: Features shape: {features.shape if hasattr(features, 'shape') else type(features)}, "
                      f"Targets: {targets.shape if hasattr(targets, 'shape') else type(targets)}")
            else:
                print(f"Batch {batch_idx}: {type(batch)}")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure that the protocol files exist:")
        print("- data/novel_protocol.txt")
        print("- data/replay_protocol.txt")
        print("\nProtocol file format should be:")
        print("utt_id subset label")
        print("Example:")
        print("file1.wav train bonafide")
        print("file2.wav train spoof")
        print("file3.wav dev bonafide")


def create_example_protocol_files():
    """Create example protocol files for demonstration"""
    os.makedirs("data", exist_ok=True)
    
    # Example novel protocol file
    novel_protocol = """novel_001.wav train bonafide
novel_002.wav train spoof
novel_003.wav train bonafide
novel_004.wav train spoof
novel_005.wav dev bonafide
novel_006.wav dev spoof
novel_007.wav eval bonafide
novel_008.wav eval spoof
"""
    
    # Example replay protocol file
    replay_protocol = """replay_001.wav train bonafide
replay_002.wav train spoof
replay_003.wav train bonafide
replay_004.wav train spoof
replay_005.wav dev bonafide
replay_006.wav dev spoof
replay_007.wav eval bonafide
replay_008.wav eval spoof
"""
    
    with open("data/novel_protocol.txt", "w") as f:
        f.write(novel_protocol)
    
    with open("data/replay_protocol.txt", "w") as f:
        f.write(replay_protocol)
    
    print("Example protocol files created:")
    print("- data/novel_protocol.txt")
    print("- data/replay_protocol.txt")


if __name__ == "__main__":
    # Uncomment the line below to create example protocol files
    # create_example_protocol_files()
    
    main() 