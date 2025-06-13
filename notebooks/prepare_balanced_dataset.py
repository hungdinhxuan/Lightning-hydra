#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare balanced dataset for continual learning experiments.

For each attack type (a01, a04, a08, -):
- Randomly select 5000 samples for each attack
- Divide 50% for train and 50% for dev
- Save the new dataset to a new csv file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_and_prepare_data():
    """Load spoof and bonafide data and merge them."""
    print("Loading spoof data...")
    spoof_df = pd.read_csv('/nvme1/hungdx/Lightning-hydra/data/0_large-corpus/AIHUB_FreeCommunication/may/june_week1_spoof.csv')
    spoof_df.columns = ['spk_id', 'path_audio', 'attack_type']
    spoof_df['label'] = 'spoof'
    spoof_df['path_audio'] = spoof_df['path_audio'].apply(lambda x: f"Spoof/{x}")

    print("Loading bonafide data...")
    bonafide_df = pd.read_csv('/nvme1/hungdx/Lightning-hydra/data/0_large-corpus/AIHUB_FreeCommunication/may/may_bona.csv')
    bonafide_df['path_audio'] = bonafide_df['path_audio'].apply(lambda x: f"Bonafide/{x}")
    bonafide_df['label'] = 'bonafide'
    bonafide_df['attack_type'] = '-'  # Bonafide samples marked with '-'

    print("Spoof columns:", spoof_df.columns.tolist())
    print("Bonafide columns:", bonafide_df.columns.tolist())

    # Concatenate instead of merge to avoid conflicts
    merged_df = pd.concat([spoof_df, bonafide_df], ignore_index=True)
    
    print(f"Total merged samples: {len(merged_df)}")
    print(f"Attack type distribution:")
    print(merged_df['attack_type'].value_counts())
    
    return merged_df

def create_balanced_dataset(merged_df, target_attacks=['a01', 'a04', 'a08', '-'], samples_per_attack=5000):
    """
    Create balanced dataset by sampling equal numbers from each attack type.
    
    Args:
        merged_df: DataFrame containing all samples
        target_attacks: List of attack types to include
        samples_per_attack: Number of samples to select per attack type
        
    Returns:
        train_df, dev_df: Training and development DataFrames
    """
    print(f"\nCreating balanced dataset with {samples_per_attack} samples per attack...")
    print(f"Target attacks: {target_attacks}")
    
    train_samples = []
    dev_samples = []
    
    for attack in target_attacks:
        print(f"\nProcessing attack type: {attack}")
        
        # Filter samples for this attack type
        attack_samples = merged_df[merged_df['attack_type'] == attack].copy()
        available_samples = len(attack_samples)
        
        print(f"Available samples for {attack}: {available_samples}")
        
        if available_samples < samples_per_attack:
            print(f"WARNING: Only {available_samples} samples available for {attack}, using all of them")
            selected_samples = attack_samples
        else:
            # Randomly sample the required number
            selected_samples = attack_samples.sample(n=samples_per_attack, random_state=42)
            print(f"Selected {len(selected_samples)} samples for {attack}")
        
        # Split 50/50 between train and dev
        n_train = len(selected_samples) // 2
        n_dev = len(selected_samples) - n_train
        
        # Shuffle and split
        shuffled_samples = selected_samples.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_attack = shuffled_samples[:n_train].copy()
        dev_attack = shuffled_samples[n_train:].copy()
        
        # Add subset labels
        train_attack['subset'] = 'train'
        dev_attack['subset'] = 'dev'
        
        train_samples.append(train_attack)
        dev_samples.append(dev_attack)
        
        print(f"Train samples for {attack}: {len(train_attack)}")
        print(f"Dev samples for {attack}: {len(dev_attack)}")
    
    # Combine all samples
    train_df = pd.concat(train_samples, ignore_index=True)
    dev_df = pd.concat(dev_samples, ignore_index=True)
    
    print(f"\nFinal dataset summary:")
    print(f"Total train samples: {len(train_df)}")
    print(f"Total dev samples: {len(dev_df)}")
    
    return train_df, dev_df

def save_datasets(train_df, dev_df, output_dir='/nvme1/hungdx/Lightning-hydra/data/cnsl_benchmark/AIHUB_FreeCommunication_balanced'):
    """Save the balanced datasets to files."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine train and dev
    full_df = pd.concat([train_df, dev_df], ignore_index=True)
    
    print(f"\nSaving datasets to {output_dir}")
    
    # Save full metadata CSV
    metadata_path = os.path.join(output_dir, 'balanced_metadata.csv')
    full_df.to_csv(metadata_path, index=False)
    print(f"Saved metadata to: {metadata_path}")
    
    # Save protocol file (path subset label format)
    protocol_df = full_df[['path_audio', 'subset', 'label']].copy()
    protocol_path = os.path.join(output_dir, 'protocol.txt')
    protocol_df.to_csv(protocol_path, header=False, sep=' ', index=False)
    print(f"Saved protocol to: {protocol_path}")
    
    # Save separate train and dev files
    train_path = os.path.join(output_dir, 'train_balanced.csv')
    train_df.to_csv(train_path, index=False)
    print(f"Saved train set to: {train_path}")
    
    dev_path = os.path.join(output_dir, 'dev_balanced.csv')
    dev_df.to_csv(dev_path, index=False)
    print(f"Saved dev set to: {dev_path}")
    
    # Print final statistics
    print(f"\nFinal dataset statistics:")
    print("Attack type distribution in full dataset:")
    print(full_df['attack_type'].value_counts())
    
    print("\nSubset distribution:")
    print(full_df['subset'].value_counts())
    
    print("\nLabel distribution:")
    print(full_df['label'].value_counts())
    
    print("\nAttack type distribution by subset:")
    subset_attack_dist = full_df.groupby(['subset', 'attack_type']).size().unstack(fill_value=0)
    print(subset_attack_dist)
    
    return metadata_path, protocol_path

def create_replay_sets(full_df, replay_ratio=0.3, output_dir='/nvme1/hungdx/Lightning-hydra/data/cnsl_benchmark/AIHUB_FreeCommunication_balanced'):
    """
    Create novel and replay sets for continual learning experiments.
    
    Args:
        full_df: Complete balanced dataset
        replay_ratio: Fraction of data to use as replay set
        output_dir: Directory to save the files
    """
    print(f"\nCreating replay sets with {replay_ratio*100}% replay ratio...")
    
    # Split by attack type to ensure balanced representation
    novel_samples = []
    replay_samples = []
    
    for attack in full_df['attack_type'].unique():
        attack_data = full_df[full_df['attack_type'] == attack].copy()
        
        # Calculate splits
        n_total = len(attack_data)
        n_replay = int(n_total * replay_ratio)
        n_novel = n_total - n_replay
        
        # Randomly shuffle and split
        shuffled = attack_data.sample(frac=1, random_state=123).reset_index(drop=True)
        
        replay_attack = shuffled[:n_replay].copy()
        novel_attack = shuffled[n_replay:].copy()
        
        replay_samples.append(replay_attack)
        novel_samples.append(novel_attack)
        
        print(f"Attack {attack}: {n_replay} replay, {n_novel} novel")
    
    # Combine splits
    replay_df = pd.concat(replay_samples, ignore_index=True)
    novel_df = pd.concat(novel_samples, ignore_index=True)
    
    # Save replay protocol
    replay_protocol = replay_df[['path_audio', 'subset', 'label']]
    replay_protocol_path = os.path.join(output_dir, 'replay_protocol.txt')
    replay_protocol.to_csv(replay_protocol_path, header=False, sep=' ', index=False)
    
    # Save novel protocol  
    novel_protocol = novel_df[['path_audio', 'subset', 'label']]
    novel_protocol_path = os.path.join(output_dir, 'novel_protocol.txt')
    novel_protocol.to_csv(novel_protocol_path, header=False, sep=' ', index=False)
    
    print(f"Saved replay protocol to: {replay_protocol_path}")
    print(f"Saved novel protocol to: {novel_protocol_path}")
    print(f"Replay set size: {len(replay_df)}")
    print(f"Novel set size: {len(novel_df)}")
    
    return replay_protocol_path, novel_protocol_path

def main():
    """Main function to create balanced dataset."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("=" * 70)
    print("Creating Balanced Dataset for Continual Learning")
    print("=" * 70)
    
    # Load and prepare data
    merged_df = load_and_prepare_data()
    
    # Define target attacks and samples per attack
    target_attacks = ['a01', 'a04', 'a08', '-']  # Include bonafide (-)
    samples_per_attack = 5000
    
    # Create balanced dataset
    train_df, dev_df = create_balanced_dataset(
        merged_df, 
        target_attacks=target_attacks, 
        samples_per_attack=samples_per_attack
    )
    
    # Save datasets
    full_df = pd.concat([train_df, dev_df], ignore_index=True)
    metadata_path, protocol_path = save_datasets(train_df, dev_df)
    
    # Create replay sets for continual learning
    replay_protocol_path, novel_protocol_path = create_replay_sets(full_df)
    
    print("\n" + "=" * 70)
    print("Dataset creation completed successfully!")
    print("=" * 70)
    print(f"Main protocol file: {protocol_path}")
    print(f"Replay protocol file: {replay_protocol_path}")
    print(f"Novel protocol file: {novel_protocol_path}")
    
    return metadata_path, protocol_path, replay_protocol_path, novel_protocol_path

if __name__ == "__main__":
    main() 