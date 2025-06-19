import os
import sys
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

# Constants
METADATA_PATH = "/nvme1/hungdx/Lightning-hydra/data/cnsl_benchmark/AIHUB_FreeCommunication_may/protocol.txt"
META_CSV_PATH = "/nvme1/hungdx/Lightning-hydra/data/cnsl_benchmark/AIHUB_FreeCommunication_may_meta.csv"
PREDICTION_FILE = "/nvme1/hungdx/Lightning-hydra/logs/results/cnsl_benchmark/ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs/AIHUB_FreeCommunication_may_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs.txt"

def load_metadata() -> pd.DataFrame:
    """Load and process metadata files."""
    try:
        metadata = pd.read_csv(METADATA_PATH, sep=" ", header=None)
        metadata.columns = ["path", "subset", "label"]
        
        meta_csv = pd.read_csv(META_CSV_PATH)
        metadata = metadata.merge(meta_csv, left_on='path', right_on='path_audio', how='left')
        
        return metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata: {str(e)}")

def process_prediction_file(score_file: str, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Process prediction file and return results DataFrame."""
    try:
        pred_df = pd.read_csv(score_file, sep=" ", header=None)
        pred_df.columns = ["path", "spoof", "score"]
        pred_df = pred_df.drop_duplicates(subset=['path'])
        
        merged_df = pred_df.merge(metadata_df, left_on='path', right_on='path_audio', how='left')
        merged_df['pred'] = merged_df.apply(
            lambda x: 'bonafide' if x['spoof'] < x['score'] else 'spoof', axis=1)
        
        return merged_df
    except Exception as e:
        raise RuntimeError(f"Failed to process prediction file {score_file}: {str(e)}")

def analyze_misclassifications(df: pd.DataFrame, target_attack_types: List[str]) -> None:
    """Analyze and print misclassified samples for specific attack types. Save paths to separate txt files."""
    # Filter for target attack types
    target_df = df[df['attack_type'].isin(target_attack_types)]
    
    # Find misclassified samples
    misclassified = target_df[target_df['label'] != target_df['pred']]

    print(f"\n{'='*80}")
    print(f"Misclassification Analysis for Attack Types: {', '.join(target_attack_types)}")
    print(f"{'='*80}")
    
    for attack_type in target_attack_types:
        attack_misclassified = misclassified[misclassified['attack_type'] == attack_type]
        
        # Save misclassified paths to separate file
        output_txt = f"misclassified_{attack_type}.txt"
        attack_paths = attack_misclassified['path_audio'].dropna().tolist()
        with open(output_txt, 'w') as f:
            for path in attack_paths:
                f.write(f"{path}\n")
        print(f"Saved misclassified file paths for {attack_type} to: {output_txt}")

        print(f"\nAttack Type: {attack_type}")
        print(f"Total samples: {len(target_df[target_df['attack_type'] == attack_type])}")
        print(f"Misclassified samples: {len(attack_misclassified)}")
        
        if len(attack_misclassified) > 0:
            print("\nMisclassified Samples Details:")
            print("-" * 80)
            for _, row in attack_misclassified.iterrows():
                print(f"File: {row['path_audio']}")
                print(f"True Label: {row['label']}")
                print(f"Predicted: {row['pred']}")
                print(f"Spoof Score: {row['spoof']:.4f}")
                print(f"Bonafide Score: {row['score']:.4f}")
                print("-" * 80)

def main() -> None:
    """Main function to run the misclassification analysis."""
    try:
        print("Loading metadata...")
        metadata_df = load_metadata()
        
        print(f"\nProcessing prediction file...")
        results_df = process_prediction_file(PREDICTION_FILE, metadata_df)
        
        # Analyze misclassifications for a04 and a08
        target_attack_types = ['a04', 'a08']
        analyze_misclassifications(results_df, target_attack_types)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 