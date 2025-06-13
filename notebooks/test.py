# import pandas as pd
# import sys
# import os
# import random
# import numpy as np
# import librosa
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm
# import functools

# os.chdir(os.path.dirname(__file__))  # Change to script's directory

# sys.path.append("../")

# from src.data.components.audio_augmentor.autotune import AutoTuneAugmentor
# from src.data.components.audio_augmentor.base import BaseAugmentor

# def audio_transform(filepath: str, aug_type: BaseAugmentor, config: dict, online: bool = False, lrs=False):
#     """
#     filepath: str, input audio file path
#     aug_type: BaseAugmentor, augmentation type object
#     config: dict, configuration dictionary
#     online: bool, if True, return the augmented audio waveform, else save the augmented audio file
#     """
#     at = aug_type(config)
#     at.load(filepath)
#     at.transform()
#     if online:
#         audio = at.augmented_audio
#         if lrs:
#             return audio
#         return pydub_to_librosa(audio)
#     else:
#         at.save()

# def autotune_v1(audio_path, args, sr=16000):
#     """
#     Apply autotune augmentation to an audio file
    
#     Parameters:
#     -----------
#     audio_path: str, path to the audio file
#     args: Args object containing configuration
#     sr: int, sample rate
    
#     Returns:
#     --------
#     waveform: numpy array, the augmented audio waveform
#     """
#     aug_dir = args.aug_dir
#     utt_id = os.path.basename(audio_path).split('.')[0]
#     args.input_path = os.path.dirname(audio_path)
#     aug_audio_path = os.path.join(aug_dir, 'autotune', utt_id + '.wav')
#     args.output_path = os.path.join(aug_dir, 'autotune')
    
#     # Create output directory if it doesn't exist
#     os.makedirs(args.output_path, exist_ok=True)
    
#     args.out_format = 'wav'
#     config = {
#         "aug_type": "autotune",
#         "output_path": args.output_path,
#         "out_format": args.out_format,
#         "noise_path": args.noise_path
#     }
    
#     if (args.online_aug):
#         waveform = audio_transform(
#             filepath=audio_path, aug_type=AutoTuneAugmentor, config=config, online=True, lrs=True)
#         return waveform
#     else:
#         if os.path.exists(aug_audio_path):
#             waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
#             return waveform
#         else:
#             audio_transform(
#                 filepath=audio_path, aug_type=AutoTuneAugmentor, config=config, online=False, lrs=True)
#             waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
#             return waveform

# class Args:
#     def __init__(self):
#         self.aug_dir = "/nvme1/hungdx/Lightning-hydra/data/0_large-corpus/aug"
#         self.online_aug = False
#         self.noise_path = "/tmp/noise.wav"

# def process_sample(audio_path, args):
#     """
#     Process a single audio sample for parallel execution
    
#     Parameters:
#     -----------
#     audio_path: str, path to the audio file
#     args: Args object containing configuration
    
#     Returns:
#     --------
#     audio_path: str, path to the processed audio file (for tracking purposes)
#     """
#     try:
#         # Fixed: Correctly call autotune_v1 with audio_path as the first argument
#         autotune_v1(audio_path, args)
#         return audio_path
#     except Exception as e:
#         return f"Error processing {audio_path}: {str(e)}"

# def main():
#     # Load dataset
#     BASE_DIR = "/nvme1/hungdx/Lightning-hydra/data/0_large-corpus"
#     df = pd.read_csv("/nvme1/hungdx/Lightning-hydra/notebooks/new_protocol_trim_vocoded_cleaned_v2.txt", 
#                      sep=" ", header=None)
#     df.columns = ["utt", "subset", "label"]
    
#     # Filter to get samples with subset is train
#     train_df = df[df["subset"] == "train"]
    
#     # Get full paths for all training samples
#     audio_paths = [os.path.join(BASE_DIR, utt) for utt in train_df["utt"].values]
    
#     # Initialize args
#     args = Args()
    
#     # Set the maximum number of workers (adjust based on your CPU capabilities)
#     max_workers = 20
    
#     # Create a partial function with fixed args
#     process_fn = functools.partial(process_sample, args=args)
    
#     # Process all files in parallel with progress bar
#     print(f"Processing {len(audio_paths)} audio files with {max_workers} workers...")
    
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         # Submit all tasks and track with tqdm
#         results = list(tqdm(
#             executor.map(process_fn, audio_paths),
#             total=len(audio_paths),
#             desc="Augmenting audio",
#             unit="file"
#         ))
    
#     # Count successful and failed operations
#     successful = sum(1 for r in results if not isinstance(r, str) or not r.startswith("Error"))
#     failed = len(results) - successful
    
#     print(f"Augmentation complete: {successful} files processed successfully, {failed} files failed")
    
#     # If there were errors, print the first few
#     if failed > 0:
#         errors = [r for r in results if isinstance(r, str) and r.startswith("Error")]
#         print(f"First {min(5, len(errors))} errors:")
#         for i, error in enumerate(errors[:5]):
#             print(f"{i+1}. {error}")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug version of evaluation script to identify missing 'label' column issue
"""

import os
import sys
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from pathlib import Path

# Constants
METADATA_PATH = "/nvme1/hungdx/Lightning-hydra/data/cnsl_benchmark/AIHUB_FreeCommunication_may_updated/protocol.txt"
META_CSV_PATH = "/nvme1/hungdx/Lightning-hydra/data/cnsl_benchmark/AIHUB_FreeCommunication_may_updated/june_week1_merged_df.csv"
BASE_DIR = "/nvme1/hungdx/Lightning-hydra/logs/eval/cnsl/largecorpus"

# old prediction file
PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/cnsl_benchmark/ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs/AIHUB_FreeCommunication_may_updated_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs.txt"

def debug_load_metadata() -> pd.DataFrame:
    """Load and debug metadata files."""
    print("=== DEBUGGING METADATA LOADING ===")
    
    try:
        print(f"Loading metadata from: {METADATA_PATH}")
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
        
        # Load protocol file
        metadata = pd.read_csv(METADATA_PATH, sep=" ", header=None)
        print(f"Protocol file shape: {metadata.shape}")
        print(f"Protocol file columns: {metadata.columns.tolist()}")
        print("First few rows of protocol file:")
        print(metadata.head())
        
        # Set column names
        metadata.columns = ["path", "subset", "label"]
        print(f"After setting column names: {metadata.columns.tolist()}")
        print(f"Unique labels: {metadata['label'].unique()}")
        print(f"Label counts: {metadata['label'].value_counts()}")
        
        # Load CSV metadata
        print(f"\nLoading CSV metadata from: {META_CSV_PATH}")
        if not os.path.exists(META_CSV_PATH):
            raise FileNotFoundError(f"CSV metadata file not found: {META_CSV_PATH}")
        
        meta_csv = pd.read_csv(META_CSV_PATH)
        print(f"CSV metadata shape: {meta_csv.shape}")
        print(f"CSV metadata columns: {meta_csv.columns.tolist()}")
        print("First few rows of CSV metadata:")
        print(meta_csv.head())
        
        # Check if 'path_audio' column exists
        if 'path_audio' not in meta_csv.columns:
            print("WARNING: 'path_audio' column not found in CSV metadata")
            print("Available columns:", meta_csv.columns.tolist())
            # Try to find similar column names
            audio_cols = [col for col in meta_csv.columns if 'path' in col.lower() or 'audio' in col.lower()]
            if audio_cols:
                print(f"Potential audio path columns: {audio_cols}")
                # Use the first one found
                meta_csv['path_audio'] = meta_csv[audio_cols[0]]
                print(f"Using '{audio_cols[0]}' as path_audio column")
        
        # Handle column conflicts before merge
        # Drop conflicting columns from meta_csv if they exist
        conflicting_cols = ['subset', 'label', 'path']
        for col in conflicting_cols:
            if col in meta_csv.columns:
                print(f"Dropping conflicting column '{col}' from CSV metadata")
                meta_csv = meta_csv.drop(columns=[col])
        
        # Perform merge
        print("\nPerforming merge...")
        merged_metadata = metadata.merge(meta_csv, left_on='path', right_on='path_audio', how='left')
        print(f"Merged metadata shape: {merged_metadata.shape}")
        print(f"Merged metadata columns: {merged_metadata.columns.tolist()}")
        
        # Ensure we have the correct column names (should be clean now)
        if 'label' not in merged_metadata.columns:
            # Check for suffixed columns and fix them
            if 'label_x' in merged_metadata.columns:
                print("Found 'label_x', renaming to 'label'")
                merged_metadata['label'] = merged_metadata['label_x']
                merged_metadata = merged_metadata.drop(columns=['label_x'])
            if 'label_y' in merged_metadata.columns:
                merged_metadata = merged_metadata.drop(columns=['label_y'])
        
        if 'subset' not in merged_metadata.columns:
            # Check for suffixed columns and fix them
            if 'subset_x' in merged_metadata.columns:
                print("Found 'subset_x', renaming to 'subset'")
                merged_metadata['subset'] = merged_metadata['subset_x']
                merged_metadata = merged_metadata.drop(columns=['subset_x'])
            if 'subset_y' in merged_metadata.columns:
                merged_metadata = merged_metadata.drop(columns=['subset_y'])
        
        # Fix path column if it got suffixed
        if 'path' not in merged_metadata.columns and 'path_x' in merged_metadata.columns:
            print("Found 'path_x', renaming to 'path'")
            merged_metadata['path'] = merged_metadata['path_x']
            merged_metadata = merged_metadata.drop(columns=['path_x'])
        if 'path_y' in merged_metadata.columns:
            merged_metadata = merged_metadata.drop(columns=['path_y'])
        
        # Check for missing values
        print(f"\nMissing values in merged data:")
        print(merged_metadata.isnull().sum())
        
        # Check if label column still exists and has values
        if 'label' in merged_metadata.columns:
            print(f"\nLabel column found after merge")
            print(f"Unique labels after merge: {merged_metadata['label'].unique()}")
            print(f"Label counts after merge: {merged_metadata['label'].value_counts()}")
        else:
            print("ERROR: Label column missing after merge!")
            print("Available columns:", merged_metadata.columns.tolist())
        
        print(f"\nFinal cleaned columns: {merged_metadata.columns.tolist()}")
        return merged_metadata
        
    except Exception as e:
        print(f"Error in debug_load_metadata: {str(e)}")
        raise

def debug_process_prediction_file(score_file: str, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Debug version of prediction file processing."""
    print(f"\n=== DEBUGGING PREDICTION FILE PROCESSING ===")
    print(f"Processing file: {score_file}")
    
    try:
        if not os.path.exists(score_file):
            raise FileNotFoundError(f"Prediction file not found: {score_file}")
        
        # Load prediction file
        pred_df = pd.read_csv(score_file, sep=" ", header=None)
        print(f"Prediction file shape: {pred_df.shape}")
        print(f"Prediction file columns: {pred_df.columns.tolist()}")
        print("First few rows of prediction file:")
        print(pred_df.head())
        
        # Set column names
        pred_df.columns = ["path", "spoof", "score"]
        print(f"After setting column names: {pred_df.columns.tolist()}")
        
        # Remove duplicates
        original_size = len(pred_df)
        pred_df = pred_df.drop_duplicates(subset=['path'])
        print(f"Removed {original_size - len(pred_df)} duplicates")
        
        # Check metadata before merge
        print(f"\nMetadata for merge:")
        print(f"Metadata shape: {metadata_df.shape}")
        print(f"Metadata columns: {metadata_df.columns.tolist()}")
        if 'label' in metadata_df.columns:
            print(f"Labels in metadata: {metadata_df['label'].unique()}")
        else:
            print("WARNING: No 'label' column in metadata!")
        
        # Perform merge
        print(f"\nMerging prediction with metadata...")
        if 'path_audio' not in metadata_df.columns:
            print("ERROR: 'path_audio' column not found in metadata_df")
            print("Available columns:", metadata_df.columns.tolist())
            # Try to find the right column
            path_cols = [col for col in metadata_df.columns if 'path' in col.lower()]
            if path_cols:
                print(f"Using '{path_cols[0]}' as path column")
                merged_df = pred_df.merge(metadata_df, left_on='path', right_on=path_cols[0], how='left')
            else:
                raise ValueError("No suitable path column found in metadata")
        else:
            merged_df = pred_df.merge(metadata_df, left_on='path', right_on='path_audio', how='left')
        
        print(f"Merged DataFrame shape: {merged_df.shape}")
        print(f"Merged DataFrame columns: {merged_df.columns.tolist()}")
        
        # Handle column naming issues after merge
        if 'label' not in merged_df.columns:
            # Check for suffixed columns and fix them
            if 'label_x' in merged_df.columns:
                print("Found 'label_x', renaming to 'label'")
                merged_df['label'] = merged_df['label_x']
                merged_df = merged_df.drop(columns=['label_x'])
            elif 'label_y' in merged_df.columns:
                print("Found 'label_y', renaming to 'label'")
                merged_df['label'] = merged_df['label_y']
                merged_df = merged_df.drop(columns=['label_y'])
            else:
                print("ERROR: 'label' column missing after merge!")
                print("Available columns:", merged_df.columns.tolist())
                return merged_df
        
        # Clean up any remaining suffixed columns
        if 'label_x' in merged_df.columns and 'label' in merged_df.columns:
            merged_df = merged_df.drop(columns=['label_x'])
        if 'label_y' in merged_df.columns and 'label' in merged_df.columns:
            merged_df = merged_df.drop(columns=['label_y'])
        
        # Fix path column if needed
        if 'path' not in merged_df.columns and 'path_x' in merged_df.columns:
            print("Found 'path_x', renaming to 'path'")
            merged_df['path'] = merged_df['path_x']
        
        # Clean up suffixed path columns
        cols_to_drop = [col for col in merged_df.columns if col in ['path_x', 'path_y'] and col != 'path']
        if cols_to_drop:
            merged_df = merged_df.drop(columns=cols_to_drop)
        
        print(f"Labels after merge: {merged_df['label'].unique()}")
        print(f"Label counts after merge: {merged_df['label'].value_counts()}")
        
        # Check for missing labels
        missing_labels = merged_df['label'].isnull().sum()
        if missing_labels > 0:
            print(f"WARNING: {missing_labels} samples have missing labels")
        
        # Create predictions
        merged_df['pred'] = merged_df.apply(
            lambda x: 'bonafide' if x['spoof'] < x['score'] else 'spoof', axis=1)
        
        print(f"Prediction counts: {merged_df['pred'].value_counts()}")
        print(f"Final DataFrame columns: {merged_df.columns.tolist()}")
        
        return merged_df
        
    except Exception as e:
        print(f"Error in debug_process_prediction_file: {str(e)}")
        raise

def main() -> None:
    """Debug main function."""
    try:
        print("=== STARTING DEBUG EVALUATION ===")
        
        # Debug metadata loading
        metadata_df = debug_load_metadata()
        
        # Check if prediction file exists
        if not os.path.exists(PREDICTION_FILE):
            print(f"ERROR: Prediction file not found: {PREDICTION_FILE}")
            return
        
        # Debug prediction file processing
        results_df = debug_process_prediction_file(PREDICTION_FILE, metadata_df)
        
        # Final check before metrics calculation
        print(f"\n=== FINAL DATAFRAME CHECK ===")
        print(f"Final DataFrame shape: {results_df.shape}")
        print(f"Final DataFrame columns: {results_df.columns.tolist()}")
        
        required_columns = ['label', 'pred', 'score']
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        
        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            print("Cannot proceed with metrics calculation")
        else:
            print("SUCCESS: All required columns present")
            print("Sample of final data:")
            print(results_df[['path', 'label', 'pred', 'score']].head())
            
            # Quick metrics test
            print(f"\nQuick metrics test:")
            print(f"Accuracy: {accuracy_score(results_df['label'], results_df['pred']) * 100:.2f}%")
        
    except Exception as e:
        print(f"Error in debug main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()