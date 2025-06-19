import os
import sys
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from pathlib import Path
import eval_metrics_DF as em

# Constants
METADATA_PATH = "/nvme1/hungdx/Lightning-hydra/data/cnsl_benchmark/AIHUB_FreeCommunication_may_updated/protocol.txt"
META_CSV_PATH = "/nvme1/hungdx/Lightning-hydra/data/cnsl_benchmark/AIHUB_FreeCommunication_may_updated/june_week1_merged_df.csv"
BASE_DIR = "/nvme1/hungdx/Lightning-hydra/logs/eval/cnsl/largecorpus"

# old prediction file
PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/cnsl_benchmark/ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs/AIHUB_FreeCommunication_may_updated_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs.txt"


class MetricsCalculator:
    @staticmethod
    def calculate_metrics(df: pd.DataFrame, group_column: Optional[str] = None) -> Dict[str, Union[float, Dict[str, Union[float, int]]]]:
        """Calculate various metrics for the given DataFrame.
        
        Args:
            df: DataFrame containing predictions and ground truth
            group_column: Optional column name to group results by
            
        Returns:
            Dictionary containing calculated metrics and sample counts
        """
        results = {
            'overall': accuracy_score(df["label"], df["pred"]) * 100,
            'f1': f1_score(df['label'], df['pred'], pos_label='bonafide'),
            'recall': recall_score(df['label'], df['pred'], pos_label='bonafide'),
            'precision': precision_score(df['label'], df['pred'], pos_label='bonafide'),
            'total_samples': len(df)
        }
        
        # Calculate overall EER
        bona_cm = df[df['label'] == 'bonafide']['score'].values
        spoof_cm = df[df['label'] == 'spoof']['score'].values
        eer_cm, _ = em.compute_eer(bona_cm, spoof_cm)
        results['overall_eer'] = eer_cm * 100
        
        if group_column and group_column in df.columns:
            group_metrics = {}
            
            # Get all bona fide samples
            all_bona = df[df['label'] == 'bonafide']
            
            # Get all spoof samples
            all_spoof = df[df['label'] == 'spoof']
            
            # Process each attack type
            for group, group_df in df.groupby(group_column):
                if group == '-':  # For bona fide samples
                    # Use all spoof samples and all bona fide samples
                    group_bona = all_bona['score'].values
                    group_spoof = all_spoof['score'].values
                else:  # For spoof attack types
                    # Get spoof samples for this attack type
                    group_spoof = group_df[group_df['label'] == 'spoof']['score'].values
                    # Randomly select equal number of bona fide samples
                    n_spoof = len(group_spoof)
                    group_bona = np.random.choice(all_bona['score'].values, size=n_spoof, replace=False)
                
                # Calculate EER for this group
                group_eer, group_threshold = em.compute_eer(group_bona, group_spoof)
                
                group_metrics[group] = {
                    'accuracy': accuracy_score(group_df["label"], group_df["pred"]) * 100,
                    'eer': group_eer * 100,
                    'threshold': group_threshold,
                    'samples': len(group_df),
                    'bona_samples': len(group_bona),
                    'spoof_samples': len(group_spoof)
                }
            results['groups'] = group_metrics
        
        return results

def load_metadata() -> pd.DataFrame:
    """Load and process metadata files with proper column handling.
    
    Returns:
        DataFrame containing merged metadata
    """
    try:
        # Load protocol file
        metadata = pd.read_csv(METADATA_PATH, sep=" ", header=None)
        metadata.columns = ["path", "subset", "label"]
        
        # Load CSV metadata
        meta_csv = pd.read_csv(META_CSV_PATH)
        
        # Handle column conflicts before merge by dropping conflicting columns from meta_csv
        conflicting_cols = ['subset', 'label', 'path']
        for col in conflicting_cols:
            if col in meta_csv.columns:
                print(f"Dropping conflicting column '{col}' from CSV metadata")
                meta_csv = meta_csv.drop(columns=[col])
        
        # Check if 'path_audio' column exists
        if 'path_audio' not in meta_csv.columns:
            # Try to find similar column names
            audio_cols = [col for col in meta_csv.columns if 'path' in col.lower() or 'audio' in col.lower()]
            if audio_cols:
                meta_csv['path_audio'] = meta_csv[audio_cols[0]]
            else:
                raise ValueError("No suitable audio path column found in CSV metadata")
        
        # Perform merge
        merged_metadata = metadata.merge(meta_csv, left_on='path', right_on='path_audio', how='left')
        
        # Clean up any remaining suffixed columns (shouldn't happen now, but just in case)
        if 'label' not in merged_metadata.columns:
            if 'label_x' in merged_metadata.columns:
                merged_metadata['label'] = merged_metadata['label_x']
                merged_metadata = merged_metadata.drop(columns=['label_x'])
            if 'label_y' in merged_metadata.columns:
                merged_metadata = merged_metadata.drop(columns=['label_y'])
        
        if 'subset' not in merged_metadata.columns:
            if 'subset_x' in merged_metadata.columns:
                merged_metadata['subset'] = merged_metadata['subset_x']
                merged_metadata = merged_metadata.drop(columns=['subset_x'])
            if 'subset_y' in merged_metadata.columns:
                merged_metadata = merged_metadata.drop(columns=['subset_y'])
        
        # Fix path column if it got suffixed
        if 'path' not in merged_metadata.columns and 'path_x' in merged_metadata.columns:
            merged_metadata['path'] = merged_metadata['path_x']
            merged_metadata = merged_metadata.drop(columns=['path_x'])
        if 'path_y' in merged_metadata.columns:
            merged_metadata = merged_metadata.drop(columns=['path_y'])
        
        return merged_metadata
        
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata: {str(e)}")

def process_prediction_file(score_file: str, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Process a single prediction file and return results DataFrame.
    
    Args:
        score_file: Path to the prediction file
        metadata_df: DataFrame containing metadata
        
    Returns:
        DataFrame containing processed predictions
    """
    try:
        pred_df = pd.read_csv(score_file, sep=" ", header=None)
        pred_df.columns = ["path", "spoof", "score"]
        pred_df = pred_df.drop_duplicates(subset=['path'])
        
        # Perform merge
        if 'path_audio' not in metadata_df.columns:
            # Try to find the right column
            path_cols = [col for col in metadata_df.columns if 'path' in col.lower()]
            if path_cols:
                merged_df = pred_df.merge(metadata_df, left_on='path', right_on=path_cols[0], how='left')
            else:
                raise ValueError("No suitable path column found in metadata")
        else:
            merged_df = pred_df.merge(metadata_df, left_on='path', right_on='path_audio', how='left')
        
        # Handle column naming issues after merge
        if 'label' not in merged_df.columns:
            if 'label_x' in merged_df.columns:
                merged_df['label'] = merged_df['label_x']
                merged_df = merged_df.drop(columns=['label_x'])
            elif 'label_y' in merged_df.columns:
                merged_df['label'] = merged_df['label_y']
                merged_df = merged_df.drop(columns=['label_y'])
            else:
                raise ValueError("No label column found after merge")
        
        # Clean up any remaining suffixed columns
        cols_to_drop = []
        for col in ['label_x', 'label_y', 'path_x', 'path_y']:
            if col in merged_df.columns and col.replace('_x', '').replace('_y', '') in merged_df.columns:
                cols_to_drop.append(col)
        if cols_to_drop:
            merged_df = merged_df.drop(columns=cols_to_drop)
        
        # Create predictions
        merged_df['pred'] = merged_df.apply(
            lambda x: 'bonafide' if x['spoof'] < x['score'] else 'spoof', axis=1)
        
        return merged_df
        
    except Exception as e:
        raise RuntimeError(f"Failed to process prediction file {score_file}: {str(e)}")

def print_results(model_results: Dict, model_name: str, original_results: Optional[Dict] = None) -> None:
    """Print evaluation results in a formatted way.
    
    Args:
        model_results: Dictionary containing model metrics
        model_name: Name of the model
        original_results: Optional dictionary containing original model results for comparison
    """
    print(f"\n{'-'*70}")
    print(f"Model: {model_name}")
    
    print(f"\nTotal Samples: {model_results['total_samples']}")
    print(f"Overall Accuracy: {model_results['overall']:.2f}%")
    print(f"Overall EER: {model_results['overall_eer']:.2f}%")
    
    if 'groups' in model_results:
        print("\nMetrics by group:")
        for group, metrics in model_results['groups'].items():
            print(f"  {group}:")
            print(f"    Accuracy: {metrics['accuracy']:.2f}%")
            print(f"    EER: {metrics['eer']:.2f}%")
            print(f"    Threshold: {metrics['threshold']:.4f}")
            print(f"    Total Samples: {metrics['samples']}")
            print(f"    Bona fide Samples: {metrics['bona_samples']}")
            print(f"    Spoof Samples: {metrics['spoof_samples']}")
    
    print("\nAdditional metrics:")
    print(f"  F1 Score: {model_results['f1']:.4f}")
    print(f"  Recall: {model_results['recall']:.4f}")
    print(f"  Precision: {model_results['precision']:.4f}")
    
    print(f"{'-'*70}")

def print_eer_csv(model_results: Dict, model_name: str):
    if 'groups' not in model_results:
        print("No group EERs to print.")
        return

    # Collect attack types and EERs
    attack_types = sorted([k for k in model_results['groups'].keys() if k != '-'])
    eers = [model_results['groups'][atk]['eer'] for atk in attack_types]

    # Calculate pooled EER (mean of all attack type EERs)
    pooled_eer = np.mean(eers) if eers else 0.0

    # Print header
    print(','.join(attack_types + ['Pooled EER']))
    # Print values
    print(','.join([f"{eer:.2f}" for eer in eers] + [f"{pooled_eer:.2f}"]))

def main() -> None:
    """Main function to run the evaluation pipeline."""
    try:
        print("Loading metadata...")
        metadata_df = load_metadata()
        print(f"Loaded metadata with shape: {metadata_df.shape}")
        print(f"Columns: {metadata_df.columns.tolist()}")
        
        prediction_files = [PREDICTION_FILE]
        prediction_files = sorted(prediction_files)
        
        all_results = {}
        
        for score_file in prediction_files:
            model_name = Path(score_file).name
            print(f"\nProcessing {model_name}...")
            
            results_df = process_prediction_file(score_file, metadata_df)
            print(f"Processed results shape: {results_df.shape}")
            print(f"Required columns present: {all(col in results_df.columns for col in ['label', 'pred', 'score'])}")
            
            metrics = MetricsCalculator.calculate_metrics(results_df, group_column='attack_type')
            all_results[model_name] = metrics
        
        for model_name, metrics in all_results.items():
            print_results(metrics, model_name)
            print_eer_csv(metrics, model_name)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 