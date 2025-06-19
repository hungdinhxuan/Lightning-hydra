import os
import sys
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from pathlib import Path
import eval_metrics_DF as em

# Constants
METADATA_PATH = "/nvme1/hungdx/Lightning-hydra/data/cnsl_benchmark/AIHUB_FreeCommunication_may/protocol.txt"
META_CSV_PATH = "/nvme1/hungdx/Lightning-hydra/data/cnsl_benchmark/AIHUB_FreeCommunication_may_meta.csv"
BASE_DIR = "/nvme1/hungdx/Lightning-hydra/logs/eval/cnsl/largecorpus"

# old prediction file
PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/cnsl_benchmark/ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs/AIHUB_FreeCommunication_may_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs.txt"
# new prediction file
PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/cnsl_benchmark/MDT_baseline_lora_exp1_june2/AIHUB_FreeCommunication_may_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_MDT_baseline_lora_exp1_june2.txt"

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
                group_eer, _ = em.compute_eer(group_bona, group_spoof)
                
                group_metrics[group] = {
                    'accuracy': accuracy_score(group_df["label"], group_df["pred"]) * 100,
                    'eer': group_eer * 100,
                    'samples': len(group_df),
                    'bona_samples': len(group_bona),
                    'spoof_samples': len(group_spoof)
                }
            results['groups'] = group_metrics
        
        return results

def load_metadata() -> pd.DataFrame:
    """Load and process metadata files.
    
    Returns:
        DataFrame containing merged metadata
    """
    try:
        metadata = pd.read_csv(METADATA_PATH, sep=" ", header=None)
        metadata.columns = ["path", "subset", "label"]
        
        meta_csv = pd.read_csv(META_CSV_PATH)
        
        metadata = metadata.merge(meta_csv, left_on='path', right_on='path_audio', how='left')
        
        return metadata
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
        
        merged_df = pred_df.merge(metadata_df, left_on='path', right_on='path_audio', how='left')
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
            print(f"    Total Samples: {metrics['samples']}")
            print(f"    Bona fide Samples: {metrics['bona_samples']}")
            print(f"    Spoof Samples: {metrics['spoof_samples']}")
    
    print("\nAdditional metrics:")
    print(f"  F1 Score: {model_results['f1']:.4f}")
    print(f"  Recall: {model_results['recall']:.4f}")
    print(f"  Precision: {model_results['precision']:.4f}")
    
    print(f"{'-'*70}")

def main() -> None:
    """Main function to run the evaluation pipeline."""
    try:
        print("Loading metadata...")
        metadata_df = load_metadata()
        
        prediction_files = [PREDICTION_FILE]
        prediction_files = sorted(prediction_files)
        
        all_results = {}
        
        for score_file in prediction_files:
            model_name = Path(score_file).name
            print(f"\nProcessing {model_name}...")
            
            results_df = process_prediction_file(score_file, metadata_df)
            metrics = MetricsCalculator.calculate_metrics(results_df, group_column='attack_type')
            all_results[model_name] = metrics
        
        for model_name, metrics in all_results.items():
            print_results(metrics, model_name)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 