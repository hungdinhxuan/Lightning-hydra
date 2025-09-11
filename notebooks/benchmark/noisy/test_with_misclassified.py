import os
import sys
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from pathlib import Path
import eval_metrics_DF as em

# Constants
METADATA_PATH = "/nvme1/hungdx/Lightning-hydra/data/noisy_env_benchmark/woongjae_dataset/protocol.txt"

META_CSV_PATH = "/nvme1/hungdx/Lightning-hydra/data/noisy_env_benchmark/protocol_hung.csv"

# old prediction file
PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/noisy_env_benchmark_interspeech2025/Conformer_MDT_DEC2024_correct/woongjae_dataset_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_infer_Conformer_MDT_DEC2024_correct.txt"
#PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/noisy_env_benchmark_interspeech2025/ConformerTCM_MDT_LoRA_exp_g1/woongjae_dataset_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g1.txt"
# PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/noisy_env_benchmark_interspeech2025/ConformerTCM_MDT_LoRA_exp_g2/woongjae_dataset_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g2.txt"
#PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/noisy_env_benchmark_interspeech2025/ConformerTCM_MDT_LoRA_exp_g3/woongjae_dataset_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g3.txt"
#PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/noisy_env_benchmark_interspeech2025/ConformerTCM_MDT_LoRA_exp_g5/woongjae_dataset_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g5.txt"
#PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/noisy_env_benchmark_interspeech2025/ConformerTCM_MDT_LoRA_exp_g6/woongjae_dataset_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g6.txt"


#### V2
#PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/noisy_env_benchmark_interspeech2025/ConformerTCM_MDT_LoRA_exp_g2_v2/woongjae_dataset_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g2_v2.txt"

#PREDICTION_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/noisy_env_benchmark_interspeech2025/ConformerTCM_MDT_LoRA_exp_g7_v2/woongjae_dataset_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g7_v2.txt"

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
        meta_csv = pd.read_csv(META_CSV_PATH, sep=",", header=None)
        meta_csv.columns = ['path', 'label', 'noisy_type']
        
        # Perform merge
        merged_metadata = metadata.merge(meta_csv, left_on='path', right_on='path', how='left')
        
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
        merged_df = pred_df.merge(metadata_df, left_on='path', right_on='path', how='left')
        
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

def print_accuracy_csv(model_results: Dict, model_name: str):
    """Print accuracy results in CSV format for each attack type."""
    if 'groups' not in model_results:
        print("No group accuracies to print.")
        return

    # Collect attack types and accuracies (excluding bonafide samples marked with '-'))
    attack_types = sorted([k for k in model_results['groups'].keys() if k != '-'])
    accuracies = [model_results['groups'][atk]['accuracy'] for atk in attack_types]

    # Calculate pooled accuracy (mean of all attack type accuracies)
    pooled_accuracy = np.mean(accuracies) if accuracies else 0.0

    # Print header
    print(','.join(attack_types + ['Pooled Accuracy']))
    # Print values
    print(','.join([f"{acc:.2f}" for acc in accuracies] + [f"{pooled_accuracy:.2f}"]))

def save_misclassified_samples(results_df: pd.DataFrame, output_file: str, model_name: str) -> None:
    """Save misclassified samples to a text file.
    
    Args:
        results_df: DataFrame containing predictions and ground truth
        output_file: Path to the output file
        model_name: Name of the model for the header
    """
    try:
        # Find misclassified samples
        misclassified = results_df[results_df['label'] != results_df['pred']].copy()
        
        if len(misclassified) == 0:
            print(f"No misclassified samples found for {model_name}")
            return
        
        print(f"Found {len(misclassified)} misclassified samples for {model_name}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write misclassified samples to file
        with open(output_file, 'w') as f:
            # Write header
            f.write(f"# Misclassified samples for model: {model_name}\n")
            f.write(f"# Total misclassified: {len(misclassified)}\n")
            f.write(f"# Format: <file_name> <noise_type> <pred_label> <true_label>\n")
            f.write("# Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            # Write each misclassified sample
            for _, row in misclassified.iterrows():
                file_name = row['path']
                noise_type = row.get('noisy_type', 'unknown')  # Use 'noisy_type' column if available
                pred_label = row['pred']
                true_label = row['label']
                
                f.write(f"{file_name} {noise_type} {pred_label} {true_label}\n")
        
        print(f"Misclassified samples saved to: {output_file}")
        
        # Print summary by noise type
        if 'noisy_type' in misclassified.columns:
            print("\nMisclassified samples by noise type:")
            noise_type_counts = misclassified['noisy_type'].value_counts()
            for noise_type, count in noise_type_counts.items():
                print(f"  {noise_type}: {count}")
        
        # Print summary by prediction error type
        print("\nMisclassified samples by error type:")
        error_types = misclassified.apply(
            lambda x: f"{x['true_label']}->{x['pred']}", axis=1
        ).value_counts()
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")
            
    except Exception as e:
        print(f"Error saving misclassified samples: {str(e)}")

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
            
            metrics = MetricsCalculator.calculate_metrics(results_df, group_column='noisy_type')
            all_results[model_name] = metrics
            
            # Save misclassified samples
            output_file = f"misclassified_samples_{Path(score_file).stem}.txt"
            save_misclassified_samples(results_df, output_file, model_name)
        
        for model_name, metrics in all_results.items():
            print_results(metrics, model_name)
            print("\nEER CSV:")
            print_eer_csv(metrics, model_name)
            print("\nAccuracy CSV:")
            print_accuracy_csv(metrics, model_name)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 