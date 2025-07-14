import os
import sys
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from pathlib import Path
import eval_metrics_DF as em

METADATA_PATH = "/nvme1/hungdx/Lightning-hydra/data/shortcutASV/merged_protocol.txt"
META_CSV_PATH = "/nvme1/hungdx/Lightning-hydra/data/shortcutASV/merged_meta.csv"

# Option to calculate EER (set to False to save time)
CALCULATE_EER = False  # Set to False to skip EER calculations and save time

# Multiple prediction files for simultaneous evaluation
PREDICTION_FILES = [
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/Conformer_MDT_DEC2024_correct/merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_Conformer_MDT_DEC2024_correct.txt",
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/Conformer_MDT_LA19/merged_scores_huggingface_benchmark_xlsr_conformertcm_mdt_lora_infer_Conformer_MDT_LA19.txt",
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/AASIST_SSL_MDT_LA19/merged_scores_huggingface_benchmark_xlsr_aasist_mdt_paper_AASIST_SSL_MDT_LA19.txt",
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/ToP_April/merged_scores_cnsl_xlsr_vib_large_corpus_ToP_April.txt",
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/ConformerTCM_MDT_LoRA_exp_g1_june27/merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g1_june27.txt",
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/ConformerTCM_MDT_LoRA_exp_g2_june27/merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g2_june27.txt",
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/ConformerTCM_MDT_LoRA_exp_g3_june27/merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g3_june27.txt",
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/ConformerTCM_MDT_LoRA_exp_g4_june27/merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g4_june27.txt",
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/ConformerTCM_MDT_LoRA_exp_g1_june29/merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g1_june29.txt",
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/ConformerTCM_MDT_LoRA_exp_g5_june29/merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_g5_june29.txt",
    "/nvme1/hungdx/Lightning-hydra/logs/results/noisy_benchmark/ConformerTCM_MDT_LoRA_exp_train-with-all_june30/merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_ConformerTCM_MDT_LoRA_exp_train-with-all_june30.txt"
]

# Filter to only include files that exist
PREDICTION_FILES = [f for f in PREDICTION_FILES if os.path.exists(f)]


class MetricsCalculator:
    @staticmethod
    def calculate_metrics(df: pd.DataFrame, group_column: Optional[str] = None, calculate_eer: bool = True) -> Dict[str, Union[float, Dict[str, Union[float, int]]]]:
        """Calculate various metrics for the given DataFrame.
        
        Args:
            df: DataFrame containing predictions and ground truth
            group_column: Optional column name to group results by
            calculate_eer: Whether to calculate EER (can be time-consuming)
            
        Returns:
            Dictionary containing calculated metrics and sample counts
        """
        # Verify that we have the expected labels
        unique_labels = df['label'].unique()
        expected_labels = ['bonafide', 'spoof']
        print(f"Unique labels found: {sorted(unique_labels)}")
        
        if not all(label in unique_labels for label in expected_labels if len(df[df['label'] == label]) > 0):
            print(f"Warning: Expected labels {expected_labels}, but found {unique_labels}")
        
        results = {
            'overall': accuracy_score(df["label"], df["pred"]) * 100,
            'f1': f1_score(df['label'], df['pred'], pos_label='bonafide'),
            'recall': recall_score(df['label'], df['pred'], pos_label='bonafide'),
            'precision': precision_score(df['label'], df['pred'], pos_label='bonafide'),
            'total_samples': len(df)
        }
        
        # Calculate overall EER only if requested
        if calculate_eer:
            bona_cm = df[df['label'] == 'bonafide']['score'].values
            spoof_cm = df[df['label'] == 'spoof']['score'].values
            if len(bona_cm) > 0 and len(spoof_cm) > 0:
                eer_cm, _ = em.compute_eer(bona_cm, spoof_cm)
                results['overall_eer'] = eer_cm * 100
            else:
                print("Warning: Missing bonafide or spoof samples for EER calculation")
                results['overall_eer'] = None
        else:
            results['overall_eer'] = None
        
        if group_column and group_column in df.columns:
            group_metrics = {}
            
            print(f"Available groups: {sorted(df[group_column].unique())}")
            
            # Process each attack type/group
            for group, group_df in df.groupby(group_column):
                # Calculate separate accuracies for bonafide and spoof within this modification
                
                # Bonafide accuracy within this modification
                bonafide_subset = group_df[group_df['label'] == 'bonafide']
                bonafide_correct = len(bonafide_subset[bonafide_subset['pred'] == 'bonafide'])
                bonafide_total = len(bonafide_subset)
                bonafide_accuracy = (bonafide_correct / bonafide_total * 100) if bonafide_total > 0 else 0.0
                
                # Spoof accuracy within this modification
                spoof_subset = group_df[group_df['label'] == 'spoof']
                spoof_correct = len(spoof_subset[spoof_subset['pred'] == 'spoof'])
                spoof_total = len(spoof_subset)
                spoof_accuracy = (spoof_correct / spoof_total * 100) if spoof_total > 0 else 0.0
                
                # Overall accuracy for this modification (combining bonafide and spoof)
                overall_accuracy = accuracy_score(group_df["label"], group_df["pred"]) * 100
                
                # Get score arrays for EER calculation
                group_bona_scores = group_df[group_df['label'] == 'bonafide']['score'].values
                group_spoof_scores = group_df[group_df['label'] == 'spoof']['score'].values
                
                group_result = {
                    'bonafide_accuracy': bonafide_accuracy,
                    'spoof_accuracy': spoof_accuracy,
                    'overall_accuracy': overall_accuracy,
                    'bonafide_correct': bonafide_correct,
                    'bonafide_total': bonafide_total,
                    'spoof_correct': spoof_correct,
                    'spoof_total': spoof_total,
                    'total_samples': len(group_df)
                }
                
                if calculate_eer:
                    # Calculate EER for this attack type using bonafide vs spoof within the group
                    if len(group_bona_scores) > 0 and len(group_spoof_scores) > 0:
                        group_eer, group_threshold = em.compute_eer(group_bona_scores, group_spoof_scores)
                        group_result.update({
                            'eer': group_eer * 100,
                            'threshold': group_threshold,
                        })
                    else:
                        group_result.update({
                            'eer': None,
                            'threshold': None,
                        })
                
                group_metrics[group] = group_result
                
            results['groups'] = group_metrics
        
        return results

def extract_attack_type_from_path(path: str) -> str:
    """Extract attack type from the file path."""
    # Extract directory structure to get attack type
    path_parts = path.split('/')
    
    # Look for the attack type in the path
    # Common patterns: asv19/wav/ATTACK_TYPE/bona-fide/... or asv19/wav/ATTACK_TYPE/spoof/...
    for i, part in enumerate(path_parts):
        if part == 'wav' and i + 1 < len(path_parts):
            attack_type = path_parts[i + 1]
            return attack_type
    
    # Fallback: use unknown
    return 'unknown'

def load_metadata() -> pd.DataFrame:
    """Load and process metadata files with proper column handling.
    
    Returns:
        DataFrame containing merged metadata
    """
    try:
        # Load protocol file
        metadata = pd.read_csv(METADATA_PATH, sep=" ", header=None)
        metadata.columns = ["path", "subset", "label"]
        
        print(f"Protocol file loaded: {len(metadata)} entries")
        print(f"Label distribution in protocol file:")
        print(metadata['label'].value_counts())
        
        # Extract attack type from path
        metadata['attack_type'] = metadata['path'].apply(extract_attack_type_from_path)
        
        print(f"Attack type distribution:")
        print(metadata['attack_type'].value_counts())
        
        # Show cross-tabulation of attack_type vs label
        print(f"\nAttack type vs Label cross-tabulation:")
        print(pd.crosstab(metadata['attack_type'], metadata['label'], margins=True))
        
        # Try to load CSV metadata if it exists and is not too large
        try:
            # Check file size first
            csv_size = os.path.getsize(META_CSV_PATH)
            print(f"CSV metadata file size: {csv_size / (1024*1024):.1f} MB")
            
            if csv_size < 50 * 1024 * 1024:  # Less than 50MB
                meta_csv = pd.read_csv(META_CSV_PATH, sep="|")
                print(f"CSV metadata loaded: {len(meta_csv)} entries")
                print(f"CSV columns: {meta_csv.columns.tolist()}")
                
                # rename columns from file_path to path_audio
                if 'file_path' in meta_csv.columns:
                    meta_csv = meta_csv.rename(columns={'file_path': 'path_audio'})
                elif 'path' in meta_csv.columns:
                    meta_csv = meta_csv.rename(columns={'path': 'path_audio'})
                
                # Handle column conflicts before merge by dropping conflicting columns from meta_csv
                conflicting_cols = ['subset', 'label', 'path', 'attack_type']
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
                        print(f"Using column '{audio_cols[0]}' as path_audio")
                    else:
                        print("Warning: No suitable audio path column found in CSV metadata")
                        return metadata  # Return without merge
                
                # Perform merge
                print("Merging protocol and CSV metadata...")
                merged_metadata = metadata.merge(meta_csv, left_on='path', right_on='path_audio', how='left')
                print(f"Merged metadata: {len(merged_metadata)} entries")
                
                return merged_metadata
            else:
                print("CSV file too large, using only protocol file")
                return metadata
                
        except Exception as e:
            print(f"Warning: Could not load CSV metadata: {str(e)}")
            print("Using only protocol file")
            return metadata
        
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
        
        print(f"Prediction file loaded: {len(pred_df)} entries")
        
        # Perform merge - try both path columns
        merge_column = 'path_audio' if 'path_audio' in metadata_df.columns else 'path'
        merged_df = pred_df.merge(metadata_df, left_on='path', right_on=merge_column, how='left')
        
        print(f"Merged prediction+metadata: {len(merged_df)} entries")
        print(f"Available columns: {merged_df.columns.tolist()}")
        
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
        
        # Ensure we have attack_type column
        if 'attack_type' not in merged_df.columns:
            print("No attack_type column found, extracting from path...")
            merged_df['attack_type'] = merged_df['path'].apply(extract_attack_type_from_path)
        
        # Create predictions
        merged_df['pred'] = merged_df.apply(
            lambda x: 'bonafide' if x['spoof'] < x['score'] else 'spoof', axis=1)
        
        # Verify label consistency
        print(f"Final label distribution: {merged_df['label'].value_counts().to_dict()}")
        print(f"Attack type distribution: {merged_df['attack_type'].value_counts().to_dict()}")
        
        # Show attack type vs label cross-tabulation for verification
        print(f"\nAttack type vs Label cross-tabulation:")
        print(pd.crosstab(merged_df['attack_type'], merged_df['label'], margins=True))
        
        # Remove rows with missing labels
        before_filter = len(merged_df)
        merged_df = merged_df.dropna(subset=['label'])
        after_filter = len(merged_df)
        if before_filter != after_filter:
            print(f"Removed {before_filter - after_filter} rows with missing labels")
        
        return merged_df
        
    except Exception as e:
        raise RuntimeError(f"Failed to process prediction file {score_file}: {str(e)}")

def print_results(model_results: Dict, model_name: str, original_results: Optional[Dict] = None) -> None:
    """Print evaluation results in a formatted way matching your reference code.
    
    Args:
        model_results: Dictionary containing model metrics
        model_name: Name of the model
        original_results: Optional dictionary containing original model results for comparison
    """
    print(f"\n{'-'*70}")
    print(f"Model: {model_name}")
    
    print(f"\nTotal Samples: {model_results['total_samples']}")
    print(f"Overall Accuracy: {model_results['overall']:.2f}%")
    
    if model_results.get('overall_eer') is not None:
        print(f"Overall EER: {model_results['overall_eer']:.2f}%")
    else:
        print("Overall EER: Not calculated (EER calculation disabled)")
    
    if 'groups' in model_results:
        print("\nBONAFIDE Accuracy by Attack Type:")
        for group, metrics in sorted(model_results['groups'].items()):
            print(f"{group},{metrics['bonafide_correct']},{metrics['bonafide_accuracy']:.2f}")
        
        print("\nSPOOF Accuracy by Attack Type:")
        for group, metrics in sorted(model_results['groups'].items()):
            print(f"{group},{metrics['spoof_correct']},{metrics['spoof_accuracy']:.2f}")
        
        print("\nDetailed Metrics by Attack Type:")
        for group, metrics in sorted(model_results['groups'].items()):
            print(f"  {group}:")
            print(f"    Bonafide: {metrics['bonafide_correct']}/{metrics['bonafide_total']} ({metrics['bonafide_accuracy']:.2f}%)")
            print(f"    Spoof: {metrics['spoof_correct']}/{metrics['spoof_total']} ({metrics['spoof_accuracy']:.2f}%)")
            print(f"    Overall: {metrics['overall_accuracy']:.2f}%")
            if metrics.get('eer') is not None:
                print(f"    EER: {metrics['eer']:.2f}%")
                print(f"    Threshold: {metrics['threshold']:.4f}")
            else:
                print(f"    EER: Not calculated")
            print(f"    Total Samples: {metrics['total_samples']}")
    
    print("\nAdditional overall metrics:")
    print(f"  F1 Score: {model_results['f1']:.4f}")
    print(f"  Recall: {model_results['recall']:.4f}")
    print(f"  Precision: {model_results['precision']:.4f}")
    
    print(f"{'-'*70}")

def print_eer_csv(model_results: Dict, model_name: str):
    if 'groups' not in model_results:
        print("No group EERs to print.")
        return

    # Check if EER was calculated
    first_group = next(iter(model_results['groups'].values()))
    if first_group.get('eer') is None:
        print("EER values not calculated (EER calculation was disabled).")
        return

    # Collect attack types and EERs
    attack_types = sorted(model_results['groups'].keys())
    eers = []
    for atk in attack_types:
        if model_results['groups'][atk].get('eer') is not None:
            eers.append(model_results['groups'][atk]['eer'])

    if not eers:
        print("No valid EER values found.")
        return

    # Calculate pooled EER (mean of all attack type EERs)
    pooled_eer = np.mean(eers)

    # Print header
    print(','.join(attack_types + ['Pooled EER']))
    # Print values
    print(','.join([f"{eer:.2f}" for eer in eers] + [f"{pooled_eer:.2f}"]))

def print_accuracy_csv(model_results: Dict, model_name: str):
    """Print accuracy results in CSV format matching your reference code."""
    if 'groups' not in model_results:
        print("No group accuracies to print.")
        return

    # Collect all attack types
    attack_types = sorted(model_results['groups'].keys())
    
    # Print bonafide accuracies
    print("=== Bonafide Accuracy CSV ===")
    bonafide_accs = [model_results['groups'][atk]['bonafide_accuracy'] for atk in attack_types]
    pooled_bonafide = np.mean(bonafide_accs) if bonafide_accs else 0.0
    print(','.join(attack_types + ['Pooled Bonafide']))
    print(','.join([f"{acc:.2f}" for acc in bonafide_accs] + [f"{pooled_bonafide:.2f}"]))
    
    # Print spoof accuracies
    print("\n=== Spoof Accuracy CSV ===")
    spoof_accs = [model_results['groups'][atk]['spoof_accuracy'] for atk in attack_types]
    pooled_spoof = np.mean(spoof_accs) if spoof_accs else 0.0
    print(','.join(attack_types + ['Pooled Spoof']))
    print(','.join([f"{acc:.2f}" for acc in spoof_accs] + [f"{pooled_spoof:.2f}"]))
    
    # Print overall accuracies per attack type
    print("\n=== Overall Attack Type Accuracy CSV ===")
    overall_accs = [model_results['groups'][atk]['overall_accuracy'] for atk in attack_types]
    pooled_overall = np.mean(overall_accs) if overall_accs else 0.0
    print(','.join(attack_types + ['Pooled Overall']))
    print(','.join([f"{acc:.2f}" for acc in overall_accs] + [f"{pooled_overall:.2f}"]))

def print_comparative_csv(all_results: Dict[str, Dict]):
    """Print comparative CSV results across all models."""
    if not all_results:
        return
    
    # Get all unique attack types from all models
    all_attack_types = set()
    for results in all_results.values():
        if 'groups' in results:
            all_attack_types.update(results['groups'].keys())
    
    attack_types = sorted(all_attack_types)
    
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS ACROSS ALL MODELS")
    print("="*80)
    
    # Check if EER was calculated
    eer_calculated = False
    for results in all_results.values():
        if 'groups' in results:
            first_group = next(iter(results['groups'].values()))
            if first_group.get('eer') is not None:
                eer_calculated = True
                break
    
    # EER Comparison (only if calculated)
    if eer_calculated:
        print("\n=== EER Comparison CSV ===")
        if attack_types:
            header = ['Model'] + attack_types + ['Pooled EER']
            print(','.join(header))
            
            for model_name, results in all_results.items():
                if 'groups' not in results:
                    continue
                
                model_short = model_name[-50:]  # Last 50 characters of the model name
                eers = []
                for atk in attack_types:
                    if atk in results['groups'] and results['groups'][atk].get('eer') is not None:
                        eers.append(f"{results['groups'][atk]['eer']:.2f}")
                    else:
                        eers.append("N/A")
                
                # Calculate pooled EER
                valid_eers = [results['groups'][atk]['eer'] for atk in attack_types 
                             if atk in results['groups'] and results['groups'][atk].get('eer') is not None]
                pooled_eer = np.mean(valid_eers) if valid_eers else 0.0
                
                row = [model_short] + eers + [f"{pooled_eer:.2f}"]
                print(','.join(row))
    else:
        print("\n=== EER Comparison CSV ===")
        print("EER calculations were disabled - no EER comparison available")
    
    # Bonafide Accuracy Comparison
    print("\n=== Bonafide Accuracy Comparison CSV ===")
    if attack_types:
        header = ['Model'] + attack_types + ['Pooled Bonafide']
        print(','.join(header))
        
        for model_name, results in all_results.items():
            if 'groups' not in results:
                continue
            
            model_short = model_name[-50:]
            row = [model_short]
            
            bonafide_accs = []
            for atk in attack_types:
                if atk in results['groups']:
                    acc = results['groups'][atk]['bonafide_accuracy']
                    row.append(f"{acc:.2f}")
                    bonafide_accs.append(acc)
                else:
                    row.append("N/A")
            
            pooled_bonafide = np.mean(bonafide_accs) if bonafide_accs else 0.0
            row.append(f"{pooled_bonafide:.2f}")
            
            print(','.join(row))
    
    # Spoof Accuracy Comparison
    print("\n=== Spoof Accuracy Comparison CSV ===")
    if attack_types:
        header = ['Model'] + attack_types + ['Pooled Spoof']
        print(','.join(header))
        
        for model_name, results in all_results.items():
            if 'groups' not in results:
                continue
            
            model_short = model_name[-50:]
            row = [model_short]
            
            spoof_accs = []
            for atk in attack_types:
                if atk in results['groups']:
                    acc = results['groups'][atk]['spoof_accuracy']
                    row.append(f"{acc:.2f}")
                    spoof_accs.append(acc)
                else:
                    row.append("N/A")
            
            pooled_spoof = np.mean(spoof_accs) if spoof_accs else 0.0
            row.append(f"{pooled_spoof:.2f}")
            
            print(','.join(row))

def main() -> None:
    """Main function to run the evaluation pipeline."""
    try:
        print(f"EER Calculation: {'Enabled' if CALCULATE_EER else 'Disabled (for faster processing)'}")
        print("Loading metadata...")
        metadata_df = load_metadata()
        print(f"Loaded metadata with shape: {metadata_df.shape}")
        print(f"Columns: {metadata_df.columns.tolist()}")
        
        print(f"\nFound {len(PREDICTION_FILES)} prediction files to process")
        
        all_results = {}
        
        for i, score_file in enumerate(PREDICTION_FILES, 1):
            model_name = Path(score_file).name
            print(f"\n[{i}/{len(PREDICTION_FILES)}] Processing {model_name}...")
            
            if not os.path.exists(score_file):
                print(f"Warning: File {score_file} does not exist, skipping...")
                continue
            
            try:
                results_df = process_prediction_file(score_file, metadata_df)
                print(f"Processed results shape: {results_df.shape}")
                print(f"Required columns present: {all(col in results_df.columns for col in ['label', 'pred', 'score'])}")
                
                metrics = MetricsCalculator.calculate_metrics(results_df, group_column='attack_type', calculate_eer=CALCULATE_EER)
                all_results[model_name] = metrics
                
            except Exception as e:
                print(f"Error processing {model_name}: {str(e)}")
                continue
        
        print(f"\nSuccessfully processed {len(all_results)} models")
        
        # Print individual results
        for model_name, metrics in all_results.items():
            print_results(metrics, model_name)
            if CALCULATE_EER:
                print("\nEER CSV:")
                print_eer_csv(metrics, model_name)
            print("\nAccuracy CSV:")
            print_accuracy_csv(metrics, model_name)
            print("\n" + "="*70)
        
        # Print comparative results
        if len(all_results) > 1:
            print_comparative_csv(all_results)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()