import os
import sys
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from pathlib import Path

# Add the parent directory to the path to import eval_metrics_DF
sys.path.append('/nvme1/hungdx/Lightning-hydra/notebooks/TTA')
import eval_metrics_DF as em

class GroupEERCalculator:
    @staticmethod
    def calculate_eer_by_group(df: pd.DataFrame, group_column: str) -> Dict[str, Union[float, Dict[str, Union[float, int]]]]:
        """Calculate EER metrics grouped by attack_model or attack_type.
        
        Args:
            df: DataFrame containing predictions, scores, labels, and group columns
            group_column: Column name to group by ('attack_model' or 'attack_type')
            
        Returns:
            Dictionary containing calculated EER metrics for each group
        """
        results = {}
        
        # Calculate overall EER first
        bona_scores = df[df['label'] == 'bonafide']['score'].values
        spoof_scores = df[df['label'] == 'spoof']['score'].values
        
        if len(bona_scores) > 0 and len(spoof_scores) > 0:
            overall_eer, overall_threshold = em.compute_eer(bona_scores, spoof_scores)
            results['overall'] = {
                'eer': overall_eer * 100,
                'threshold': overall_threshold,
                'bona_samples': len(bona_scores),
                'spoof_samples': len(spoof_scores),
                'total_samples': len(df)
            }
        
        # Calculate EER for each group
        group_metrics = {}
        
        # Get all bonafide samples for comparison
        all_bona_scores = df[df['label'] == 'bonafide']['score'].values
        
        # Process each group
        for group_name, group_df in df.groupby(group_column):
            if group_name in ['none', '-', 'bonafide']:  # Handle bonafide samples
                # For bonafide group, use all bonafide vs all spoof
                group_bona = all_bona_scores
                group_spoof = df[df['label'] == 'spoof']['score'].values
            else:  # For specific attack types/models
                # Get spoof samples for this specific attack
                group_spoof = group_df[group_df['label'] == 'spoof']['score'].values
                
                # Use all bonafide samples for comparison
                group_bona = all_bona_scores
                
                # Alternative: Use equal number of random bonafide samples
                # if len(group_spoof) > 0:
                #     n_spoof = len(group_spoof)
                #     if n_spoof <= len(all_bona_scores):
                #         group_bona = np.random.choice(all_bona_scores, size=n_spoof, replace=False)
                #     else:
                #         group_bona = all_bona_scores
            
            # Calculate EER for this group
            if len(group_bona) > 0 and len(group_spoof) > 0:
                group_eer, group_threshold = em.compute_eer(group_bona, group_spoof)
                
                # Calculate accuracy for this group
                group_df_copy = group_df.copy()
                group_df_copy['pred'] = group_df_copy.apply(
                    lambda x: 'bonafide' if x['score'] > group_threshold else 'spoof', axis=1
                )
                
                group_accuracy = accuracy_score(group_df_copy['label'], group_df_copy['pred']) * 100
                
                group_metrics[group_name] = {
                    'eer': group_eer * 100,
                    'threshold': group_threshold,
                    'accuracy': group_accuracy,
                    'bona_samples': len(group_bona),
                    'spoof_samples': len(group_spoof),
                    'total_samples': len(group_df),
                    'spoof_in_group': len(group_df[group_df['label'] == 'spoof']),
                    'bona_in_group': len(group_df[group_df['label'] == 'bonafide'])
                }
            else:
                print(f"Warning: Insufficient data for group '{group_name}' (bona: {len(group_bona)}, spoof: {len(group_spoof)})")
        
        results['groups'] = group_metrics
        return results
    
    @staticmethod
    def print_results(results: Dict, group_column: str) -> None:
        """Print EER results in a formatted way."""
        print(f"\n{'='*80}")
        print(f"EER Analysis by {group_column.upper()}")
        print(f"{'='*80}")
        
        # Print overall results
        if 'overall' in results:
            overall = results['overall']
            print(f"\nOVERALL RESULTS:")
            print(f"  EER: {overall['eer']:.2f}%")
            print(f"  Threshold: {overall['threshold']:.6f}")
            print(f"  Total Samples: {overall['total_samples']}")
            print(f"  Bonafide Samples: {overall['bona_samples']}")
            print(f"  Spoof Samples: {overall['spoof_samples']}")
        
        # Print group results
        if 'groups' in results:
            print(f"\nRESULTS BY {group_column.upper()}:")
            print(f"{'Group':<20} {'EER (%)':<10} {'Accuracy (%)':<12} {'Threshold':<12} {'Samples':<10} {'Spoof':<8} {'Bona':<8}")
            print("-" * 90)
            
            for group_name, metrics in results['groups'].items():
                print(f"{group_name:<20} {metrics['eer']:<10.2f} {metrics['accuracy']:<12.2f} "
                      f"{metrics['threshold']:<12.6f} {metrics['total_samples']:<10} "
                      f"{metrics['spoof_in_group']:<8} {metrics['bona_in_group']:<8}")
    
    @staticmethod
    def print_eer_csv(results: Dict, group_column: str) -> None:
        """Print EER results in CSV format."""
        if 'groups' not in results:
            print("No group EERs to print.")
            return
        
        # Get all groups except 'none' for attacks
        attack_groups = sorted([k for k in results['groups'].keys() if k not in ['none', '-', 'bonafide']])
        
        if not attack_groups:
            print("No attack groups found.")
            return
        
        # Get EER values
        eers = [results['groups'][group]['eer'] for group in attack_groups]
        
        # Calculate pooled EER (mean of all attack EERs)
        pooled_eer = np.mean(eers) if eers else 0.0
        
        print(f"\nCSV Format - EER by {group_column}:")
        print(','.join(attack_groups + ['Pooled_EER']))
        print(','.join([f"{eer:.2f}" for eer in eers] + [f"{pooled_eer:.2f}"]))

def load_and_process_data(result_file: str, metadata_file: str) -> pd.DataFrame:
    """Load and merge result file with metadata."""
    # Load result file
    df = pd.read_csv(result_file, sep=' ', header=None)
    df.columns = ['utt', 'spoof', 'score']
    
    # Load metadata file
    metadata = pd.read_csv(metadata_file, sep=' ', header=None)
    metadata.columns = ['utt', 'label', 'attack_model', 'attack_type']
    
    # Merge dataframes
    merged_df = pd.merge(df, metadata, on='utt', how='left')
    
    print(f"Loaded {len(merged_df)} samples")
    print(f"Unique labels: {merged_df['label'].unique()}")
    print(f"Unique attack_models: {merged_df['attack_model'].unique()}")
    print(f"Unique attack_types: {merged_df['attack_type'].unique()}")
    
    return merged_df

def main():
    """Main function to demonstrate the EER calculation."""
    # Example file paths - update these to your actual file paths
    result_file = '/nvme1/hungdx/Lightning-hydra/logs/results/TTA_benchmark_test_det_show_lts/ToP_LA19/ADV_2025_cnsl_xlsr_vib_paper_ToP_LA19.txt'
    metadata_file = '/nvme1/hungdx/Lightning-hydra/data/TTA_benchmark_2025/ADV_2025/ADV/protocol.txt'
    
    try:
        # Load and process data
        print("Loading data...")
        df = load_and_process_data(result_file, metadata_file)
        
        # Display first few rows
        print("\nFirst 5 rows of merged data:")
        print(df.head())
        
        # Calculate EER by attack_model
        print("\nCalculating EER by attack_model...")
        calculator = GroupEERCalculator()
        results_by_model = calculator.calculate_eer_by_group(df, 'attack_model')
        calculator.print_results(results_by_model, 'attack_model')
        calculator.print_eer_csv(results_by_model, 'attack_model')
        
        # Calculate EER by attack_type
        print("\n\nCalculating EER by attack_type...")
        results_by_type = calculator.calculate_eer_by_group(df, 'attack_type')
        calculator.print_results(results_by_type, 'attack_type')
        calculator.print_eer_csv(results_by_type, 'attack_type')
        
        # Additional analysis: show distribution
        print(f"\n\nData Distribution:")
        print(f"Label distribution:")
        print(df['label'].value_counts())
        print(f"\nAttack model distribution:")
        print(df['attack_model'].value_counts())
        print(f"\nAttack type distribution:")
        print(df['attack_type'].value_counts())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 