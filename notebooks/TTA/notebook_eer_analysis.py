# Notebook-friendly EER Analysis Code
# Use this code directly in your Jupyter notebook

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import eval_metrics_DF as em

def calculate_group_eer(df, group_column='attack_type'):
    """
    Calculate EER for each group in the dataset.
    
    Args:
        df: DataFrame with columns ['utt', 'spoof', 'score', 'label', 'attack_model', 'attack_type']
        group_column: Column to group by ('attack_model' or 'attack_type')
    
    Returns:
        Dictionary with overall and group-specific EER results
    """
    results = {}
    
    # Overall EER calculation
    bona_scores = df[df['label'] == 'bonafide']['score'].values
    spoof_scores = df[df['label'] == 'spoof']['score'].values
    
    if len(bona_scores) > 0 and len(spoof_scores) > 0:
        overall_eer, overall_threshold = em.compute_eer(bona_scores, spoof_scores)
        results['overall'] = {
            'eer': overall_eer * 100,
            'threshold': overall_threshold,
            'samples': len(df)
        }
    
    # Group-specific EER calculation
    group_results = {}
    all_bona_scores = df[df['label'] == 'bonafide']['score'].values
    
    for group_name, group_df in df.groupby(group_column):
        if group_name in ['none', '-']:  # Skip bonafide group
            continue
            
        # Get spoof samples for this attack
        group_spoof_scores = group_df[group_df['label'] == 'spoof']['score'].values
        
        if len(group_spoof_scores) > 0 and len(all_bona_scores) > 0:
            # Calculate EER for this specific attack vs all bonafide
            group_eer, group_threshold = em.compute_eer(all_bona_scores, group_spoof_scores)
            
            # Calculate accuracy using the threshold
            group_df_copy = group_df.copy()
            group_df_copy['pred'] = group_df_copy.apply(
                lambda x: 'bonafide' if x['score'] > group_threshold else 'spoof', axis=1
            )
            group_accuracy = accuracy_score(group_df_copy['label'], group_df_copy['pred']) * 100
            
            group_results[group_name] = {
                'eer': group_eer * 100,
                'threshold': group_threshold,
                'accuracy': group_accuracy,
                'samples': len(group_df),
                'spoof_samples': len(group_spoof_scores),
                'bona_samples': len(all_bona_scores)
            }
    
    results['groups'] = group_results
    return results

def print_eer_summary(results, group_column):
    """Print formatted EER results."""
    print(f"\n{'='*60}")
    print(f"EER Analysis by {group_column}")
    print(f"{'='*60}")
    
    # Overall results
    if 'overall' in results:
        print(f"Overall EER: {results['overall']['eer']:.2f}%")
        print(f"Overall Threshold: {results['overall']['threshold']:.6f}")
        print(f"Total Samples: {results['overall']['samples']}")
    
    # Group results
    if 'groups' in results and results['groups']:
        print(f"\nGroup-specific results:")
        print(f"{'Group':<25} {'EER (%)':<10} {'Accuracy (%)':<12} {'Samples':<10}")
        print("-" * 65)
        
        for group, metrics in results['groups'].items():
            print(f"{group:<25} {metrics['eer']:<10.2f} {metrics['accuracy']:<12.2f} {metrics['samples']:<10}")
        
        # Calculate pooled EER (average of all attack EERs)
        attack_eers = [metrics['eer'] for metrics in results['groups'].values()]
        pooled_eer = np.mean(attack_eers)
        print(f"\nPooled EER (average): {pooled_eer:.2f}%")
        
        # CSV format for easy copying
        print(f"\nCSV format:")
        groups = list(results['groups'].keys())
        eers = [results['groups'][g]['eer'] for g in groups]
        print(','.join(groups + ['Pooled_EER']))
        print(','.join([f"{eer:.2f}" for eer in eers] + [f"{pooled_eer:.2f}"]))

# Example usage with your data:
"""
# Load your data (assuming you already have df loaded)
result_file = '/path/to/your/result.txt'
metadata_file = '/path/to/your/protocol.txt'

# Load result file
df = pd.read_csv(result_file, sep=' ', header=None)
df.columns = ['utt', 'spoof', 'score']

# Load metadata
metadata = pd.read_csv(metadata_file, sep=' ', header=None)
metadata.columns = ['utt', 'label', 'attack_model', 'attack_type']

# Merge
df = pd.merge(df, metadata, on='utt', how='left')

# Calculate EER by attack_type
results_by_type = calculate_group_eer(df, 'attack_type')
print_eer_summary(results_by_type, 'attack_type')

# Calculate EER by attack_model  
results_by_model = calculate_group_eer(df, 'attack_model')
print_eer_summary(results_by_model, 'attack_model')
""" 