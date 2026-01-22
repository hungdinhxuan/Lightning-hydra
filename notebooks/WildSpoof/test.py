import os
# Fix OpenBLAS thread limit issue
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

embeddings_path = "/home/hungdx/code/Lightning-hydra/logs/wildspoof_eval_embs/XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_and_vocoded_Nov16"

protocol_path = "/home/hungdx/code/Lightning-hydra/data/WildSpoof_Final_Eval/Final_eval/protocol.txt"

protocol_df = pd.read_csv(protocol_path, sep=" ", header=None, names=["utt_id", "subset", "label"])

# If label is spoof then set it to unknown
protocol_df.loc[protocol_df['label'] == 'spoof', 'label'] = 'unknown'
protocol_df['utt_id'] = protocol_df['utt_id'].str.replace('data_v2.0/', '')
protocol_df['utt_id'] = protocol_df['utt_id'].str.replace('.flac', '')
protocol_df

# OPTIMIZED: Load embeddings and match with protocol
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
from pathlib import Path

print("🚀 Starting optimized embedding loading...")
start_time = time.time()

# Get all .npy files efficiently using pathlib
embedding_files = list(Path(embeddings_path).glob('*.npy'))
print(f"Found {len(embedding_files)} embedding files")

# Pre-allocate lists for better memory efficiency
utt_ids = []
embeddings_list = []

# Load embeddings with progress indication
print("Loading embeddings...")
for i, file_path in enumerate(embedding_files):
    if i % 10000 == 0:  # Progress indicator
        print(f"  Processed {i}/{len(embedding_files)} files...")
    
    utt_id = file_path.stem  # More efficient than string replace
    utt_ids.append(utt_id)
    
    # Load embedding
    emb = np.load(file_path)
    embeddings_list.append(emb)

# Convert to numpy array efficiently
embeddings = np.array(embeddings_list)
load_time = time.time()
print(f"✅ Loaded {len(embeddings)} embeddings with shape: {embeddings.shape}")
print(f"   Loading time: {load_time - start_time:.2f} seconds")

# Create dataframe with embeddings and utterance IDs
emb_df = pd.DataFrame({'utt_id': utt_ids})

# Merge with protocol to get labels
data_df = emb_df.merge(protocol_df, on='utt_id', how='inner')
print(f"Matched {len(data_df)} utterances with protocol")
print(f"Label distribution:\n{data_df['label'].value_counts()}")

data_df.head()

# OPTIMIZED: Filter embeddings to match the protocol data using vectorized operations
start_time = time.time()

print("Optimizing data matching with vectorized operations...")

# Create embedding dataframe for efficient merging
emb_df_full = pd.DataFrame({
    'utt_id': utt_ids,
    'emb_idx': range(len(utt_ids))
})

# Use efficient pandas merge instead of nested loops (O(n log n) vs O(n²))
merged_df = emb_df_full.merge(data_df[['utt_id', 'label']], on='utt_id', how='inner')

# Extract matched indices and labels in vectorized way
matched_indices = merged_df['emb_idx'].values
matched_labels = merged_df['label'].values

# Get the matched embeddings using advanced indexing (very fast)
matched_embeddings = embeddings[matched_indices]

print(f"Final dataset: {len(matched_embeddings)} samples")
print(f"Embedding dimension: {matched_embeddings.shape[1]}")
print(f"Label counts: {np.unique(matched_labels, return_counts=True)}")

# Standardize embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(matched_embeddings)

end_time = time.time()
print(f"Embeddings loaded and preprocessed successfully!")
print(f"⚡ Optimization complete: processed {len(matched_embeddings)} samples in {end_time - start_time:.2f} seconds")
# T-SNE Visualization
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
np.random.seed(42)

def perform_tsne(embeddings, n_components=2, perplexity=30, max_iter=1000, random_state=42):
    """
    Perform t-SNE dimensionality reduction on embeddings.
    
    Args:
        embeddings: Array of embeddings to reduce
        n_components: Number of dimensions for t-SNE output (default: 2)
        perplexity: Perplexity parameter for t-SNE (default: 30)
        max_iter: Maximum number of iterations (default: 1000)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        tsne_results: 2D array of t-SNE coordinates
    """
    print(f"Original embedding shape: {embeddings.shape}")
    
    # Reshape if 3D
    if len(embeddings.shape) == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        print(f"Reshaped embedding shape: {embeddings.shape}")
    
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Adjust perplexity if needed (must be less than n_samples)
    n_samples = embeddings_scaled.shape[0]
    adjusted_perplexity = min(perplexity, n_samples - 1)
    if adjusted_perplexity < perplexity:
        print(f"Adjusted perplexity from {perplexity} to {adjusted_perplexity} (n_samples={n_samples})")
    
    # Apply t-SNE
    print("Performing t-SNE...")
    # Limit n_jobs to avoid OpenBLAS thread limit issues
    tsne = TSNE(
        n_components=n_components,
        perplexity=adjusted_perplexity,
        max_iter=max_iter,
        random_state=random_state,
        learning_rate='auto',
        n_jobs=4  # Limited to 4 threads to avoid OpenBLAS issues
    )
    tsne_results = tsne.fit_transform(embeddings_scaled)
    print(f"t-SNE results shape: {tsne_results.shape}")
    
    return tsne_results

def plot_tsne(tsne_results, labels, title="t-SNE Visualization of Embeddings", figsize=(12, 8), save_path=None):
    """
    Plot t-SNE results with colored labels.
    
    Args:
        tsne_results: 2D array of t-SNE coordinates
        labels: Array of labels for coloring
        title: Plot title
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    print(f"t-SNE results shape: {tsne_results.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    plt.figure(figsize=figsize)
    
    # Color palette
    unique_labels = np.unique(labels)
    colors = {
        'bonafide': '#2E8B57',  # Sea green
        'unknown': '#DC143C',   # Crimson
        'spoof': '#FF6347'      # Tomato
    }
    
    # Create scatter plot
    for label in unique_labels:
        mask = labels == label
        color = colors.get(label, None)
        if color is None:
            # Use seaborn palette for other labels
            color = sns.color_palette("husl", n_colors=len(unique_labels))[list(unique_labels).index(label)]
        
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                   c=[color], label=label.capitalize(), alpha=0.6, s=50)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return plt

print("✅ T-SNE functions loaded!")
# Perform t-SNE on the embeddings
print("🎨 Performing t-SNE dimensionality reduction...")
print("="*60)

# Use a subset for faster computation if dataset is very large
# For full dataset, this may take a while
USE_SUBSET = False  # Set to True to use a subset for faster testing
SUBSET_SIZE = 10000  # Number of samples to use if USE_SUBSET is True

if USE_SUBSET and len(embeddings_scaled) > SUBSET_SIZE:
    print(f"Using subset of {SUBSET_SIZE} samples for faster computation...")
    indices = np.random.choice(len(embeddings_scaled), SUBSET_SIZE, replace=False)
    embeddings_subset = embeddings_scaled[indices]
    labels_subset = matched_labels[indices]
else:
    embeddings_subset = embeddings_scaled
    labels_subset = matched_labels
    print(f"Using full dataset: {len(embeddings_subset)} samples")

# Perform t-SNE
tsne_results = perform_tsne(
    embeddings_subset,
    n_components=2,
    perplexity=30,
    max_iter=1000,
    random_state=42
)

print(f"✅ t-SNE completed!")
print(f"t-SNE embedding shape: {tsne_results.shape}")
print(f"Label distribution in subset:")
unique_labels, counts = np.unique(labels_subset, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"  {label}: {count} ({count/len(labels_subset)*100:.1f}%)")
