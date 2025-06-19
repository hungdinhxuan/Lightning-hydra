# Replay Module for Continual Learning

This replay module implements a continual learning approach that combines novel data with a fixed replay buffer to prevent catastrophic forgetting. The module is designed to work with PyTorch Lightning and supports audio data processing with multi-view augmentation.

## Features

- **Fixed Replay Set**: Maintains a fixed set of replay data defined by a protocol file
- **Novel Set**: Handles new/novel data defined by a separate protocol file  
- **Configurable Ratios**: Allows custom ratios (x:y) for novel vs replay samples in each batch
- **Random Replay Sampling**: Randomly samples from the fixed replay set for each batch
- **Multi-view Support**: Compatible with existing multi-view data processing pipeline
- **Lightning Integration**: Full PyTorch Lightning DataModule implementation

## Key Components

### 1. ReplayDataset
Custom dataset class that handles both novel and replay data:
- Stores separate lists for novel and replay samples
- Provides methods to sample from both sets
- Supports all existing augmentation methods

### 2. ReplaySampler
Custom sampler that creates mixed batches:
- Calculates samples per batch based on configured ratios
- Ensures proper mixing of novel and replay data
- Handles random sampling from replay buffer

### 3. ReplayDataLoader
Custom DataLoader with replay-aware batching:
- Uses ReplaySampler for batch composition
- Handles the custom collate function
- Maintains compatibility with existing pipeline

### 4. ReplayDataModule
Lightning DataModule for the complete replay system:
- Manages both novel and replay protocol files
- Configures ratios and batch composition
- Provides train/val/test dataloaders

## Usage

### Basic Setup

```python
from src.data.replay_multiview_datamodule import ReplayDataModule

# Configuration
args = {
    'views': 4,
    'wav_samp_rate': 16000,
    'padding_type': 'repeat',
    'random_start': True,
    'trim_length': 66800,
    'augmentation_methods': [],
    # ... other parameters
}

# Create datamodule
datamodule = ReplayDataModule(
    data_dir="data/",
    batch_size=32,
    args=args,
    novel_ratio=0.7,        # 70% novel samples
    replay_ratio=0.3,       # 30% replay samples
    novel_protocol_path="data/novel_protocol.txt",
    replay_protocol_path="data/replay_protocol.txt",
)
```

### Protocol File Format

Both novel and replay protocol files should follow the same format:
```
utt_id subset label
```

Example:
```
file001.wav train bonafide
file002.wav train spoof
file003.wav dev bonafide
file004.wav eval spoof
```

Where:
- `utt_id`: Audio file identifier (relative to data_dir)
- `subset`: Data split (`train`, `dev`, `eval`)
- `label`: Class label (`bonafide` or `spoof`)

### Configuration Parameters

#### Ratio Configuration
- `novel_ratio`: Fraction of batch composed of novel samples (0.0-1.0)
- `replay_ratio`: Fraction of batch composed of replay samples (0.0-1.0)
- **Constraint**: `novel_ratio + replay_ratio ≤ 1.0`

#### File Paths
- `novel_protocol_path`: Path to novel set protocol file
- `replay_protocol_path`: Path to replay set protocol file
- `data_dir`: Base directory containing audio files

#### Batch Composition Example
For `batch_size=32`, `novel_ratio=0.7`, `replay_ratio=0.3`:
- Novel samples per batch: 22
- Replay samples per batch: 9
- Total samples per batch: 31 (may be less than batch_size)

### Integration with Training

```python
import lightning as L

# Create trainer
trainer = L.Trainer(
    max_epochs=100,
    devices=1,
    accelerator="gpu",
)

# Setup datamodule
datamodule.setup(stage="fit")

# Train model with replay datamodule
# trainer.fit(model, datamodule)
```

## Implementation Details

### Batch Creation Process

1. **Novel Sampling**: Sample novel data sequentially (with shuffling)
2. **Replay Sampling**: Randomly sample from fixed replay buffer
3. **Batch Mixing**: Combine novel and replay samples
4. **Shuffling**: Optionally shuffle the mixed batch

### Memory Efficiency

- Replay buffer is fixed in size (no growth over time)
- Random sampling prevents overfitting to specific replay samples
- Efficient data loading with PyTorch's DataLoader

### Reproducibility

- Random sampling is controlled by PyTorch's random state
- Consistent batch composition across epochs when using fixed seeds

## Example Files

### Creating Protocol Files

```python
def create_protocol_files():
    # Novel protocol
    novel_data = """
    novel_001.wav train bonafide
    novel_002.wav train spoof
    novel_003.wav dev bonafide
    """
    
    # Replay protocol  
    replay_data = """
    replay_001.wav train bonafide
    replay_002.wav train spoof
    replay_003.wav dev bonafide
    """
    
    with open("data/novel_protocol.txt", "w") as f:
        f.write(novel_data)
    
    with open("data/replay_protocol.txt", "w") as f:
        f.write(replay_data)
```

### Running the Example

```bash
python example_replay_usage.py
```

## Validation and Testing

- **Validation**: Uses only novel data (standard evaluation)
- **Testing**: Uses only novel data (standard evaluation)
- **Training**: Uses mixed novel + replay data according to configured ratios

## Notes

- The epoch length is determined by the novel set size
- Replay samples are sampled with replacement
- Supports all existing augmentation methods
- Compatible with multi-view processing pipeline
- Maintains backward compatibility with existing codebase

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure protocol files exist and paths are correct
2. **Ratio Error**: Ensure `novel_ratio + replay_ratio ≤ 1.0`
3. **Empty Datasets**: Check protocol files have correct format and valid subset labels
4. **Memory Issues**: Reduce batch size or number of workers if running out of memory

### Debug Tips

- Enable verbose logging to see batch composition
- Check dataset sizes after setup
- Validate protocol file format and file paths
- Monitor replay buffer sampling distribution 