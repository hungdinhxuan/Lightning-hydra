# Continual Learning with XLSR-ConformerTCM

This project implements a replay-based continual learning system for audio spoofing detection using the XLSR-ConformerTCM model and the Avalanche framework.

## Project Structure

```
continual_learning/
├── src/
│   ├── models/
│   │   └── xlsr_conformertcm_cl.py
│   ├── data/
│   │   └── avalanche_datamodule.py
│   └── train.py
├── configs/
├── experiments/
└── requirements.txt
```

## Features

- Replay-based continual learning using Avalanche framework
- XLSR-ConformerTCM model for audio spoofing detection
- Reservoir sampling for replay buffer management
- TensorBoard logging for training metrics
- Checkpoint saving and loading

## Installation

1. Create a new virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in the required format:
   - Audio files in a directory
   - Protocol file with format: `utterance_id subset label`

2. Train the model:
```bash
python src/train.py \
    --data_dir /path/to/data \
    --protocol_path /path/to/protocol.txt \
    --ssl_pretrained_path /path/to/ssl/model \
    --conformer_config '{"config": "here"}' \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --replay_buffer_size 1000
```

## Model Architecture

The model combines:
- XLSR-ConformerTCM as the base model
- Replay buffer for storing past experiences
- Reservoir sampling for efficient memory management
- Continual learning strategies from Avalanche

## Training Strategy

The training process:
1. Loads data using Avalanche's benchmark system
2. Initializes the model with replay buffer
3. Trains on each experience while maintaining past knowledge
4. Evaluates performance on test data
5. Saves checkpoints after each experience

## Monitoring

Training progress can be monitored using:
- Interactive console output
- TensorBoard logs in the `tb_data` directory
- Checkpoints in the specified checkpoint directory

## License

This project is licensed under the MIT License - see the LICENSE file for details. 