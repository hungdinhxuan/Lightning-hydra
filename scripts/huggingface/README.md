# Hugging Face Integration

This directory contains scripts and utilities for uploading models from the Lightning-hydra project to Hugging Face Hub.

## Files

- `push_to_hf.py` - Main function for uploading models to Hugging Face Hub
- `upload_model.py` - Command-line interface for easy model uploads
- `example_usage.py` - Example usage of the upload functionality
- `conformer_tcm.py` - Example model implementation with PyTorchModelHubMixin
- `README.md` - This documentation file

## Quick Start

### 1. Install Dependencies

```bash
pip install huggingface_hub torch transformers
```

### 2. Set up Authentication

Get your Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and set it as an environment variable:

```bash
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx
```

### 3. Upload a Model

#### Using the CLI (Recommended)

```bash
python upload_model.py \
    --model_name conformer_tcm \
    --config_path /path/to/your/config.yaml \
    --checkpoint_path /path/to/your/checkpoint.pth \
    --repo_name your-username/model-name
```

#### Using Python Function

```python
from push_to_hf import push_model_to_hf

success = push_model_to_hf(
    model_name="conformer_tcm",
    config_path="/path/to/config.yaml",
    checkpoint_path="/path/to/checkpoint.pth",
    repo_name="your-username/model-name"
)
```

## Function Parameters

The `push_model_to_hf` function accepts the following parameters:

### Required Parameters

- **`model_name`** (str): Name of the model class (should match filename without .py)
- **`config_path`** (str): Path to the YAML configuration file
- **`checkpoint_path`** (str): Path to the model checkpoint file
- **`repo_name`** (str): Name of the Hugging Face repository (e.g., "username/model-name")

### Optional Parameters

- **`private`** (bool): Whether to create a private repository (default: False)
- **`commit_message`** (str): Commit message for the upload (default: "Upload model")
- **`token`** (Optional[str]): Hugging Face token (if None, will use cached token)

## Supported Model Types

The function supports several model types out of the box:

1. **`conformer_tcm`** - Conformer TCM models
2. **`xlsr_conformertcm_baseline`** - XLSR Conformer TCM baseline models
3. **Custom models** - Any model class in the `scripts/huggingface/` directory

## Model Requirements

For your model to work with this integration, it should:

1. **Inherit from PyTorchModelHubMixin**:
   ```python
   from huggingface_hub import PyTorchModelHubMixin
   import torch.nn as nn
   
   class YourModel(nn.Module, PyTorchModelHubMixin):
       def __init__(self, **kwargs):
           super().__init__()
           # Your model initialization
       
       def forward(self, x):
           # Your forward pass
           return output
   ```

2. **Be compatible with the configuration format** used in your YAML config files

3. **Handle checkpoint loading** properly (the function will try to load from `state_dict` or direct checkpoint)

## Configuration Format

Your YAML configuration should follow this structure:

```yaml
model:
  args:
    conformer:
      # Conformer-specific parameters
    ssl_pretrained_path: "/path/to/ssl/model"
    # Other model parameters
```

## Examples

### Example 1: Basic Upload

```bash
python upload_model.py \
    --model_name conformer_tcm \
    --config_path configs/experiment/conformer_config.yaml \
    --checkpoint_path checkpoints/best_model.pth \
    --repo_name myusername/conformer-model
```

### Example 2: Private Repository

```bash
python upload_model.py \
    --model_name xlsr_conformertcm_baseline \
    --config_path configs/experiment/xlsr_config.yaml \
    --checkpoint_path checkpoints/xlsr_model.pth \
    --repo_name myusername/private-model \
    --private
```

### Example 3: With Custom Token and Message

```bash
python upload_model.py \
    --model_name conformer_tcm \
    --config_path config.yaml \
    --checkpoint_path model.pth \
    --repo_name myusername/model \
    --token hf_xxxxxxxxxxxx \
    --commit_message "Updated model with better performance"
```

## Error Handling

The function includes comprehensive error handling for:

- Missing files (config or checkpoint)
- Invalid configuration files
- Model import errors
- Network issues during upload
- Authentication problems

## Generated Model Card

The function automatically generates a model card (README.md) that includes:

- Model metadata and tags
- Configuration details
- Usage examples
- Training information
- Citation information

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure your model class is properly defined and inherits from `PyTorchModelHubMixin`

2. **Authentication Issues**: Verify your Hugging Face token is valid and has the necessary permissions

3. **File Not Found**: Ensure all file paths are correct and files exist

4. **Configuration Errors**: Check that your YAML config file is valid and contains the expected structure

### Debug Mode

Use the `--verbose` flag to enable detailed logging:

```bash
python upload_model.py --verbose --model_name conformer_tcm --config_path config.yaml --checkpoint_path model.pth --repo_name username/model
```

## Advanced Usage

### Custom Model Loading

If you have a custom model that doesn't fit the predefined patterns, you can:

1. Create a Python file in the `scripts/huggingface/` directory
2. Name it after your model (e.g., `my_custom_model.py`)
3. Implement your model class with `PyTorchModelHubMixin`
4. Use the model name in the upload function

### Batch Uploads

For uploading multiple models, you can create a script that calls the function multiple times:

```python
models_to_upload = [
    {
        "model_name": "conformer_tcm",
        "config_path": "config1.yaml",
        "checkpoint_path": "model1.pth",
        "repo_name": "username/model1"
    },
    {
        "model_name": "xlsr_conformertcm_baseline",
        "config_path": "config2.yaml",
        "checkpoint_path": "model2.pth",
        "repo_name": "username/model2"
    }
]

for model_info in models_to_upload:
    success = push_model_to_hf(**model_info)
    print(f"Upload {'successful' if success else 'failed'} for {model_info['repo_name']}")
```

## Contributing

When adding new model types or improving the upload functionality:

1. Update the `model_configs` dictionary in `_load_model_from_name`
2. Add appropriate error handling
3. Update this documentation
4. Test with your specific model configuration

## License

This integration follows the same license as the main Lightning-hydra project.
