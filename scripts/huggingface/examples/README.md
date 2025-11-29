# Model Registry Examples

This directory contains examples of how to use the decoupled model registry system for Hugging Face Hub integration.

## Files

- `custom_model_config.py` - Shows how to create custom model configurations
- `usage_example.py` - Demonstrates how to use the system with both built-in and custom models

## How to Add Your Own Models

### 1. Create a Model Configuration Class

Create a new file (e.g., `my_model_config.py`) and define your model configuration:

```python
from model_registry import ModelConfig

class MyModelConfig(ModelConfig):
    @property
    def model_name(self) -> str:
        return "my_model"
    
    @property
    def import_path(self) -> str:
        return "/path/to/your/model.py"
    
    @property
    def class_name(self) -> str:
        return "MyModel"
    
    def get_init_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "param1": config.get('model', {}).get('args', {}).get('param1'),
            "param2": config.get('model', {}).get('args', {}).get('param2')
        }
```

### 2. Register Your Model

```python
from model_registry import register_model_config
from my_model_config import MyModelConfig

# Register your model
register_model_config(MyModelConfig())
```

### 3. Use Your Model

```python
from push_to_hf import push_model_to_hf

success = push_model_to_hf(
    model_name="my_model",
    config_path="/path/to/config.yaml",
    checkpoint_path="/path/to/checkpoint.pth",
    repo_name="username/model-name"
)
```

## Benefits of the New System

1. **No Code Modification**: Users don't need to modify the core upload script
2. **Easy Extension**: Simply create a new configuration class and register it
3. **Clean Separation**: Model-specific logic is separated from the upload logic
4. **Fallback Support**: If a model isn't registered, the system tries to load it from the file system
5. **Type Safety**: Abstract base class ensures all required methods are implemented

## Built-in Models

The system comes with a pre-configured model:
- `xlsr_conformer_tcm` - XLSR Conformer TCM model

## Adding Models to the Registry

You can add models to the registry in several ways:

1. **At runtime** (recommended for user code):
   ```python
   from model_registry import register_model_config
   register_model_config(MyModelConfig())
   ```

2. **By modifying the registry directly** (for permanent additions):
   Edit `model_registry.py` and add your configuration to the `_register_default_configs` method.

3. **Using environment variables** (for configuration-based registration):
   You can extend the system to read model configurations from files or environment variables.
