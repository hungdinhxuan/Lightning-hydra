# Benchmark Python Package

A clean Python refactor of the original bash benchmark scripts. This package provides easier maintenance, better error handling, and improved extensibility.

## 🎯 Why Python?

The original bash scripts were difficult to maintain and debug. This Python implementation offers:

- **Better readability**: Clear object-oriented structure
- **Easier maintenance**: Modular design with single responsibility per module
- **Type safety**: Type hints for better IDE support
- **Error handling**: Comprehensive exception handling
- **Testing**: Easy to write unit tests
- **Extensibility**: Simple to add new features

## 📁 Project Structure

```
benchmark_py/
├── __init__.py              # Package initialization
├── constants.py             # Configuration constants (environment variable support)
├── utils.py                 # Utility functions (colors, progress bars, UI)
├── validation.py            # Score file validation
├── protocol.py              # Protocol file operations
├── scores.py                # Score file merging operations
├── execution.py             # Benchmark command execution
├── eer.py                   # EER calculations (pooled and average)
├── merge.py                 # Merged protocol/score file creation
├── benchmark.py             # Main orchestration script
└── README.md                # This file
```

## 🚀 Quick Start

### Basic Usage

Run a benchmark with the same parameters as the bash version:

```bash
python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c cnsl/xlsr_vib_large_corpus \
    -b /path/to/benchmark/folder \
    -m /path/to/model \
    -r /path/to/results \
    -n test_run
```

### With Optional Parameters

```bash
python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c cnsl/xlsr_vib_large_corpus \
    -b /path/to/benchmark/folder \
    -m /path/to/model \
    -r /path/to/results \
    -n test_run \
    -a /path/to/adapter \
    -l true \
    -s true \
    -t 64000
```

## 📝 Command Line Arguments

### Required Arguments

| Argument | Short | Description | Example |
|----------|-------|-------------|---------|
| `--gpu` | `-g` | GPU identifier (index or MIG UUID) | `0`, `1`, `MIG-xxxx` |
| `--config` | `-c` | YAML config file path | `cnsl/xlsr_vib_large_corpus` |
| `--benchmark-folder` | `-b` | Bulk benchmark folder path | `/data/benchmarks` |
| `--model-path` | `-m` | Base model path | `/models/checkpoint.ckpt` |
| `--results-folder` | `-r` | Results folder path | `/results` |
| `--comment` | `-n` | Comment/experiment name | `test_run` |

### Optional Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--adapter-paths` | `-a` | None | Adapter/LoRA paths |
| `--is-ln` | `-l` | `true` | Use Lightning checkpoint loading |
| `--random-start` | `-s` | `true` | Use random start for data processing |
| `--trim-length` | `-t` | `64000` | Trim length for audio processing |

## ⚙️ Configuration

### Environment Variables

You can configure defaults via environment variables:

```bash
# Protocol subset configuration
export PROTOCOL_SUBSET="test"  # Default: "eval"

# Execution defaults
export DEFAULT_BATCH_SIZE=128  # Default: 64
export DEFAULT_TRIM_LENGTH=32000  # Default: 64000

# UI configuration
export PROGRESS_BAR_WIDTH=60  # Default: 50

# Run benchmark with custom configuration
python scripts/benchmark_py/benchmark.py -g 0 -c config -b folder -m model -r results -n test
```

### Programmatic Configuration

You can also modify the constants in code:

```python
from benchmark_py.constants import CONSTANTS

# Modify constants
CONSTANTS.protocol_subset = "test"
CONSTANTS.default_batch_size = 128
CONSTANTS.progress_bar_width = 60
```

## 📦 Module Details

### 1. `constants.py` - Configuration Constants

Stores all configurable constants with environment variable support.

**Key Features:**
- Protocol subset configuration
- Default execution parameters
- File naming patterns
- Environment variable overrides

**Example:**
```python
from benchmark_py.constants import CONSTANTS

# Check if protocol subset should be used
if CONSTANTS.should_use_protocol_subset(protocol_file):
    subset_name = CONSTANTS.get_protocol_subset_name()
```

### 2. `utils.py` - Utility Functions

Provides colored output, progress tracking, and UI functions.

**Key Features:**
- Colored terminal output
- Progress bar display
- Banner and usage information
- File cleanup utilities

**Example:**
```python
from benchmark_py.utils import print_color, Color, display_progress

print_color(Color.GREEN, "✓ Success!")
display_progress(current=5, total=10)
```

### 3. `validation.py` - Score File Validation

Handles score file validation and completeness checking.

**Key Features:**
- Score file existence check
- Line count validation
- Protocol subset awareness

**Example:**
```python
from benchmark_py.validation import validate_score_file

result = validate_score_file(score_file, protocol_file)
if result.is_valid:
    print(f"Valid: {result.score_lines}/{result.expected_lines} lines")
```

### 4. `protocol.py` - Protocol File Operations

Handles protocol file operations including missing entry detection.

**Key Features:**
- Missing entry detection (sequential evaluation optimized)
- Protocol subset extraction
- Temporary protocol file creation

**Example:**
```python
from benchmark_py.protocol import create_missing_protocol

missing_count = create_missing_protocol(score_file, protocol_file, temp_protocol)
if missing_count > 0:
    print(f"Found {missing_count} missing entries")
```

### 5. `scores.py` - Score File Operations

Handles score file merging and parsing.

**Key Features:**
- Score file parsing (handles paths with spaces)
- Score file merging with deduplication
- Backup creation

**Example:**
```python
from benchmark_py.scores import merge_score_files

success = merge_score_files(original_score, new_score, merged_score)
```

### 6. `execution.py` - Benchmark Execution

Handles benchmark command construction and execution.

**Key Features:**
- Type-safe configuration with dataclasses
- Subprocess execution with GPU assignment
- Result evaluation

**Example:**
```python
from benchmark_py.execution import BenchmarkConfig, execute_benchmark

config = BenchmarkConfig(
    gpu_number="0",
    yaml_config="cnsl/xlsr_vib_large_corpus",
    score_save_path=Path("scores.txt"),
    # ... other parameters
)

success = execute_benchmark(config)
```

### 7. `eer.py` - EER Calculations

Handles pooled and average EER calculations.

**Key Features:**
- Pooled EER calculation across multiple datasets
- Average EER calculation
- Integration with existing Python scripts

**Example:**
```python
from benchmark_py.eer import calculate_pooled_eer, calculate_average_eer

pooled_result = calculate_pooled_eer(results_folder, normalized_yaml, comment, summary_file, subdirs)
average_eer = calculate_average_eer(summary_file)
```

### 8. `merge.py` - Merged File Creation

Handles creation of merged protocol and score files.

**Key Features:**
- Merged protocol/score file creation
- Metadata file generation
- Dataset prefixing for unique identification

**Example:**
```python
from benchmark_py.merge import create_merged_protocol

success = create_merged_protocol(
    results_folder, normalized_yaml, comment,
    yaml_config, base_model_path, summary_file, subdirs
)
```

### 9. `benchmark.py` - Main Script

Main orchestration script that ties everything together.

**Key Features:**
- Argument parsing with argparse
- Dataset iteration and processing
- Progress tracking
- Result summarization

## 🔄 Migration from Bash

### Comparison

| Feature | Bash Version | Python Version |
|---------|--------------|----------------|
| **Execution** | `./scripts/benchmark/benchmark.sh` | `python scripts/benchmark_py/benchmark.py` |
| **Arguments** | Same | Same |
| **Configuration** | Edit files or env vars | Env vars or programmatic |
| **Modularity** | Bash sourcing | Python imports |
| **Error Handling** | Exit codes | Exceptions + exit codes |
| **Testing** | Difficult | Easy with unittest/pytest |

### What's the Same?

- **Command-line interface**: Identical arguments and behavior
- **Functionality**: All features from bash version are preserved
- **Output format**: Same colored output and progress bars
- **File formats**: Compatible with existing score/protocol files

### What's Better?

- **Maintainability**: Clear module boundaries and responsibilities
- **Debugging**: Python stack traces vs bash script debugging
- **IDE Support**: Auto-completion, type checking, refactoring
- **Error Messages**: More detailed and helpful error messages
- **Extensibility**: Easy to add new features or modify existing ones

## 🧪 Testing (Future)

The modular structure makes it easy to add unit tests:

```python
# Example test (not yet implemented)
import unittest
from benchmark_py.validation import validate_score_file
from pathlib import Path

class TestValidation(unittest.TestCase):
    def test_validate_complete_score_file(self):
        result = validate_score_file(
            Path("test_data/complete_scores.txt"),
            Path("test_data/protocol.txt")
        )
        self.assertTrue(result.is_valid)
```

## 📊 Output Files

The Python version generates the same output files as the bash version:

1. **Summary File**: `summary_results.txt`
   - Individual dataset EERs
   - Pooled EER
   - Average EER
   - Configuration information

2. **Score Files**: `{dataset}_{config}_{comment}.txt`
   - Per-dataset score files

3. **Merged Protocol**: `merged_protocol_{config}_{comment}.txt`
   - Combined protocol from all datasets

4. **Merged Scores**: `merged_scores_{config}_{comment}.txt`
   - Combined scores from all datasets

5. **Metadata**: `pooled_merged_protocol_{config}_{comment}.txt`
   - Metadata about merged files

## 🐛 Troubleshooting

### Common Issues

**Issue**: Import errors when running the script
```bash
# Solution: Run from the scripts directory or add to PYTHONPATH
cd /path/to/Lightning-hydra
python scripts/benchmark_py/benchmark.py [args]
```

**Issue**: Permission denied
```bash
# Solution: Make the script executable
chmod +x scripts/benchmark_py/benchmark.py
```

**Issue**: Environment variables not being picked up
```bash
# Solution: Export them before running
export PROTOCOL_SUBSET="test"
python scripts/benchmark_py/benchmark.py [args]
```

## 🚧 Future Improvements

Potential enhancements for future versions:

- [ ] Add comprehensive unit tests
- [ ] Add integration tests
- [ ] Parallel dataset processing (multiprocessing)
- [ ] Progress persistence (resume interrupted runs)
- [ ] JSON output format option
- [ ] Web-based dashboard for results
- [ ] Docker containerization
- [ ] CI/CD integration examples

## 📄 License

Same license as the parent project.

## 🤝 Contributing

When contributing to this module:

1. Follow PEP 8 style guidelines
2. Add type hints to new functions
3. Update this README for new features
4. Add docstrings to all functions
5. Keep modules focused on single responsibility

## 📚 Additional Resources

- Original bash scripts: `scripts/benchmark/`
- Related Python scripts:
  - `scripts/score_file_to_eer.py` - EER evaluation
  - `scripts/calculate_pooled_eer.py` - Pooled EER calculation
  - `scripts/eval_metrics_DF.py` - Evaluation metrics

## 💬 Support

For issues or questions:
1. Check this README first
2. Review the original bash scripts for behavior comparison
3. Check the docstrings in each module
4. Examine the example usage in `benchmark.py`
