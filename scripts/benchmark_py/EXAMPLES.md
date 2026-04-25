# Usage Examples

Practical examples for common benchmark scenarios.

## 📚 Table of Contents

1. [Basic Usage](#basic-usage)
2. [Advanced Usage](#advanced-usage)
3. [Configuration Examples](#configuration-examples)
4. [Integration Examples](#integration-examples)
5. [Troubleshooting Examples](#troubleshooting-examples)

## Basic Usage

### Example 1: Simple Benchmark

Run a basic benchmark on a single GPU:

```bash
python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c cnsl/xlsr_vib_large_corpus \
    -b /data/benchmarks/wildspoof \
    -m /models/checkpoint.ckpt \
    -r /results \
    -n baseline_test
```

**Expected output:**
```
┌─────────────────────────────────────────────────────────────────┐
│               🚀 BULK BENCHMARK RUNNER TOOL 🚀                  │
└─────────────────────────────────────────────────────────────────┘

✓ Starting benchmark with device 0 and config cnsl/xlsr_vib_large_corpus
✓ Results will be saved to /results/baseline_test

┌─────────────────────────────────────────────────────────────────┐
│ Processing dataset: dataset1
└─────────────────────────────────────────────────────────────────┘
[==================================================] 100% (1/3)
...
```

### Example 2: With LoRA Adapter

Run benchmark with a LoRA adapter:

```bash
python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c cnsl/xlsr_vib_large_corpus \
    -b /data/benchmarks/wildspoof \
    -m /models/base_model.ckpt \
    -r /results \
    -n lora_test \
    -a /models/lora_adapter.pt
```

### Example 3: Custom Trim Length

Use a shorter trim length for faster processing:

```bash
python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c cnsl/xlsr_vib_large_corpus \
    -b /data/benchmarks/wildspoof \
    -m /models/checkpoint.ckpt \
    -r /results \
    -n fast_test \
    -t 32000
```

## Advanced Usage

### Example 4: Multiple GPUs in Parallel

Run different experiments on different GPUs simultaneously:

```bash
# Terminal 1 - GPU 0
python scripts/benchmark_py/benchmark.py \
    -g 0 -c config1 -b /data/benchmarks -m /models/model1.ckpt \
    -r /results -n exp_gpu0 &

# Terminal 2 - GPU 1
python scripts/benchmark_py/benchmark.py \
    -g 1 -c config2 -b /data/benchmarks -m /models/model2.ckpt \
    -r /results -n exp_gpu1 &

# Wait for both to complete
wait
```

### Example 5: MIG GPU Usage

Run on MIG (Multi-Instance GPU) partitions:

```bash
# Find MIG UUIDs
nvidia-smi -L

# Use specific MIG instance
python scripts/benchmark_py/benchmark.py \
    -g MIG-bf1c6c1f-6556-5c33-8db0-3c4f69ebd15e \
    -c cnsl/xlsr_vib_large_corpus \
    -b /data/benchmarks \
    -m /models/checkpoint.ckpt \
    -r /results \
    -n mig_test
```

### Example 6: Batch Processing Multiple Models

Test multiple models on the same benchmark:

```bash
#!/bin/bash
# test_multiple_models.sh

MODELS=(
    "/models/baseline.ckpt"
    "/models/augmented.ckpt"
    "/models/fine_tuned.ckpt"
)

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    name="model_$(basename $model .ckpt)"
    
    echo "Testing model: $name"
    
    python scripts/benchmark_py/benchmark.py \
        -g 0 \
        -c cnsl/xlsr_vib_large_corpus \
        -b /data/benchmarks \
        -m "$model" \
        -r /results \
        -n "$name"
done
```

## Configuration Examples

### Example 7: Using Different Protocol Subsets

```bash
# Use "eval" subset (default)
python scripts/benchmark_py/benchmark.py [args] -n eval_test

# Use "test" subset
export PROTOCOL_SUBSET="test"
python scripts/benchmark_py/benchmark.py [args] -n test_subset

# Use "dev" subset
export PROTOCOL_SUBSET="dev"
python scripts/benchmark_py/benchmark.py [args] -n dev_subset

# Use all lines (no subset filtering)
export PROTOCOL_SUBSET=""
python scripts/benchmark_py/benchmark.py [args] -n all_lines
```

### Example 8: Custom Configuration File

Create a reusable configuration:

```bash
# config_production.env
export PROTOCOL_SUBSET="eval"
export DEFAULT_BATCH_SIZE=64
export DEFAULT_TRIM_LENGTH=64000
export PROGRESS_BAR_WIDTH=50

# Load and use
source config_production.env
python scripts/benchmark_py/benchmark.py [args]
```

### Example 9: Development vs Production Settings

```bash
# Development (fast, test subset)
export PROTOCOL_SUBSET="dev"
export DEFAULT_BATCH_SIZE=128
export DEFAULT_TRIM_LENGTH=32000

python scripts/benchmark_py/benchmark.py [args] -n dev_run

# Production (complete, eval subset)
export PROTOCOL_SUBSET="eval"
export DEFAULT_BATCH_SIZE=64
export DEFAULT_TRIM_LENGTH=64000

python scripts/benchmark_py/benchmark.py [args] -n prod_run
```

## Integration Examples

### Example 10: Integration with Shell Script

```bash
#!/bin/bash
# benchmark_runner.sh - Production wrapper script

set -e  # Exit on error

# Configuration
GPU="0"
CONFIG="cnsl/xlsr_vib_large_corpus"
BENCHMARK_ROOT="/data/benchmarks"
MODEL_ROOT="/models"
RESULTS_ROOT="/results"
DATE=$(date +%Y%m%d_%H%M%S)

# Function to run benchmark
run_benchmark() {
    local model_name=$1
    local experiment_name="${model_name}_${DATE}"
    
    echo "========================================="
    echo "Running benchmark: $experiment_name"
    echo "Model: ${MODEL_ROOT}/${model_name}.ckpt"
    echo "========================================="
    
    python scripts/benchmark_py/benchmark.py \
        -g "$GPU" \
        -c "$CONFIG" \
        -b "$BENCHMARK_ROOT" \
        -m "${MODEL_ROOT}/${model_name}.ckpt" \
        -r "$RESULTS_ROOT" \
        -n "$experiment_name"
    
    echo "✅ Completed: $experiment_name"
    echo ""
}

# Run benchmarks
run_benchmark "baseline"
run_benchmark "augmented"
run_benchmark "fine_tuned"

echo "🎉 All benchmarks completed!"
```

### Example 11: Python Wrapper for Automation

```python
#!/usr/bin/env python3
# automated_benchmark.py

import subprocess
import sys
from pathlib import Path
from datetime import datetime

class BenchmarkRunner:
    def __init__(self, gpu, config, benchmark_root, results_root):
        self.gpu = gpu
        self.config = config
        self.benchmark_root = benchmark_root
        self.results_root = results_root
    
    def run(self, model_path, experiment_name):
        """Run a single benchmark"""
        cmd = [
            'python', 'scripts/benchmark_py/benchmark.py',
            '-g', self.gpu,
            '-c', self.config,
            '-b', str(self.benchmark_root),
            '-m', str(model_path),
            '-r', str(self.results_root),
            '-n', experiment_name
        ]
        
        print(f"🚀 Running: {experiment_name}")
        result = subprocess.run(cmd, check=True)
        print(f"✅ Completed: {experiment_name}\n")
        return result.returncode == 0

def main():
    runner = BenchmarkRunner(
        gpu='0',
        config='cnsl/xlsr_vib_large_corpus',
        benchmark_root=Path('/data/benchmarks'),
        results_root=Path('/results')
    )
    
    models = [
        (Path('/models/baseline.ckpt'), 'baseline'),
        (Path('/models/augmented.ckpt'), 'augmented'),
        (Path('/models/fine_tuned.ckpt'), 'fine_tuned'),
    ]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for model_path, model_name in models:
        experiment = f"{model_name}_{timestamp}"
        runner.run(model_path, experiment)
    
    print("🎉 All benchmarks completed!")

if __name__ == '__main__':
    main()
```

### Example 12: Integration with MLflow

```python
#!/usr/bin/env python3
# benchmark_with_mlflow.py

import mlflow
import subprocess
from pathlib import Path

def run_benchmark_with_tracking(
    gpu, config, benchmark_folder, model_path, 
    results_folder, experiment_name
):
    """Run benchmark and log to MLflow"""
    
    with mlflow.start_run(run_name=experiment_name):
        # Log parameters
        mlflow.log_param("gpu", gpu)
        mlflow.log_param("config", config)
        mlflow.log_param("model_path", model_path)
        
        # Run benchmark
        cmd = [
            'python', 'scripts/benchmark_py/benchmark.py',
            '-g', gpu,
            '-c', config,
            '-b', str(benchmark_folder),
            '-m', str(model_path),
            '-r', str(results_folder),
            '-n', experiment_name
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Parse and log results
        summary_file = Path(results_folder) / experiment_name / 'summary_results.txt'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                for line in f:
                    if '|' in line and 'EER' not in line:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 2 and parts[0] not in ['Dataset', '']:
                            dataset = parts[0]
                            eer = float(parts[1])
                            mlflow.log_metric(f"{dataset}_EER", eer)
            
            # Log summary file as artifact
            mlflow.log_artifact(str(summary_file))

# Usage
mlflow.set_experiment("benchmark_experiments")

run_benchmark_with_tracking(
    gpu='0',
    config='cnsl/xlsr_vib_large_corpus',
    benchmark_folder='/data/benchmarks',
    model_path='/models/checkpoint.ckpt',
    results_folder='/results',
    experiment_name='test_run'
)
```

## Troubleshooting Examples

### Example 13: Debugging Import Errors

```bash
# If you get import errors, check Python path
cd /nvme2/hungdx/code/Lightning-hydra

# Test imports
python3 -c "
import sys
sys.path.insert(0, 'scripts')
from benchmark_py import constants
print('✅ Imports work!')
"

# If it works, run benchmark
python scripts/benchmark_py/benchmark.py [args]
```

### Example 14: Handling Missing Protocol Files

```bash
# Check if protocol files exist
for dir in /data/benchmarks/*; do
    if [ ! -f "$dir/protocol.txt" ]; then
        echo "⚠️ Missing protocol.txt in: $dir"
    fi
done

# Create dummy protocol if needed (for testing)
echo "file1.wav eval bonafide" > /data/benchmarks/test/protocol.txt
```

### Example 15: Resume Failed Benchmark

The Python version automatically detects and resumes incomplete benchmarks:

```bash
# First run (interrupted)
python scripts/benchmark_py/benchmark.py \
    -g 0 -c config -b /data/benchmarks \
    -m /models/model.ckpt -r /results -n exp1
# ^C (interrupted)

# Resume (automatically detects completed datasets)
python scripts/benchmark_py/benchmark.py \
    -g 0 -c config -b /data/benchmarks \
    -m /models/model.ckpt -r /results -n exp1
# Will skip already completed datasets!
```

## Comparison Examples

### Example 16: Side-by-Side Bash vs Python

```bash
# Run bash version
time ./scripts/benchmark/benchmark.sh \
    -g 0 -c config -b /data/benchmarks \
    -m /models/model.ckpt -r /results/bash -n test

# Run Python version
time python scripts/benchmark_py/benchmark.py \
    -g 0 -c config -b /data/benchmarks \
    -m /models/model.ckpt -r /results/python -n test

# Compare results
diff -u results/bash/test/summary_results.txt \
        results/python/test/summary_results.txt

# Should show no differences (or minimal floating-point differences)
```

### Example 17: Performance Comparison

```bash
#!/bin/bash
# compare_performance.sh

echo "Testing Bash version..."
time ./scripts/benchmark/benchmark.sh [args] > /tmp/bash.log 2>&1

echo "Testing Python version..."
time python scripts/benchmark_py/benchmark.py [args] > /tmp/python.log 2>&1

echo "Bash log:"
tail /tmp/bash.log

echo "Python log:"
tail /tmp/python.log
```

## Production Examples

### Example 18: CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Run Benchmarks

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  benchmark:
    runs-on: self-hosted  # GPU runner
    steps:
      - uses: actions/checkout@v2
      
      - name: Run benchmark
        run: |
          python scripts/benchmark_py/benchmark.py \
            -g 0 \
            -c cnsl/xlsr_vib_large_corpus \
            -b /data/benchmarks \
            -m /models/latest.ckpt \
            -r /results \
            -n ci_${GITHUB_RUN_NUMBER}
      
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: results/ci_*/
```

### Example 19: Docker Integration

```dockerfile
# Dockerfile.benchmark
FROM pytorch/pytorch:latest

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "scripts/benchmark_py/benchmark.py"]
```

```bash
# Build image
docker build -f Dockerfile.benchmark -t benchmark:latest .

# Run benchmark in container
docker run --gpus all \
    -v /data/benchmarks:/benchmarks \
    -v /models:/models \
    -v /results:/results \
    benchmark:latest \
    -g 0 -c config -b /benchmarks \
    -m /models/model.ckpt -r /results -n docker_test
```

### Example 20: Scheduled Benchmarks

```bash
# Add to crontab
crontab -e

# Run benchmark every Sunday at midnight
0 0 * * 0 cd /nvme2/hungdx/code/Lightning-hydra && \
    python scripts/benchmark_py/benchmark.py \
    -g 0 -c config -b /data/benchmarks \
    -m /models/latest.ckpt -r /results \
    -n weekly_$(date +\%Y\%m\%d) >> /var/log/benchmark.log 2>&1
```

## Summary

These examples cover:
- ✅ Basic single-GPU usage
- ✅ Advanced multi-GPU scenarios
- ✅ Configuration management
- ✅ Shell script integration
- ✅ Python automation
- ✅ MLflow tracking
- ✅ CI/CD integration
- ✅ Docker deployment
- ✅ Scheduled runs

Choose the example that best fits your use case and adapt it to your needs!
