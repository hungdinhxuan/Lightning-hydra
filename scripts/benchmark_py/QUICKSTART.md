# Quick Start Guide

Get started with the Python benchmark scripts in 5 minutes!

## 🚀 Installation

No installation needed! The scripts use only Python standard library (except for dependencies already in your project).

## ⚡ Quick Usage

### 1. Basic Example

```bash
cd /nvme1/hungdx/code/Lightning-hydra

python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c cnsl/xlsr_vib_large_corpus \
    -b /path/to/benchmark/folder \
    -m /path/to/model \
    -r /path/to/results \
    -n my_experiment
```

### 2. With All Options

```bash
python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c cnsl/xlsr_vib_large_corpus \
    -b /path/to/benchmark/folder \
    -m /path/to/model \
    -r /path/to/results \
    -n my_experiment \
    -a /path/to/adapter \
    -l true \
    -s true \
    -t 64000
```

### 3. Using a Different Protocol Subset

```bash
# Use "test" subset instead of default "eval"
export PROTOCOL_SUBSET="test"

python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c cnsl/xlsr_vib_large_corpus \
    -b /path/to/benchmark/folder \
    -m /path/to/model \
    -r /path/to/results \
    -n my_experiment
```

## 📋 Command Reference

### Required Arguments

| Flag | Description | Example |
|------|-------------|---------|
| `-g` | GPU number or UUID | `0`, `1`, `MIG-xxxx` |
| `-c` | YAML config path | `cnsl/xlsr_vib_large_corpus` |
| `-b` | Benchmark folder | `/data/benchmarks` |
| `-m` | Model checkpoint | `/models/checkpoint.ckpt` |
| `-r` | Results folder | `/results` |
| `-n` | Experiment name | `test_run` |

### Optional Arguments

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `-a` | Adapter/LoRA paths | None | `/adapters/lora.pt` |
| `-l` | Use Lightning checkpoint | `true` | `true`/`false` |
| `-s` | Random start | `true` | `true`/`false` |
| `-t` | Trim length | `64000` | `32000` |

## 🔧 Configuration

### Environment Variables

```bash
# Protocol subset (default: "eval")
export PROTOCOL_SUBSET="test"

# Batch size (default: 64)
export DEFAULT_BATCH_SIZE=128

# Trim length (default: 64000)
export DEFAULT_TRIM_LENGTH=32000

# Progress bar width (default: 50)
export PROGRESS_BAR_WIDTH=60
```

### Create a Config File

```bash
# create config.env
cat > config.env << EOF
export PROTOCOL_SUBSET="eval"
export DEFAULT_BATCH_SIZE=64
export DEFAULT_TRIM_LENGTH=64000
EOF

# Use it
source config.env
python scripts/benchmark_py/benchmark.py [args]
```

## 📂 Expected Folder Structure

```
benchmark_folder/
├── dataset1/
│   ├── protocol.txt
│   └── [audio files]
├── dataset2/
│   ├── protocol.txt
│   └── [audio files]
└── dataset3/
    ├── protocol.txt
    └── [audio files]
```

## 📊 Output Files

After running, you'll find in your results folder:

```
results/
└── my_experiment/
    ├── summary_results.txt           # Main results summary
    ├── dataset1_config_experiment.txt  # Per-dataset scores
    ├── dataset2_config_experiment.txt
    ├── dataset3_config_experiment.txt
    ├── merged_protocol_config_experiment.txt  # Combined protocol
    ├── merged_scores_config_experiment.txt    # Combined scores
    └── pooled_merged_protocol_config_experiment.txt  # Metadata
```

## 💡 Common Use Cases

### 1. Quick Test on Single Dataset

```bash
# Test with a single small dataset first
python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c my_config \
    -b /data/benchmarks/single_dataset \
    -m /models/model.ckpt \
    -r /results/test \
    -n quick_test
```

### 2. Batch Processing Multiple Configs

```bash
#!/bin/bash
configs=("config1" "config2" "config3")

for config in "${configs[@]}"; do
    python scripts/benchmark_py/benchmark.py \
        -g 0 \
        -c "$config" \
        -b /data/benchmarks \
        -m /models/model.ckpt \
        -r /results \
        -n "exp_${config}"
done
```

### 3. Using Different GPUs

```bash
# GPU 0
python scripts/benchmark_py/benchmark.py -g 0 -c config1 [other args]

# GPU 1
python scripts/benchmark_py/benchmark.py -g 1 -c config2 [other args]

# MIG GPU
python scripts/benchmark_py/benchmark.py -g MIG-xxxx -c config3 [other args]
```

## 🐛 Troubleshooting

### Problem: Import Error

```
ModuleNotFoundError: No module named 'benchmark_py'
```

**Solution**: Run from project root
```bash
cd /nvme1/hungdx/code/Lightning-hydra
python scripts/benchmark_py/benchmark.py [args]
```

### Problem: File Not Found

```
Error: Benchmark folder '/path/to/folder' does not exist
```

**Solution**: Use absolute paths or check the path
```bash
# Use absolute path
python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c config \
    -b /absolute/path/to/benchmark \
    [other args]
```

### Problem: Permission Denied

```bash
# Make script executable
chmod +x scripts/benchmark_py/benchmark.py

# Then run
./scripts/benchmark_py/benchmark.py [args]
```

## 📖 Learn More

- [README.md](README.md) - Detailed documentation
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migrating from bash
- [COMPARISON.md](COMPARISON.md) - Bash vs Python comparison

## ✅ Quick Checklist

Before running:

- [ ] You're in the project root directory
- [ ] Benchmark folder exists and contains subdirectories
- [ ] Each subdirectory has a `protocol.txt` file
- [ ] Model checkpoint path is correct
- [ ] GPU is available
- [ ] Results folder is writable

## 🎉 Success!

After running, check:

```bash
# View summary
cat results/my_experiment/summary_results.txt

# Check individual dataset results
ls results/my_experiment/

# View detailed logs
tail -f results/my_experiment/summary_results.txt
```

That's it! You're ready to run benchmarks with Python. 🚀
