# Quick Reference Card

## 🚀 Basic Usage

```bash
python scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c cnsl/xlsr_vib_large_corpus \
    -b /path/to/benchmark \
    -m /path/to/model.ckpt \
    -r /path/to/results \
    -n experiment_name
```

## 🔄 Resume Incomplete Benchmark

```bash
# Just re-run the SAME command
# Will automatically:
# • Skip completed datasets
# • Resume incomplete datasets from where they stopped
```

## 🐛 Debug Incomplete Score

```bash
PROTOCOL_SUBSET=dev python scripts/benchmark_py/debug_validation.py \
    logs/results/dataset_score.txt \
    data/dataset/protocol.txt
```

## ⚙️ Protocol Subset

```bash
# Use "dev" subset
export PROTOCOL_SUBSET="dev"
python scripts/benchmark_py/benchmark.py [args]

# Use "eval" subset (default)
export PROTOCOL_SUBSET="eval"
python scripts/benchmark_py/benchmark.py [args]

# Use all lines (no filtering)
export PROTOCOL_SUBSET=""
python scripts/benchmark_py/benchmark.py [args]
```

## 🎯 Partial Results (NEW!)

```bash
# Accept partial results if >= 95% complete (default)
python scripts/benchmark_py/benchmark.py [args]

# Accept if >= 80% complete
export MIN_COMPLETION_RATE=80
python scripts/benchmark_py/benchmark.py [args]

# Only accept 100% complete (strict mode)
export MIN_COMPLETION_RATE=100
python scripts/benchmark_py/benchmark.py [args]
```

## 📊 Arguments

### Required
- `-g` GPU (0, 1, or MIG-xxxx)
- `-c` Config (cnsl/xlsr_vib_large_corpus)
- `-b` Benchmark folder
- `-m` Model checkpoint
- `-r` Results folder
- `-n` Experiment name/comment

### Optional
- `-a` Adapter paths (LoRA)
- `-l` Use Lightning checkpoint (true/false, default: true)
- `-s` Random start (true/false, default: true)
- `-t` Trim length (default: 64000)

## 🔍 Common Issues

### Issue: "Score file incomplete"
**Solution**: Re-run with SAME command to resume

### Issue: "Completion rate < 95%"
**Solution**: 
```bash
# Option 1: Re-run to try again
PROTOCOL_SUBSET=dev ./benchmark.py [args]

# Option 2: Accept partial results
export MIN_COMPLETION_RATE=80
PROTOCOL_SUBSET=dev ./benchmark.py [args]
```

### Issue: "Protocol subset mismatch"
**Solution**: Use same PROTOCOL_SUBSET value

### Issue: "Missing protocol creation failed"
**Solution**: Use debug tool to check protocol file

## 📂 Output Files

```
results/
└── experiment_name/
    ├── summary_results.txt
    ├── dataset1_config_experiment.txt
    ├── dataset2_config_experiment.txt
    ├── merged_protocol_config_experiment.txt
    ├── merged_scores_config_experiment.txt
    └── pooled_merged_protocol_config_experiment.txt
```

## 💡 Tips

1. **Same parameters**: Keep all parameters identical when resuming
2. **Protocol subset**: Most important - must be the same
3. **Different experiments**: Use different `-n` comment for different runs
4. **Debug first**: Use debug tool before asking for help

## 📚 Documentation

- `README.md` - Complete documentation
- `QUICKSTART.md` - 5-minute quick start
- `BUGFIX_RESUME.md` - Resume feature details
- `EXAMPLES.md` - Usage examples
- `MIGRATION_GUIDE.md` - Migrating from bash

## 🆘 Getting Help

1. Check error message details (now more verbose)
2. Use debug tool
3. Check this quick reference
4. Read detailed docs

---

**Quick tip**: Save this command in your notes! 📝
