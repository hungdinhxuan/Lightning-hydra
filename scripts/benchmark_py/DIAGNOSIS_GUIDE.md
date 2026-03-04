# Missing Files Diagnosis Guide

## 🎉 Tiến triển lớn!

Sau khi fix parsing:
- **Before**: 66817/85128 (78.5%) - thiếu 18311 files
- **After**: 66817/67641 (98.8%) - thiếu chỉ 824 files

→ **Fix parsing đã giúp giảm missing từ 21.5% xuống 1.2%!** ✨

## 🔍 Tìm nguyên nhân 824 files còn lại

### Option 1: Chạy diagnosis script (Recommended)

```bash
# Auto-run với configuration sẵn
./scripts/benchmark_py/run_diagnosis.sh
```

**Output**:
```
🔍 DIAGNOSING MISSING FILES (824 files)
...
📄 Full report saved to: missing_files_diagnosis_20260123_163000.log

DIAGNOSIS SUMMARY:
  FILE_NOT_FOUND: 500 files (60.7%)
  CORRUPTED: 200 files (24.3%)
  FORMAT_ISSUE: 100 files (12.1%)
  UNKNOWN: 24 files (2.9%)
```

### Option 2: Chạy manual với custom paths

```bash
python scripts/benchmark_py/diagnose_missing_files.py \
    --score-file logs/results/CNSL_Q1_2026_benchmarks_dev/XLSR_ConformerTCM_MDT_RawboostLA_DF/2026_JAN_14_CNSL_DATA_*.txt \
    --protocol-file data/CNSL_Q1_2026_benchmarks/2026_JAN_14_CNSL_DATA/protocol.txt \
    --data-dir data/CNSL_Q1_2026_benchmarks/2026_JAN_14_CNSL_DATA \
    --subset dev \
    --output missing_diagnosis.log
```

## 📊 Log File Format

Diagnosis log sẽ chứa:

### 1. Header (Metadata)
```
================================================================================
MISSING FILES DIAGNOSIS REPORT
================================================================================
Generated: 2026-01-23 16:30:00
Protocol file: data/.../protocol.txt
Score file: logs/.../score.txt
Data directory: data/.../
Protocol subset: dev

Total protocol entries: 67641
Scored entries: 66817
Missing entries: 824 (1.22%)
```

### 2. Detailed Analysis (Per File)
```
[FILE_NOT_FOUND] Line 12345
  File ID: AIHub Elevenlabs/missing_file.wav
  Subset: dev, Label: spoof
  Expected path: data/.../AIHub Elevenlabs/missing_file.wav

[CORRUPTED] Line 23456
  File ID: corrupted_audio.wav
  Subset: dev, Label: bonafide
  File path: data/.../corrupted_audio.wav
  File size: 0 bytes
  Error: WAV file header corrupted

[FORMAT_ISSUE] Line 34567
  File ID: invalid_format.mp3
  Subset: dev, Label: spoof
  File path: data/.../invalid_format.mp3
  File size: 1234567 bytes
  Error: Not a WAV file | Error: unsupported format

[UNKNOWN] Line 45678
  File ID: mystery_file.wav
  Subset: dev, Label: bonafide
  File exists: data/.../mystery_file.wav
  Audio info: {'channels': 1, 'sample_width': 2, 'framerate': 16000, ...}
  Note: File is valid but wasn't processed (check benchmark logs)
```

### 3. Summary by Category
```
================================================================================
SUMMARY BY CATEGORY
================================================================================

FILE_NOT_FOUND: 500 files (60.7%)
CORRUPTED: 200 files (24.3%)
FORMAT_ISSUE: 100 files (12.1%)
UNKNOWN: 24 files (2.9%)
```

### 4. Recommendations
```
================================================================================
RECOMMENDATIONS
================================================================================

1. FILE_NOT_FOUND (500 files):
   - Check if data directory is correct
   - Check if files were moved or deleted
   - Consider removing these entries from protocol

2. CORRUPTED (200 files):
   - Re-download these files if possible
   - Remove from protocol if cannot fix
   - Check disk for errors

... etc
```

## 🎯 Actions Based on Results

### If most are FILE_NOT_FOUND

```bash
# Option 1: Fix protocol file (remove missing entries)
python scripts/benchmark_py/create_clean_protocol.py \
    --input data/.../protocol.txt \
    --output data/.../protocol_clean.txt \
    --missing-log missing_diagnosis.log

# Then re-run benchmark with clean protocol
```

### If most are CORRUPTED

```bash
# Option 1: Re-download corrupted files
# Option 2: Remove from protocol
# Option 3: Accept partial results (98.8% is excellent!)
```

### If most are UNKNOWN

```bash
# Check benchmark logs for OOM or timeout errors
grep -i "error\|oom\|timeout\|killed" benchmark.log

# May need to:
# - Increase memory
# - Increase timeout
# - Process in smaller batches
```

## 📈 Expected Results

### Scenario 1: Mostly FILE_NOT_FOUND (60%+)

```
FILE_NOT_FOUND: 500 files (60.7%)
CORRUPTED: 200 files (24.3%)
FORMAT_ISSUE: 100 files (12.1%)
UNKNOWN: 24 files (2.9%)
```

**Action**: Remove missing files from protocol or re-download

### Scenario 2: Mostly UNKNOWN (50%+)

```
FILE_NOT_FOUND: 50 files (6%)
CORRUPTED: 20 files (2.4%)
FORMAT_ISSUE: 10 files (1.2%)
UNKNOWN: 744 files (90.3%)
```

**Action**: Check benchmark logs, might be OOM/timeout issues

### Scenario 3: Mixed Issues

```
FILE_NOT_FOUND: 200 files (24%)
CORRUPTED: 200 files (24%)
FORMAT_ISSUE: 200 files (24%)
UNKNOWN: 224 files (27%)
```

**Action**: Multiple issues, need comprehensive fix

## 💡 Recommendations

### For 98.8% Completion (Your Case)

**Option 1: Accept Results** ⭐ RECOMMENDED

```bash
# 98.8% is excellent! Just accept it
# Results are already evaluated and in summary
# Just document the 1.2% missing rate
```

**Option 2: Investigate and Fix**

```bash
# Run diagnosis
./scripts/benchmark_py/run_diagnosis.sh

# Review log
cat missing_files_diagnosis_*.log

# Fix issues
# - Remove missing files from protocol
# - Fix corrupted files
# - Convert format issues

# Re-run benchmark
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
```

### For Production (Need 100%)

```bash
# 1. Diagnose
./scripts/benchmark_py/run_diagnosis.sh

# 2. Fix all issues in log

# 3. Re-run with strict mode
export MIN_COMPLETION_RATE=100
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
```

## 🚀 Quick Commands

### 1. Run Diagnosis Now

```bash
cd /nvme2/hungdx/code/Lightning-hydra
./scripts/benchmark_py/run_diagnosis.sh
```

### 2. View Results

```bash
# View full log
cat logs/results/*/missing_files_diagnosis_*.log

# View summary only
tail -50 logs/results/*/missing_files_diagnosis_*.log

# Search for specific file
grep "missing_file.wav" logs/results/*/missing_files_diagnosis_*.log
```

### 3. Create Clean Protocol (Remove Missing)

```bash
# Extract only successfully scored files
python scripts/benchmark_py/create_clean_protocol.py \
    --score-file logs/results/.../score.txt \
    --protocol-file data/.../protocol.txt \
    --output data/.../protocol_clean_dev.txt \
    --subset dev
```

## 📝 Log File Locations

After running diagnosis:

```
logs/results/CNSL_Q1_2026_benchmarks_dev/XLSR_ConformerTCM_MDT_RawboostLA_DF/
├── missing_files_diagnosis_20260123_163000.log  # Detailed diagnosis
├── missing_files.txt                            # List of missing file IDs
└── summary_results.txt                          # Benchmark results
```

## 🎯 Decision Tree

```
98.8% completion?
│
├─ For Research/Testing → ✅ Accept (excellent rate!)
│   └─ Document: "98.8% completion, 1.2% missing due to [reasons]"
│
├─ For Production → 🔍 Investigate
│   ├─ Run diagnosis → Find categories
│   ├─ Fix issues → Re-run
│   └─ Verify 100% → Accept
│
└─ Need to understand why → 🔍 Run diagnosis
    ├─ Review log file
    ├─ Check patterns
    └─ Make informed decision
```

## ✅ Success Metrics

| Completion Rate | Status | Action |
|----------------|--------|--------|
| 100% | Perfect | Use as-is |
| 95-99.9% | Excellent | Accept with documentation |
| 90-95% | Good | Accept or investigate |
| 80-90% | Acceptable | Investigate recommended |
| < 80% | Poor | Must investigate |

**Your case: 98.8% = EXCELLENT!** ✅

## 🎊 Conclusion

Với 98.8% completion:
- ✅ **Results đã được evaluate và add vào summary**
- ✅ **Marked as (PARTIAL - 98.8%)** để rõ ràng
- ✅ **98.8% là tỉ lệ rất tốt cho research/testing**
- 🔍 **Chạy diagnosis nếu muốn biết chi tiết 1.2% missing**

---

**Chạy diagnosis ngay:** `./scripts/benchmark_py/run_diagnosis.sh` 🔍
