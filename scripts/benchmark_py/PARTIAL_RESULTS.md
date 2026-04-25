# Partial Results Feature

## 🎯 Vấn đề

Khi benchmark một dataset lớn, có thể một số files không xử lý được do:
- Audio files bị corrupted
- Audio files missing
- Format không hỗ trợ
- Out of memory errors
- Timeout issues

Điều này dẫn đến score file **incomplete** (thiếu một số entries).

## ✅ Giải pháp: Partial Results

Script giờ có thể **accept và evaluate partial results** nếu:
1. Đã cố gắng resume (run missing entries)
2. Completion rate >= minimum threshold (default: 95%)

## 🔧 Cách hoạt động

### Kịch bản 1: Completion rate >= 95% (Accept)

```bash
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]

# Output:
⚠️ Warning: Score file is incomplete for dataset
  Score lines: 66817/85128 (78.5%)
  Missing: 18311 lines
  
📝 Note: Just attempted resume but still incomplete
  This likely means some files could not be processed
  
❌ Completion rate (78.5%) < minimum threshold (95.0%)
  To accept partial results, set: export MIN_COMPLETION_RATE=78
```

### Kịch bản 2: Với MIN_COMPLETION_RATE thấp hơn

```bash
# Accept nếu có >= 78% completion
export MIN_COMPLETION_RATE=78
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]

# Output:
⚠️ Warning: Score file is incomplete for dataset
  Score lines: 66817/85128 (78.5%)
  
✓ Completion rate (78.5%) >= minimum threshold (78.0%)
💡 Evaluating with 66817 available samples...
⚠️ Results for dataset (PARTIAL - 78.5% complete)
```

### Kịch bản 3: Completion rate >= 95% (Auto-accept)

```bash
# Nếu >= 95%, tự động accept
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]

# Output:
⚠️ Warning: Score file is incomplete for dataset
  Score lines: 81072/85128 (95.2%)
  
✓ Completion rate (95.2%) >= minimum threshold (95.0%)
💡 Evaluating with 81072 available samples...
⚠️ Results for dataset (PARTIAL - 95.2% complete)
```

## ⚙️ Configuration

### Environment Variable

```bash
# Default: 95%
export MIN_COMPLETION_RATE=95

# Chấp nhận partial results với 80% completion
export MIN_COMPLETION_RATE=80

# Chỉ chấp nhận 100% complete (strict mode)
export MIN_COMPLETION_RATE=100

# Chấp nhận bất kỳ % nào (not recommended)
export MIN_COMPLETION_RATE=0
```

### Trong Code

```python
from benchmark_py.constants import CONSTANTS

# Check current setting
print(CONSTANTS.min_completion_rate)  # 95.0

# Set programmatically (before running benchmark)
CONSTANTS.min_completion_rate = 80.0
```

## 📊 Use Cases

### Use Case 1: Large Dataset với một số files corrupted

```bash
# Dataset có 100K files, 5% corrupted (5K files)
# Completion: 95K/100K = 95%

# Default settings sẽ accept
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# ✓ Auto-accepted (95% >= 95%)
```

### Use Case 2: Dataset với nhiều files lỗi

```bash
# Dataset có 100K files, 25% lỗi
# Completion: 75K/100K = 75%

# Default settings sẽ reject
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# ❌ Rejected (75% < 95%)

# Nếu muốn accept:
export MIN_COMPLETION_RATE=75
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# ✓ Accepted (75% >= 75%)
```

### Use Case 3: Production (Strict mode)

```bash
# Trong production, chỉ accept 100% complete
export MIN_COMPLETION_RATE=100
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# ❌ Rejected nếu thiếu bất kỳ file nào
```

## 🎯 Best Practices

### 1. Development (Relaxed)

```bash
# Accept 90% để test nhanh
export MIN_COMPLETION_RATE=90
```

### 2. Research (Balanced)

```bash
# Accept 95% (default) - reasonable cho research
export MIN_COMPLETION_RATE=95  # or không set (default)
```

### 3. Production (Strict)

```bash
# Chỉ accept 100% complete
export MIN_COMPLETION_RATE=100
```

## 📝 Trong Summary File

Partial results sẽ được đánh dấu rõ ràng:

```
Dataset | EER | min_score | max_score | Threshold | Accuracy
dataset1 | 21.958151 | -6.526936 | 6.579559 | 1.112642 | 75.596797
dataset2 (PARTIAL-95.2%) | 22.123456 | -6.234567 | 6.345678 | 1.234567 | 76.123456
dataset3 | 20.345678 | -5.987654 | 6.123456 | 1.098765 | 77.654321
```

**Note**: Partial results sẽ được add comment trong summary file để dễ identify.

## ⚠️ Lưu ý

### 1. Impact lên Results

Partial results có thể ảnh hưởng đến:
- **EER**: Có thể cao hoặc thấp hơn nếu missing files có distribution khác
- **Pooled EER**: Sẽ bị ảnh hưởng bởi missing data
- **Statistical significance**: Giảm với ít samples hơn

### 2. Khi nào nên accept partial?

✅ **Nên accept khi**:
- Missing files là random (không có bias)
- Completion rate >= 95%
- Chỉ dùng để preliminary testing
- Biết rõ nguyên nhân của missing files

❌ **Không nên accept khi**:
- Missing files có pattern (ví dụ: chỉ missing spoof files)
- Completion rate < 90%
- Cần results chính thức cho paper
- Không rõ tại sao files missing

### 3. Debugging Missing Files

Để biết files nào bị missing, check logs:

```bash
# Run và save logs
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args] 2>&1 | tee benchmark.log

# Check for errors
grep -i "error\|failed\|corrupted" benchmark.log

# Check missing files
# Compare protocol.txt với score file để tìm missing files
```

## 🔍 Troubleshooting

### Issue 1: Tại sao completion rate thấp?

```bash
# Debug tool
PROTOCOL_SUBSET=dev python scripts/benchmark_py/debug_validation.py \
    score_file.txt protocol.txt

# Sẽ hiển thị:
# - Total protocol lines
# - Score file lines
# - Missing count
# - First few entries
```

### Issue 2: Làm sao biết files nào missing?

```python
# Script to find missing files
import sys

protocol_file = sys.argv[1]
score_file = sys.argv[2]

# Read protocol file IDs
protocol_ids = set()
with open(protocol_file, 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            file_id = line.split()[0]
            protocol_ids.add(file_id)

# Read score file IDs
score_ids = set()
with open(score_file, 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            file_id = line.split()[0]
            score_ids.add(file_id)

# Find missing
missing = protocol_ids - score_ids
print(f"Missing {len(missing)} files:")
for file_id in sorted(missing)[:10]:  # Show first 10
    print(f"  {file_id}")
```

### Issue 3: Re-run vẫn không complete

Nếu sau nhiều lần re-run vẫn thiếu cùng một số files:
1. Files đó thực sự có vấn đề (corrupted, missing, etc.)
2. Không thể xử lý được
3. Nên accept partial results hoặc fix files

## 📚 Summary

| Setting | When to Use | Accept Rate |
|---------|------------|-------------|
| `MIN_COMPLETION_RATE=100` | Production, paper results | 100% only |
| `MIN_COMPLETION_RATE=95` (default) | Research, normal use | >= 95% |
| `MIN_COMPLETION_RATE=90` | Development, testing | >= 90% |
| `MIN_COMPLETION_RATE=80` | Quick tests, debugging | >= 80% |
| `MIN_COMPLETION_RATE=0` | Accept anything (not recommended) | Any % |

**Recommendation**: Giữ default (95%) cho hầu hết các use cases.

## 🎉 Example Workflow

```bash
# Step 1: Run benchmark
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# Một số datasets incomplete

# Step 2: Check completion rate
# Nếu >= 95%, được accept tự động
# Nếu < 95%, bị reject

# Step 3: Nếu muốn accept lower completion rate
export MIN_COMPLETION_RATE=80
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# Sẽ accept datasets với >= 80% completion

# Step 4: Check results
cat results/summary_results.txt
# Sẽ thấy (PARTIAL-xx.x%) cho các datasets incomplete
```

---

**Tip**: Luôn document completion rate trong báo cáo results! 📝
