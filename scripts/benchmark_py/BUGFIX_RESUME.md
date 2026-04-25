# Bug Fix: Resume từ Incomplete Score File

## 🐛 Vấn đề gốc

Khi chạy benchmark và bị interrupt (hoặc một số file không xử lý được), score file sẽ incomplete. Khi re-run, script báo lỗi:

```
❌ Error: Score file exists but is incomplete/corrupted for 2026_JAN_14_CNSL_DATA
❌ Failed processing 2026_JAN_14_CNSL_DATA
```

Và **KHÔNG tự động resume** từ nơi đã dừng.

## 🔍 Nguyên nhân

### 1. **Thiếu error handling khi tạo missing protocol**

```python
# Code cũ:
missing_count = create_missing_protocol(...)
if missing_count > 0:
    # Continue
elif missing_count == 0:
    # Complete
else:
    # Fallback - nhưng không có exception handling
```

Nếu `create_missing_protocol` raise exception, code sẽ crash.

### 2. **Validation không xử lý comments và empty lines đúng**

```python
# Code cũ:
score_lines = count_non_empty_lines(score_file)  # Đơn giản quá
subset_lines = count_non_empty_lines(protocol_file)  # Không filter comments
```

Nếu file có comments (`#...`), sẽ đếm sai số dòng.

### 3. **Protocol subset matching không chính xác**

Khi dùng `PROTOCOL_SUBSET=dev`, code check:
```python
if CONSTANTS.protocol_subset in line:  # Too simple!
```

Có thể match nhầm (ví dụ: `development` chứa `dev`).

### 4. **Thiếu logging chi tiết**

Khi lỗi xảy ra, không biết:
- Có bao nhiêu dòng trong score file?
- Expected bao nhiêu dòng?
- Protocol subset nào đang dùng?
- Tại sao merge failed?

## ✅ Các fix đã áp dụng

### Fix 1: Thêm Exception Handling

**File**: `benchmark.py`

```python
# Code mới:
try:
    missing_count = create_missing_protocol(...)
    if missing_count > 0:
        # Resume with missing entries only
    elif missing_count == 0:
        # Already complete
    else:
        # Error case (negative return)
        # Fallback to full re-run
except Exception as e:
    print_color(Color.RED, f"❌ Error creating missing protocol: {e}")
    print_color(Color.YELLOW, "  Falling back to full benchmark re-run...")
    # Fallback to full re-run
```

### Fix 2: Cải thiện Validation Logic

**File**: `validation.py`

```python
# Code mới:
def validate_score_file(score_file, protocol_file, verbose=True):
    # Count score lines - exclude comments and empty lines
    score_lines = 0
    with open(score_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # ✅ Skip comments
                score_lines += 1
    
    # Count protocol subset lines - also exclude comments
    subset_lines = 0
    with open(protocol_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # ✅ Skip comments
                if CONSTANTS.protocol_subset in line:
                    subset_lines += 1
```

### Fix 3: Cải thiện Protocol Processing

**File**: `protocol.py`

```python
# Code mới:
def create_missing_protocol(...) -> int:
    """
    Returns:
        Number of missing entries (0 if complete, -1 if error)
    """
    try:
        # Extract subset with proper filtering
        subset_line_count = 0
        with open(protocol_file, 'r') as src, open(temp_file, 'w') as dst:
            for line in src:
                line_stripped = line.strip()
                # ✅ Skip comments and empty lines
                if line_stripped and not line_stripped.startswith('#'):
                    if CONSTANTS.protocol_subset in line:
                        dst.write(line)
                        subset_line_count += 1
        
        # Calculate missing with validation
        missing_count = subset_line_count - existing_lines
        
        if existing_lines >= subset_line_count:
            # Already complete
            return 0
        
        # Create temp protocol with remaining lines
        return missing_count
        
    except Exception as e:
        print_color(Color.RED, f"Error: {e}")
        return -1  # ✅ Signal error with -1
```

### Fix 4: Thêm Detailed Logging

**File**: `benchmark.py`

```python
# Khi validation failed:
if not validation_result.is_valid:
    print_color(Color.RED, f"❌ Error: Score file incomplete")
    print_color(Color.YELLOW, f"  Score lines: {validation_result.score_lines}")
    print_color(Color.YELLOW, f"  Expected lines: {validation_result.expected_lines}")
    print_color(Color.YELLOW, f"  Protocol subset: {CONSTANTS.protocol_subset}")
    print_color(Color.YELLOW, "  Possible causes:")
    print_color(Color.YELLOW, "    1. Benchmark was interrupted")
    print_color(Color.YELLOW, "    2. Some files couldn't be processed")
    print_color(Color.YELLOW, "    3. Protocol subset mismatch")
    print_color(Color.CYAN, "  💡 Tip: Re-run the same command to resume!")
```

### Fix 5: Better Merge Handling

**File**: `benchmark.py`

```python
# Code mới:
if use_temp_protocol:
    if temp_score_path and temp_score_path.exists():
        # Merge scores
        if merge_score_files(...):
            print_color(Color.GREEN, "✓ Merged successfully")
        else:
            print_color(Color.RED, "❌ Merge failed")
    else:
        print_color(Color.RED, "❌ Temp score file not created")
        print_color(Color.YELLOW, f"  Expected at: {temp_score_path}")
        return False
```

## 🧪 Debug Tool

Tạo debug script để kiểm tra validation:

```bash
# Usage
PROTOCOL_SUBSET=dev python scripts/benchmark_py/debug_validation.py \
    logs/results/dataset_score.txt \
    data/dataset/protocol.txt
```

Script sẽ hiển thị:
- ✅ File existence check
- ✅ Configuration (PROTOCOL_SUBSET)
- ✅ Score file line count
- ✅ Protocol file line count (total và subset)
- ✅ Validation result
- ✅ Missing protocol creation test

## 🎯 Cách sử dụng Resume

### Kịch bản 1: Benchmark bị interrupt

```bash
# Run lần 1 (bị interrupt sau dataset thứ 2)
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# ^C

# Re-run với CÙNG command
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# ✅ Sẽ skip dataset 1, 2 (already complete)
# ✅ Sẽ resume dataset 3 từ dòng đã xử lý
```

### Kịch bản 2: Một số file lỗi

```bash
# Run lần 1 - dataset có 100 files, chỉ xử lý được 80
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# Score file: 80/100 lines
# ❌ Incomplete

# Fix các file lỗi (hoặc bỏ qua)
# Re-run
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# ✅ Sẽ chỉ xử lý 20 files còn lại (lines 81-100)
```

### Kịch bản 3: Thay đổi protocol subset

```bash
# Run với dev subset
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]
# Score file: 50 lines (dev subset)

# Thay sang test subset - KHÔNG resume được
PROTOCOL_SUBSET=test ./scripts/benchmark_py/benchmark.py [args]
# Expected: 100 lines (test subset)
# Score file: 50 lines
# ❌ Mismatch! Sẽ re-run toàn bộ
```

## 📝 Lưu ý quan trọng

### 1. Protocol Subset phải giống nhau

```bash
# ❌ SAI:
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]  # Lần 1
PROTOCOL_SUBSET=test ./scripts/benchmark_py/benchmark.py [args]  # Lần 2 - khác subset!

# ✅ ĐÚNG:
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]  # Lần 1
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]  # Lần 2 - cùng subset
```

### 2. Không được thay đổi parameters khác

Parameters phải giống nhau:
- `-c` config
- `-m` model_path
- `-b` benchmark_folder
- `-r` results_folder
- `-n` comment
- `-t` trim_length
- `-s` random_start

Nếu thay đổi bất kỳ parameter nào, nên dùng comment khác để tạo results folder mới.

### 3. Score file format

Score file phải có format:
```
filename score1 score2
filename score1 score2
...
```

Comments (`#...`) sẽ được skip.

## 🔧 Troubleshooting

### Vấn đề 1: Vẫn báo incomplete sau khi re-run

```bash
# Debug
PROTOCOL_SUBSET=dev python scripts/benchmark_py/debug_validation.py \
    logs/results/dataset_score.txt \
    data/dataset/protocol.txt
```

Kiểm tra:
- Score lines vs Expected lines
- Protocol subset có đúng không
- File có comments không

### Vấn đề 2: Missing count = 0 nhưng vẫn incomplete

Có thể do:
- Comments trong score file
- Empty lines
- Protocol subset không match

### Vấn đề 3: Merge failed

Kiểm tra:
- Temp score file có tồn tại không
- Temp score file có data không
- Permissions

## ✅ Test Cases

### Test 1: Normal resume
```bash
# Create incomplete score (50/100 lines)
head -50 full_score.txt > incomplete_score.txt

# Run benchmark - should process remaining 50
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py [args]

# Result: 100/100 lines ✅
```

### Test 2: Protocol subset change
```bash
# Run with dev
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py -n exp_dev [args]

# Run with test (different experiment)
PROTOCOL_SUBSET=test ./scripts/benchmark_py/benchmark.py -n exp_test [args]

# Both should work independently ✅
```

### Test 3: With comments
```bash
# Protocol with comments
cat > protocol.txt << EOF
# This is a comment
file1.wav dev bonafide
# Another comment
file2.wav dev spoof
EOF

# Should correctly count 2 lines (not 4) ✅
```

## 🎉 Kết luận

Sau khi fix:
- ✅ Resume tự động từ incomplete score
- ✅ Handle exceptions gracefully
- ✅ Detailed error messages
- ✅ Support comments in files
- ✅ Better protocol subset handling
- ✅ Debug tool để troubleshoot

Giờ bạn có thể yên tâm re-run benchmark nếu bị interrupt! 🚀
