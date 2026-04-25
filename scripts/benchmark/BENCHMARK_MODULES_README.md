# Benchmark Script Modules

Script `benchmark.sh` đã được refactor thành các module nhỏ để dễ bảo trì và tái sử dụng.

## Cấu trúc Module

### 0. `benchmark_constants.sh` ⭐ NEW
**Chức năng:** Lưu trữ các constants có thể cấu hình
- Protocol subset configuration (thay vì hard-code 'eval')
- Default values cho các parameters
- File naming patterns
- Temporary file patterns

**Constants:**
- `PROTOCOL_SUBSET` - Subset name để filter protocol (default: "eval", có thể set qua environment variable)
- `DEFAULT_BATCH_SIZE` - Default batch size (default: 64)
- `DEFAULT_TRIM_LENGTH` - Default trim length (default: 64000)
- `DEFAULT_IS_BASE_MODEL_PATH_LN` - Default Lightning checkpoint loading (default: true)
- `DEFAULT_IS_RANDOM_START` - Default random start (default: true)
- `PROGRESS_BAR_WIDTH` - Progress bar width (default: 50)
- Và nhiều constants khác cho file naming patterns

**Functions:**
- `should_use_protocol_subset(protocol_file)` - Check xem có nên dùng subset không
- `get_protocol_subset_name()` - Get subset name để display

**Cách sử dụng:**
```bash
# Set protocol subset qua environment variable trước khi chạy script
export PROTOCOL_SUBSET="test"  # Thay vì "eval"
./scripts/benchmark/benchmark.sh -g 0 -c config -b folder -m model -r results -n comment

# Hoặc sửa trực tiếp trong file benchmark_constants.sh
```

### 1. `benchmark_utils.sh`
**Chức năng:** Utility functions cơ bản
- Color definitions và print functions
- Progress bar display
- Banner và usage information
- Spinner cho long-running commands
- Cleanup functions

**Functions:**
- `print_color(color, text)` - In text với màu
- `display_progress(current, total)` - Hiển thị progress bar
- `show_usage()` - Hiển thị usage information
- `print_banner()` - In banner
- `run_with_spinner(cmd, description)` - Chạy command với spinner
- `cleanup_temp_files(results_folder)` - Dọn dẹp temp files

### 2. `benchmark_config.sh`
**Chức năng:** Configuration và argument parsing
- Parse command line arguments
- Validate required arguments
- Set default values
- Initialize results directory và summary file

**Functions:**
- `parse_arguments()` - Parse command line arguments
- `validate_arguments()` - Validate required arguments
- `set_defaults()` - Set default values cho optional parameters
- `initialize_results()` - Khởi tạo results directory và summary file

### 3. `benchmark_validation.sh`
**Chức năng:** Score file validation
- Validate score file completeness
- Check số lượng entries

**Functions:**
- `validate_score_file(score_file, protocol_file)` - Validate score file

### 4. `benchmark_protocol.sh`
**Chức năng:** Protocol file operations
- Tạo temporary protocol files cho missing entries
- Extract protocol subset từ protocol (sử dụng `PROTOCOL_SUBSET` constant)

**Functions:**
- `create_missing_protocol(score_file, protocol_file, temp_protocol)` - Tạo protocol cho missing entries
- `extract_eval_subset(protocol_file, output_file)` - Extract protocol subset (tên function giữ nguyên để backward compatible)

### 5. `benchmark_scores.sh`
**Chức năng:** Score file merging
- Merge score files từ multiple runs
- Handle paths với spaces

**Functions:**
- `merge_score_files(original_score, new_score, merged_score)` - Merge score files

### 6. `benchmark_execution.sh`
**Chức năng:** Benchmark execution
- Construct benchmark command
- Execute benchmark với spinner
- Evaluate results và extract metrics

**Functions:**
- `construct_benchmark_command(...)` - Tạo command string
- `execute_benchmark(cmd)` - Execute benchmark
- `evaluate_results(score_file, protocol_file, summary_file, dataset_name)` - Evaluate và extract metrics

### 7. `benchmark_eer.sh`
**Chức năng:** EER calculations
- Calculate pooled EER
- Calculate average EER

**Functions:**
- `calculate_pooled_eer(results_folder, normalized_yaml, comment, summary_file, subdirs...)` - Calculate pooled EER
- `calculate_average_eer(summary_file)` - Calculate average EER

### 8. `benchmark_merge.sh`
**Chức năng:** Merged protocol creation
- Tạo merged protocol và score files
- Generate metadata files

**Functions:**
- `create_merged_protocol(results_folder, normalized_yaml, comment, yaml_config, base_model_path, summary_file, subdirs...)` - Tạo merged files

### 9. `benchmark.sh` (Main Script)
**Chức năng:** Main orchestration script
- Import tất cả modules
- Orchestrate toàn bộ benchmark process
- Handle main loop cho từng dataset

## Cách sử dụng

Script chính vẫn được sử dụng như cũ:

```bash
./scripts/benchmark.sh \
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

## Lợi ích của modular structure

1. **Dễ bảo trì:** Mỗi module có trách nhiệm rõ ràng
2. **Tái sử dụng:** Các functions có thể được sử dụng trong scripts khác
3. **Dễ test:** Có thể test từng module riêng biệt
4. **Dễ đọc:** Main script ngắn gọn và dễ hiểu flow
5. **Dễ mở rộng:** Thêm features mới chỉ cần thêm/sửa module liên quan

## Dependencies

Các modules được source theo thứ tự:
1. `benchmark_constants.sh` - Constants (được source đầu tiên trong main script)
2. `benchmark_utils.sh` - Base utilities (được source bởi tất cả modules khác)
3. `benchmark_config.sh` - Config (source utils và constants)
4. `benchmark_validation.sh` - Validation (source utils và constants)
5. `benchmark_protocol.sh` - Protocol (source utils và constants)
6. `benchmark_scores.sh` - Scores (source utils)
7. `benchmark_execution.sh` - Execution (source utils và constants)
8. `benchmark_eer.sh` - EER (source utils)
9. `benchmark_merge.sh` - Merge (source utils và protocol)

## Configuration via Constants

Thay vì hard-code các giá trị, bạn có thể cấu hình qua file `benchmark_constants.sh`:

### Thay đổi Protocol Subset

**Cách 1: Sửa file constants**
```bash
# Edit benchmark_constants.sh
PROTOCOL_SUBSET="test"  # Thay vì "eval"
```

**Cách 2: Set environment variable**
```bash
export PROTOCOL_SUBSET="test"
./scripts/benchmark/benchmark.sh ...
```

### Thay đổi Default Values

```bash
# Trong benchmark_constants.sh
DEFAULT_BATCH_SIZE=128
DEFAULT_TRIM_LENGTH=32000
PROGRESS_BAR_WIDTH=60
```

## Notes

- Tất cả modules đều có shebang `#!/bin/bash`
- Các modules sử dụng `SCRIPT_DIR` để source dependencies
- Variables được pass giữa functions thông qua parameters
- Arrays được pass trực tiếp trong main script để tránh complexity
- Constants có thể được override qua environment variables (sử dụng `${VAR:-default}` syntax)