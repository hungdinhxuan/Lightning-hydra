## Bottleneck debug notes (Thien exp, Apr 2026)

File chạy debug: `scripts/cnsl/thien-exp/2_debug.sh`

### Mục tiêu

- Chỉ ra bottleneck chính dựa trên số liệu profiler (không đoán mò)
- Tách bạch: **DataLoader / I/O** vs **GPU compute** vs **Validation/Checkpoint**

### Setup profiling

- Profiler: Lightning `SimpleProfiler` (`debug=profiler` → `trainer.profiler=simple`)
- Short run: giới hạn số batch để ra report nhanh
- Chạy trên GPU thật (`trainer.accelerator=cuda`, `+trainer.precision=bf16-mixed`)
- Tự chọn GPU rảnh (tránh OOM do GPU bận)

### Kết quả chính (Baseline)

Run: `num_workers=16`, `pin_memory=true`, `augmentation_methods=[RawBoost12, none]`, `limit_train_batches=100`, `limit_val_batches=20`

Trích từ `FIT Profiler Report`:

- **`run_training_batch`**: mean ~0.394s × 100 → ~39.39s (**51.3%** tổng thời gian)
- **`optimizer_step`**: ~39.38s (**51.29%**)
- **`training_step`**: mean ~0.201s × 100 → ~20.08s (**26.15%**)
- **`backward`**: mean ~0.187s × 100 → ~18.69s (**24.34%**)
- **`train_dataloader_next`**: mean ~0.0737s × 100 → ~7.37s (**9.61%**)
- **Validation loop (`val_next`)**: ~8.29s (**10.80%**)
- **Checkpoint save**: ~3.57s (**4.65%**)
- **`batch_to_device`**: ~0.072s (**0.094%**) → **không** phải bottleneck

**Diễn giải**:

- Tổng quan: pipeline hiện tại **compute-heavy** (forward/backward/optimizer_step chiếm phần lớn).
- Nhưng **DataLoader vẫn đáng kể** (~10% wall time) → có dư địa tối ưu data path.
- Overhead validation + checkpoint trong debug run chiếm thêm ~15% tổng thời gian (tùy tần suất val).

### A/B test 1: Tắt augmentation (chỉ `"none"`)

Run: `augmentation_methods=["none"]`, `num_workers=16`, `limit_train_batches=50`, `limit_val_batches=10`

Trích profiler:

- **`train_dataloader_next`**: mean ~0.129s × 50 → ~6.48s (**12.64%**)
- **`training_step`**: mean ~0.190s × 50 → ~9.51s (**18.56%**)
- **`backward`**: mean ~0.189s × 50 → ~9.44s (**18.43%**)

**Kết luận tạm thời**:

- Việc “tắt augmentation” **không làm DataLoader nhanh hơn** trong run này (thậm chí `train_dataloader_next` cao hơn).
- Điều này gợi ý: bottleneck data path không chỉ nằm ở `RawBoost12` (hoặc RawBoost đang không phải chi phí lớn so với decode/pad/multi-view, hoặc có cost khác như IO/decode/numba warmup).

### A/B test 2: `num_workers=0` (single-process dataloading)

Ban đầu test này bị lỗi vì `train_dataloader()` hardcode `persistent_workers=True`.
Đã fix trong `src/data/normal_multiview_datamodule.py`:

- `persistent_workers=True if num_workers > 0 else False`

Sau khi fix, run lại `num_workers=0` cho thấy:

- **`train_dataloader_next`**: mean ~2.562s × 50 → **128.11s (72.2%)**
- **`training_step`**: mean ~0.161s × 50 → ~8.07s (4.55%)
- **`backward`**: mean ~0.185s × 50 → ~9.24s (5.21%)

**Kết luận chắc chắn**:

- Nếu thiếu worker, pipeline bị **input-bound** cực mạnh.
- Vì vậy `num_workers`/`pin_memory`/prefetch là nút điều chỉnh quan trọng; DataLoader hoàn toàn có thể “ăn” phần lớn thời gian nếu cấu hình sai.

### Bottleneck hiện tại là gì?

Theo baseline (cấu hình hợp lý với workers):

1. **Compute / training step** là bottleneck chính (forward/backward/optimizer_step).
2. **Data pipeline** là bottleneck phụ đáng kể (≈10%); vẫn nên tối ưu vì dễ “mất GPU util” khi data chậm.
3. **Validation + checkpoint** là overhead không nhỏ trong debug mode; cần kiểm soát tần suất khi profiling.

### Khuyến nghị tối ưu (ưu tiên theo tác động)

- **Giữ `num_workers > 0`** và dùng `persistent_workers=true` (hiện đã an toàn khi `num_workers=0`).
- **Tối ưu decode/load audio**:
  - `load_audio()` đang dùng `librosa.load()` (CPU + resample). Nếu audio đã 16kHz, cân nhắc load nhanh hơn (ví dụ `soundfile`/`torchaudio`) hoặc cache `.npy`.
  - Nếu dataset đã được đặt trên **`/dev/shm`** và bạn không muốn tốn thêm CPU/đĩa cho việc ghi `.npy`, thì **không cần bật cache**. Lúc này ưu tiên là đảm bảo pipeline *thật sự* đọc từ `/dev/shm` (không fallback sang path khác) và tune DataLoader (`num_workers/pin_memory/prefetch`).
- **Giảm chi phí multi-view**:
  - `collate_fn` hiện pad/stack cho 4 views mỗi sample → CPU/memory traffic cao.
  - Nếu mục tiêu profiling compute, thử giảm views tạm thời (chỉ để đo compute path), hoặc tối ưu collate/pad (vector hóa/torch ops).
- **Kiểm soát overhead validation/checkpoint khi profiling**:
  - Tăng `val_check_interval` hoặc giới hạn `limit_val_batches` nhỏ khi chỉ cần step-time.
  - Tắt checkpoint trong profiling micro-run nếu không cần.

### Next steps (đề xuất)

- Chạy thêm A/B: `num_workers=8/16/24` để tìm “sweet spot” của máy.
- Vì dataset đã ở `/dev/shm`, bỏ qua A/B “bật cache”; tập trung vào `num_workers/pin_memory/batch_size` và giảm overhead validation/checkpoint khi đo step-time.
- Nếu vẫn cần bóc sâu: dùng `torch.profiler` 50–100 step để tách rõ: `load_audio` / `pad` / model forward/backward.

### Gợi ý chạy A/B nhanh (không cần sửa file)

`2_debug.sh` hỗ trợ override qua biến môi trường:

- `NUM_WORKERS` (mặc định 16)
- `PIN_MEMORY` (mặc định `true`)
- `BATCH_SIZE` (mặc định 16)
- `LIMIT_TRAIN_BATCHES`, `LIMIT_VAL_BATCHES`, `VAL_CHECK_INTERVAL`, `OMP_THREADS`

Ví dụ:

- Sweep DataLoader workers:
  - `NUM_WORKERS=8  bash scripts/cnsl/thien-exp/2_debug.sh`
  - `NUM_WORKERS=16 bash scripts/cnsl/thien-exp/2_debug.sh`
  - `NUM_WORKERS=24 bash scripts/cnsl/thien-exp/2_debug.sh`
- Test pin_memory:
  - `PIN_MEMORY=false bash scripts/cnsl/thien-exp/2_debug.sh`
  - `PIN_MEMORY=true  bash scripts/cnsl/thien-exp/2_debug.sh`
- Test batch size (compute-bound check):
  - `BATCH_SIZE=16 bash scripts/cnsl/thien-exp/2_debug.sh`
  - `BATCH_SIZE=20 bash scripts/cnsl/thien-exp/2_debug.sh`
  - `BATCH_SIZE=24 bash scripts/cnsl/thien-exp/2_debug.sh`

