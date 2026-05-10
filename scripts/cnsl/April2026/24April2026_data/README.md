# Inference


# Create April_2026_benchmark (rsync to RAM)
## veo3

### Server 3
```bash
rsync -avr --info=progress2 /nvme2/hungdx/preprocessors/data/dataprocessedveo3_generation_wav_processed/ /dev/shm/April_2026_benchmark/veo3_hf/
rsync -avr --info=progress2 /nvme2/hungdx/preprocessors/data/dataprocessedveo3_generation_wav_processed_protocols/protocol.txt /dev/shm/April_2026_benchmark/veo3_hf/protocol.txt
```

## kling
```bash
rsync -avr --info=progress2 /nvme2/hungdx/preprocessors/data/downloads_wav_processed/ /dev/shm/April_2026_benchmark/kling-ai/
rsync -avr --info=progress2 /nvme2/hungdx/preprocessors/data/downloads_wav_processed_protocols/protocol.txt /dev/shm/April_2026_benchmark/kling-ai/protocol.txt

```

### Server 6
```bash
rsync -avr --info=progress2 /nvme3/Datasets/April_2026_benchmark /dev/shm/
```

# benchmark old models

## DEC 2024
```bash
DEFAULT_BATCH_SIZE=512 uv run ./scripts/benchmark_py/benchmark.py -g 1 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b /dev/shm/April_2026_benchmark -m pretrained/S_241214_conf-1.pth -r logs/results/April_2026_benchmark -n "S_241214_conf-1" -l false
 ```

## May 2025

```bash
DEFAULT_BATCH_SIZE=512 uv run ./scripts/benchmark_py/benchmark.py -g 1 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs -b /dev/shm/April_2026_benchmark -m pretrained/S_241214_conf-1.pth -a pretrained/MDT_241214_lora_250501 -r logs/results/April_2026_benchmark -n "LoRAMay2025" -l false
 ```

## Feb 2026

```bash
DEFAULT_BATCH_SIZE=256 uv run ./scripts/benchmark_py/benchmark.py -g MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b /dev/shm/April_2026_benchmark -m /NAS1_pretrained_lab/06feb26_xlsr_conformertcm_mdt_vad.pt -r logs/results/April_2026_benchmark -n "06feb26_xlsr_conformertcm_mdt_vad" -l false
 ```

## 21 April 2026

### re-train
```bash
DEFAULT_BATCH_SIZE=128 uv run ./scripts/benchmark_py/benchmark.py -g 1 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/April_2026_benchmark -m /data/hungdx/lighning-hydra-train-runs/runs/2026-04-21_01-46-37/checkpoints/averaged_top5.ckpt -r logs/results/April_2026_benchmark -n "21April26_xlsr_conformertcm_mdt_bf16-mixed" -l true +trainer.precision=bf16-mixed
 ```
### lora
```bash
DEFAULT_BATCH_SIZE=128 uv run ./scripts/benchmark_py/benchmark.py -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/April_2026_benchmark -m /NAS1_pretrained_lab/06feb26_xlsr_conformertcm_mdt_vad.pt -a /data/hungdx/lighning-hydra-train-runs/runs/2026-04-21_01-40-54/checkpoints/epoch_006.ckpt -r logs/results/April_2026_benchmark -n "21April26_xlsr_conformertcm_mdt_lora_from_feb_bf16-mixed" -l false +trainer.precision=bf16-mixed
 ```

 ## April quick check
 ```bash
DEFAULT_BATCH_SIZE=128 uv run ./scripts/benchmark_py/benchmark.py -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/April_2026_benchmark_quick_check -m /data/hungdx/lighning-hydra-train-runs/runs/2026-04-21_01-46-37/checkpoints/averaged_top5.ckpt -r logs/results/April_2026_benchmark_quick_check -n "21April26_xlsr_conformertcm_mdt_bf16-mixed" -l true +trainer.precision=bf16-mixed
 ```
### lora
```bash
DEFAULT_BATCH_SIZE=128 uv run ./scripts/benchmark_py/benchmark.py -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/April_2026_benchmark_quick_check -m /NAS1_pretrained_lab/06feb26_xlsr_conformertcm_mdt_vad.pt -a /data/hungdx/lighning-hydra-train-runs/runs/2026-04-21_01-40-54/checkpoints/epoch_006.ckpt -r logs/results/April_2026_benchmark_quick_check -n "21April26_xlsr_conformertcm_mdt_lora_from_feb_bf16-mixed" -l false +trainer.precision=bf16-mixed
 ```

### re-train avg
```bash
DEFAULT_BATCH_SIZE=128 uv run ./scripts/benchmark_py/benchmark.py -g 1 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/April_2026_benchmark -m /nvme2/hungdx/Lightning-hydra/logs/train/runs/2026-04-24_22-43-39/checkpoints/averaged_top5.ckpt -r logs/results/April_2026_benchmark -n "24April26_xlsr_conformertcm_mdt_bf16-mixed" -l true +trainer.precision=bf16-mixed
 ```

 ### re-train best
```bash
DEFAULT_BATCH_SIZE=128 uv run ./scripts/benchmark_py/benchmark.py -g 1 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/April_2026_benchmark -m /nvme2/hungdx/Lightning-hydra/logs/train/runs/2026-04-24_22-43-39/checkpoints/epoch_000-v1.ckpt -r logs/results/April_2026_benchmark -n "24April26_xlsr_conformertcm_mdt_best_bf16-mixed" -l true +trainer.precision=bf16-mixed
 ```

 ### lora replay
 ```bash
DEFAULT_BATCH_SIZE=128 uv run ./scripts/benchmark_py/benchmark.py -g 1 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/April_2026_benchmark -m /NAS1_pretrained_lab/06feb26_xlsr_conformertcm_mdt_vad.pt -a /data/hungdx/lighning-hydra-train-runs/runs/2026-04-26_16-36-55/checkpoints/epoch_010.ckpt -r logs/results/April_2026_benchmark -n "24April26_xlsr_conformertcm_mdt_lora_replay_from_06feb26_xlsr_conformertcm_mdt_vad_bf16-mixed" -l false +trainer.precision=bf16-mixed
 ```


  ### lora replay conf-2
 ```bash
DEFAULT_BATCH_SIZE=128 uv run ./scripts/benchmark_py/benchmark.py -g 1 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/April_2026_benchmark -m /NAS1_pretrained_lab/06feb26_xlsr_conformertcm_mdt_vad.pt -a /data/hungdx/lighning-hydra-train-runs/runs/2026-04-26_22-17-41/checkpoints/epoch_009.ckpt -r logs/results/April_2026_benchmark -n "24April26_xlsr_conformertcm_mdt_lora_replay-conf-2_from_06feb26_xlsr_conformertcm_mdt_vad_bf16-mixed" -l false +trainer.precision=bf16-mixed
 ```