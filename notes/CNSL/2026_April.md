

# Create April_2026_benchmark (rsync to RAM)
## veo3
```bash
rsync -avr --info=progress2 /nvme2/hungdx/preprocessors/data/dataprocessedveo3_generation_wav_processed/ /dev/shm/April_2026_benchmark/veo3_hf/
rsync -avr --info=progress2 /nvme2/hungdx/preprocessors/data/dataprocessedveo3_generation_wav_processed_protocols/protocol.txt /dev/shm/April_2026_benchmark/veo3_hf/protocol.txt
```

## kling
```bash
rsync -avr --info=progress2 /nvme2/hungdx/preprocessors/data/downloads_wav_processed/ /dev/shm/April_2026_benchmark/kling-ai/
rsync -avr --info=progress2 /nvme2/hungdx/preprocessors/data/downloads_wav_processed_protocols/protocol.txt /dev/shm/April_2026_benchmark/kling-ai/protocol.txt

```

# benchmark old models

## DEC 2024
```bash
 DEFAULT_BATCH_SIZE=196 uv run ./scripts/benchmark_py/benchmark.py -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b /dev/shm/April_2026_benchmark -m /data/hungdx/lighning-hydra-train-runs/runs/2026-02-24_23-46-15/checkpoints/averaged_top5.ckpt -r logs/results/Kipot_benchmark -n "26feb26_xlsr_conformertcm_mdt"
 ```

## May 2025

```bash
 DEFAULT_BATCH_SIZE=196 uv run ./scripts/benchmark_py/benchmark.py -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b /dev/shm/April_2026_benchmark -m /data/hungdx/lighning-hydra-train-runs/runs/2026-02-24_23-46-15/checkpoints/averaged_top5.ckpt -r logs/results/Kipot_benchmark -n "26feb26_xlsr_conformertcm_mdt"
 ```

## Feb 2026

```bash
DEFAULT_BATCH_SIZE=128 uv run ./scripts/benchmark_py/benchmark.py -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b /dev/shm/April_2026_benchmark -m /NAS1_pretrained_lab/06feb26_xlsr_conformertcm_mdt_vad.pt -r logs/results/April_2026_benchmark -n "06feb26_xlsr_conformertcm_mdt_vad" -l false
 ```