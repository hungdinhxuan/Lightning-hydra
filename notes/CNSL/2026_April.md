

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
DEFAULT_BATCH_SIZE=512 uv run ./scripts/benchmark_py/benchmark.py -g 1 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b /dev/shm/April_2026_benchmark -m /NAS1_pretrained_lab/06feb26_xlsr_conformertcm_mdt_vad.pt -r logs/results/April_2026_benchmark -n "06feb26_xlsr_conformertcm_mdt_vad" -l false
 ```
