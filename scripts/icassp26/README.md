# Exp1 (exp1) Single lora
- aasist_ssl_exp1.sh


+ eval
./scripts/benchmark.sh -g 2 -c icassp26/aasist_ssl/xlsr_aasist_single_lora -b data/ICASSP25_benchmark_noise -m /nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth -a /home/hungdx/logs/train/runs/2025-09-10_11-31-31/checkpoints/epoch_009.ckpt -r logs/results/ICASSP25_benchmark_noise -n "xlsr_aasist_single_lora_datasmall_correct" -l false

- conformertcm_exp1.sh
./scripts/benchmark.sh -g 3 -c icassp26/conformertcm/xlsr_conformertcm_single_lora -b data/ICASSP25_benchmark_noise -m /nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth -a /home/hungdx/logs/train/runs/2025-09-10_11-31-33/checkpoints/epoch_006.ckpt -r logs/results/ICASSP25_benchmark_noise -n "xlsr_conformertcm_single_lora_correct" -l false


# Dynamic LoRa

## Background_music_noise (g1)

## Autotune

## Bandpass-filter

## Echo

## manipulation

## Gaussian Noise

## Reverberation (g7)