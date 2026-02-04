# MDT training (1%)
```bash
CUDA_VISIBLE_DEVICES='MIG-6e4275af-2db0-51f1-a601-7ad8a1002745' OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/xlsr_conformertcm_mdt ++model_averaging=True +model.score_save_path="logs/eval/wildspoof/spoofceleb_eval_xlsr_conformertcm_MDT_large_corpus_clean_4s.txt" +trainer.limit_train_batches=0.01
```

- Average ckpt: /nvme1/hungdx/logs/train/runs/2025-09-23_21-57-59/checkpoints/averaged_top5.ckpt

- eval code
```bash
./scripts/benchmark_old.sh -g MIG-6e4275af-2db0-51f1-a601-7ad8a1002745 -c wildspoof/xlsr_conformertcm_mdt -b data/benchmark_kd -m /nvme1/hungdx/logs/train/runs/2025-09-23_21-57-59/checkpoints/averaged_top5.ckpt -r logs/results/benchmark_wildspoof -n "xlsr_conformertcm_mdt"
```

```bash
./scripts/benchmark_old.sh -g 2 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme1/hungdx/pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_rawboost3_spoofceleb"
```


# MDT training (5%)
```bash
CUDA_VISIBLE_DEVICES='MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494' OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/xlsr_conformertcm_mdt ++model_averaging=True +model.score_save_path="logs/eval/wildspoof/spoofceleb_eval_xlsr_conformertcm_MDT_large_corpus_clean_4s.txt" +trainer.limit_train_batches=0.05
```

# Variable-length training

=================================================== Lastest update (Oct-5-2025) ===================================================
========================================Merge dataset from Voxceleb2 + SpoofCeleb (Balacned classes training) ====================

# MDT training (Rawboostfull)
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/xlsr_conformertcm_mdt ++model_averaging=True +model.score_save_path="logs/eval/wildspoof/spoofceleb_eval_xlsr_conformertcm_MDT_large_corpus_clean_4s.txt"
```

Time for training: from 2025-10-04 18:08 to 2025-10-06 04:11:10 (~26h)
Average checkpoint: /home/hungdx/logs/train/runs/2025-10-04_18-08-07/checkpoints/averaged_top5.ckpt
- eval code
```bash
./scripts/benchmark_old.sh -g 0 -c wildspoof/xlsr_conformertcm_mdt -b data/benchmark_kd -m /home/hungdx/logs/train/runs/2025-10-04_18-08-07/checkpoints/averaged_top5.ckpt -r logs/results/benchmark_wildspoof -n "xlsr_conformertcm_mdt"
```

# Standard training (no rawboost)
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/xlsr_conformertcm_wo_rawboost ++model_averaging=True +model.score_save_path="logs/eval/wildspoof/spoofceleb_eval_xlsr_conformertcm_wo_rawboost_large_corpus_clean_4s.txt"
```
./scripts/benchmark_old.sh -g 3 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme1/hungdx/pretrained/xlsr_conformertcm_mdt_rawboostfull_wildspoof-noisevoxceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_rawboostfull_voxcelebaug-spoofceleb"
```

# MDT training (Rawboost3)
Time for training: from 2025-10-07 03:17 to 2025-10-09 11:28 (~55h)
Average checkpoint: /nvme1/hungdx/pretrained/xlsr_conformertcm_mdt_rawboost3_wildspoof-noisevoxceleb.pt
- eval code
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme1/hungdx/pretrained/xlsr_conformertcm_mdt_rawboost3_wildspoof-noisevoxceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_rawboost3_voxcelebaug-spoofceleb"
```

# MDT training (w/o Rawboost)
Time for training: from 2025-10-08 08:20:01 to 2025-10-09 13:46 (~29h)
Average checkpoint: /nvme1/hungdx/pretrained/xlsr_conformertcm_mdt_wo_rawboost_wildspoof-noisevoxceleb.pt
- eval code
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme1/hungdx/pretrained/xlsr_conformertcm_mdt_wo_rawboost_wildspoof-noisevoxceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_wo_rawboost_voxcelebaug-spoofceleb"
```

# Standard training (w/o Rawboost)
Time for training: from 2025-10-08 08:20:01 to 2025-10-09 13:46 (~29h)
Average checkpoint: /nvme1/hungdx/pretrained/xlsr_conformertcm_no_rawboost_wildspoof-noisevoxceleb.pt
- eval code
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme1/hungdx/pretrained/xlsr_conformertcm_no_rawboost_wildspoof-noisevoxceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_wo_rawboost_voxcelebaug-spoofceleb"
```

# XLSR-VIB SCL + MDT
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python src/train.py experiment=wildspoof/xlsr_vib_scl_mdt ++model_averaging=True +model.score_save_path="logs/eval/wildspoof/spoofceleb_eval_xlsr_vib_scl_mdt_large_corpus_clean_4s.txt"


======================= Post-processing SSBolt ========================================

## SSBolt post-process

### xlsr_conformertcm_1p_combined_protocol_ssbolt79
```bash
./scripts/benchmark_old.sh -g 2 -c wildspoof/Oct_25/xlsr_conformertcm_mdt_infer -b data/wildspoof_challenge_benchmark -m /data/Datasets/wild_spoof_ckpt/xlsr_conformertcm_1p_combined_protocol_ssbolt79.ckpt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_1p_combined_protocol_ssbolt79"
```

### xlsr_conformertcm_1p_spoofceleb_protocol_ssbolt79
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/Oct_25/xlsr_conformertcm_mdt_infer -b data/wildspoof_challenge_benchmark -m /data/Datasets/wild_spoof_ckpt/xlsr_conformertcm_1p_spoofceleb_protocol_ssbolt79.ckpt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_1p_spoofceleb_protocol_ssbolt79"
```

/nvme1/hungdx/pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt

### xlsr_conformertcm_mdt_rawboost3_spoofceleb + ssboll79
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/Oct_25/xlsr_conformertcm_mdt_infer -b data/wildspoof_challenge_benchmark -m /nvme1/hungdx/pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_rawboost3_spoofceleb_ssbolt79"
```


./scripts/benchmark_old.sh -g 3 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme1/hungdx/pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_rawboost3_spoofceleb"


### X_C_TCM_AUX_BEATS_MDT
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/Oct31/xlsr_beats_seres2net_conformertcm_mdt_aux_infer -b data/wildspoof_challenge_benchmark -m /data/Datasets/wild_spoof_ckpt/XLSR_ConformerTCM_MDT_AUX_spoofceleb_noise.ckpt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_aux_beats_spoofceleb_noise"
```

### X_C_TCM_AUX_BEATS
```bash
./scripts/benchmark_old.sh -g 2 -c wildspoof/Oct31/xlsr_beats_seres2net_conformertcm_aux_infer -b data/wildspoof_challenge_benchmark -m /data/Datasets/wild_spoof_ckpt/XLSR_ConformerTCM_AUX_spoofceleb_noise.ckpt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_aux_beats_spoofceleb_noise"
```

### X_C_TCM_AUX_BEATS_FIXED
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/Oct31/xlsr_beats_seres2net_conformertcm_aux_infer -b data/wildspoof_challenge_benchmark -m /data/Datasets/wild_spoof_ckpt/XLSR_ConformerTCM_AUX_spoofceleb_noise_fixed.ckpt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_aux_beats_spoofceleb_noise_fixed"
```

### X_C_TCM_AUX_BEATS_FIXED SWAPPROTOCOL
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/Oct31/xlsr_beats_seres2net_conformertcm_aux_infer -b data/wildspoof_challenge_benchmark -m /data/Datasets/wild_spoof_ckpt/XLSR_ConformerTCM_AUX_spoofceleb_noise_fixed_swapprotocol.ckpt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_aux_beats_spoofceleb_noise_fixed_swapprotocol"
```

### X_C_TCM Spoofceleb_noisy
```bash
./scripts/benchmark_old.sh -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/wildspoof_challenge_benchmark -m /data/Datasets/wild_spoof_ckpt/XLSR_ConformerTCM_spoofceleb_noise.ckpt -r logs/results/wildspoof_challenge_benchmark -n "XLSR_ConformerTCM_spoofceleb_noise"
```


### ASV19_denoiser
```bash
./scripts/benchmark_old.sh -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/asv19_denoiser -m /data/Datasets/wild_spoof_ckpt/XLSR_ConformerTCM_spoofceleb_noise.ckpt -r logs/results/asv19_denoiser -n "XLSR_ConformerTCM_spoofceleb_noise"
```

### X_C_TCM Spoofceleb_noisy (new balance protocol Nov 12)
```bash
./scripts/benchmark_old.sh -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b data/wildspoof_challenge_benchmark -m /data/Datasets/wild_spoof_ckpt/XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_Nov12.ckpt -r logs/results/wildspoof_challenge_benchmark -n "XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_Nov12"
```

### X_C_TCM_AUX_BEATS_FT 
```bash
./scripts/benchmark_old.sh -g 2 -c wildspoof/Nov3/xlsr_beats_conformertcm_aux_28g_infer -b data/wildspoof_challenge_benchmark -m /data/Datasets/wild_spoof_ckpt/XLSR_ConformerTCM_pre_AUX_spoofceleb_clean_ft_new_noise_Nov12.ckpt -r logs/results/wildspoof_challenge_benchmark -n "XLSR_ConformerTCM_pre_AUX_spoofceleb_clean_ft_new_noise_Nov12"
```

### X_C_TCM_FIXED_AUX_BEATS_FT 
```bash
./scripts/benchmark_old.sh -g 2 -c wildspoof/Nov3/xlsr_beats_conformertcm_aux_28g_infer -b data/wildspoof_challenge_benchmark -m /data/Datasets/wild_spoof_ckpt/XLSR_ConformerTCM_pre_AUX_spoofceleb_noise_fixed_new_Nov12.ckpt -r logs/results/wildspoof_challenge_benchmark -n "XLSR_ConformerTCM_pre_AUX_spoofceleb_noise_fixed_new_Nov12"
```

# Lora with lossy codec
## conf-1
```bash
./scripts/benchmark.sh -g 3 -c wildspoof/Nov3/xlsr_conformertcm_lora -b data/wildspoof_challenge_benchmark -m pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -a /data/Datasets/lora_config-1/epoch_001-v1.ckpt -r logs/results/wildspoof_challenge_benchmark -n "XLSR_ConformerTCM_LoRA_lossy_codec-conf-1" -l true
```

## conf-1-v2
```bash
./scripts/benchmark.sh -g 2 -c wildspoof/Nov3/xlsr_conformertcm_lora -b data/wildspoof_challenge_benchmark -m pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -a /data/Datasets/lora_config-1-v2/epoch_003.ckpt -r logs/results/wildspoof_challenge_benchmark -n "XLSR_ConformerTCM_LoRA_lossy_codec-conf-1-v2" -l true
```

## conf-2
```bash
./scripts/benchmark.sh -g 2 -c wildspoof/Nov3/xlsr_conformertcm_lora -b data/wildspoof_challenge_benchmark -m pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -a /data/Datasets/lora_config-2/epoch_004.ckpt -r logs/results/wildspoof_challenge_benchmark -n "XLSR_ConformerTCM_LoRA_lossy_codec-conf-2" -l true
```

## conf-3
```bash
./scripts/benchmark.sh -g 2 -c wildspoof/Nov3/xlsr_conformertcm_lora -b data/wildspoof_challenge_benchmark -m pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -a /data/Datasets/lora_config-3/epoch_000.ckpt -r logs/results/wildspoof_challenge_benchmark -n "XLSR_ConformerTCM_LoRA_lossy_codec-conf-3" -l true
```


```bash
./scripts/benchmark_old.sh -g 2 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme1/hungdx/pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_rawboost3_spoofceleb"
```


./scripts/benchmark_old.sh -g 2 -c wildspoof/xlsr_conformertcm_mdt -b data/Wild_Spoof_Dataset -m /data/Datasets/wild_spoof_ckpt/XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_and_vocoded_Nov16.ckpt -r logs/results/wildspoof_challenge_benchmark -n "T3_dev"