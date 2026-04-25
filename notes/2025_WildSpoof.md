# MDT training (1%)
```bash
CUDA_VISIBLE_DEVICES='MIG-6e4275af-2db0-51f1-a601-7ad8a1002745' OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/xlsr_conformertcm_mdt ++model_averaging=True +model.score_save_path="logs/eval/wildspoof/spoofceleb_eval_xlsr_conformertcm_MDT_large_corpus_clean_4s.txt" +trainer.limit_train_batches=0.01
```

- Average ckpt: /nvme2/hungdx/logs/train/runs/2025-09-23_21-57-59/checkpoints/averaged_top5.ckpt

- eval code
```bash
./scripts/benchmark_old.sh -g MIG-6e4275af-2db0-51f1-a601-7ad8a1002745 -c wildspoof/xlsr_conformertcm_mdt -b data/benchmark_kd -m /nvme2/hungdx/logs/train/runs/2025-09-23_21-57-59/checkpoints/averaged_top5.ckpt -r logs/results/benchmark_wildspoof -n "xlsr_conformertcm_mdt"
```

```bash
./scripts/benchmark_old.sh -g 2 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme2/hungdx/pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_rawboost3_spoofceleb"
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
./scripts/benchmark_old.sh -g 3 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme2/hungdx/pretrained/xlsr_conformertcm_mdt_rawboostfull_wildspoof-noisevoxceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_rawboostfull_voxcelebaug-spoofceleb"
```

# MDT training (Rawboost3)
Time for training: from 2025-10-07 03:17 to 2025-10-09 11:28 (~55h)
Average checkpoint: /nvme2/hungdx/pretrained/xlsr_conformertcm_mdt_rawboost3_wildspoof-noisevoxceleb.pt
- eval code
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme2/hungdx/pretrained/xlsr_conformertcm_mdt_rawboost3_wildspoof-noisevoxceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_rawboost3_voxcelebaug-spoofceleb"
```

# MDT training (w/o Rawboost)
Time for training: from 2025-10-08 08:20:01 to 2025-10-09 13:46 (~29h)
Average checkpoint: /nvme2/hungdx/pretrained/xlsr_conformertcm_mdt_wo_rawboost_wildspoof-noisevoxceleb.pt
- eval code
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme2/hungdx/pretrained/xlsr_conformertcm_mdt_wo_rawboost_wildspoof-noisevoxceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_mdt_wo_rawboost_voxcelebaug-spoofceleb"
```

# Standard training (w/o Rawboost)
Time for training: from 2025-10-08 08:20:01 to 2025-10-09 13:46 (~29h)
Average checkpoint: /nvme2/hungdx/pretrained/xlsr_conformertcm_no_rawboost_wildspoof-noisevoxceleb.pt
- eval code
```bash
./scripts/benchmark_old.sh -g 3 -c wildspoof/xlsr_conformertcm_mdt -b data/wildspoof_challenge_benchmark -m /nvme2/hungdx/pretrained/xlsr_conformertcm_no_rawboost_wildspoof-noisevoxceleb.pt -r logs/results/wildspoof_challenge_benchmark -n "xlsr_conformertcm_wo_rawboost_voxcelebaug-spoofceleb"
```

# XLSR-VIB SCL + MDT
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python src/train.py experiment=wildspoof/xlsr_vib_scl_mdt ++model_averaging=True +model.score_save_path="logs/eval/wildspoof/spoofceleb_eval_xlsr_vib_scl_mdt_large_corpus_clean_4s.txt"
