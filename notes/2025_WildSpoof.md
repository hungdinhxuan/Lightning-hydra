# MDT training (1%)
```bash
CUDA_VISIBLE_DEVICES='MIG-6e4275af-2db0-51f1-a601-7ad8a1002745' OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/xlsr_conformertcm_mdt ++model_averaging=True +model.score_save_path="logs/eval/wildspoof/spoofceleb_eval_xlsr_conformertcm_MDT_large_corpus_clean_4s.txt" +trainer.limit_train_batches=0.01
```

- Average ckpt: /nvme1/hungdx/logs/train/runs/2025-09-23_21-57-59/checkpoints/averaged_top5.ckpt

- eval code
```bash
./scripts/benchmark_old.sh -g MIG-6e4275af-2db0-51f1-a601-7ad8a1002745 -c wildspoof/xlsr_conformertcm_mdt -b data/benchmark_kd -m /nvme1/hungdx/logs/train/runs/2025-09-23_21-57-59/checkpoints/averaged_top5.ckpt -r logs/results/benchmark_wildspoof -n "xlsr_conformertcm_mdt"
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