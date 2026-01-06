# clean
```bash
./scripts/benchmark_old.sh -g "MIG-57de94a5-be15-5b5a-b67e-e118352d8a59" -c wildspoof/Nov3/xlsr_conformertcm -b data/WildSpoof_Final_Eval -m pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -r logs/results/WildSpoof_Final_Eval -n "xlsr_conformertcm_mdt_rawboost3_spoofceleb"
```

## Extract embedding
```bash
CUDA_VISIBLE_DEVICES="MIG-57de94a5-be15-5b5a-b67e-e118352d8a59" OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/Nov3/xlsr_conformertcm ++model.score_save_path="logs/tmp.txt" ++data.data_dir=data/WildSpoof_Final_Eval/Final_eval ++data.args.protocol_path=data/WildSpoof_Final_Eval/Final_eval/protocol.txt ++train=False ++test=True ++model.spec_eval=True ++data.args.trim_length=64000 ckpt_path=pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt ++model.last_emb=True ++model.emb_save_path="logs/wildspoof_eval_embs/xlsr_conformertcm_mdt_rawboost3_spoofceleb"
```

# unbalanced noise ft
```bash
./scripts/benchmark_old.sh -g "MIG-57de94a5-be15-5b5a-b67e-e118352d8a59" -c wildspoof/Nov3/xlsr_conformertcm -b data/WildSpoof_Final_Eval -m pretrained/XLSR_ConformerTCM_spoofceleb_noise_ft.ckpt -r logs/results/WildSpoof_Final_Eval -n "XLSR_ConformerTCM_spoofceleb_noise_ft"
```

## Extract embedding
```bash
CUDA_VISIBLE_DEVICES="MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd" OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/Nov3/xlsr_conformertcm ++model.score_save_path="logs/tmp.txt" ++data.data_dir=data/WildSpoof_Final_Eval/Final_eval ++data.args.protocol_path=data/WildSpoof_Final_Eval/Final_eval/protocol.txt ++train=False ++test=True ++model.spec_eval=True ++data.args.trim_length=64000 ckpt_path=pretrained/XLSR_ConformerTCM_spoofceleb_noise_ft.ckpt ++model.last_emb=True ++model.emb_save_path="logs/wildspoof_eval_embs/XLSR_ConformerTCM_spoofceleb_noise_ft"
```


# unbalanced noise
```bash
./scripts/benchmark_old.sh -g "MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd" -c wildspoof/Nov3/xlsr_conformertcm -b data/WildSpoof_Final_Eval -m pretrained/XLSR_ConformerTCM_spoofceleb_noise.ckpt -r logs/results/WildSpoof_Final_Eval -n "XLSR_ConformerTCM_spoofceleb_noise"
```

## Extract embedding
```bash
CUDA_VISIBLE_DEVICES="MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd" OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/Nov3/xlsr_conformertcm ++model.score_save_path="logs/tmp.txt" ++data.data_dir=data/WildSpoof_Final_Eval/Final_eval ++data.args.protocol_path=data/WildSpoof_Final_Eval/Final_eval/protocol.txt ++train=False ++test=True ++model.spec_eval=True ++data.args.trim_length=64000 ckpt_path=pretrained/XLSR_ConformerTCM_spoofceleb_noise.ckpt ++model.last_emb=True ++model.emb_save_path="logs/wildspoof_eval_embs/XLSR_ConformerTCM_spoofceleb_noise"
```

# balance noise ft
```bash
./scripts/benchmark_old.sh -g "MIG-57de94a5-be15-5b5a-b67e-e118352d8a59" -c wildspoof/Nov3/xlsr_conformertcm -b data/WildSpoof_Final_Eval -m pretrained/XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_balacne_ce_Nov16.ckpt -r logs/results/WildSpoof_Final_Eval -n "XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_balacne_ce_Nov16"
```
## Extract embedding
```bash
CUDA_VISIBLE_DEVICES="MIG-57de94a5-be15-5b5a-b67e-e118352d8a59" OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/Nov3/xlsr_conformertcm ++model.score_save_path="logs/tmp.txt" ++data.data_dir=data/WildSpoof_Final_Eval/Final_eval ++data.args.protocol_path=data/WildSpoof_Final_Eval/Final_eval/protocol.txt ++train=False ++test=True ++model.spec_eval=True ++data.args.trim_length=64000 ckpt_path=pretrained/XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_balacne_ce_Nov16.ckpt ++model.last_emb=True ++model.emb_save_path="logs/wildspoof_eval_embs/XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_balacne_ce_Nov16"
```

# vocoded ckpt ft
```bash
./scripts/benchmark_old.sh -g "MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd" -c wildspoof/Nov3/xlsr_conformertcm -b data/WildSpoof_Final_Eval -m pretrained/XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_and_vocoded_Nov16.ckpt -r logs/results/WildSpoof_Final_Eval -n "XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_and_vocoded_Nov16"
```

## Extract embedding
```bash
CUDA_VISIBLE_DEVICES="MIG-57de94a5-be15-5b5a-b67e-e118352d8a59" OMP_NUM_THREADS=5 python src/train.py experiment=wildspoof/Nov3/xlsr_conformertcm ++model.score_save_path="logs/tmp.txt" ++data.data_dir=data/WildSpoof_Final_Eval/Final_eval ++data.args.protocol_path=data/WildSpoof_Final_Eval/Final_eval/protocol.txt ++train=False ++test=True ++model.spec_eval=True ++data.args.trim_length=64000 ckpt_path=pretrained/XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_and_vocoded_Nov16.ckpt ++model.last_emb=True ++model.emb_save_path="logs/wildspoof_eval_embs/XLSR_ConformerTCM_spoofceleb_clean_ft_new_noise_and_vocoded_Nov16"
```

# Lossy codec

# Lora with lossy codec
## conf-1
```bash
./scripts/benchmark.sh -g "MIG-8cdeef83-092c-5a8d-a748-452f299e1df0" -c wildspoof/Nov3/xlsr_conformertcm_lora -b data/WildSpoof_Final_Eval -m pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -a /aisrc3_data/lora_config-1/epoch_001-v1.ckpt -r logs/results/WildSpoof_Final_Eval -n "XLSR_ConformerTCM_LoRA_lossy_codec-conf-1" -l true
```

## conf-1-v2
```bash
./scripts/benchmark.sh -g "MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494" -c wildspoof/Nov3/xlsr_conformertcm_lora -b data/WildSpoof_Final_Eval -m pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -a /aisrc3_data/lora_config-1-v2/epoch_003.ckpt -r logs/results/WildSpoof_Final_Eval -n "XLSR_ConformerTCM_LoRA_lossy_codec-conf-1-v2" -l true
```

## conf-2
```bash
./scripts/benchmark.sh -g "MIG-6e4275af-2db0-51f1-a601-7ad8a1002745" -c wildspoof/Nov3/xlsr_conformertcm_lora -b data/WildSpoof_Final_Eval -m pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -a /aisrc3_data/lora_config-2/epoch_004.ckpt -r logs/results/WildSpoof_Final_Eval -n "XLSR_ConformerTCM_LoRA_lossy_codec-conf-2" -l true
```

## conf-3
```bash
./scripts/benchmark.sh -g "MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd" -c wildspoof/Nov3/xlsr_conformertcm_lora -b data/WildSpoof_Final_Eval -m pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt -a /aisrc3_data/lora_config-3/epoch_000.ckpt -r logs/results/WildSpoof_Final_Eval -n "XLSR_ConformerTCM_LoRA_lossy_codec-conf-3" -l true
```