 # In-the-wild benchmark
 
 ## ToP April
 ```bash
./scripts/benchmark.sh -g 3 -c cnsl/xlsr_vib_large_corpus -b $(pwd)/data/Nov_benchmark -m pretrained/vib_conf-5_gelu_acmccs_apr3_moreko_telephone_epoch22.pth -r logs/results/Nov_benchmark -n "ToP_April" -l false
 ```

## MDT 
 ```bash
./scripts/benchmark.sh -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b $(pwd)/data/Nov_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -r logs/results/Nov_benchmark -n "MDT"
 ```

## MDT LoRA
 ```bash
./scripts/benchmark.sh -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs -b data/Nov_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -a pretrained/MDT_241214_lora_250501 -r logs/results/Nov_benchmark -n "MDT_LoRA" 
 ```

