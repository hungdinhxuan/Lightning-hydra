# Exp1 (exp1) Single lora
- aasist_ssl_exp1.sh


+ eval


++Single Lora
./scripts/benchmark.sh -g 2 -c icassp26/aasist_ssl/xlsr_aasist_single_lora -b data/ICASSP25_benchmark_noise -m /nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth -a /home/hungdx/logs/train/runs/2025-09-10_11-31-31/checkpoints/epoch_009.ckpt -r logs/results/ICASSP25_benchmark_noise -n "xlsr_aasist_single_lora_datasmall_correctv2" -l false -s false

++Dynamic LoRA
./scripts/benchmark_var.sh -g 2 -c icassp26/aasist_ssl/xlsr_aasist_mul_lora -b data/ICASSP26_benchmark_noise -m /nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth -r logs/results/ICASSP26_benchmark_noise -n "xlsr_aasist_dynamic_lora_datasmall" -l false -s false

- conformertcm_exp1.sh
./scripts/benchmark.sh -g 3 -c icassp26/conformertcm/xlsr_conformertcm_single_lora -b data/ICASSP25_benchmark_noise -m /nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth -a /home/hungdx/logs/train/runs/2025-09-10_11-31-33/checkpoints/epoch_006.ckpt -r logs/results/ICASSP26_benchmark_noise -n "xlsr_conformertcm_single_lora_correctv2" -l false -s false
++Dynamic LoRA
./scripts/benchmark_var.sh -g 2 -c icassp26/conformertcm/xlsr_conformertcm_mul_lora -b data/ICASSP26_benchmark_noise -m /nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth -r logs/results/ICASSP26_benchmark_noise -n "xlsr_conformertcm_dynamic_lora_datasmall" -l false -s false



# Dynamic LoRa

================== AASIST-SSL ===========================================

## Background_music_noise (g1) 
/home/hungdx/logs/train/runs/2025-09-11_02-19-50/checkpoints/epoch_001.ckpt

/home/hungdx/logs/train/runs/2025-09-12_14-07-14/checkpoints/epoch_010.ckpt (dev 99.855%)

## benchmark
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/aasist_ssl/xlsr_aasist_single_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_aasist_dynamic_lora_datasmall/asv19_g1_icassp26_aasist_ssl_xlsr_aasist_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++train=False ++test=True ++model.spec_eval=True  ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_02-19-50/checkpoints/epoch_001.ckpt ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/background_music_noise.txt"

## Autotune (g2)
/home/hungdx/logs/train/runs/2025-09-11_02-21-04/checkpoints/epoch_018.ckpt

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/aasist_ssl/xlsr_aasist_single_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_aasist_dynamic_lora_datasmall/asv19_g2_icassp26_aasist_ssl_xlsr_aasist_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++train=False ++test=True ++model.spec_eval=True  ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_02-21-04/checkpoints/epoch_018.ckpt ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/auto_tune.txt"

## Bandpass-filter (g3)
/home/hungdx/logs/train/runs/2025-09-11_04-04-11/checkpoints/epoch_029.ckpt

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/aasist_ssl/xlsr_aasist_single_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_aasist_dynamic_lora_datasmall/asv19_g3_icassp26_aasist_ssl_xlsr_aasist_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++train=False ++test=True ++model.spec_eval=True  ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_04-04-11/checkpoints/epoch_029.ckpt ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/band_pass_filter.txt"


## Echo (g4)
/home/hungdx/logs/train/runs/2025-09-11_04-03-13/checkpoints/epoch_028.ckpt

/home/hungdx/logs/train/runs/2025-09-13_04-27-17/checkpoints/epoch_022.ckpt (Dev: 99.7%)

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/aasist_ssl/xlsr_aasist_single_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_aasist_dynamic_lora_datasmall/asv19_g4_icassp26_aasist_ssl_xlsr_aasist_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++train=False ++test=True ++model.spec_eval=True  ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_04-03-13/checkpoints/epoch_028.ckpt ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/echo.txt"

## manipulation (g5) _
/home/hungdx/logs/train/runs/2025-09-11_04-28-50/checkpoints/epoch_019.ckpt
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/aasist_ssl/xlsr_aasist_single_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_aasist_dynamic_lora_datasmall/asv19_g5_icassp26_aasist_ssl_xlsr_aasist_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++train=False ++test=True ++model.spec_eval=True  ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_04-28-50/checkpoints/epoch_019.ckpt ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/manipulation.txt"


/home/hungdx/logs/train/runs/2025-09-12_07-55-41/checkpoints/epoch_010.ckpt (97% dev) > _ change lr


/home/hungdx/logs/train/runs/2025-09-13_03-14-46/checkpoints/epoch_019.ckpt (98.8% dev)

/home/hungdx/logs/train/runs/2025-09-13_10-52-28/checkpoints/epoch_014.ckpt (99.7%)



## Gaussian Noise (g6) __
/home/hungdx/logs/train/runs/2025-09-11_04-47-15/checkpoints/epoch_029.ckpt

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/aasist_ssl/xlsr_aasist_single_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_aasist_dynamic_lora_datasmall/asv19_g6_icassp26_aasist_ssl_xlsr_aasist_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++train=False ++test=True ++model.spec_eval=True  ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_04-47-15/checkpoints/epoch_029.ckpt ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/gaussian_noise.txt"

/home/hungdx/logs/train/runs/2025-09-13_07-38-15/checkpoints/epoch_009.ckpt



## Reverberation (g7)
/home/hungdx/logs/train/runs/2025-09-11_05-11-40/checkpoints/epoch_028.ckpt (91%)

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/aasist_ssl/xlsr_aasist_single_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_aasist_dynamic_lora_datasmall/asv19_g7_icassp26_aasist_ssl_xlsr_aasist_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++train=False ++test=True ++model.spec_eval=True  ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_05-11-40/checkpoints/epoch_028.ckpt ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/reverberation.txt"


/home/hungdx/logs/train/runs/2025-09-12_07-21-18/checkpoints/epoch_008.ckpt (95% dev)


/home/hungdx/logs/train/runs/2025-09-12_08-00-50/checkpoints/epoch_008.ckpt (98% dev)


/home/hungdx/logs/train/runs/2025-09-13_05-31-56/checkpoints/epoch_009.ckpt (98.7%)

/home/hungdx/logs/train/runs/2025-09-13_09-54-56/checkpoints/epoch_012.ckpt (99.4%)


/home/hungdx/logs/train/runs/2025-09-13_10-52-28/checkpoints/epoch_014.ckpt,/home/hungdx/logs/train/runs/2025-09-13_09-54-56/checkpoints/epoch_012.ckpt

================== ConformerTCM-SSL ===========================================

## Background_music_noise (g1) 
/home/hungdx/logs/train/runs/2025-09-11_05-38-35/checkpoints/epoch_008.ckpt

/home/hungdx/logs/train/runs/2025-09-12_14-09-27/checkpoints/epoch_019.ckpt (99.903 %)


CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/conformertcm/xlsr_conformertcm_mul_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_conformertcm_dynamic_lora_datasmall/asv19_g1_icassp26_conformertcm_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/background_music_noise.txt" ++train=False ++test=True ++model.spec_eval=True ++data.args.random_start=false ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_05-38-35/checkpoints/epoch_008.ckpt

## Autotune (g2) --
/home/hungdx/logs/train/runs/2025-09-11_05-38-53/checkpoints/epoch_029.ckpt

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/conformertcm/xlsr_conformertcm_mul_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_conformertcm_dynamic_lora_datasmall/asv19_g2_icassp26_conformertcm_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/auto_tune.txt" ++train=False ++test=True ++model.spec_eval=True ++data.args.random_start=false ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_05-38-53/checkpoints/epoch_029.ckpt


## Bandpass-filter (g3) +
/home/hungdx/logs/train/runs/2025-09-11_06-02-37/checkpoints/epoch_011.ckpt

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/conformertcm/xlsr_conformertcm_mul_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_conformertcm_dynamic_lora_datasmall/asv19_g3_icassp26_conformertcm_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/band_pass_filter.txt" ++train=False ++test=True ++model.spec_eval=True ++data.args.random_start=false ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_06-02-37/checkpoints/epoch_011.ckpt


## Echo (g4) +
/home/hungdx/logs/train/runs/2025-09-11_06-07-41/checkpoints/epoch_029.ckpt

/home/hungdx/logs/train/runs/2025-09-13_04-26-58/checkpoints/epoch_014.ckpt (Dev 99.4%)


CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/conformertcm/xlsr_conformertcm_mul_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_conformertcm_dynamic_lora_datasmall/asv19_g4_icassp26_conformertcm_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/echo.txt" ++train=False ++test=True ++model.spec_eval=True ++data.args.random_start=false ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_06-07-41/checkpoints/epoch_029.ckpt

## manipulation (g5) +
/home/hungdx/logs/train/runs/2025-09-11_06-29-53/checkpoints/epoch_017.ckpt


CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/conformertcm/xlsr_conformertcm_mul_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_conformertcm_dynamic_lora_datasmall/asv19_g5_icassp26_conformertcm_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/manipulation.txt" ++train=False ++test=True ++model.spec_eval=True ++data.args.random_start=false ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_06-29-53/checkpoints/epoch_017.ckpt


/home/hungdx/logs/train/runs/2025-09-12_08-04-50/checkpoints/epoch_014.ckpt (98% dev)

/home/hungdx/logs/train/runs/2025-09-13_03-16-14/checkpoints/epoch_014.ckpt (98.6% dev)


### Extract-emb
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/conformertcm/xlsr_conformertcm_mul_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_conformertcm_dynamic_lora_datasmall/asv19_g5_icassp26_conformertcm_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/manipulation.txt" ++train=False ++test=True ++model.spec_eval=True ++data.args.random_start=false ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth" ++model.is_base_model_path_ln=false ++model.last_emb=True ++model.emb_save_path="logs/icassp25_conformertcm_emb_g5" 

### extract-emb-2
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/conformertcm/xlsr_conformertcm_single_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_conformertcm_dynamic_lora_datasmall/asv19_g4_icassp26_conformertcm_single_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/manipulation.txt" ++train=False ++test=True ++model.spec_eval=True ++data.args.random_start=false ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-13_03-16-14/checkpoints/epoch_014.ckpt ++model.last_emb=True ++model.emb_save_path="logs/icassp25_conformertcm_single_emb_g5" 


## Gaussian Noise (g6) +
/home/hungdx/logs/train/runs/2025-09-11_06-36-06/checkpoints/epoch_019.ckpt

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/conformertcm/xlsr_conformertcm_mul_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_conformertcm_dynamic_lora_datasmall/asv19_g6_icassp26_conformertcm_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/gaussian_noise.txt" ++train=False ++test=True ++model.spec_eval=True ++data.args.random_start=false ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_06-36-06/checkpoints/epoch_019.ckpt

/home/hungdx/logs/train/runs/2025-09-13_07-46-07/checkpoints/epoch_008.ckpt

/home/hungdx/logs/train/runs/2025-09-13_10-52-27/checkpoints/epoch_005.ckpt (99.7%)

## Reverberation (g7)
/home/hungdx/logs/train/runs/2025-09-11_07-05-42/checkpoints/epoch_021.ckpt


CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 python src/train.py experiment=icassp26/conformertcm/xlsr_conformertcm_mul_lora ++model.score_save_path="logs/results/ICASSP25_benchmark_noise/xlsr_conformertcm_dynamic_lora_datasmall/asv19_g7_icassp26_conformertcm_mul_lora_xlsr_aasist_dynamic_lora_datasmall.txt" ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/reverberation.txt" ++train=False ++test=True ++model.spec_eval=True ++data.args.random_start=false ++data.args.trim_length=64000 ++model.base_model_path="/nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth" ++model.is_base_model_path_ln=false ++model.adapter_paths=/home/hungdx/logs/train/runs/2025-09-11_07-05-42/checkpoints/epoch_021.ckpt


/home/hungdx/logs/train/runs/2025-09-12_07-36-19/checkpoints/epoch_021.ckpt (98% dev)

/home/hungdx/logs/train/runs/2025-09-13_05-32-14/checkpoints/epoch_007.ckpt (98.3% dev)


/home/hungdx/logs/train/runs/2025-09-13_09-54-51/checkpoints/epoch_008.ckpt (99.4% dev)


/home/hungdx/logs/train/runs/2025-09-13_10-52-27/checkpoints/epoch_005.ckpt,/home/hungdx/logs/train/runs/2025-09-13_09-54-51/checkpoints/epoch_008.ckpt
(latest update):
- 5,7