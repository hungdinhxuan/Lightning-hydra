# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

conda activate myenv && conda install -y gcc_linux-64 gxx_linux-64 ninja cmake
cd fairseq_lib && TORCH_CUDA_VERSION=cu121 FORCE_CUDA=1 pip install -e .

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python src/train.py experiment=partialspoof/aasistssl_multiview_conf-beta-1.1 ++model_averaging=True +model.score_save_path="logs/eval/ps/ps_xlsr_aasist_multiview_conf-beta-1.1_eval_4s.txt"  ++train=False ++test=True ++ckpt_path='/nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-29_23-09-16-beta-1.1/checkpoints' callbacks=none ++data.args.cut=64600 ++data.args.random_start=False  ++data.batch_size=64 ++data.args.protocols_path="/nvme1/hungdx/Lightning-hydra/data/PartialSpoof/database/protocol.txt"  ++data.data_dir="/nvme1/hungdx/Lightning-hydra/data/PartialSpoof/database/" ++data.chunking_eval=False ++data.args.eval_set='partialspoof' ++data.args.eval=True ++data.args.chunk_size=64000 ++data.args.overlap_size=32000 

## Docker Build Instructions

To build with the default options, simply run docker buildx bake.
To build a specific target, use docker buildx bake <target>.
To specify the platform, use docker buildx bake <target> --set <target>.platform=linux/amd64.

```bash
docker buildx bake 280-py311-cuda1281-cudnn-devel-ubuntu2204 --set 280-py311-cuda1281-cudnn-devel-ubuntu2204.platform=linux/amd64
```

## April (ToP) cnsl
```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 python src/train.py experiment=cnsl/xlsr_vib_large_corpus +model.score_save_path="/nvme1/hungdx/Lightning-hydra/logs/eval/cnsl/KoreanReadSpeechCorpus_april_xlsr_vib_large_corpus_s202412.txt"  ++data.data_dir="/nvme1/hungdx/Lightning-hydra/data/KoreanReadSpeechCorpus" ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/data/KoreanReadSpeechCorpus/KoreanReadSpeechCorpus_protocol.txt" ++train=False ++test=True ++model.spec_eval=True ++data.batch_size=64
 ```

# Benchmark ToP april
 ```bash
./scripts/benchmark.sh -g 3 -c cnsl/xlsr_vib_large_corpus -b $(pwd)/data/huggingface_benchrmark_Speech-DF-Arena -m /datad/hungdx/KDW2V-AASISTL/pretrained/vib_conf-5_gelu_acmccs_apr3_moreko_telephone_epoch22.pth -r logs/results/huggingface_benchrmark_Speech-DF-Arena -n "ToP_April"
 ```

 # Benchmark Conformer + MDT
 ```bash
./scripts/benchmark.sh -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b $(pwd)/data/huggingface_benchrmark_Speech-DF-Arena -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -r logs/results/huggingface_benchrmark_Speech-DF-Arena -n "Conformer_MDT_DEC2024_correct"
 ```

 # Benchmark ToP (LA19)
 ```bash
 ./scripts/benchmark.sh -g 2 -c cnsl/xlsr_vib_paper -b $(pwd)/data/huggingface_benchrmark_Speech-DF-Arena -m /datad/pretrained/AudioDeepfakeCMs/vib/vib_asvspoof2019_epoch13.pth -r logs/results/huggingface_benchrmark_Speech-DF-Arena -n "ToP_LA19"
 ```

 # Benchmark AASIST-SSL + MDT(LA19)
 ```bash
./scripts/benchmark.sh -g 2 -c huggingface_benchmark/xlsr_aasist_mdt_paper -b $(pwd)/data/huggingface_benchrmark_Speech-DF-Arena -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-10-16_21-04-31-conf-2/checkpoints/averaged_top5.ckpt -r logs/results/huggingface_benchrmark_Speech-DF-Arena -n "AASIST_SSL_MDT_LA19"
 ```

  # Benchmark ConformerTCM + MDT (LA19)
 ```bash
./scripts/benchmark.sh -g 2 -c huggingface_benchmark/xlsr_conformertcm_mdt_lora_infer -b $(pwd)/data/huggingface_benchrmark_Speech-DF-Arena -m /nvme1/hungdx/tcm_add/models/Conformer_w_TCM_LA_WCE_1e-06_ES144_H4_NE4_KS31_AUG3_w_sin_pos_multiview/best/avg_5_best_4.pth -r logs/results/huggingface_benchrmark_Speech-DF-Arena -n "ConformerTCM_MDT_LA19"
 ```

 # Benchrmark ConformerTCM + MDT LoRA (Large corpus + More elevenlabs)
 ```bash
./scripts/benchmark.sh -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs -b $(pwd)/data/cnsl_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -a /nvme1/hungdx/Lightning-hydra/logs/train/runs/2025-04-29_11-38-10-v4-corrected/checkpoints/epoch_020.ckpt -r logs/results/cnsl_benchmark -n "ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs"
 ```

  # Benchrmark ConformerTCM + MDT LoRA (Large corpus + More elevenlabs) on Huggingface
 ```bash
./scripts/benchmark.sh -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs -b $(pwd)/data/huggingface_benchrmark_Speech-DF-Arena -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -a /nvme1/hungdx/Lightning-hydra/logs/train/runs/2025-04-29_11-38-10-v4-corrected/checkpoints/epoch_020.ckpt -r logs/results/huggingface_benchrmark_Speech-DF-Arena -n "ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs"
 ```

# CNSL benchmark
## ConformerTCM + MDT LoRA (Large corpus + More elevenlabs)
```bash
./scripts/benchmark.sh -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs -b $(pwd)/data/cnsl_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -a /nvme1/hungdx/Lightning-hydra/logs/train/runs/2025-04-29_11-38-10-v4-corrected/checkpoints/epoch_020.ckpt -r logs/results/cnsl_benchmark -n "ConformerTCM_MDT_LoRA_LargeCorpus_MoreElevenlabs"
 ```
 ## ToP April
 ```bash
./scripts/benchmark.sh -g 3 -c cnsl/xlsr_vib_large_corpus -b $(pwd)/data/cnsl_benchmark -m /datad/hungdx/KDW2V-AASISTL/pretrained/vib_conf-5_gelu_acmccs_apr3_moreko_telephone_epoch22.pth -r logs/results/cnsl_benchmark -n "ToP_April"
 ```

 # 2B benchmark
 ```bash
./scripts/benchmark2.sh -g 2 -c cnsl/xlsr_2b_conformertcm_layerwise_24_mdt_large_corpus -b $(pwd)/data/huggingface_benchrmark_Speech-DF-Arena -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2025-05-02_02-31-32-2b/checkpoints/epoch_018.ckpt -r logs/results/huggingface_benchrmark_Speech-DF-Arena -n "ConformerTCM_MDT_2B_LargeCorpus_e18"
 ```

 # 
```bash
./scripts/benchmark.sh -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b $(pwd)/data/cnsl_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -r logs/results/cnsl_benchmark -n "Conformer_MDT_DEC2024_correct"
```


## benchmark again merged lora model
```bash
./scripts/benchmark.sh -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b $(pwd)/data/cnsl_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/best_ckpt/MDT_241214_lora_250501_merged.pth -r logs/results/cnsl_benchmark -n "MDT_241214_lora_250501_merged_lora_exp_2_june2"
```


# Benchmark learnable aasist-ssl (LA19)
 ```bash
./scripts/benchmark.sh -g 2 -c aasistssl_mdt_conf-2_learnable -b $(pwd)/data/huggingface_benchrmark_Speech-DF-Arena -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2025-05-29_07-48-34/checkpoints/averaged_top5.ckpt -r logs/results/huggingface_benchrmark_Speech-DF-Arena -n "aasistssl_mdt_leranble_MDT_LA19"
 ```


# Merge lora weights to base model
```bash
python scripts/inference/merge_lora_to_base.py --checkpoint_path="/nvme1/hungdx/Lightning-hydra/logs/best_ckpt/MDT_241214_lora_250501_merged.pth" --config_path="/nvme1/hungdx/Lightning-hydra/configs/experiment/cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs.yaml" --lora_path="/nvme1/hungdx/Lightning-hydra/logs/train/runs/2025-06-02_02-18-43/checkpoints/epoch_000.ckpt" --output_path="/nvme1/hungdx/Lightning-hydra/logs/best_ckpt/MDT_241214_lora_250501_merged_lora_exp2_june2.pth" --device='cpu'
```

# Merge lora weights to base model
```bash
python scripts/inference/merge_lora_to_base.py --checkpoint_path="/datad/pretrained/AudioDeepfakeCMs/S_241214_conf-1.pth" --config_path="/nvme1/hungdx/Lightning-hydra/configs/experiment/cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs.yaml" --lora_path="/nvme1/hungdx/Lightning-hydra/logs/train/runs/2025-06-01_13-16-39/checkpoints/epoch_000.ckpt" --output_path="/nvme1/hungdx/Lightning-hydra/logs/best_ckpt/MDT_241214_baseline_lora_exp1_june1.pth" --device='cpu'
```



python tests/test_merge_lora.py \
    --checkpoint_path /datad/pretrained/AudioDeepfakeCMs/S_241214_conf-1.pth \
    --config_path /nvme1/hungdx/Lightning-hydra/configs/experiment/cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs.yaml \
    --lora_path /datad/pretrained/AudioDeepfakeCMs/lora/MDT_241214_lora_250418 \
    --seed 42

# June 2

## exp1_june1_25
```bash
./scripts/benchmark.sh -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b $(pwd)/data/cnsl_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/best_ckpt/MDT_241214_baseline_lora_exp1_june1.pth -r logs/results/cnsl_benchmark -n "MDT_241214_baseline_lora_exp_1_june1" -l false
```

## exp2_june2_25
```bash
./scripts/benchmark.sh -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b $(pwd)/data/huggingface_benchrmark_Speech-DF-Arena -m /nvme1/hungdx/Lightning-hydra/logs/best_ckpt/MDT_241214_lora_250501_merged_lora_exp2_june2.pth -r logs/results/huggingface_benchrmark_Speech-DF-Arena -n "MDT_241214_lora_250501_merged_lora_exp_2_june2" -l false
```

## exp1_june2_25
```bash
./scripts/benchmark.sh -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer -b $(pwd)/data/huggingface_benchrmark_Speech-DF-Arena -m /nvme1/hungdx/Lightning-hydra/logs/best_ckpt/MDT_baseline_lora_exp1_june2.pth -r logs/results/huggingface_benchrmark_Speech-DF-Arena -n "MDT_baseline_lora_exp1_june2" -l false
```

# CL benchmark
## exp_1.1_june9
```bash
./scripts/benchmark.sh -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs -b $(pwd)/data/CL_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -a /datad/Lightning-hydra/runs/2025-06-09_14-55-39/checkpoints/epoch_049.ckpt -r logs/results/CL_benchmark -n "ConformerTCM_MDT_LoRA_exp_1.1_june9"
```

## exp_1.2_june9
```bash
./scripts/benchmark.sh -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs -b $(pwd)/data/CL_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -a /datad/Lightning-hydra/runs/2025-06-09_14-55-41/checkpoints/epoch_025.ckpt -r logs/results/CL_benchmark -n "ConformerTCM_MDT_LoRA_exp_1.2_june9"
```

## exp_1.3_june10
```bash
./scripts/benchmark.sh -g 2 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs -b $(pwd)/data/CL_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -a /nvme1/hungdx/Lightning-hydra/logs/train/runs/2025-06-10_13-03-58/checkpoints/epoch_035.ckpt -r logs/results/CL_benchmark -n "ConformerTCM_MDT_LoRA_exp_1.3_june10"
```

## exp_2.1_june10
```bash
./scripts/benchmark.sh -g 3 -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs -b $(pwd)/data/CL_benchmark -m /nvme1/hungdx/Lightning-hydra/logs/train/runs/2024-12-14_08-35-06-large-corpus-conf-1/checkpoints/averaged_top5.ckpt -a /nvme1/hungdx/Lightning-hydra/logs/train/runs/2025-06-10_13-14-47/checkpoints/epoch_058.ckpt -r logs/results/CL_benchmark -n "ConformerTCM_MDT_LoRA_exp_2.1_june10"
```