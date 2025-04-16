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