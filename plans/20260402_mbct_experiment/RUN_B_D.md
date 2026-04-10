# Running EXP B and EXP D

See `EXP.md` for the experiment matrix. This repo adds **dedicated Hydra entries and scripts** so you do not have to reuse generic names in WandB.

| EXP | Role | Hydra experiment | Scripts |
| --- | ---- | ----------------- | ------- |
| **B** | MBCT on, LoRA off (full model from SSL) | `experiment=cnsl/April2026/exp_b_xlsr_conformertcm_mbct` | `exp_b_0_dry_run.sh`, `exp_b_1_full_train.sh`, `exp_b_2_full_strength_nohup.sh` |
| **D** | Pretrained A + MBCT + LoRA | `experiment=cnsl/April2026/exp_d_xlsr_conformertcm_mbct_lora` | `exp_d_0_dry_run.sh`, `exp_d_1_full_train.sh`, `exp_d_2_full_strength_nohup.sh` |

### MDT + MBCT (duration views × bands)

Use :class:`src.data.normal_mbct_mdt_datamodule.NormalMBCTMDTDataModule` with composite batch keys `1_normal` … `4_wideband` (see `DEFAULT_MBCT_MDT_COMPOSITE_KEYS` in `src/models/base/mbct_module.py`).

| EXP | Hydra experiment |
| --- | ---------------- |
| **B** (full model) | `experiment=cnsl/April2026/exp_b_xlsr_conformertcm_mbct_mdt` |
| **D** (LoRA) | `experiment=cnsl/April2026/exp_d_xlsr_conformertcm_mbct_mdt_lora` |

**EXP D, two MBCT bands only** (same MDT four duration views; eight `weighted_views` keys):

| Variant | Bands | Hydra experiment |
| ------- | ----- | ---------------- |
| normal + narrowband | `mbct_band_configs`: normal, narrowband | `experiment=cnsl/April2026/exp_d_xlsr_conformertcm_mbct_mdt_lora_2band_normal_narrowband` |
| normal + wideband | `mbct_band_configs`: normal, wideband (7800 Hz) | `experiment=cnsl/April2026/exp_d_xlsr_conformertcm_mbct_mdt_lora_2band_normal_wideband` |

Shell scripts for the two-band variants (dry run + foreground full train, matching `exp_d_0` / `exp_d_1`):

- `scripts/cnsl/April2026/exp_d_0_dry_run_2band_normal_narrowband.sh`
- `scripts/cnsl/April2026/exp_d_0_dry_run_2band_normal_wideband.sh`
- `scripts/cnsl/April2026/exp_d_1_full_train_2band_normal_narrowband.sh`
- `scripts/cnsl/April2026/exp_d_1_full_train_2band_normal_wideband.sh`

The `exp_d_1_full_train_2band_*` scripts pass `logger=wandb`, `+trainer.val_check_interval=0.5`, and `++model_averaging=True` (tune `val_check_interval` by protocol size per trial-monitoring skill).

Data config: `configs/data/normal_largecorpus_MBCT_MDT_cnsl.yaml`. Scripts above still target **MBCT-only** datamodules; for MDT×MBCT runs, copy the same Hydra overrides from `exp_b_1_full_train.sh` but set `experiment=.../exp_b_xlsr_conformertcm_mbct_mdt` (or `exp_d_..._mbct_mdt_lora`) and keep `batch_size` modest (default 8 in those YAMLs).

Data paths match `DATA.md` (training corpus + protocol; benchmark under `data/benchmark_telephony` for eval).

## Prerequisites

- `export XLSR_PRETRAINED_MODEL_PATH=...` (SSL checkpoint for building the network)
- For **D**, baseline weights `pretrained/S_241214_conf-1.pth` (EXP A reference) are set in the model config

## Quick smoke tests

```bash
export XLSR_PRETRAINED_MODEL_PATH=pretrained/xlsr2_300m.pt
bash scripts/cnsl/April2026/exp_b_0_dry_run.sh -d 0
bash scripts/cnsl/April2026/exp_d_0_dry_run.sh -d 0
```

## Full-strength training (monitoring skill)

Foreground:

```bash
bash scripts/cnsl/April2026/exp_b_1_full_train.sh -d <GPU_OR_MIG>
bash scripts/cnsl/April2026/exp_d_1_full_train.sh -d <GPU_OR_MIG>
```

Detached `nohup` + log + PID:

```bash
bash scripts/cnsl/April2026/exp_b_2_full_strength_nohup.sh -d <GPU_OR_MIG>
bash scripts/cnsl/April2026/exp_d_2_full_strength_nohup.sh -d <GPU_OR_MIG>
```

Logs: `logs/train/exp_B_full_strength_mbct/` and `logs/train/exp_D_full_strength_mbct_lora/`.

These scripts pass `logger=wandb`, `+trainer.val_check_interval=0.5`, and `++model_averaging=True` as in the existing April2026 MBCT LoRA full-strength script.
