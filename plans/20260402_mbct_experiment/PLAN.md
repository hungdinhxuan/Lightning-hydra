# MBCT experiment plan (fix-duration + `NormalMBCTDataModule`)

This plan follows the repo integration workflow: data path is isolated in
`src/data/normal_mbct_datamodule.py`, model in `MBCTLitModule` / `XLSRConformertcmMBCTLitModule`,
configs under `configs/data/normal_largecorpus_MBCT_cnsl.yaml` and
`configs/experiment/cnsl/April2026/xlsr_conformertcm_mbct*.yaml`.

## What was implemented

- **Fix-duration pipeline** (same idea as `normal_datamodule.py`): `Dataset_for` pads/crops to `trim_length` in `__getitem__`.
- **`NormalMBCTDataModule`**: subclasses `NormalDataModule`, attaches `mbct_collate_fn` on train/val. By default `mbct_fix_duration_in_dataset: true` so collate uses `max_length_sec=None` and only applies band transforms (no second crop/pad).
- **`mbct_module.py`**: documents default band keys, adds `DEFAULT_MBCT_BAND_KEYS` and `band_keys_match_weighted_views()` helper.
- **Multiview datamodule**: MBCT branch removed; use `NormalMBCTDataModule` for MBCT.

## Next steps (recommended order)

1. **Environment**  
   - Set `XLSR_PRETRAINED_MODEL_PATH` (e.g. `pretrained/xlsr2_300m.pt`), and optionally `LARGE_CORPUS_FOR_CNSL` / `LARGE_CORPUS_FOR_CNSL_PROTOCOLS` if not overriding paths in the experiment YAML.

2. **Dry-run training** *(done — LoRA MBCT smoke test)*  
   - Script: **`scripts/cnsl/April2026/0_mbct_dry_run.sh`**  
     ```bash
     export XLSR_PRETRAINED_MODEL_PATH=pretrained/xlsr2_300m.pt
     bash scripts/cnsl/April2026/0_mbct_dry_run.sh -d 0
     ```  
   - Uses `+trainer.fast_dev_run=2`, `callbacks=default_lora_loss_earlystop` (avoids `LearningRateMonitor` when loggers are suppressed), `++test=False` (test loop needs `score_save_path`).  
   - Success: logs show `val/view_normal_*`, `val/view_narrowband_*`, `val/view_wideband_*` (per-band MBCT) and exit code 0.

3. **Full training** *(LoRA launched — trial-experiment-monitoring)*  
   - **Foreground** (same flags as nohup): **`scripts/cnsl/April2026/1_xlsr_conformertcm_mbct_lora.sh`** — includes `logger=wandb`, `+trainer.val_check_interval=0.5`, `++model_averaging=True`.  
   - **Background (nohup)**: **`scripts/cnsl/April2026/2_mbct_lora_full_strength_nohup.sh`** — writes `logs/train/full_strength_mbct_lora/train_<stamp>.log` and `.pid`.  
   - From-scratch (no LoRA): `experiment=cnsl/April2026/xlsr_conformertcm_mbct`.  
   - Config: `experiment=cnsl/April2026/xlsr_conformertcm_mbct_lora` (`pretrained/S_241214_conf-1.pth`, `is_base_model_path_ln: false`).

4. **Checkpoint**  
   - Use your usual averaging / best-ckpt selection (same as MDT runs).

5. **Benchmark**  
   - Run the existing telephony benchmark on `data/benchmark_telephony`.  
   - Compare **m_ailabs_v7_only_la_codec_aug** EER to baseline in `logs/results/benchmark_telephony/S_241214_conf-1/summary_results.txt`.

6. **Optional ablations**  
   - `mbct_fix_duration_in_dataset: false` + set `mbct_max_length_sec` to move duration control entirely into collate (closer to the old multiview+MBCT idea).  
   - Tune `weighted_views` per band or `cross_entropy_weight`.

7. **Trial / queue** (if using `trial-experiment-monitoring`)  
   - Register the experiment name, log `trial_id`, point eval scripts at the merged benchmark protocol for the new run.

## Config touchpoints

| Piece | Location |
|--------|----------|
| Datamodule | `configs/data/normal_largecorpus_MBCT_cnsl.yaml` → `_target_: NormalMBCTDataModule` |
| MBCT collate args | `args.mbct_fix_duration_in_dataset`, `args.mbct_band_configs`, `args.trim_length` |
| Model bands | `model.weighted_views` keys must match collate band names |

## Risk / sanity checks

- **Double duration fix**: keep `mbct_fix_duration_in_dataset: true` unless you intentionally want collate-time `_fix_length`.  
- **torchaudio**: narrowband/wideband paths require torchaudio (same as `mbct_collate_fn` tests).  
- **DDP**: batch size divisible by world size (unchanged from other datamodules).
