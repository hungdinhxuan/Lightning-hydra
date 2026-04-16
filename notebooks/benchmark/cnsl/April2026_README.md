# Post-analysis

## This is list of models that test on multiple datasets 
➜  Lightning-hydra git:(icassp26-wildspoof) ✗ ls /home/hungdx/code/Lightning-hydra/logs/results/April_2026_benchmark 
06feb26_xlsr_conformertcm_mdt_vad  LoRAMay2025  S_241214_conf-1  VIB_April_2024

## Benchmark datasets
Example layout under `--benchmark-root` (e.g. `/dev/shm/April_2026_benchmark/`):

```text
2026_April_Intern_collect   artificialanalysis_audios   commonvoice26_de_en_ko_4000
2026_April_Synthesizer_Hung itw-real-collections        kling-ai
spoof-collections_Hung      veo3_hf
```

- `2026_April_Intern_collect` — intern-collected social clips (TikTok/YouTube); filepath hierarchy drives `generator` / `collection_id` in CSV exports.
- `itw-real-collections` — in-the-wild **bonafide** audio; top-level folders (`Emilia`, `Yodas`, `CommonVoice26`, …) map to the `collection` column in `itw_real_collections_detailed.csv`.
- `spoof-collections_Hung` — **spoof-only** TTS mix; top-level folder names (e.g. `MagpieTTS`, `Voxtral_TTS`, `Qwen3-TTS`, …) map to `engine` in `spoof_collections_Hung_detailed.csv`.
- `2026_April_Synthesizer_Hung` — **spoof-only** suite (`OmniVoice`, `MiraTTS`, …); subfolders under each engine become `mode` / `variant` in detailed CSVs; see `2026_April_Synthesizer_Hung_engine_aggregate.csv` for per-engine rollups.

## each models contain different bechmark set,  summary_results.txt contains whole results from all of dataset (EER, Accuracy)
➜  Lightning-hydra git:(icassp26-wildspoof) ✗ tree -L 2 /home/hungdx/code/Lightning-hydra/logs/results/April_2026_benchmark
/home/hungdx/code/Lightning-hydra/logs/results/April_2026_benchmark
├── 06feb26_xlsr_conformertcm_mdt_vad
│   ├── artificialanalysis_audios_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_06feb26_xlsr_conformertcm_mdt_vad.txt
│   ├── kling-ai_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_06feb26_xlsr_conformertcm_mdt_vad.txt
│   ├── merged_protocol_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_06feb26_xlsr_conformertcm_mdt_vad.txt
│   ├── merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_06feb26_xlsr_conformertcm_mdt_vad.txt
│   ├── pooled_merged_protocol_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_06feb26_xlsr_conformertcm_mdt_vad.txt
│   ├── summary_results.txt
│   └── veo3_hf_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_06feb26_xlsr_conformertcm_mdt_vad.txt
├── LoRAMay2025
│   ├── artificialanalysis_audios_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_LoRAMay2025.txt
│   ├── kling-ai_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_LoRAMay2025.txt
│   ├── merged_protocol_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_LoRAMay2025.txt
│   ├── merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_LoRAMay2025.txt
│   ├── pooled_merged_protocol_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_LoRAMay2025.txt
│   ├── summary_results.txt
│   └── veo3_hf_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_more_elevenlabs_LoRAMay2025.txt
├── S_241214_conf-1
│   ├── artificialanalysis_audios_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_S_241214_conf-1.txt
│   ├── kling-ai_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_S_241214_conf-1.txt
│   ├── merged_protocol_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_S_241214_conf-1.txt
│   ├── merged_scores_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_S_241214_conf-1.txt
│   ├── pooled_merged_protocol_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_S_241214_conf-1.txt
│   ├── summary_results.txt
│   └── veo3_hf_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_S_241214_conf-1.txt
└── VIB_April_2024
    ├── artificialanalysis_audios_cnsl_xlsr_vib_large_corpus_VIB_April_2024.txt
    ├── kling-ai_cnsl_xlsr_vib_large_corpus_VIB_April_2024.txt
    ├── merged_protocol_cnsl_xlsr_vib_large_corpus_VIB_April_2024.txt
    ├── merged_scores_cnsl_xlsr_vib_large_corpus_VIB_April_2024.txt
    ├── pooled_merged_protocol_cnsl_xlsr_vib_large_corpus_VIB_April_2024.txt
    ├── summary_results.txt
    └── veo3_hf_cnsl_xlsr_vib_large_corpus_VIB_April_2024.txt

## Implemented
- Script: `scripts/benchmark_py/export_april2026_tables.py`
- Purpose: export benchmark CSV tables per model, including summary metrics and **hierarchical** breakdowns where merged protocols use a stable relative path under each dataset folder.

**Inputs**
- `--results-root`: directory with one subfolder per model (each containing `summary_results.txt` with `MERGED_PROTOCOL` / `MERGED_SCORES`).
- `--benchmark-root`: dataset root (printed for context only; paths are resolved from the merged protocol stored under `results-root`).

**Outputs** (written to `<results-root>/csv_reports/` by default):

| File | Contents |
|------|----------|
| `summary_by_model_dataset.csv` | One row per `(model, dataset)` from each `summary_results.txt`. |
| `artificialanalysis_detailed.csv` | `(model, generator, gender, country)` from path depth under `artificialanalysis_audios/`. |
| `artificialanalysis_generator_aggregate.csv` | Aggregated per `(model, generator)`. |
| `intern_collect_detailed.csv` | `(model, generator, collection_id)` under `2026_April_Intern_collect/`. |
| `intern_collect_generator_aggregate.csv` | Aggregated per `(model, generator)`. |
| `itw_real_collections_detailed.csv` | `(model, collection)` — first path segment under `itw-real-collections/` (bonafide-only slices; EER often blank). |
| `spoof_collections_Hung_detailed.csv` | `(model, engine)` — first segment under `spoof-collections_Hung/` (spoof-only; EER often blank). |
| `2026_April_Synthesizer_Hung_detailed.csv` | `(model, engine, mode, variant)` — segments 1–3 after `2026_April_Synthesizer_Hung/` (e.g. OmniVoice scenario × language). |
| `2026_April_Synthesizer_Hung_engine_aggregate.csv` | Aggregated per `(model, engine)` within that dataset. |

**Path example for `artificialanalysis_audios`:**

```text
/dev/shm/April_2026_benchmark/artificialanalysis_audios
├── Async_Flash_v1.0          # generator
│   ├── female                # gender
│   │   ├── UK                # country
│   │   │   ├── 02d19590-e6d9-4a7d-912f-93ea1bd29993.mp3
│   │   │   └── ...
```

## Usage
```bash
uv run python scripts/benchmark_py/export_april2026_tables.py \
  --results-root logs/results/April_2026_benchmark \
  --benchmark-root /dev/shm/April_2026_benchmark
```

## Output CSV files

All paths below are relative to `--results-root` (example: `logs/results/April_2026_benchmark/csv_reports/`).

- `summary_by_model_dataset.csv` — one row per `(model, dataset)` from each `summary_results.txt`.
- `artificialanalysis_detailed.csv` — one row per `(model, generator, gender, country)`; columns include `samples`, `accuracy`, `eer`, `threshold`, `min_score`, `max_score`.
- `artificialanalysis_generator_aggregate.csv` — one row per `(model, generator)`.
- `intern_collect_detailed.csv` — one row per `(model, generator, collection_id)` for `2026_April_Intern_collect`.
- `intern_collect_generator_aggregate.csv` — one row per `(model, generator)` for that intern set.
- `itw_real_collections_detailed.csv` — one row per `(model, collection)` for `itw-real-collections` (typically **bonafide-only** groups; use `accuracy` as acceptance of real speech).
- `spoof_collections_Hung_detailed.csv` — one row per `(model, engine)` for `spoof-collections_Hung` (**spoof-only**; use `accuracy` as spoof detection rate).
- `2026_April_Synthesizer_Hung_detailed.csv` — one row per `(model, engine, mode, variant)` under `2026_April_Synthesizer_Hung` (fine-grained; many rows for OmniVoice subfolders).
- `2026_April_Synthesizer_Hung_engine_aggregate.csv` — one row per `(model, engine)` for the same dataset (coarser rollup).

## Notes
- EER is empty/`NaN` when a group has only one class label (for example **bonafide-only** or **spoof-only** subsets).
- The script resolves `MERGED_PROTOCOL` and `MERGED_SCORES` from each model's `summary_results.txt`.
- Social media source tags (TikTok/YouTube) are part of the dataset curation metadata; intern CSVs group by generator and `collection_id` from filepath hierarchy.
- For `2026_April_Synthesizer_Hung`, shallow paths (e.g. `MiraTTS/<lang>/<file>.wav`) fill only `engine` and `mode`; `variant` may repeat per utterance—use `2026_April_Synthesizer_Hung_engine_aggregate.csv` for high-level comparisons.