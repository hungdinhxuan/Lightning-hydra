
```bash
python3 scripts/convert_to_wds.py \
  --input_wav_dir data/DVC_DSD-Large-Corpus/raw/0_large-corpus_toys \
  --protocol_path data/protocols/cnsl/new_protocol_trim_vocoded_cleaned_v4_corrected.txt \
  --output_dir /dev/shm/dsd_corpus_dec_2024 \
  --shard_size_mb 256
```