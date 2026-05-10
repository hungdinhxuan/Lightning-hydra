# Deepfake Whisper Features LCNN Integration

## Source

- Repository: `https://github.com/piotrkawa/deepfake-whisper-features`
- Local reference: `references/deepfake-whisper-features`
- Commit: `5829735a54aca3b37b48ceb59bb206cbb41560a1`

## Environment Check

The reference README asks for Python 3.8, PyTorch 1.11.0, torchaudio 0.11,
`asteroid-filterbanks==0.4.0`, `librosa==0.9.2`, and OpenAI Whisper pinned to
`7858aa9c08d98f75575035ecd6481f462d66ca27`.

This project currently pins Python 3.9.21, PyTorch 2.8.0, torchaudio 2.8.0, and
librosa 0.9.2. Because PyTorch/Python versions conflict, the external
environment was not installed. The LCNN and MFCC/LFCC frontend code are ported
directly onto the existing repo dependencies.

## Integration Scope

- Add LCNN model component adapted from the reference LCNN baseline.
- Add Lightning module config reusing the existing `AASISTLitModule` interface.
- Add feature datamodule for protocol files in `<file_path> <subset> <label>`
  format.
- Support `mfcc` and `lfcc` double-delta features in the dataloader.
- Add smoke tests for model forward, feature dataloader, Hydra instantiation, and
  a one-step CPU training loop on synthetic WAV files.

## Files

- `src/models/components/lcnn.py`
- `src/data/feature_datamodule.py`
- `configs/model/lcnn.yaml`
- `configs/data/feature_lcnn_protocol.yaml`
- `configs/experiment/lcnn_feature_smoke.yaml`
- `tests/test_lcnn_feature_pipeline.py`

## Smoke Command

```bash
uv run pytest tests/test_lcnn_feature_pipeline.py
```

