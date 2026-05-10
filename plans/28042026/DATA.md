# LCNN Feature Data Contract

Training data uses a root directory plus a protocol file. Each protocol line is
space-separated:

```text
<relative_or_absolute_audio_path> <subset> <label>
```

Supported subsets:

- `train`
- `dev`
- `eval`
- `test`

Supported labels:

- `bonafide` -> class `1`
- `spoof` -> class `0`

The LCNN feature datamodule loads each audio file, resamples to 16 kHz, computes
MFCC or LFCC with deltas and double-deltas, then pads or truncates the feature
sequence to `max_frames`.

