# Checkpoints

Place the trained 3-channel ADTOF Frame-RNN checkpoint here.

Expected formats:

- `*.ckpt` — PyTorch Lightning checkpoint (keys prefixed with `model.`).
  This is the format produced by the fine-tuning script used in the paper.
- `*.pth` — bare PyTorch state dict whose keys match `ADTOFFrameRNN`.

The model architecture is built with `output_classes=3`, so the checkpoint's
output head must have 3 channels (kick, snare, hihat).

Example invocation:

```
python ../transcribe.py path/to/audio.wav --checkpoint adtof_3ch.ckpt
```
