"""Transcribe a drum audio file into kick / snare / hihat onset times.

Loads a 3-channel ADTOF Frame-RNN checkpoint, runs the model on the input
spectrogram, and applies madmom-style peak picking to produce per-instrument
onset times in seconds.

Usage:
    python transcribe.py audio.wav --checkpoint checkpoints/adtof_3ch.ckpt
    python transcribe.py audio.wav --checkpoint checkpoints/adtof_3ch.ckpt \
        --threshold-kick 0.5 --threshold-snare 0.5 --threshold-hihat 0.5 \
        --output onsets.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from adtof_pytorch import (
    create_frame_rnn_model,
    calculate_n_bins,
    load_pytorch_weights,
    get_default_weights_path,
)
from adtof_pytorch.audio import create_adtof_processor
from adtof_pytorch.post_processing import NotePeakPickingProcessor


TRACK_NAMES = ["kick", "snare", "hihat"]
DEFAULT_FPS = 100

# Peak-picking parameters tuned for the 3-channel ADT model. Channels
# are ordered [kick, snare, hihat] in the 3-class output head.
DEFAULT_PP_PARAMS_3CH: Sequence[Dict[str, float]] = (
    dict(threshold=0.65, post_max=0.05, pre_avg=0.14, pre_max=0.03, post_avg=0.01, combine=0.02),
    dict(threshold=0.75, post_max=0.07, pre_avg=0.16, pre_max=0.04, post_avg=0.01, combine=0.02),
    dict(threshold=0.65, post_max=0.04, pre_avg=0.10, pre_max=0.03, post_avg=0.01, combine=0.02),
)

def load_3ch_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> torch.nn.Module:
    """Load a 3-channel ADTOF model.

    Supports two formats:
      - Bare PyTorch state dict (.pth) — same key names as the model.
      - Lightning checkpoint (.ckpt) — keys prefixed with "model."; we strip
        the prefix here so we don't need pytorch_lightning at inference time.
    """
    n_bins = calculate_n_bins()
    model = create_frame_rnn_model(n_bins=n_bins, output_classes=3)

    if checkpoint_path.endswith(".ckpt"):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip the Lightning "model." prefix.
        bare = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                bare[k[len("model."):]] = v
        missing, unexpected = model.load_state_dict(bare, strict=False)
        if missing:
            print(f"  warning: {len(missing)} missing keys", file=sys.stderr)
        if unexpected:
            print(f"  warning: {len(unexpected)} unexpected keys", file=sys.stderr)
    else:
        # Fall back to the original ADTOF .pth loader.
        model = load_pytorch_weights(model, checkpoint_path, strict=False)

    return model.eval().to(device)


def transcribe(
    audio_path: str,
    checkpoint_path: str,
    thresholds: Sequence[float] | None = None,
    pp_params: Sequence[Dict[str, float]] | None = None,
    device: str = "cuda",
    fps: int = DEFAULT_FPS,
) -> Dict[str, np.ndarray]:
    """Run the full transcription pipeline on a single audio file.

    Returns:
        dict with keys "kick", "snare", "hihat" mapping to np.ndarray of onset
        times in seconds.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_3ch_checkpoint(checkpoint_path, device=device)

    processor = create_adtof_processor()
    spec = processor.process_audio(audio_path)  # (T, n_bins)

    spec_t = torch.from_numpy(spec).float().unsqueeze(0).to(device)
    with torch.no_grad():
        env = model(spec_t)[0].cpu().numpy()  # (T, 3)

    pp = list(pp_params) if pp_params is not None else [dict(p) for p in DEFAULT_PP_PARAMS_3CH]
    if thresholds is not None:
        for ch_idx, th in enumerate(thresholds):
            if th is not None:
                pp[ch_idx]["threshold"] = float(th)
    for p in pp:
        p["fps"] = fps

    detected: Dict[str, np.ndarray] = {}
    for ch_idx, name in enumerate(TRACK_NAMES):
        picker = NotePeakPickingProcessor(**pp[ch_idx])
        peaks = picker.process(env[:, ch_idx])
        detected[name] = np.array([t for t, _ in peaks], dtype=np.float32)
    return detected


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe drum audio to per-instrument onset times.",
    )
    parser.add_argument("audio", type=str, help="Path to input audio file (wav/mp3/flac).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to 3-channel ADTOF checkpoint (.ckpt or .pth). "
            "If omitted, uses the packaged 5-channel original ADTOF weights "
            "(this is a backwards-compat fallback and not the 3-channel model)."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--threshold-kick", type=float, default=None)
    parser.add_argument("--threshold-snare", type=float, default=None)
    parser.add_argument("--threshold-hihat", type=float, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set, write detected onsets to this JSON file.",
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        default_w = get_default_weights_path()
        if default_w and os.path.exists(default_w):
            print(
                "[warn] No --checkpoint provided. The packaged weights are the "
                "original 5-channel ADTOF and will not load into the 3-channel head.",
                file=sys.stderr,
            )
        raise SystemExit("Please pass --checkpoint <path-to-3ch-ckpt>.")

    thresholds = [args.threshold_kick, args.threshold_snare, args.threshold_hihat]
    if all(t is None for t in thresholds):
        thresholds = None

    onsets = transcribe(
        args.audio,
        args.checkpoint,
        thresholds=thresholds,
        device=args.device,
    )

    print(f"Audio:      {args.audio}")
    print(f"Checkpoint: {args.checkpoint}")
    for name in TRACK_NAMES:
        ts = onsets[name]
        print(f"  {name:6s}: {len(ts):3d} onsets  {np.round(ts, 3).tolist()}")

    if args.output:
        out = {name: onsets[name].tolist() for name in TRACK_NAMES}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
