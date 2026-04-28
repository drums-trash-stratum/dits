"""Render a 16-step drum loop from a JSON or CLI configuration.

Synthesizes a 4-second monophonic kick / snare / hihat loop from per-step
velocity vectors, three one-shot wav files, a tempo, and per-track swing
amounts. Two output modes are supported:

  --mode 1   "basic"     — raw render, then divide by max-abs if clipping.
  --mode 4   "fx+lufs"   — RMS-aware pre-gain → per-track FX (HP, peak EQ,
                           compressor, optional reverb) → master bus
                           (compressor + limiter) → LUFS normalize to -16
                           → clamp to [-1, 1].

Mode 4 chooses FX parameters from instrument-specific random ranges; pass
`--seed` for reproducible renders.

Usage examples:

  python render_loop.py --config examples/config_basic_house.json \\
      --mode 4 --seed 42 --output out/house_fx.wav

  python render_loop.py \\
      --kick examples/one_shots/kick/kick.wav \\
      --snare examples/one_shots/snare/snare.wav \\
      --hihat examples/one_shots/hihat/hh.wav \\
      --tempo 120 --beat-type 8th \\
      --kick-velo  1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 \\
      --snare-velo 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 \\
      --hihat-velo 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 \\
      --mode 1 --output out/loop.wav
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchaudio

from drum_machine import (
    DrumMachine,
    load_one_shot,
    detect_and_normalize_clipping,
    lufs_normalize_loop,
    build_drum_fx,
    build_master_fx,
)


# ----------------------------------------------------------------------
# Config schema
# ----------------------------------------------------------------------
def _validate_config(cfg: Dict) -> Dict:
    required = [
        "kick_path", "snare_path", "hihat_path",
        "kick_velo", "snare_velo", "hihat_velo",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    cfg.setdefault("tempo", 120.0)
    cfg.setdefault("beat_type", "8th")
    cfg.setdefault("swing_kick", 0.5)
    cfg.setdefault("swing_snare", 0.5)
    cfg.setdefault("swing_hihat", 0.5)
    cfg.setdefault("sample_rate", 16000)
    cfg.setdefault("loop_length_secs", 4.0)
    cfg.setdefault("one_shot_length_secs", 0.8)
    cfg.setdefault("target_lufs", -16.0)
    if cfg["beat_type"] not in ("8th", "16th"):
        raise ValueError("beat_type must be '8th' or '16th'")
    for name in ("kick_velo", "snare_velo", "hihat_velo"):
        v = cfg[name]
        if len(v) != 16:
            raise ValueError(f"{name} must have length 16, got {len(v)}")
        for x in v:
            if not (0.0 <= float(x) <= 1.0):
                raise ValueError(f"{name} contains value outside [0, 1]: {x}")
    return cfg


def _parse_velo(values: List[str], name: str) -> List[float]:
    if len(values) != 16:
        raise ValueError(f"--{name} expects 16 values, got {len(values)}")
    return [float(v) for v in values]


# ----------------------------------------------------------------------
# Render modes
# ----------------------------------------------------------------------
def _render_mode1(
    sum_track: torch.Tensor, tracks: torch.Tensor
) -> Dict:
    """Mode 1: clipping normalization only."""
    sum_norm, max_abs = detect_and_normalize_clipping(sum_track)
    tracks_norm = tracks / max_abs
    return {
        "sum_track": sum_norm,
        "tracks": tracks_norm,
        "log": {"clipping_normalizer": float(max_abs)},
    }


def _render_mode4(
    sum_track: torch.Tensor,
    tracks: torch.Tensor,
    sample_rate: int,
    target_lufs: float,
    seed: int | None,
) -> Dict:
    """Mode 4: RMS-matched pre-gain → FX → master bus → LUFS norm → clamp."""
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    fx_logs: Dict = {}
    inst_names = ["kick", "snare", "hihat"]

    processed_tracks = []
    for idx, name in enumerate(inst_names):
        track_np = tracks[idx, 0].cpu().numpy().astype(np.float32)
        board, log = build_drum_fx(
            name, input_signal=track_np, rms_pre_gain=True, rng=rng,
        )
        processed_tracks.append(board(track_np, sample_rate=sample_rate))
        fx_logs[name] = log

    tracks_fx = torch.tensor(np.stack(processed_tracks), device=tracks.device).unsqueeze(1)
    sum_fx = tracks_fx.sum(dim=0)

    master_board, master_log = build_master_fx()
    sum_fx_np = sum_fx.cpu().numpy().squeeze(0)
    sum_fx_np = master_board(sum_fx_np, sample_rate=sample_rate)
    fx_logs["master"] = master_log

    sum_fx = torch.tensor(sum_fx_np, device=tracks.device).unsqueeze(0)
    sum_fx_lufs, gain = lufs_normalize_loop(sum_fx, sample_rate, target_lufs=target_lufs)
    sum_fx_lufs = torch.clamp(sum_fx_lufs, -0.999, 0.999)
    tracks_fx_lufs = torch.clamp(gain * tracks_fx, -0.999, 0.999)

    return {
        "sum_track": sum_fx_lufs,
        "tracks": tracks_fx_lufs,
        "log": {
            "fx": fx_logs,
            "lufs_norm_gain": float(gain),
            "target_lufs": target_lufs,
            "seed": seed,
        },
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Render a 16-step drum loop from a config or CLI flags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")

    parser.add_argument("--kick", dest="kick_path", type=str, default=None)
    parser.add_argument("--snare", dest="snare_path", type=str, default=None)
    parser.add_argument("--hihat", dest="hihat_path", type=str, default=None)

    parser.add_argument("--kick-velo", nargs=16, default=None, metavar="V")
    parser.add_argument("--snare-velo", nargs=16, default=None, metavar="V")
    parser.add_argument("--hihat-velo", nargs=16, default=None, metavar="V")

    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--beat-type", type=str, choices=["8th", "16th"], default="8th")
    parser.add_argument("--swing-kick", type=float, default=0.5)
    parser.add_argument("--swing-snare", type=float, default=0.5)
    parser.add_argument("--swing-hihat", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--loop-length-secs", type=float, default=4.0)
    parser.add_argument("--one-shot-length-secs", type=float, default=0.8)

    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 4],
        default=1,
        help=(
            "Render mode: 1 = basic (clipping norm only), "
            "4 = RMS pre-gain + per-track FX + master bus + LUFS norm. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-16.0,
        help="Target integrated loudness for mode 4 (default -16 LUFS).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for mode 4's FX parameter sampling. Required for reproducibility.",
    )

    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output mixed wav. Stems are saved alongside if --save-stems.",
    )
    parser.add_argument("--save-stems", action="store_true",
                        help="Also save kick / snare / hihat stems next to the output mix.")
    parser.add_argument("--save-log", action="store_true",
                        help="Also write a sibling JSON file with the render log.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()

    # Build config: --config takes precedence over CLI flags.
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        for required in ("kick_path", "snare_path", "hihat_path"):
            if getattr(args, required) is None:
                raise SystemExit(
                    f"--{required.replace('_path', '')} required when --config is not given."
                )
        for required in ("kick_velo", "snare_velo", "hihat_velo"):
            if getattr(args, required) is None:
                raise SystemExit(f"--{required.replace('_', '-')} required when --config is not given.")
        cfg = {
            "kick_path": args.kick_path,
            "snare_path": args.snare_path,
            "hihat_path": args.hihat_path,
            "kick_velo": _parse_velo(args.kick_velo, "kick-velo"),
            "snare_velo": _parse_velo(args.snare_velo, "snare-velo"),
            "hihat_velo": _parse_velo(args.hihat_velo, "hihat-velo"),
            "tempo": args.tempo,
            "beat_type": args.beat_type,
            "swing_kick": args.swing_kick,
            "swing_snare": args.swing_snare,
            "swing_hihat": args.swing_hihat,
            "sample_rate": args.sample_rate,
            "loop_length_secs": args.loop_length_secs,
            "one_shot_length_secs": args.one_shot_length_secs,
            "target_lufs": args.target_lufs,
        }

    cfg = _validate_config(cfg)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Render the dry per-track audio.
    machine = DrumMachine(
        num_tracks=3, num_steps=16, steps_per_beat=4,
        sample_rate=cfg["sample_rate"],
        loop_length_secs=cfg["loop_length_secs"],
        device=args.device,
    )
    one_shots = torch.stack([
        load_one_shot(cfg["kick_path"],  cfg["sample_rate"], cfg["one_shot_length_secs"]),
        load_one_shot(cfg["snare_path"], cfg["sample_rate"], cfg["one_shot_length_secs"]),
        load_one_shot(cfg["hihat_path"], cfg["sample_rate"], cfg["one_shot_length_secs"]),
    ]).to(args.device)
    velo = torch.tensor(
        [cfg["kick_velo"], cfg["snare_velo"], cfg["hihat_velo"]],
        dtype=torch.float32, device=args.device,
    )
    swings = [cfg["swing_kick"], cfg["swing_snare"], cfg["swing_hihat"]]
    tracks_dry, _ = machine.render(one_shots, velo, cfg["tempo"], swings, beat_type=cfg["beat_type"])
    sum_dry = tracks_dry.sum(dim=0)

    # Apply the requested mode.
    if args.mode == 1:
        result = _render_mode1(sum_dry, tracks_dry)
    elif args.mode == 4:
        result = _render_mode4(
            sum_dry, tracks_dry,
            sample_rate=cfg["sample_rate"],
            target_lufs=cfg["target_lufs"],
            seed=args.seed,
        )
    else:
        raise SystemExit(f"Unsupported --mode {args.mode}")

    # Save outputs.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sum_track = torch.clamp(result["sum_track"].cpu(), -1.0, 1.0)
    torchaudio.save(str(out_path), sum_track, cfg["sample_rate"])
    print(
        f"Wrote {out_path}  (mode={args.mode}, tempo={cfg['tempo']}, "
        f"{cfg['beat_type']}, sr={cfg['sample_rate']})"
    )

    if args.save_stems:
        names = ["kick", "snare", "hihat"]
        stems = result["tracks"]
        for i, name in enumerate(names):
            stem_path = out_path.with_name(f"{out_path.stem}_{name}.wav")
            stem = torch.clamp(stems[i].cpu(), -1.0, 1.0)
            torchaudio.save(str(stem_path), stem, cfg["sample_rate"])
            print(f"  stem: {stem_path}")

    if args.save_log:
        log_path = out_path.with_suffix(".json")
        with open(log_path, "w") as f:
            json.dump(
                {"config": cfg, "mode": args.mode, "render": result["log"]},
                f, indent=2,
            )
        print(f"  log:  {log_path}")


if __name__ == "__main__":
    main()
