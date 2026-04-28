"""Audio post-processing helpers used by render mode 4 (FX + LUFS norm).

Mode 1 only needs `detect_and_normalize_clipping`. Mode 4 additionally uses
`build_drum_fx`, `build_master_fx`, and `lufs_normalize_loop`.

The drum-FX chain (per-track HP filter, peak EQ, compressor, optional reverb)
samples its parameters randomly from instrument-specific ranges; pass a NumPy
random state to `build_drum_fx` for reproducibility.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pyloudnorm as pyln
import torch
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    Limiter,
    PeakFilter,
    Pedalboard,
    Reverb,
)


# ----------------------------------------------------------------------
# Clipping / LUFS
# ----------------------------------------------------------------------
def detect_and_normalize_clipping(
    audio: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Divide by max-abs if it exceeds 1.0 (no-op otherwise)."""
    max_abs = torch.max(torch.abs(audio))
    if max_abs > 1:
        return audio / max_abs, max_abs
    return audio, torch.tensor(1.0)


def lufs_normalize_loop(
    audio: torch.Tensor, sample_rate: int, target_lufs: float = -16.0
) -> Tuple[torch.Tensor, float]:
    """Scale `audio` to `target_lufs` integrated loudness; returns the gain.

    Silent input (loudness = -inf) is returned unchanged with gain=1.0.
    """
    meter = pyln.Meter(sample_rate)
    audio_np = audio.squeeze().cpu().numpy()
    input_loudness = meter.integrated_loudness(audio_np)
    if not np.isfinite(input_loudness):
        return audio, 1.0
    gain = float(np.power(10.0, (target_lufs - input_loudness) / 20.0))
    return gain * audio, gain


# ----------------------------------------------------------------------
# RMS pre-gain
# ----------------------------------------------------------------------
_TARGET_RMS_DB_RANGES = {
    "kick":  (-17.0, -13.0),
    "snare": (-19.0, -15.0),
    "hihat": (-25.0, -19.0),
}


def _rms_db(signal: np.ndarray, eps: float = 1e-8) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(signal ** 2) + eps) + eps)


def _build_rms_pregain(
    signal: np.ndarray, track_type: str, rng: np.random.RandomState
) -> Tuple[Gain, Dict[str, float]]:
    lo, hi = _TARGET_RMS_DB_RANGES.get(track_type, (-18.0, -18.0))
    target_rms = float(rng.uniform(lo, hi))
    gain_db = float(np.clip(target_rms - _rms_db(signal), -12.0, 12.0))
    return Gain(gain_db=gain_db), {
        "target_rms_db": target_rms,
        "applied_gain_db": gain_db,
    }


# ----------------------------------------------------------------------
# Per-track FX chain
# ----------------------------------------------------------------------
def build_drum_fx(
    track_type: str,
    input_signal: np.ndarray,
    rms_pre_gain: bool = True,
    rng: np.random.RandomState | None = None,
) -> Tuple[Pedalboard, Dict]:
    """Build a per-track FX chain (RMS pre-gain → EQ → compression → reverb).

    All EQ/compressor/reverb parameters are sampled from instrument-specific
    ranges. Pass a `rng` for reproducible renders.
    """
    if rng is None:
        rng = np.random.RandomState()

    log: Dict = {}
    board = Pedalboard([])

    if rms_pre_gain:
        pregain, pregain_log = _build_rms_pregain(input_signal, track_type, rng)
        board.append(pregain)
        log["rms_pregain"] = pregain_log

    # EQ
    if track_type == "kick":
        if rng.rand() < 0.33:
            hp = float(rng.uniform(20, 40))
            board.append(HighpassFilter(hp))
            log["highpass_hz"] = hp
        freq = float(rng.uniform(50, 90))
        gain = float(rng.uniform(-3, 4))
        q = float(rng.uniform(0.7, 1.5))
        board.append(PeakFilter(freq, gain, q))
        log["peak_eq"] = dict(freq=freq, gain_db=gain, q=q)

    elif track_type == "snare":
        if rng.rand() < 0.33:
            hp = float(rng.uniform(20, 80))
            board.append(HighpassFilter(hp))
            log["highpass_hz"] = hp
        freq = float(rng.uniform(180, 250))
        gain = float(rng.uniform(-3, 5))
        q = float(rng.uniform(0.8, 2.0))
        board.append(PeakFilter(freq, gain, q))
        log["peak_eq"] = dict(freq=freq, gain_db=gain, q=q)

    elif track_type == "hihat":
        hp = float(rng.uniform(200, 2000))
        board.append(HighpassFilter(hp))
        log["highpass_hz"] = hp
        freq = float(rng.uniform(8000, 12000))
        gain = float(rng.uniform(-4, 4))
        q = float(rng.uniform(0.5, 1.2))
        board.append(PeakFilter(freq, gain, q))
        log["peak_eq"] = dict(freq=freq, gain_db=gain, q=q)

    # Compression (70% probability)
    if rng.rand() < 0.7:
        threshold_db = float(rng.uniform(-18, -10))
        ratio = float(rng.uniform(2, 5))
        attack_ms = float(rng.uniform(5, 20))
        release_ms = float(rng.uniform(50, 150))
        board.append(
            Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
        )
        log["compressor"] = dict(
            threshold_db=threshold_db, ratio=ratio,
            attack_ms=attack_ms, release_ms=release_ms,
        )

    # Subtle send-like reverb on snare/hihat (33% probability)
    if track_type != "kick" and rng.rand() < 0.33:
        room_size = float(rng.uniform(0.1, 0.3))
        damping = float(rng.uniform(0.4, 0.7))
        wet_level = float(rng.uniform(0.05, 0.25))
        board.append(
            Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_level,
                dry_level=1.0,
            )
        )
        log["reverb"] = dict(
            room_size=room_size, damping=damping,
            wet_level=wet_level, dry_level=1.0,
        )

    return board, log


# ----------------------------------------------------------------------
# Master bus
# ----------------------------------------------------------------------
def build_master_fx() -> Tuple[Pedalboard, Dict]:
    """Master bus: gentle compression + brick-wall limiter at -1 dBFS."""
    board = Pedalboard([
        Compressor(threshold_db=-12, ratio=2, attack_ms=30, release_ms=200),
        Limiter(threshold_db=-1.0),
    ])
    log = {
        "compressor": {
            "threshold_db": -12, "ratio": 2,
            "attack_ms": 30, "release_ms": 200,
        },
        "limiter": {"threshold_db": -1.0},
    }
    return board, log
