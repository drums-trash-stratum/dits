"""Minimal drum machine for synthesizing 16-step drum loops.

Renders monophonic kick/snare/hihat tracks from per-step velocity vectors
with configurable tempo, swing, and one-shot samples. Optional post-render
modes apply RMS-aware drum FX, master bus, and LUFS normalization.
"""

from .sequencer import DrumMachine, render_loop, load_one_shot
from .audio_processing import (
    detect_and_normalize_clipping,
    lufs_normalize_loop,
    build_drum_fx,
    build_master_fx,
)

__all__ = [
    "DrumMachine",
    "render_loop",
    "load_one_shot",
    "detect_and_normalize_clipping",
    "lufs_normalize_loop",
    "build_drum_fx",
    "build_master_fx",
]
