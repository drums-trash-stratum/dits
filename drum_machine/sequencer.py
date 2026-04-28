"""Differentiable monophonic drum sequencer.

Given per-step velocity vectors and tempo, position one-shot samples on a
16-step grid (with optional MPC-style swing) and render a mixed mono loop.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchaudio


# Default 8th- and 16th-note swing index sets used by KonSequencer. The swing
# offset shifts the swing-affected steps by `(2 * swing_amount - 1) * step_len`
# samples, matching the convention used in MPC-style drum machines.
SWING_INDICES_8TH = [2, 6, 10, 14]
SWING_INDICES_16TH = [1, 3, 5, 7, 9, 11, 13, 15]


def _resolve_swing_indices(
    beat_type: str, num_steps: int = 16
) -> List[int]:
    if beat_type == "8th":
        return [i for i in SWING_INDICES_8TH if i < num_steps]
    if beat_type == "16th":
        return [i for i in SWING_INDICES_16TH if i < num_steps]
    raise ValueError(f"beat_type must be '8th' or '16th', got {beat_type!r}")


class DrumMachine:
    """Monophonic drum loop renderer.

    Args:
        num_tracks: number of percussive voices (default 3 = kick/snare/hihat).
        num_steps: total step count per loop (default 16).
        steps_per_beat: subdivision (default 4 for 16-step loops).
        sample_rate: output sample rate in Hz (default 16000).
        loop_length_secs: total rendered loop duration in seconds (default 4).
        device: torch device ("cpu" or "cuda").
    """

    def __init__(
        self,
        num_tracks: int = 3,
        num_steps: int = 16,
        steps_per_beat: int = 4,
        sample_rate: int = 16000,
        loop_length_secs: float = 4.0,
        device: str = "cpu",
    ):
        self.num_tracks = num_tracks
        self.num_steps = num_steps
        self.steps_per_beat = steps_per_beat
        self.sample_rate = sample_rate
        self.loop_length = int(round(sample_rate * loop_length_secs))
        self.device = device

        # Click/pop suppression around triggers.
        self.min_attack_ms = 0.2
        self.min_attack_len = max(1, int(sample_rate * self.min_attack_ms / 1000))
        self.fade_ms = 2.0
        self.fade_len = max(1, int(self.fade_ms * sample_rate / 1000))
        self.loop_end_fade_out_ms = 4.0
        self.loop_end_fade_out_len = max(
            1, int(self.loop_end_fade_out_ms * sample_rate / 1000)
        )

    # ------------------------------------------------------------------
    # Step grid -> sample-domain activation tensor
    # ------------------------------------------------------------------
    def _samples_per_step(self, tempo: float) -> int:
        samples_per_beat = math.floor(self.sample_rate * 60 / tempo)
        return max(1, samples_per_beat // self.steps_per_beat)

    def generate_activation_vectors(
        self,
        velo_vectors: torch.Tensor,
        tempo: float,
        swing_amounts: Sequence[float],
        beat_type: str = "8th",
    ) -> torch.Tensor:
        """Convert per-step velocities into a sample-domain activation matrix.

        Args:
            velo_vectors: (num_tracks, num_steps) float tensor in [0, 1].
            tempo: BPM.
            swing_amounts: list/tuple of length num_tracks. Values in [0.5, 0.75].
                0.5 = no swing; 0.75 = strong (triplet-feel) swing.
            beat_type: "8th" or "16th"; controls which step indices are swung.

        Returns:
            activation_vectors: (num_tracks, loop_length) tensor.
        """
        device = velo_vectors.device
        num_tracks, num_steps = velo_vectors.shape
        samples_per_step = self._samples_per_step(tempo)
        loop_one_period = samples_per_step * self.num_steps

        swing_indices = _resolve_swing_indices(beat_type, num_steps)
        if len(swing_amounts) != num_tracks:
            raise ValueError(
                f"swing_amounts must have length {num_tracks}, got {len(swing_amounts)}"
            )

        positions = (
            torch.arange(num_steps, device=device).unsqueeze(0).expand(num_tracks, -1)
            * samples_per_step
        ).clone()

        for t in range(num_tracks):
            offset = int(round((2 * float(swing_amounts[t]) - 1) * samples_per_step))
            positions[t, swing_indices] += offset

        positions = torch.clamp(positions, max=loop_one_period - 1).round().long()

        activation_one_loop = torch.zeros(num_tracks, loop_one_period, device=device)
        flat_positions = positions.reshape(-1)
        flat_velocities = velo_vectors.reshape(-1).float()
        flat_track_indices = torch.arange(
            num_tracks, device=device
        ).repeat_interleave(num_steps)
        activation_one_loop.index_put_(
            (flat_track_indices, flat_positions), flat_velocities, accumulate=True
        )

        # Tile to fill loop_length.
        num_repeats = math.ceil(self.loop_length / loop_one_period)
        tiled = activation_one_loop.repeat(1, num_repeats)
        return tiled[:, : self.loop_length]

    # ------------------------------------------------------------------
    # Monophonic rendering (per-track: a new trigger truncates the previous one)
    # ------------------------------------------------------------------
    def render(
        self,
        one_shots: torch.Tensor,
        velo_vectors: torch.Tensor,
        tempo: float,
        swing_amounts: Sequence[float],
        beat_type: str = "8th",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render multi-track monophonic drums.

        Args:
            one_shots: (num_tracks, 1, K) float tensor. K is one-shot length in samples.
            velo_vectors: (num_tracks, num_steps) float tensor in [0, 1].
            tempo: BPM (float).
            swing_amounts: list of length num_tracks with values in [0.5, 0.75].
            beat_type: "8th" or "16th".

        Returns:
            tracks: (num_tracks, 1, loop_length) rendered per-track audio.
            activation_vectors: (num_tracks, loop_length) sample-domain triggers.
        """
        activation_vectors = self.generate_activation_vectors(
            velo_vectors, tempo, swing_amounts, beat_type
        )
        B, T = activation_vectors.shape
        _, _, K = one_shots.shape
        device = activation_vectors.device

        trigger_positions = torch.where(
            activation_vectors > 0,
            torch.arange(T, device=device).view(1, -1).expand(B, -1),
            torch.full((B, T), -1, device=device),
        )
        trigger_velocities = torch.where(
            activation_vectors > 0,
            activation_vectors,
            torch.full((B, T), -1.0, device=device),
        )

        # Carry the most recent trigger position/velocity forward in time.
        last_trigger = torch.zeros_like(trigger_positions)
        last_trigger[:, 0] = trigger_positions[:, 0]
        for t in range(1, T):
            last_trigger[:, t] = torch.maximum(last_trigger[:, t - 1], trigger_positions[:, t])

        last_velocity = torch.zeros_like(trigger_velocities)
        last_velocity[:, 0] = torch.clamp_min(trigger_velocities[:, 0], 0.0)
        for t in range(1, T):
            last_velocity[:, t] = torch.where(
                trigger_velocities[:, t] >= 0,
                trigger_velocities[:, t],
                last_velocity[:, t - 1],
            )

        time_idx = torch.arange(T, device=device).view(1, -1).expand(B, -1)
        offset = time_idx - last_trigger

        # Soft attack to avoid clicks.
        env = torch.ones_like(offset, dtype=torch.float)
        attack_mask = (offset >= 0) & (offset < self.min_attack_len)
        env = torch.where(attack_mask, offset.float() / self.min_attack_len, env)
        env = torch.clamp(env, 0.0, 1.0)

        valid = (last_trigger >= 0) & (offset >= 0) & (offset < K)

        sample_flat = one_shots.reshape(B * K)
        gather_index = (
            torch.arange(B, device=device).view(-1, 1) * K + offset.clamp(0, K - 1)
        ).reshape(-1)
        sample_vals = sample_flat[gather_index].view(B, T)
        sample_vals = sample_vals.to(device) * last_velocity * env

        # Cross-fade at retrigger boundaries.
        fade = torch.ones_like(offset, dtype=torch.float)
        fade_in = offset < self.fade_len
        fade = torch.where(fade_in, offset.float() / self.fade_len, fade)
        next_trigger = torch.zeros_like(trigger_positions)
        next_trigger[:, :-1] = trigger_positions[:, 1:]
        fade_out = (next_trigger >= 0) & (offset >= 0) & (offset < self.fade_len)
        fade = torch.where(fade_out, 1.0 - (offset.float() / self.fade_len), fade)
        sample_vals = sample_vals * fade

        output = torch.where(valid, sample_vals, torch.zeros_like(sample_vals)).unsqueeze(1)
        # Remove DC offset and apply loop-end fade.
        output = output - output.mean(dim=-1, keepdim=True)
        end_fade = torch.ones(T, device=device)
        end_fade[-self.loop_end_fade_out_len :] = torch.linspace(
            1.0, 0.0, self.loop_end_fade_out_len, device=device
        )
        output = output * end_fade.view(1, 1, -1)
        return output, activation_vectors


# ----------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------
def load_one_shot(
    path: str,
    target_sample_rate: int = 16000,
    target_length_secs: float = 0.8,
    mono: bool = True,
) -> torch.Tensor:
    """Load a one-shot wav, resample, and pad/trim to a fixed length.

    Returns:
        Tensor of shape (1, K) where K = target_length_secs * target_sample_rate.
    """
    wav, sr = torchaudio.load(path)
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    K = int(round(target_length_secs * target_sample_rate))
    if wav.shape[1] >= K:
        wav = wav[:, :K]
    else:
        pad = torch.zeros(wav.shape[0], K - wav.shape[1])
        wav = torch.cat([wav, pad], dim=1)
    return wav


def render_loop(
    kick_path: str,
    snare_path: str,
    hihat_path: str,
    velo_kick: Sequence[float],
    velo_snare: Sequence[float],
    velo_hihat: Sequence[float],
    tempo: float = 120.0,
    swing_kick: float = 0.5,
    swing_snare: float = 0.5,
    swing_hihat: float = 0.5,
    beat_type: str = "8th",
    sample_rate: int = 16000,
    loop_length_secs: float = 4.0,
    one_shot_length_secs: float = 0.8,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """High-level wrapper: load three one-shots and render a 4-second loop.

    Returns:
        sum_track: (1, loop_length) mono mix.
        tracks: (3, 1, loop_length) per-track mono audio (kick, snare, hihat).
    """
    machine = DrumMachine(
        num_tracks=3,
        num_steps=16,
        steps_per_beat=4,
        sample_rate=sample_rate,
        loop_length_secs=loop_length_secs,
        device=device,
    )

    one_shots = torch.stack([
        load_one_shot(kick_path, sample_rate, one_shot_length_secs),
        load_one_shot(snare_path, sample_rate, one_shot_length_secs),
        load_one_shot(hihat_path, sample_rate, one_shot_length_secs),
    ]).to(device)  # (3, 1, K)

    velo = torch.tensor(
        [list(velo_kick), list(velo_snare), list(velo_hihat)],
        dtype=torch.float32,
        device=device,
    )
    swings = [float(swing_kick), float(swing_snare), float(swing_hihat)]

    tracks, _ = machine.render(one_shots, velo, tempo, swings, beat_type=beat_type)
    sum_track = tracks.sum(dim=0)  # (1, loop_length)
    return sum_track, tracks