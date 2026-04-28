"""
Clean audio processing implementation for ADTOF without madmom dependency.
"""

from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 44100,
        fps: int = 100,
        frame_size: int = 2048,
        bands_per_octave: int = 12,
        fmin: float = 20.0,
        fmax: float = 20000.0,
        n_channels: int = 1,
        normalize: bool = False,
    ):
        self.sample_rate = sample_rate
        self.fps = fps
        self.frame_size = frame_size
        self.bands_per_octave = bands_per_octave
        self.fmin = fmin
        self.fmax = fmax
        self.n_channels = n_channels
        self.normalize = normalize
        self.hop_length = int(np.round(sample_rate / fps))
        self.n_fft = frame_size
        self._setup_filterbank()

    def _setup_filterbank(self) -> None:
        target_frequencies = self._log_frequencies(self.bands_per_octave, self.fmin, self.fmax)
        fft_freqs = np.fft.fftfreq(self.n_fft, 1 / self.sample_rate)[: self.n_fft // 2]
        bins = self._frequencies_to_bins(target_frequencies, fft_freqs, unique_bins=True)
        self.filterbank = self._create_madmom_filterbank(bins, len(fft_freqs)).astype(np.float32, copy=False)
        self.n_bins = self.filterbank.shape[0]

    def _log_frequencies(self, bands_per_octave: int, fmin: float, fmax: float) -> np.ndarray:
        freqs = []
        factor = 2.0 ** (1.0 / bands_per_octave)
        f = fmin
        while f <= fmax * (1.0 + 1e-12):
            freqs.append(f)
            f *= factor
        return np.array(freqs, dtype=float)

    def _frequencies_to_bins(self, frequencies: np.ndarray, fft_freqs: np.ndarray, unique_bins: bool = True) -> np.ndarray:
        bins = np.empty(len(frequencies), dtype=int)
        for i, f in enumerate(frequencies):
            bins[i] = int(np.argmin(np.abs(fft_freqs - f)))
        if unique_bins:
            unique_bins_array = []
            last_bin = -1
            for bin_idx in bins:
                if bin_idx > last_bin:
                    unique_bins_array.append(int(bin_idx))
                    last_bin = int(bin_idx)
            bins = np.array(unique_bins_array, dtype=int)
        return bins

    def _create_madmom_filterbank(self, bins: np.ndarray, n_fft_bins: int) -> np.ndarray:
        n_filters = len(bins) - 2
        filterbank = np.zeros((n_filters, n_fft_bins), dtype=np.float32)
        for i in range(n_filters):
            left_bin = int(bins[i])
            center_bin = int(bins[i + 1])
            right_bin = int(bins[i + 2])
            if right_bin - left_bin < 2:
                if 0 <= left_bin < n_fft_bins:
                    filterbank[i, left_bin] = 1.0
                continue
            if center_bin > left_bin:
                for b in range(left_bin, center_bin):
                    filterbank[i, b] = (b - left_bin) / float(center_bin - left_bin)
            if 0 <= center_bin < n_fft_bins:
                filterbank[i, center_bin] = 1.0
            if right_bin > center_bin + 0:
                for b in range(center_bin + 1, min(right_bin, n_fft_bins)):
                    filterbank[i, b] = (right_bin - b) / float(right_bin - center_bin)
        filter_sums = np.sum(filterbank, axis=1, keepdims=True)
        filter_sums[filter_sums == 0] = 1
        filterbank = filterbank / filter_sums
        return filterbank

    def load_audio(self, audio_path: str) -> np.ndarray:
        # Use soundfile to load (no finite-check), then sanitize and resample
        audio, sr = sf.read(audio_path, dtype='float32')
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # soundfile returns (samples, channels); mix to mono if needed
        if self.n_channels == 1:
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        elif self.n_channels == 2:
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            elif audio.ndim > 1:
                audio = audio.T  # (channels, samples) for librosa convention
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        if self.normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        audio = audio.astype(np.float32, copy=False)
        return audio

    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=np.hanning(self.n_fft).astype(np.float32),
            center=True,
            pad_mode='constant',
        )
        magnitude = np.abs(stft)[: self.n_fft // 2, :].astype(np.float32, copy=False)
        return magnitude

    def apply_filterbank(self, spectrogram: np.ndarray) -> np.ndarray:
        filtered = (self.filterbank @ spectrogram.astype(np.float32, copy=False)).astype(np.float32, copy=False)
        filtered = np.log10(1.0 + filtered).astype(np.float32, copy=False)
        return filtered

    def process_audio(self, audio_path: str) -> np.ndarray:
        audio = self.load_audio(audio_path)
        return self._process_audio_array(audio)

    def process_waveform(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Process a waveform array directly (same as process_audio but without file loading).

        Args:
            waveform: numpy array of shape (samples,) for mono or (samples, channels) for stereo
            sr: sample rate of the input waveform

        Returns:
            Spectrogram array of shape (T, n_bins, 1) for mono
        """
        audio = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        # Handle mono/stereo conversion
        if self.n_channels == 1:
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        elif self.n_channels == 2:
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            elif audio.ndim > 1:
                audio = audio.T  # (channels, samples) for librosa convention
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        if self.normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        audio = audio.astype(np.float32, copy=False)
        return self._process_audio_array(audio)

    def _process_audio_array(self, audio: np.ndarray) -> np.ndarray:
        """Internal method to process a loaded/prepared audio array."""
        if self.n_channels == 1:
            stft = self.compute_stft(audio)
            filtered = self.apply_filterbank(stft)
            result = filtered.T.astype(np.float32, copy=False)
            result = result[:, :, np.newaxis]
        else:
            results = []
            for ch in range(self.n_channels):
                stft = self.compute_stft(audio[ch])
                filtered = self.apply_filterbank(stft)
                results.append(filtered.T)
            result = np.stack(results, axis=2).astype(np.float32, copy=False)
        return result

    def get_n_bins(self) -> int:
        return self.n_bins


def create_adtof_processor(**kwargs) -> AudioProcessor:
    defaults = {
        'sample_rate': 44100,
        'fps': 100,
        'frame_size': 2048,
        'bands_per_octave': 12,
        'fmin': 20.0,
        'fmax': 20000.0,
        'n_channels': 1,
        'normalize': False,
    }
    defaults.update(kwargs)
    return AudioProcessor(**defaults)


def process_audio_file(audio_path: str, **kwargs) -> Tuple[np.ndarray, int]:
    processor = create_adtof_processor(**kwargs)
    spectrogram = processor.process_audio(audio_path)
    return spectrogram, processor.get_n_bins()
