"""NSP reading, resampling, and log-mel features shared by training and export."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

SR = 16_000
N_FFT = 512
WIN = 400
HOP = 160
N_MELS = 60
FMIN = 0.0
FMAX = 8_000.0
TARGET = 20_000  # 1.25 s at 16 kHz


def read_nsp(path: Path | bytes) -> tuple[np.ndarray, int]:
    data = path.read_bytes() if isinstance(path, Path) else path
    if data[:4] != b"FORM":
        raise ValueError("not an NSP file")
    idx = data.find(b"SDA_")
    if idx < 12:
        raise ValueError("NSP data chunk missing")
    sample_rate = struct.unpack_from("<I", data, idx - 12)[0]
    nbytes = struct.unpack_from("<I", data, idx + 4)[0]
    pcm = np.frombuffer(data[idx + 8 : idx + 8 + nbytes], dtype="<i2")
    audio = pcm.astype(np.float32) / 32768.0
    return audio, int(sample_rate)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int = SR) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = int(np.gcd(src_sr, dst_sr))
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def loudest_window(audio: np.ndarray, length: int = TARGET) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio - float(audio.mean()) if audio.size else audio
    if audio.size <= length:
        out = np.zeros(length, dtype=np.float32)
        start = (length - audio.size) // 2
        out[start : start + audio.size] = audio
        return out
    frame = HOP
    n = 1 + (audio.size - length) // frame
    starts = np.arange(n) * frame
    energy = np.empty(n, dtype=np.float32)
    for i, start in enumerate(starts):
        chunk = audio[start : start + length]
        energy[i] = np.dot(chunk, chunk)
    start = int(starts[int(np.argmax(energy))])
    return audio[start : start + length].copy()


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int = SR,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """HTK mel filterbank, shape (n_mels, n_fft // 2 + 1)."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_freqs)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    weights = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, center):
            if 0 <= j < n_freqs and center != left:
                weights[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < n_freqs and right != center:
                weights[i, j] = (right - j) / (right - center)
    return weights.astype(np.float32)


def hann(n: int = WIN) -> np.ndarray:
    i = np.arange(n, dtype=np.float32)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))


def log_mel(audio: np.ndarray, filters: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram, shape (frames, n_mels)."""
    if audio.size < WIN:
        raise ValueError("audio shorter than the window")
    n_frames = 1 + (audio.size - WIN) // HOP
    spec = np.empty((n_frames, filters.shape[0]), dtype=np.float32)
    padded = np.zeros(N_FFT, dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP
        padded[:WIN] = audio[start : start + WIN] * window
        padded[WIN:] = 0
        power = np.abs(np.fft.rfft(padded, n=N_FFT)) ** 2
        spec[i] = np.log(np.maximum(filters @ power, 1e-10))
    return spec


def n_frames(length: int = TARGET) -> int:
    return 1 + (length - WIN) // HOP
