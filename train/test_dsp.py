"""Write a fixture the browser frontend is checked against."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio import hann, log_mel, mel_filterbank  # noqa: E402

rng = np.random.default_rng(0)
audio = rng.standard_normal(20_000).astype(np.float32)
filters = mel_filterbank()
window = hann()
mel = log_mel(audio, filters, window)
mean = mel.mean(axis=0)
std = np.maximum(mel.std(axis=0), 1e-4)
norm = (mel - mean) / std
out = Path("/tmp/talksik-dsp.json")
out.write_text(
    json.dumps(
        {
            "audio": audio.astype(float).tolist(),
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "filterbank": filters.astype(float).tolist(),
            "window_fn": window.astype(float).tolist(),
            "window": 400,
            "hop": 160,
            "n_fft": 512,
            "mel": norm.astype(float).tolist(),
        }
    )
)
print(out, norm.shape)
