"""Zero-phase filters and a simple bad-segment detector for ECG."""

from __future__ import annotations

import numpy as np
from scipy import signal


def _apply(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=float))


def highpass(x, fs: float, cutoff: float = 0.5, order: int = 2) -> np.ndarray:
    return _apply(signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos"), x)


def lowpass(x, fs: float, cutoff: float = 40.0, order: int = 4) -> np.ndarray:
    return _apply(signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos"), x)


def bandpass(x, fs: float, low: float, high: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return _apply(sos, x)


def notch(x, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    b, a = signal.iirnotch(freq, q, fs=fs)
    return signal.filtfilt(b, a, np.asarray(x, dtype=float))


def preprocess(
    x, fs: float, hp: float = 0.5, lp: float = 40.0, notch_hz: float | None = None
) -> np.ndarray:
    """Baseline removal (high-pass) -> optional mains notch -> low-pass."""
    y = highpass(x, fs, hp)
    if notch_hz:
        y = notch(y, fs, notch_hz)
    return lowpass(y, fs, min(lp, 0.45 * fs))


def bad_segment_mask(
    x, fs: float, win_s: float = 2.0, low_ratio: float = 0.05, high_ratio: float = 8.0
) -> np.ndarray:
    """Boolean mask (True = unusable) from peak-to-peak amplitude per window.

    A window is flagged if its peak-to-peak range is far below (flat line, lead off) or far
    above (saturation, motion artifact) the median window range of the record.
    These are heuristics: tune ``low_ratio`` / ``high_ratio`` on real data.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = max(int(win_s * fs), 1)
    nwin = int(np.ceil(n / w))
    p2p = np.array([np.ptp(x[i * w : (i + 1) * w]) for i in range(nwin)])
    positive = p2p[p2p > 0]
    ref = float(np.median(positive)) if positive.size else 0.0
    bad = (p2p <= low_ratio * ref) | (p2p >= high_ratio * ref)
    return np.repeat(bad, w)[:n]
