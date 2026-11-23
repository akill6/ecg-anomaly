"""Cut a recording into beats and compute RR-interval features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_beats(
    x,
    rpeaks,
    fs: float,
    pre_s: float = 0.25,
    post_s: float = 0.45,
    normalize: str = "zscore",
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-length windows around each R peak.

    Returns ``(beats, kept_peaks)``: beats has shape (n, pre+post); peaks too close to the
    record edges are dropped. ``normalize='zscore'`` standardises each beat (this discards
    absolute amplitude); use ``'none'`` to keep it.
    """
    x = np.asarray(x, dtype=float)
    rpeaks = np.asarray(rpeaks, dtype=int)
    pre, post = int(round(pre_s * fs)), int(round(post_s * fs))
    ok = (rpeaks - pre >= 0) & (rpeaks + post <= len(x))
    peaks = rpeaks[ok]
    if peaks.size == 0:
        return np.empty((0, pre + post)), peaks
    beats = np.stack([x[p - pre : p + post] for p in peaks])
    if normalize == "zscore":
        beats = (beats - beats.mean(axis=1, keepdims=True)) / (beats.std(axis=1, keepdims=True) + 1e-8)
    elif normalize != "none":
        raise ValueError("normalize must be 'zscore' or 'none'")
    return beats, peaks


def rr_features(rpeaks, fs: float, median_beats: int = 11) -> dict[str, np.ndarray]:
    """Per-beat RR features: previous/next RR [s], local median RR and their ratios."""
    rpeaks = np.asarray(rpeaks)
    if len(rpeaks) < 2:
        empty = np.empty(0)
        return {"rr_prev": empty, "rr_next": empty, "rr_median": empty,
                "ratio_prev": empty, "ratio_next": empty}
    rr = np.diff(rpeaks) / fs
    rr_prev = np.r_[rr[0], rr]
    rr_next = np.r_[rr, rr[-1]]
    local = pd.Series(rr_prev).rolling(median_beats, center=True, min_periods=1).median().to_numpy()
    return {
        "rr_prev": rr_prev,
        "rr_next": rr_next,
        "rr_median": local,
        "ratio_prev": rr_prev / local,
        "ratio_next": rr_next / local,
    }
