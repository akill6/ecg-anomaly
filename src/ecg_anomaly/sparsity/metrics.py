"""Reconstruction-quality metrics, identical to the definitions in the MATLAB thesis app."""

from __future__ import annotations

import numpy as np


def prd(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Percent root difference: sqrt(sum(e^2) / sum(x^2)) * 100 (no mean removal)."""
    x = np.asarray(x, dtype=float)
    e = x - np.asarray(x_hat, dtype=float)
    return float(100.0 * np.sqrt(np.sum(e**2) / np.sum(x**2)))


def snr_db(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Signal-to-noise ratio in dB: 10 * log10(sum(x^2) / sum(e^2)). inf if error is zero."""
    x = np.asarray(x, dtype=float)
    e = x - np.asarray(x_hat, dtype=float)
    err = float(np.sum(e**2))
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10(np.sum(x**2) / err))
