import numpy as np
import pytest


def synthetic_ecg(fs=250, duration=60.0, hr=70.0, noise=0.02, drift=0.3, seed=0):
    """Gaussian-sum ECG with RR variability, baseline wander and noise.

    Returns (signal, true_r_peak_samples).
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    beats, tcur = [], 0.5
    while tcur < duration - 0.5:
        beats.append(tcur)
        tcur += 60.0 / hr * max(0.6, 1 + 0.05 * rng.standard_normal())
    x = np.zeros(n)
    for bt in beats:
        for amp, mu, sig in [(0.12, -0.2, 0.025), (-0.15, -0.04, 0.010), (1.0, 0.0, 0.012),
                             (-0.25, 0.04, 0.010), (0.3, 0.25, 0.05)]:
            x += amp * np.exp(-0.5 * ((t - bt - mu) / sig) ** 2)
    x += drift * np.sin(2 * np.pi * 0.25 * t) + noise * rng.standard_normal(n)
    return x, np.round(np.array(beats) * fs).astype(int)


@pytest.fixture
def ecg():
    return synthetic_ecg()
