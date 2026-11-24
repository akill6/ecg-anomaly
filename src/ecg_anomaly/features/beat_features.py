"""Hand-made per-beat features: RR timing, morphology and DCT sparsity (thesis idea)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..beats.segmentation import rr_features
from .spectral import TRANSFORMS, all_sparsity_features, sparsity_cols, sparsity_features

RR_COLS = ["rr_prev", "rr_next", "rr_median", "ratio_prev", "ratio_next"]
RHYTHM_COLS = ["rr_sum_ratio", "rr_prev2_ratio", "rr_next_prev", "rr_cv", "rr_dprev", "rr_dnext"]
P_COLS = ["p_ratio", "p_energy_frac"]
TPL_COLS = ["tpl_corr", "tpl_rmse", "tpl_rmse_p", "tpl_rmse_qrs", "tpl_rmse_t"]
MORPH_COLS = ["amp_rel", "qrs_width_s", "area_rel", "qrs_energy_frac", "skew", "kurt",
              "pre_mean", "post_mean"]
SPARSITY_COLS = sparsity_cols("dct")  # k_prd5, k_prd10, top5_energy (DCT, as in the original model)
SPARSITY_BY = {t: sparsity_cols(t) for t in TRANSFORMS}  # one column triple per basis
_BASE = RR_COLS + RHYTHM_COLS + MORPH_COLS + P_COLS + TPL_COLS
FEATURE_SETS = {
    "rr": RR_COLS,
    "rhythm": RR_COLS + RHYTHM_COLS,
    "morph": MORPH_COLS,
    "template": P_COLS + TPL_COLS,
    "sparsity": SPARSITY_COLS,
    "morph+sparsity": MORPH_COLS + SPARSITY_COLS,
    "all_v1": RR_COLS + MORPH_COLS + SPARSITY_COLS,   # the phase-2a feature set
    "all": RR_COLS + RHYTHM_COLS + MORPH_COLS + SPARSITY_COLS + P_COLS + TPL_COLS,
}
CORE_SETS = tuple(FEATURE_SETS)  # the sets of the selection grid (kept fixed for reproducibility)
# The same model with the sparsity basis of your choice (``sparsity`` / ``all`` are the DCT ones);
# compare them with scripts/phase4_transforms.py.
for _t in TRANSFORMS[1:]:
    FEATURE_SETS[f"sparsity_{_t}"] = SPARSITY_BY[_t]
    FEATURE_SETS[f"all_{_t}"] = _BASE + SPARSITY_BY[_t]
FEATURE_SETS["sparsity_all"] = [c for t in TRANSFORMS for c in SPARSITY_BY[t]]
FEATURE_SETS["all_multi"] = _BASE + FEATURE_SETS["sparsity_all"]
# One specialised detector per group of features; their scores are fused (see models.fusion).
FUSION_GROUPS = {
    "rhythm": RR_COLS + RHYTHM_COLS,
    "morph": MORPH_COLS,
    "template": P_COLS + TPL_COLS,
    "sparsity": SPARSITY_COLS,
}
FUSIONS = {"fusion-max": "max", "fusion-mean": "mean"}


def dct_sparsity_features(beats_z, targets_prd=(5.0, 10.0), top=5) -> dict[str, np.ndarray]:
    """DCT sparsity features (kept for compatibility); see ``spectral.sparsity_features``."""
    return sparsity_features(beats_z, "dct", targets_prd, top)


def morphology_features(beats_raw, beats_z, pre: int, fs: float) -> dict[str, np.ndarray]:
    """Amplitude, QRS width / area / energy share and shape statistics of each beat.

    ``beats_raw`` are filtered but not normalised windows (R peak at index ``pre``);
    ``beats_z`` the z-scored version. Amplitudes are made relative to the record's median
    R amplitude so they are comparable between patients.
    """
    raw = np.asarray(beats_raw, dtype=float)
    z = np.asarray(beats_z, dtype=float)
    a = np.abs(raw - np.median(raw, axis=1, keepdims=True))
    r = max(int(0.06 * fs), 1)
    lo, hi = max(pre - r, 0), pre + r + 1
    peak = a[:, lo:hi].max(axis=1)
    amp = peak / (np.median(peak) + 1e-12)

    width = np.empty(len(a))
    for i in range(len(a)):
        above = a[i] >= 0.5 * peak[i]
        j = lo + int(np.argmax(a[i, lo:hi]))
        left = j
        while left > 0 and above[left - 1]:
            left -= 1
        right = j
        while right < a.shape[1] - 1 and above[right + 1]:
            right += 1
        width[i] = (right - left + 1) / fs
    area = a.sum(axis=1) / (peak + 1e-12) / fs

    qrs = (z[:, lo:hi] ** 2).sum(axis=1) / ((z**2).sum(axis=1) + 1e-12)
    pre_w = slice(0, max(pre - int(0.1 * fs), 1))
    post_w = slice(pre + int(0.15 * fs), pre + int(0.40 * fs))
    return {
        "amp_rel": amp,
        "qrs_width_s": width,
        "area_rel": area,
        "qrs_energy_frac": qrs,
        "skew": stats.skew(z, axis=1),
        "kurt": stats.kurtosis(z, axis=1),
        "pre_mean": z[:, pre_w].mean(axis=1),
        "post_mean": z[:, post_w].mean(axis=1),
    }


def rhythm_features(rpeaks, fs: float, window: int = 11) -> dict[str, np.ndarray]:
    """Rhythm context of every beat, from the R-peak times only.

    * ``rr_sum_ratio``: (RR before + RR after) / (2 * local median RR). About 1 after a premature
      beat with a *full* compensatory pause (typical for V), below 1 for an incomplete one (S).
    * ``rr_prev2_ratio``: RR before / the RR before that (prematurity of this beat).
    * ``rr_next_prev``: RR after / RR before.
    * ``rr_cv``: local variability of RR (std / median over ``window`` beats).
    * ``rr_dprev`` / ``rr_dnext``: |change of RR| into / out of this beat, relative to the median.
    """
    rpeaks = np.asarray(rpeaks)
    if len(rpeaks) < 2:
        return {c: np.empty(0) for c in RHYTHM_COLS}
    rr = np.diff(rpeaks) / fs
    rr_prev = np.r_[rr[0], rr]
    rr_next = np.r_[rr, rr[-1]]
    rr_prev2 = np.r_[rr_prev[0], rr_prev[:-1]]
    ser = pd.Series(rr_prev)
    med = ser.rolling(window, center=True, min_periods=1).median().to_numpy()
    cv = (ser.rolling(window, center=True, min_periods=2).std().bfill().to_numpy()) / med
    return {
        "rr_sum_ratio": (rr_prev + rr_next) / (2 * med),
        "rr_prev2_ratio": rr_prev / rr_prev2,
        "rr_next_prev": rr_next / rr_prev,
        "rr_cv": cv,
        "rr_dprev": np.abs(rr_prev - rr_prev2) / med,
        "rr_dnext": np.abs(rr_next - rr_prev) / med,
    }


def causal_template(beats_z, window: int = 300, step: int = 50, min_beats: int = 30) -> np.ndarray:
    """Template (median beat) of the patient for every beat, built from PAST beats only.

    The template is refreshed every ``step`` beats from the previous ``window`` beats, so the
    features stay usable on a live stream. The very first beats use the first ``min_beats`` beats.
    """
    b = np.asarray(beats_z, dtype=float)
    n = len(b)
    tpl = np.empty_like(b)
    for i in range(0, n, step):
        if i < min_beats:
            t = np.median(b[: min(n, min_beats)], axis=0)
        else:
            t = np.median(b[max(0, i - window) : i], axis=0)
        tpl[i : i + step] = t
    return tpl


def wave_features(beats_raw, beats_z, template, pre: int, fs: float) -> dict[str, np.ndarray]:
    """P-wave presence and deviation from the patient's own template, per segment.

    ``p_ratio``: peak-to-peak amplitude in the P-wave window (80-220 ms before R) relative to the
    QRS amplitude; ``p_energy_frac``: share of beat energy there. ``tpl_*``: correlation with the
    template and RMS residual overall and in the P, QRS and ST-T segments.
    """
    raw = np.asarray(beats_raw, dtype=float)
    z = np.asarray(beats_z, dtype=float)
    L = z.shape[1]
    p_lo, p_hi = max(pre - int(0.22 * fs), 0), max(pre - int(0.08 * fs), 1)
    q_lo, q_hi = max(pre - int(0.06 * fs), 0), min(pre + int(0.08 * fs) + 1, L)
    t_lo = min(pre + int(0.10 * fs), L - 1)

    dev = raw - np.median(raw, axis=1, keepdims=True)
    qrs_amp = np.abs(dev[:, q_lo:q_hi]).max(axis=1) + 1e-12
    p_seg = dev[:, p_lo:p_hi]
    p_ratio = (p_seg.max(axis=1) - p_seg.min(axis=1)) / qrs_amp
    p_energy = (z[:, p_lo:p_hi] ** 2).sum(axis=1) / ((z**2).sum(axis=1) + 1e-12)

    zc = z - z.mean(axis=1, keepdims=True)
    tc = template - template.mean(axis=1, keepdims=True)
    corr = (zc * tc).sum(axis=1) / (np.linalg.norm(zc, axis=1) * np.linalg.norm(tc, axis=1) + 1e-12)
    res = z - template
    rmse = lambda a, b: np.sqrt((res[:, a:b] ** 2).mean(axis=1))
    return {
        "p_ratio": p_ratio,
        "p_energy_frac": p_energy,
        "tpl_corr": corr,
        "tpl_rmse": rmse(0, L),
        "tpl_rmse_p": rmse(p_lo, p_hi),
        "tpl_rmse_qrs": rmse(q_lo, q_hi),
        "tpl_rmse_t": rmse(t_lo, L),
    }


def beat_feature_table(beats_raw, beats_z, rpeaks, fs: float, pre: int) -> pd.DataFrame:
    """All hand-made features for the beats of one record (row i belongs to ``rpeaks[i]``)."""
    cols: dict[str, np.ndarray] = {}
    cols.update(rr_features(rpeaks, fs))
    cols.update(rhythm_features(rpeaks, fs))
    cols.update(morphology_features(beats_raw, beats_z, pre, fs))
    cols.update(wave_features(beats_raw, beats_z, causal_template(beats_z), pre, fs))
    cols.update(all_sparsity_features(beats_z))
    return pd.DataFrame(cols)
