"""Per-patient feature normalisation: compare every beat with the same patient's own norm."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .beat_features import FEATURE_SETS
from .spectral import sparsity_cols

ALL_FEATURES = sorted({c for cols in FEATURE_SETS.values() for c in cols})

# Features whose scale depends on the patient (heart rate, QRS width, lead gain, signal shape).
# Chosen by this principle, not by test results. The others are already ratios or deviations from
# the patient's own context (RR ratios, rhythm features, relative amplitude, template residuals);
# dividing them by their IQR again mostly amplifies noise in steady recordings.
ABSOLUTE_COLS = [
    "rr_prev", "rr_next", "rr_median", "qrs_width_s", "area_rel", "qrs_energy_frac",
    "skew", "kurt", "pre_mean", "post_mean", "p_ratio", "p_energy_frac",
    "k_prd5", "k_prd10", "top5_energy",
    *[c for t in ("dft", "haar", "db2") for c in sparsity_cols(t)],
]


def per_record_normalize(table: pd.DataFrame, cols=None, window: int | None = None,
                         min_periods: int = 30) -> pd.DataFrame:
    """Robust z-score of each feature inside its own record: ``(x - median) / IQR``.

    ``window=None`` uses the whole record (offline analysis of a finished recording).
    ``window=N`` uses only the previous N beats (causal, i.e. usable on a live stream); the
    first beats of a record use the statistics of the first ``min_periods`` beats.
    No labels are used. Median and IQR are barely moved by the few anomalous beats.
    """
    cols = list(cols or ALL_FEATURES)
    out = table.copy()
    out[cols] = out[cols].astype(float)  # integer columns (e.g. k_prd5) must hold fractions
    for _, idx in table.groupby("record", sort=False).groups.items():
        X = table.loc[idx, cols]
        if window is None:
            med = X.median()
            iqr = X.quantile(0.75) - X.quantile(0.25)
            med = pd.DataFrame([med.to_numpy()] * len(X), index=X.index, columns=cols)
            iqr = pd.DataFrame([iqr.to_numpy()] * len(X), index=X.index, columns=cols)
        else:
            r = X.rolling(window, min_periods=min(min_periods, window))
            med = r.median().bfill()
            iqr = (r.quantile(0.75) - r.quantile(0.25)).bfill()
        scale = iqr.where(iqr > 1e-9, 1.0)
        out.loc[idx, cols] = ((X - med) / scale).to_numpy()
    return out
