"""Synthetic multi-patient beat table for tests / smoke runs (no real data needed)."""

import numpy as np
import pandas as pd

from ecg_anomaly.features import ALL_FEATURES


def synthetic_table(n_records=14, beats=700, seed=0):
    rng = np.random.default_rng(seed)
    rows, durations = [], {}
    splits = ["train"] * 6 + ["val"] * 2 + ["test"] * (n_records - 8)
    for i in range(n_records):
        shift = rng.normal(0, 3, len(ALL_FEATURES))   # each patient has its own "normal"
        scale = rng.uniform(0.5, 2.0, len(ALL_FEATURES))
        cls = np.where(rng.random(beats) < 0.12, rng.choice(["S", "V", "F"], beats), "N")
        X = rng.normal(0, 1, (beats, len(ALL_FEATURES)))
        X[cls == "V"] += rng.normal(2.5, 0.5, len(ALL_FEATURES))
        X[cls == "S"][:, :4] += 1.0
        X = X * scale + shift
        d = pd.DataFrame(X, columns=ALL_FEATURES)
        d.insert(0, "cls", cls)
        d.insert(0, "split", splits[i])
        d.insert(0, "record", f"r{100 + i}")
        rows.append(d)
        durations[f"r{100 + i}"] = beats * 0.8
    return pd.concat(rows, ignore_index=True), durations
