"""Final evaluation of pre-selected configurations on data not used for selection."""

from __future__ import annotations

import pandas as pd

from .experiment import fit_score, make_variants, summarize_scores
from .metrics import dominated_by, paired_patient_bootstrap


def evaluate_final(train_table: pd.DataFrame, eval_sets: dict, configs: dict, baseline: str,
                   fp_targets=(30, 100), n_boot: int = 500, seed: int = 0, log=print):
    """Fit every configuration on ``train_table`` (no labels) and score each evaluation set.

    ``eval_sets``: ``name -> (table, durations)``; ``configs``: ``label -> (norm, model, features)``;
    ``baseline``: label of the reference configuration. Returns ``(results, scores)`` where
    ``scores[(set, label)]`` follows the row order of that set's table. Results carry paired
    patient-bootstrap differences against the baseline (same patients resampled for both).
    """
    scores = {}
    for label, (norm, model, fs) in configs.items():
        tr = make_variants(train_table, [norm])[norm]
        evs = {n: make_variants(t, [norm])[norm] for n, (t, _) in eval_sets.items()}
        _, out = fit_score(model, fs, tr, list(evs.values()), seed)
        for (name, _), s in zip(evs.items(), out):
            scores[(name, label)] = s
        log(f"fitted {label}: {norm} / {model} / {fs}")

    rows = []
    for name, (table, durations) in eval_sets.items():
        hours = sum(durations[r] for r in table["record"].unique()) / 3600
        df = table[["record", "cls"]].copy()
        for label in configs:
            df[label] = scores[(name, label)]
        for label, (norm, model, fs) in configs.items():
            d = df.rename(columns={label: "score"})[["record", "cls", "score"]]
            row = {"dataset": name, "config": label, "norm": norm, "model": model,
                   "features": fs, "patients": table["record"].nunique(), "hours": hours}
            row.update(summarize_scores(d, hours, fp_targets, n_boot, seed))
            if label != baseline:
                row.update(paired_patient_bootstrap(df, baseline, label, n_boot, seed))
            row["s_dominated"] = ",".join(dominated_by(table, "S"))
            rows.append(row)
    return pd.DataFrame(rows), scores
