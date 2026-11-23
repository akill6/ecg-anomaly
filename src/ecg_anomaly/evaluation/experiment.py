"""Run baseline configurations (normalisation x model x feature set) and collect metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import (
    ABSOLUTE_COLS,
    ALL_FEATURES,
    FEATURE_SETS,
    FUSION_GROUPS,
    FUSIONS,
    per_record_normalize,
)
from ..models import BaselineScorer, FusionScorer
from .metrics import (
    auc_scores,
    dominated_by,
    patient_bootstrap,
    per_record_auroc,
    sens_at_fp_rate,
)

NORMS = ("global", "patient", "patient-abs", "patient-abs-causal")
ALL_SETS = tuple(FEATURE_SETS) + tuple(FUSIONS)


def make_variants(table: pd.DataFrame, norms=NORMS, causal_window: int = 300) -> dict:
    """Feature tables for each normalisation.

    ``global``: raw features. ``patient``: all features scaled inside the patient's record.
    ``patient-abs``: only the patient-dependent features (``ABSOLUTE_COLS``) are scaled.
    ``*-causal``: the same, but using only the last ``causal_window`` beats (live stream).
    """
    spec = {
        "global": None,
        "patient": (ALL_FEATURES, None),
        "patient-abs": (ABSOLUTE_COLS, None),
        "patient-causal": (ALL_FEATURES, causal_window),
        "patient-abs-causal": (ABSOLUTE_COLS, causal_window),
    }
    out = {}
    for n in norms:
        if n not in spec:
            raise ValueError(f"unknown normalisation {n!r}")
        out[n] = table if spec[n] is None else per_record_normalize(table, spec[n][0],
                                                                    window=spec[n][1])
    return out


def fit_score(model: str, fs: str, tr: pd.DataFrame, evals: list[pd.DataFrame], seed: int = 0,
              n_estimators: int = 200):
    """Fit one configuration on ``tr`` (no labels) and score train and every table in ``evals``.

    ``fs`` is a feature-set name or a fusion name. Returns ``(train_scores, [eval_scores])``.
    """
    if fs in FUSIONS:
        sc = FusionScorer(FUSION_GROUPS, model, FUSIONS[fs], seed, n_estimators).fit(tr)
        return sc.score(tr), [sc.score(e) for e in evals]
    cols = FEATURE_SETS[fs]
    sc = BaselineScorer(model, seed=seed, n_estimators=n_estimators).fit(tr[cols].to_numpy())
    return sc.score(tr[cols].to_numpy()), [sc.score(e[cols].to_numpy()) for e in evals]


def summarize_scores(te: pd.DataFrame, hours: float, fp_targets=(30, 100), n_boot: int = 200,
                     seed: int = 0) -> dict:
    """All metrics for one scored table ``te`` (columns record, cls, score).

    ``*_adj`` leave out records where S outnumbers N (their 'normal' is not N).
    """
    s_dom = dominated_by(te, "S")
    keep = ~te.record.isin(s_dom).to_numpy()
    a = auc_scores(te.cls, te.score)
    ci = patient_bootstrap(te, n_boot=n_boot, seed=seed)
    pr = per_record_auroc(te)
    row = {
        "auroc": a["auroc"], "auroc_lo": ci["auroc"][0], "auroc_hi": ci["auroc"][1],
        "auprc": a["auprc"], "auprc_lo": ci["auprc"][0], "auprc_hi": ci["auprc"][1],
        "auroc_V": auc_scores(te.cls, te.score, ("V",))["auroc"],
        "auroc_S": auc_scores(te.cls, te.score, ("S",))["auroc"],
        "auroc_S_adj": auc_scores(te.cls[keep], te.score[keep], ("S",))["auroc"],
        "auroc_patient_median": float(pr.auroc.median()) if len(pr) else np.nan,
        "prevalence": a["prevalence"],
    }
    for t in fp_targets:
        s = sens_at_fp_rate(te.cls, te.score, hours, t)
        adj = sens_at_fp_rate(te.cls, te.score, hours, t, keep=keep)
        row[f"sens_all@{t}"] = s["sens_anomaly"]
        for c in ("V", "S", "F"):
            row[f"sens_{c}@{t}"] = s[f"sens_{c}"]
        row[f"sens_all_adj@{t}"] = adj["sens_anomaly"]
        row[f"sens_S_adj@{t}"] = adj["sens_S"]
        # end to end: false detections of the R-peak detector (class X) also count as false alarms
        e2e = sens_at_fp_rate(te.cls, te.score, hours, t, keep=keep, fp_classes=("N", "X"))
        row[f"sens_all_e2e@{t}"] = e2e["sens_anomaly"]
        row[f"sens_V_e2e@{t}"] = e2e["sens_V"]
        row[f"sens_S_e2e@{t}"] = e2e["sens_S"]
    return row


def run_baselines(table, durations, models=("iforest", "gmm"), sets=None, norms=NORMS,
                  fp_targets=(30, 100), n_boot=200, alarm_rate=0.05, seed=0, log=print):
    """Fit on train (no labels), evaluate on test. Returns ``(results, scores)``.

    ``scores`` maps ``(norm, model, feature_set)`` to the test-beat scores (rows follow
    ``table[table.split == 'test']``). ``transfer`` is the share of test beats above the
    threshold that flags ``alarm_rate`` of the train beats: 1.0 means the threshold carries
    over to unseen patients. NOTE: exploring many configurations on the test split is only for
    development; choose the final configuration with ``cross_validate`` instead.
    """
    sets = sets or list(ALL_SETS)
    te_mask = (table.split == "test").to_numpy()
    tr_mask = (table.split == "train").to_numpy()
    hours = sum(durations[r] for r in table.loc[te_mask, "record"].unique()) / 3600
    rows, scores = [], {}
    for norm, tab in make_variants(table, norms).items():
        tr, te = tab[tr_mask], tab[te_mask].copy()
        for model in models:
            for fs in sets:
                s_tr, (s_te,) = fit_score(model, fs, tr, [te], seed)
                te["score"] = s_te
                scores[(norm, model, fs)] = s_te.copy()
                row = {"norm": norm, "model": model, "features": fs}
                row.update(summarize_scores(te, hours, fp_targets, n_boot, seed))
                thr = np.quantile(s_tr, 1 - alarm_rate)
                row["transfer"] = float((s_te > thr).mean() / alarm_rate)
                rows.append(row)
                log(f"{norm:18s} {model:7s} {fs:15s} AUROC {row['auroc']:.3f}  "
                    f"AUPRC {row['auprc']:.3f}")
    res = pd.DataFrame(rows)
    res.attrs["s_dominated_records"] = dominated_by(table[te_mask], "S")
    res.attrs["test_hours"] = hours
    return res, scores
