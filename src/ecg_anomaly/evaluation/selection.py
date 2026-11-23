"""Model selection by patient-wise cross-validation on the development records (DS1).

The test split and external databases are never used here: the configuration is chosen from
leave-patients-out folds, then evaluated once (see ``final.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import FEATURE_SETS, FUSION_GROUPS, FUSIONS
from .experiment import ALL_SETS, NORMS, fit_score, make_variants
from .metrics import ANOMALY_CLASSES, auc_scores

NORM_COMPLEXITY = {"global": 0, "patient": 1, "patient-abs": 2, "patient-causal": 3,
                   "patient-abs-causal": 3}


def make_folds(table: pd.DataFrame, k: int = 5) -> dict[str, int]:
    """Assign whole records to ``k`` folds, balancing the number of anomalous beats (snake order).

    Records never share a fold with other folds' training data, so every patient is unseen
    when it is scored.
    """
    n_anom = (table.assign(a=table.cls.isin(ANOMALY_CLASSES)).groupby("record")["a"].sum()
              .sort_values(ascending=False, kind="stable"))
    folds = {}
    for i, rec in enumerate(n_anom.index):
        r = i % (2 * k)
        folds[rec] = r if r < k else 2 * k - 1 - r
    return folds


def default_configs(norms=NORMS, models=("iforest", "gmm"), sets=None):
    sets = sets or list(ALL_SETS)
    return [(n, m, s) for n in norms for m in models for s in sets]


def n_features(fs: str) -> int:
    if fs in FUSIONS:
        return sum(len(c) for c in FUSION_GROUPS.values())
    return len(FEATURE_SETS[fs])


def cross_validate(table: pd.DataFrame, configs, k: int = 5, seed: int = 0,
                   n_estimators: int = 100, log=print) -> pd.DataFrame:
    """Patient-wise k-fold CV. Returns one row per (configuration, fold).

    ``table`` holds the development records only (train + val). Models are fitted on the other
    folds without labels; labels are used only to score the held-out fold.
    """
    fold_of = table["record"].map(make_folds(table, k)).to_numpy()
    variants = make_variants(table, sorted({c[0] for c in configs}, key=NORM_COMPLEXITY.get))
    rows = []
    for ci, (norm, model, fs) in enumerate(configs, 1):
        tab = variants[norm]
        for f in range(k):
            tr, te = tab[fold_of != f], tab[fold_of == f]
            _, (s,) = fit_score(model, fs, tr, [te], seed, n_estimators)
            a = auc_scores(te.cls, s)
            rows.append({"norm": norm, "model": model, "features": fs, "fold": f,
                         "auroc": a["auroc"], "auprc": a["auprc"], "prevalence": a["prevalence"],
                         "auroc_V": auc_scores(te.cls, s, ("V",))["auroc"],
                         "auroc_S": auc_scores(te.cls, s, ("S",))["auroc"],
                         "auprc_S": auc_scores(te.cls, s, ("S",))["auprc"]})
        m = pd.DataFrame(rows[-k:])
        log(f"[{ci}/{len(configs)}] {norm:18s} {model:7s} {fs:15s} "
            f"AUPRC {m.auprc.mean():.3f}  AUROC {m.auroc.mean():.3f}")
    return pd.DataFrame(rows)


def summarize_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard error and worst fold of AUROC / AUPRC for every configuration."""
    g = cv.groupby(["norm", "model", "features"])
    out = g.agg(auprc=("auprc", "mean"), auprc_sd=("auprc", "std"), auprc_min=("auprc", "min"),
                auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), auroc_min=("auroc", "min"),
                auroc_V=("auroc_V", "mean"), auroc_S=("auroc_S", "mean"),
                auprc_S=("auprc_S", "mean"),
                folds=("fold", "count")).reset_index()
    out["auprc_se"] = out.auprc_sd / np.sqrt(out.folds)
    out["auroc_se"] = out.auroc_sd / np.sqrt(out.folds)
    out["n_features"] = out.features.map(n_features)
    return out.sort_values(["auprc", "auroc"], ascending=False).reset_index(drop=True)


def select_best(summary: pd.DataFrame) -> dict:
    """Two choices from the CV summary.

    ``best``: highest mean AUPRC (ties: AUROC). ``one_se``: the simplest configuration (fewest
    features, then simplest normalisation / model) whose mean AUPRC is within one standard error
    of the best, a standard guard against picking noise.
    """
    best = summary.iloc[0]
    ok = summary[summary.auprc >= best.auprc - best.auprc_se].copy()
    ok["cx"] = list(zip(ok.n_features, ok.norm.map(NORM_COMPLEXITY), ok.model != "iforest"))
    simple = ok.sort_values("cx").iloc[0]
    fmt = lambda r: {"norm": r.norm, "model": r.model, "features": r.features,
                     "cv_auprc": float(r.auprc), "cv_auprc_se": float(r.auprc_se),
                     "cv_auroc": float(r.auroc)}
    return {"best": fmt(best), "one_se": fmt(simple)}
