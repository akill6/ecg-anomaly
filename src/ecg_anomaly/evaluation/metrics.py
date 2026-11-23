"""Anomaly-detection metrics: AUROC / AUPRC, sensitivity at fixed false alarms per hour,
per-patient breakdown and patient-level bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ANOMALY_CLASSES = ("S", "V", "F")  # Q (paced / unclassifiable) and X (false detection) are excluded


def eval_mask(cls) -> np.ndarray:
    """Beats used for evaluation: normal (N) and anomalous (S, V, F)."""
    return np.isin(np.asarray(cls), ("N",) + ANOMALY_CLASSES)


def auc_scores(cls, score, positives=ANOMALY_CLASSES) -> dict[str, float]:
    """AUROC and AUPRC of ``score`` for ``positives`` vs N (other classes ignored)."""
    cls, score = np.asarray(cls), np.asarray(score, dtype=float)
    m = np.isin(cls, ("N",) + tuple(positives))
    y = np.isin(cls[m], positives).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"auroc": np.nan, "auprc": np.nan, "prevalence": y.mean() if len(y) else np.nan}
    return {"auroc": roc_auc_score(y, score[m]), "auprc": average_precision_score(y, score[m]),
            "prevalence": float(y.mean())}


def threshold_for_fp_per_hour(cls, score, hours: float, target: float,
                              fp_classes=("N",)) -> float:
    """Smallest threshold such that non-anomalous beats scoring above it occur <= ``target`` per
    hour. ``fp_classes=("N", "X")`` also counts false detections (class X) as false alarms."""
    s = np.sort(np.asarray(score)[np.isin(np.asarray(cls), fp_classes)])[::-1]
    allowed = int(np.floor(target * hours))
    if allowed >= len(s):
        return -np.inf
    return float(s[allowed])  # beats with score > thr are alarms


def sens_at_threshold(cls, score, thr: float, hours: float, keep=None,
                      fp_classes=("N",)) -> dict[str, float]:
    """False alarms per hour (always over all beats) and sensitivities.

    ``keep`` (boolean mask) restricts the *sensitivities* to some beats, e.g. to leave out records
    where S outnumbers N; false alarms stay pooled over everything.
    """
    cls, score = np.asarray(cls), np.asarray(score)
    keep = np.ones(len(cls), dtype=bool) if keep is None else np.asarray(keep)
    alarm = score > thr
    anom = np.isin(cls, ANOMALY_CLASSES) & keep
    out = {"fp_per_hour": float((alarm & np.isin(cls, fp_classes)).sum() / hours),
           "sens_anomaly": float(alarm[anom].mean()) if anom.any() else np.nan}
    for c in ANOMALY_CLASSES:
        m = (cls == c) & keep
        out[f"sens_{c}"] = float(alarm[m].mean()) if m.any() else np.nan
    return out


def sens_at_fp_rate(cls, score, hours: float, target: float, keep=None,
                    fp_classes=("N",)) -> dict[str, float]:
    """Operating point of the score curve: sensitivity when non-anomalous beats raise ``target``
    false alarms per hour (pooled over all records). Comparable across models."""
    thr = threshold_for_fp_per_hour(cls, score, hours, target, fp_classes)
    return sens_at_threshold(cls, score, thr, hours, keep, fp_classes)


def dominated_by(df: pd.DataFrame, cls_name: str = "S") -> list[str]:
    """Records where class ``cls_name`` outnumbers N (their 'normal' is not N, e.g. MIT-BIH 232)."""
    c = df.groupby(["record", "cls"]).size().unstack(fill_value=0)
    if cls_name not in c or "N" not in c:
        return []
    return c.index[c[cls_name] > c["N"]].tolist()


def per_record_auroc(df: pd.DataFrame, score_col: str = "score", min_pos: int = 10,
                     min_neg: int = 100) -> pd.DataFrame:
    """AUROC inside each record (anomalous vs normal beats of the same patient)."""
    rows = []
    for rec, g in df.groupby("record"):
        cls = g["cls"].to_numpy()
        n_pos, n_neg = int(np.isin(cls, ANOMALY_CLASSES).sum()), int((cls == "N").sum())
        if n_pos >= min_pos and n_neg >= min_neg:
            rows.append({"record": rec, "n_anomalous": n_pos, "n_normal": n_neg,
                         "auroc": auc_scores(cls, g[score_col].to_numpy())["auroc"]})
    return pd.DataFrame(rows, columns=["record", "n_anomalous", "n_normal", "auroc"])


def patient_bootstrap(df: pd.DataFrame, score_col: str = "score", n_boot: int = 500,
                      seed: int = 0, positives=ANOMALY_CLASSES) -> dict[str, tuple[float, float]]:
    """95 % CI of AUROC / AUPRC, resampling whole records (patients) with replacement."""
    rng = np.random.default_rng(seed)
    groups = {r: (g["cls"].to_numpy(), g[score_col].to_numpy()) for r, g in df.groupby("record")}
    names = list(groups)
    vals = {"auroc": [], "auprc": []}
    for _ in range(n_boot):
        pick = rng.choice(len(names), size=len(names), replace=True)
        cls = np.concatenate([groups[names[i]][0] for i in pick])
        sc = np.concatenate([groups[names[i]][1] for i in pick])
        res = auc_scores(cls, sc, positives)
        if not np.isnan(res["auroc"]):
            vals["auroc"].append(res["auroc"])
            vals["auprc"].append(res["auprc"])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v
            else (np.nan, np.nan) for k, v in vals.items()}


def paired_patient_bootstrap(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int = 500,
                             seed: int = 0, positives=ANOMALY_CLASSES) -> dict[str, float]:
    """Difference ``metric(col_b) - metric(col_a)`` with a 95 % CI, resampling the SAME patients
    for both scorers (paired), plus the share of resamples where ``col_b`` is better."""
    rng = np.random.default_rng(seed)
    groups = {r: (g["cls"].to_numpy(), g[col_a].to_numpy(), g[col_b].to_numpy())
              for r, g in df.groupby("record")}
    names = list(groups)
    d = {"auroc": [], "auprc": []}
    for _ in range(n_boot):
        pick = rng.choice(len(names), size=len(names), replace=True)
        cls = np.concatenate([groups[names[i]][0] for i in pick])
        a = auc_scores(cls, np.concatenate([groups[names[i]][1] for i in pick]), positives)
        b = auc_scores(cls, np.concatenate([groups[names[i]][2] for i in pick]), positives)
        if not np.isnan(a["auroc"]):
            d["auroc"].append(b["auroc"] - a["auroc"])
            d["auprc"].append(b["auprc"] - a["auprc"])
    out = {}
    for k, v in d.items():
        v = np.asarray(v)
        out[f"d_{k}"] = float(np.mean(v)) if len(v) else np.nan
        out[f"d_{k}_lo"] = float(np.percentile(v, 2.5)) if len(v) else np.nan
        out[f"d_{k}_hi"] = float(np.percentile(v, 97.5)) if len(v) else np.nan
        out[f"p_better_{k}"] = float((v > 0).mean()) if len(v) else np.nan
    return out
