"""Evaluate R-peak detector settings on a set of annotated records, overall and per AAMI class."""

from __future__ import annotations

import numpy as np

from .rpeaks import detect_rpeaks, matched_flags

CLASSES = ("N", "S", "V", "F")


def evaluate_config(records: list[dict], cfg: dict, tol_s: float = 0.1) -> dict:
    """Pooled detection quality of ``detect_rpeaks(**cfg)`` over ``records``.

    Each record is a dict with ``x`` (preprocessed signal), ``fs``, ``ann_samples`` and
    ``ann_classes`` (AAMI letter per annotated beat). Returns beats, pooled sensitivity / PPV
    and sensitivity per class (NaN if the class is absent).
    """
    found = {c: 0 for c in CLASSES}
    total = {c: 0 for c in CLASSES}
    tp = fp = n_ann = 0
    for rec in records:
        peaks = detect_rpeaks(rec["x"], rec["fs"], **cfg)
        flags = matched_flags(rec["ann_samples"], peaks, tol=tol_s * rec["fs"])
        classes = np.asarray(rec["ann_classes"])
        tp += int(flags.sum())
        fp += len(peaks) - int(flags.sum())
        n_ann += len(flags)
        for c in CLASSES:
            m = classes == c
            total[c] += int(m.sum())
            found[c] += int(flags[m].sum())
    out = {
        "beats": n_ann,
        "sens_all": tp / n_ann if n_ann else float("nan"),
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
    }
    for c in CLASSES:
        out[f"sens_{c}"] = found[c] / total[c] if total[c] else float("nan")
    return out


ALL_CLASSES = ("N", "S", "V", "F", "Q")


def record_detection_summary(peaks, ann_samples, ann_classes, fs: float, tol_s: float = 0.1) -> dict:
    """Detection counts for one record: annotated / detected / true / false / missed beats and,
    for every AAMI class, how many annotated beats exist and how many the detector missed."""
    flags = matched_flags(ann_samples, peaks, tol=tol_s * fs)
    classes = np.asarray(ann_classes)
    tp = int(flags.sum())
    out = {"annotated": len(flags), "detected": len(peaks), "tp": tp,
           "fp": len(peaks) - tp, "fn": len(flags) - tp}
    for c in ALL_CLASSES:
        m = classes == c
        out[f"n_{c}"] = int(m.sum())
        out[f"missed_{c}"] = int((~flags[m]).sum())
    return out
