import json

import numpy as np
import pandas as pd
import pytest
from synth_table import synthetic_table

from ecg_anomaly.evaluation import (
    cross_validate,
    default_configs,
    evaluate_final,
    make_folds,
    paired_patient_bootstrap,
    select_best,
    summarize_cv,
)
from ecg_anomaly.evaluation.metrics import sens_at_threshold
from ecg_anomaly.features import ABSOLUTE_COLS, ALL_FEATURES, per_record_normalize


def _dev(n=12, beats=500, seed=0):
    tab, dur = synthetic_table(n_records=n, beats=beats, seed=seed)
    tab["split"] = "train"
    return tab, dur


def test_absolute_columns_are_valid_and_normalised_selectively():
    assert set(ABSOLUTE_COLS) <= set(ALL_FEATURES) and len(ABSOLUTE_COLS) < len(ALL_FEATURES)
    tab, _ = _dev(4)
    out = per_record_normalize(tab, ABSOLUTE_COLS)
    untouched = [c for c in ALL_FEATURES if c not in ABSOLUTE_COLS]
    assert np.allclose(out[untouched], tab[untouched])
    assert out.groupby("record")[ABSOLUTE_COLS].median().abs().to_numpy().max() < 1e-9


def test_make_folds_keeps_patients_together_and_balances_anomalies():
    tab, _ = _dev(12)
    folds = make_folds(tab, k=4)
    assert set(folds) == set(tab.record.unique()) and set(folds.values()) == {0, 1, 2, 3}
    per_fold = (tab.assign(f=tab.record.map(folds), a=tab.cls != "N").groupby("f")["a"].sum())
    assert per_fold.max() / per_fold.min() < 1.5


def test_cross_validate_summary_and_selection():
    tab, _ = _dev(10, 400)
    configs = default_configs(("global", "patient-abs"), ("iforest",), ["morph", "all"])
    cv = cross_validate(tab, configs, k=3, n_estimators=30, log=lambda *a: None)
    assert len(cv) == len(configs) * 3 and set(cv.fold) == {0, 1, 2}
    summ = summarize_cv(cv)
    assert summ.auprc.is_monotonic_decreasing and (summ.folds == 3).all()
    sel = select_best(summ)
    assert sel["best"]["cv_auprc"] == summ.auprc.iloc[0]
    # the 1-SE choice is never more complex than the best and stays within one SE
    assert sel["one_se"]["cv_auprc"] >= sel["best"]["cv_auprc"] - sel["best"]["cv_auprc_se"] - 1e-12


def test_select_best_prefers_simpler_within_one_se():
    rows = [("global", "iforest", "all", 0.50, 0.03, 0.8), ("global", "iforest", "rr", 0.49, 0.03, 0.7)]
    summ = pd.DataFrame(rows, columns=["norm", "model", "features", "auprc", "auprc_se", "auroc"])
    summ["n_features"] = [30, 5]
    assert select_best(summ)["best"]["features"] == "all"
    assert select_best(summ)["one_se"]["features"] == "rr"


def test_sens_at_threshold_can_exclude_records_from_sensitivity_only():
    cls = np.array(["N"] * 10 + ["S"] * 10)
    score = np.r_[np.zeros(10), np.r_[np.ones(5), np.zeros(5)]]
    keep = np.r_[np.ones(10, bool), np.r_[np.zeros(5, bool), np.ones(5, bool)]]
    full = sens_at_threshold(cls, score, 0.5, hours=1.0)
    adj = sens_at_threshold(cls, score, 0.5, hours=1.0, keep=keep)
    assert full["sens_S"] == 0.5 and adj["sens_S"] == 0.0 and full["fp_per_hour"] == adj["fp_per_hour"]


def test_paired_bootstrap_detects_a_better_scorer():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(10):
        cls = np.where(rng.random(300) < 0.15, "V", "N")
        good = rng.random(300) + 1.5 * (cls == "V")
        bad = rng.random(300) + 0.2 * (cls == "V")
        rows.append(pd.DataFrame({"record": str(r), "cls": cls, "a": bad, "b": good}))
    d = paired_patient_bootstrap(pd.concat(rows), "a", "b", n_boot=100)
    assert d["d_auroc"] > 0.2 and d["d_auroc_lo"] > 0 and d["p_better_auroc"] == 1.0


def test_final_evaluation_and_report(tmp_path):
    from ecg_anomaly.evaluation.plots import final_markdown, make_final_figure

    dev, _ = _dev(10, 500, seed=1)
    test, dt = synthetic_table(6, 500, seed=2)
    ext, de = synthetic_table(6, 500, seed=3)
    ext["record"] = "x" + ext.record
    de = {"x" + k: v for k, v in de.items()}
    eval_sets = {"MIT-BIH DS2": (test, dt), "SVDB (external)": (ext, de)}
    configs = {"baseline": ("global", "iforest", "all_v1"), "selected": ("patient-abs", "iforest", "all")}
    res, scores = evaluate_final(dev, eval_sets, configs, "baseline", n_boot=20, log=lambda *a: None)
    assert len(res) == 4 and "d_auprc" in res and res[res.config == "selected"].d_auprc.notna().all()
    assert res[res.config == "baseline"].d_auprc.isna().all()
    assert scores[("SVDB (external)", "selected")].shape == (len(ext),)
    cv = pd.DataFrame({"norm": ["global", "patient-abs"], "model": ["iforest"] * 2,
                       "features": ["all_v1", "all"], "auprc": [0.3, 0.4], "auprc_se": [0.02] * 2})
    png = make_final_figure(res, scores, eval_sets, cv, configs, tmp_path / "f.png", dpi=40)
    assert (tmp_path / "f.png").stat().st_size > 5000
    md = final_markdown(res)
    assert "SVDB (external)" in md and "paired" in md


def test_final_script_lock_blocks_second_run(tmp_path, monkeypatch):
    import importlib.util
    import pathlib
    import sys

    spec = importlib.util.spec_from_file_location(
        "phase3_final", pathlib.Path(__file__).parents[1] / "scripts" / "phase3_final.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "final_done.lock").write_text("2026-01-01 configs={}", encoding="utf8")
    monkeypatch.setattr(sys, "argv", ["x", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "already done" in str(e.value)
    json.dumps({})  # (keeps json import used)
