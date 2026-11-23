import numpy as np
import pandas as pd

from ecg_anomaly.beats import detect_rpeaks, extract_beats
from ecg_anomaly.evaluation import (
    auc_scores,
    patient_bootstrap,
    sens_at_threshold,
    threshold_for_fp_per_hour,
)
from ecg_anomaly.features import dct_sparsity_features
from ecg_anomaly.features.beat_features import FEATURE_SETS, beat_feature_table
from ecg_anomaly.models import BaselineScorer
from ecg_anomaly.preprocessing import preprocess
from ecg_anomaly.sparsity import sparse_approximation

FS, PRE = 250, 62


def _beats(ecg, wide=()):
    x, truth = ecg
    y = preprocess(x, FS)
    peaks = detect_rpeaks(y, FS)
    raw, kept = extract_beats(y, peaks, FS, normalize="none")
    z, _ = extract_beats(y, peaks, FS)
    return raw, z, kept


def test_vectorised_dct_sparsity_matches_sparse_approximation(ecg):
    _, z, _ = _beats(ecg)
    f = dct_sparsity_features(z[:10], targets_prd=(5.0, 10.0))
    for i in range(10):
        for t in (5, 10):
            ref = sparse_approximation(z[i], "dct", target_prd=t, mode="strict")
            assert ref.prd <= t + 1e-9
            assert abs(f[f"k_prd{t}"][i] - ref.nnz) <= 1  # DC is always kept by the reference


def test_feature_table_shape_and_finiteness(ecg):
    raw, z, kept = _beats(ecg)
    tab = beat_feature_table(raw, z, kept, FS, PRE)
    assert len(tab) == len(kept)
    cols = sorted({c for v in FEATURE_SETS.values() for c in v})
    assert set(cols) <= set(tab.columns)
    assert np.isfinite(tab[cols].to_numpy()).all()
    assert 0.015 < tab.qrs_width_s.median() < 0.2


def test_abnormal_beats_score_higher(ecg):
    raw, z, kept = _beats(ecg)
    raw, z = raw.copy(), z.copy()
    rng = np.random.default_rng(0)
    bad = rng.choice(len(raw), size=len(raw) // 10, replace=False)
    t = np.arange(raw.shape[1]) / FS
    for i in bad:  # wide, inverted, large "ventricular-like" complex
        width = rng.uniform(0.035, 0.06)
        raw[i] = -rng.uniform(1.5, 3.0) * np.exp(-0.5 * ((t - PRE / FS) / width) ** 2)
        raw[i] += 0.03 * rng.standard_normal(len(t))
        z[i] = (raw[i] - raw[i].mean()) / raw[i].std()
    tab = beat_feature_table(raw, z, kept, FS, PRE)
    cls = np.array(["N"] * len(tab), dtype=object)
    cls[bad] = "V"
    # one GMM component: with ~70 beats more components would just model the abnormal beats
    cols = FEATURE_SETS["morph"]   # ~70 beats: too few for a covariance over all 30+ features
    for model, kw in (("iforest", {}), ("gmm", {"n_components": 1})):
        s = BaselineScorer(model, seed=0, **kw).fit(tab[cols].to_numpy()).score(tab[cols].to_numpy())
        assert auc_scores(cls, s)["auroc"] > 0.9


def test_auc_ignores_other_classes():
    cls = np.array(["N", "N", "V", "Q", "X"])
    s = np.array([0.1, 0.2, 0.9, 5.0, 5.0])
    r = auc_scores(cls, s)
    assert r["auroc"] == 1.0 and r["auprc"] == 1.0


def test_threshold_for_fp_per_hour_and_sensitivity():
    cls = np.array(["N"] * 100 + ["V"] * 10)
    score = np.r_[np.linspace(0, 1, 100), np.full(10, 2.0)]
    thr = threshold_for_fp_per_hour(cls, score, hours=1.0, target=5)
    res = sens_at_threshold(cls, score, thr, hours=1.0)
    assert res["fp_per_hour"] <= 5 and res["sens_V"] == 1.0


def test_patient_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(8):
        cls = np.where(rng.random(200) < 0.2, "V", "N")
        rows.append(pd.DataFrame({"record": str(r), "cls": cls,
                                  "score": rng.random(200) + (cls == "V")}))
    df = pd.concat(rows)
    ci = patient_bootstrap(df, n_boot=100)["auroc"]
    point = auc_scores(df.cls, df.score)["auroc"]
    assert ci[0] <= point <= ci[1]


# ---- per-patient normalisation, experiment runner, report -------------------------------

def test_per_record_normalize_removes_patient_offsets():
    from synth_table import synthetic_table

    from ecg_anomaly.features import ALL_FEATURES, per_record_normalize

    tab, _ = synthetic_table(n_records=4, beats=400)
    norm = per_record_normalize(tab, ALL_FEATURES)
    med = norm.groupby("record")[ALL_FEATURES].median().abs().to_numpy()
    assert med.max() < 1e-9                       # every patient is centred on its own median
    raw_med = tab.groupby("record")[ALL_FEATURES].median().abs().to_numpy()
    assert raw_med.max() > 1                      # ...while the raw table was not
    assert np.array_equal(norm.cls, tab.cls)      # metadata untouched


def test_causal_normalisation_uses_only_the_past():
    from ecg_anomaly.features import per_record_normalize

    rng = np.random.default_rng(0)
    base = pd.DataFrame({"record": "a", "f": rng.normal(size=300)})
    changed = base.copy()
    changed.loc[200:, "f"] += 50                  # change only the future
    a = per_record_normalize(base, ["f"], window=50)
    b = per_record_normalize(changed, ["f"], window=50)
    assert np.allclose(a.f[:200], b.f[:200])


def test_dominated_by_and_per_record_auroc():
    from ecg_anomaly.evaluation import dominated_by, per_record_auroc

    df = pd.DataFrame({"record": ["a"] * 150 + ["b"] * 150,
                       "cls": ["N"] * 140 + ["V"] * 10 + ["S"] * 100 + ["N"] * 50,
                       "score": np.r_[np.zeros(140), np.ones(10), np.ones(100), np.zeros(50)]})
    assert dominated_by(df, "S") == ["b"]
    pr = per_record_auroc(df, min_pos=10, min_neg=40)
    assert pr.auroc.tolist() == [1.0, 1.0]


def test_run_baselines_and_report(tmp_path):
    from synth_table import synthetic_table

    from ecg_anomaly.evaluation import ALL_SETS, NORMS, run_baselines
    from ecg_anomaly.evaluation.plots import make_report_figure, results_markdown

    tab, dur = synthetic_table(n_records=16, beats=800)
    res, scores = run_baselines(tab, dur, models=("iforest", "gmm"), n_boot=10,
                                log=lambda *a: None)
    assert len(res) == len(NORMS) * 2 * len(ALL_SETS)
    g = res[(res.model == "iforest") & (res.features == "all")].set_index("norm")
    assert g.loc["patient", "auprc"] > g.loc["global", "auprc"]      # patient offsets hurt global
    assert abs(g.loc["patient", "transfer"] - 1) < abs(g.loc["global", "transfer"] - 1)
    te = tab[tab.split == "test"].reset_index(drop=True)
    png = make_report_figure(res, scores, te, out=tmp_path / "r.png", dpi=40)
    assert (tmp_path / "r.png").stat().st_size > 5000
    assert "AUROC" in results_markdown(res)


def test_normalize_handles_integer_columns():
    from ecg_anomaly.features import per_record_normalize

    df = pd.DataFrame({"record": ["a"] * 60 + ["b"] * 60, "k": np.arange(120) % 17})
    assert df.k.dtype.kind == "i"
    for window in (None, 20):
        out = per_record_normalize(df, ["k"], window=window)
        assert out.k.dtype.kind == "f" and np.isfinite(out.k).all()


# ---- phase 2c: rhythm, wave / template features, fusion ----------------------------------

def test_rhythm_features_flag_a_premature_beat():
    from ecg_anomaly.features import rhythm_features

    peaks = np.cumsum([200] * 10 + [120, 280] + [200] * 10)      # beat 10 is early (S/V-like)
    f = rhythm_features(peaks, fs=250)
    k = 10                                                       # index of the early beat
    assert f["rr_prev2_ratio"][k] < 0.7
    assert f["rr_next_prev"][k] > 2
    assert abs(f["rr_sum_ratio"][k] - 1.0) < 0.05                # full compensatory pause
    assert f["rr_dprev"][k] > 0.3 and f["rr_cv"][2] < 0.05
    assert all(np.isfinite(v).all() for v in f.values())


def test_causal_template_uses_only_past_beats():
    from ecg_anomaly.features import causal_template

    rng = np.random.default_rng(0)
    b = rng.normal(size=(400, 20))
    c = b.copy()
    c[250:] += 100                                               # alter the future only
    ta, tc = causal_template(b), causal_template(c)
    assert np.allclose(ta[:250], tc[:250])
    assert not np.allclose(ta[300:], tc[300:])


def test_wave_features_detect_missing_p_wave_and_odd_shape(ecg):
    from ecg_anomaly.features import causal_template, wave_features

    raw, z, kept = _beats(ecg)
    raw, z = raw.copy(), z.copy()
    t = np.arange(raw.shape[1]) / FS
    p_win = (t > (PRE / FS - 0.22)) & (t < (PRE / FS - 0.08))
    bad = np.arange(40, len(raw), 7)
    for i in bad:                                                # remove the P wave
        raw[i, p_win] = raw[i, p_win].mean()
        z[i] = (raw[i] - raw[i].mean()) / raw[i].std()
    f = wave_features(raw, z, causal_template(z), PRE, FS)
    good = np.setdiff1d(np.arange(40, len(raw)), bad)
    assert np.median(f["p_ratio"][bad]) < 0.5 * np.median(f["p_ratio"][good])
    assert np.median(f["tpl_rmse_p"][bad]) > np.median(f["tpl_rmse_p"][good])
    assert np.median(f["tpl_corr"][good]) > 0.95


def test_fusion_catches_anomalies_that_live_in_one_group():
    from ecg_anomaly.models import FusionScorer

    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))
    cls = np.array(["N"] * n, dtype=object)
    a_idx, b_idx = np.arange(0, n, 20), np.arange(10, n, 20)     # 5 % anomalies in each group
    df.loc[a_idx, ["a", "b", "c"]] += 5
    df.loc[b_idx, ["d", "e", "f"]] += 5
    cls[a_idx], cls[b_idx] = "V", "S"
    groups = {"g1": list("abc"), "g2": list("def")}
    fused = FusionScorer(groups, "iforest", "max").fit(df).score(df)
    single = BaselineScorer("iforest").fit(df[list("abc")].to_numpy()).score(df[list("abc")].to_numpy())
    assert auc_scores(cls, fused)["auroc"] > 0.97
    assert auc_scores(cls, fused)["auroc"] > auc_scores(cls, single)["auroc"] + 0.1
