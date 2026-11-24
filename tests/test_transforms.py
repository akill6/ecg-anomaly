import numpy as np
import pandas as pd
import pytest
from scipy.fft import dct

from ecg_anomaly.features import FEATURE_SETS, CORE_SETS, ABSOLUTE_COLS, ALL_FEATURES
from ecg_anomaly.features.dataset import ensure_sparsity_columns
from ecg_anomaly.features.spectral import (
    TRANSFORMS, WAVELETS, dwt_coeffs, sparsity_cols, sparsity_features,
)


def _beats(n=40, L=175, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, L)
    b = np.exp(-(t / 0.08) ** 2)[None] * rng.uniform(0.8, 1.2, (n, 1)) + 0.02 * rng.normal(size=(n, L))
    return (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)


@pytest.mark.parametrize("w", list(WAVELETS))
@pytest.mark.parametrize("level", [1, 3, 5])
def test_dwt_is_orthonormal(w, level):
    x = _beats()
    c = dwt_coeffs(x, w, level)
    assert np.allclose((c**2).sum(1), (x**2).sum(1))  # Parseval
    # orthogonality: random vectors keep inner products
    y = _beats(seed=3)
    assert np.allclose((dwt_coeffs(x, w, level) * dwt_coeffs(y, w, level)).sum(1), (x * y).sum(1))


def test_haar_matches_hand_computation_and_db2_kills_linear_trends():
    c = dwt_coeffs(np.array([[1.0, 3.0, 5.0, 11.0]]), "haar", 1)
    assert np.allclose(sorted(c[0]), sorted([(1 - 3) / 2**0.5, (5 - 11) / 2**0.5,
                                             4 / 2**0.5, 16 / 2**0.5]))
    ramp = np.tile(np.arange(64.0), (1, 1))
    d = dwt_coeffs(ramp, "db2", 1)[0, 32:]  # detail band; db2 has 2 vanishing moments
    assert np.allclose(d[:-1], 0, atol=1e-9)  # last value touches the periodic wrap


def test_dct_features_equal_original_formula():
    x = _beats()
    got = sparsity_features(x, "dct")
    c = dct(x, norm="ortho", axis=1)
    e = np.sort(c**2, axis=1)[:, ::-1]
    total = e.sum(1, keepdims=True) + 1e-12
    prd = 100 * np.sqrt(np.maximum(total - np.cumsum(e, 1), 0) / total)
    assert np.array_equal(got["k_prd5"], 1 + (prd > 5).sum(1))
    assert np.array_equal(got["k_prd10"], 1 + (prd > 10).sum(1))
    assert np.allclose(got["top5_energy"], e[:, :5].sum(1) / total[:, 0])


@pytest.mark.parametrize("t", TRANSFORMS)
def test_features_are_sane_and_k_matches_brute_force_prd(t):
    x = _beats(10)
    f = sparsity_features(x, t)
    k5, k10, top = (f[c] for c in sparsity_cols(t))
    assert (k10 <= k5).all() and (k5 >= 1).all() and ((top > 0) & (top <= 1 + 1e-9)).all()
    if t in ("dct", "haar", "db2"):  # brute force: keep k largest, measure PRD in the signal domain
        for i in range(3):
            c = dct(x[i], norm="ortho") if t == "dct" else dwt_coeffs(x[i:i+1], t, 5)[0]
            keep = np.argsort(-np.abs(c))[: int(k5[i])]
            r = np.zeros_like(c); r[keep] = c[keep]
            prd = 100 * np.sqrt(((c - r) ** 2).sum() / (c**2).sum())
            assert prd <= 5 + 1e-9
            keep = np.argsort(-np.abs(c))[: int(k5[i]) - 1]
            r = np.zeros_like(c); r[keep] = c[keep]
            assert 100 * np.sqrt(((c - r) ** 2).sum() / (c**2).sum()) > 5 - 1e-9


def test_dft_counts_conjugate_pairs_as_two_parameters():
    n = 64
    pure = np.cos(2 * np.pi * 5 * np.arange(n) / n)[None]  # one conjugate pair
    assert sparsity_features(pure, "dft")["k_prd5_dft"][0] == 2
    assert sparsity_features(pure, "dct")["k_prd5"][0] > 2  # DCT is not compact for a pure DFT bin


def test_sparse_beat_is_cheaper_than_noise_in_every_basis():
    rng = np.random.default_rng(1)
    clean, noise = _beats(20)[:, :], rng.normal(size=(20, 175))
    for t in TRANSFORMS:
        c, nz = (sparsity_features(a, t)[sparsity_cols(t)[0]] for a in (clean, noise))
        assert c.mean() < nz.mean()


def test_feature_sets_and_back_compat():
    assert set(CORE_SETS) == {"rr", "rhythm", "morph", "template", "sparsity", "morph+sparsity",
                              "all_v1", "all"}
    for t in ("dft", "haar", "db2"):
        assert len(FEATURE_SETS[f"all_{t}"]) == len(FEATURE_SETS["all"])
        assert set(sparsity_cols(t)) <= set(ABSOLUTE_COLS)
    assert len(FEATURE_SETS["all_multi"]) == len(FEATURE_SETS["all"]) + 9
    assert set(ABSOLUTE_COLS) <= set(ALL_FEATURES)


def test_old_table_gets_missing_columns():
    x = _beats(30)
    old = pd.DataFrame(sparsity_features(x, "dct"))
    new = ensure_sparsity_columns(old, x)
    assert set(c for t in TRANSFORMS for c in sparsity_cols(t)) <= set(new.columns)
    assert np.array_equal(new["k_prd5"], old["k_prd5"])


def test_feature_set_for_gives_valid_names():
    from ecg_anomaly.features.spectral import feature_set_for
    for t in TRANSFORMS:
        for scope in ("all", "sparsity"):
            assert feature_set_for(t, scope) in FEATURE_SETS
    assert feature_set_for("multi") == "all_multi" and feature_set_for("dct") == "all"
    with pytest.raises(ValueError):
        feature_set_for("fourier")
