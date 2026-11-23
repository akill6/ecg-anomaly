import numpy as np
import pytest
from scipy.fft import dct

from ecg_anomaly.sparsity import make_basis, prd, snr_db, sparse_approximation

try:
    import pywt  # noqa: F401

    HAS_PYWT = True
except ImportError:  # pragma: no cover
    HAS_PYWT = False

_needs_pywt = pytest.mark.skipif(not HAS_PYWT, reason="PyWavelets not installed")
CASES = [
    ("dft", {}),
    ("dct", {}),
    pytest.param("dwt", {"wavelet": "haar"}, marks=_needs_pywt),
    pytest.param("dwt", {"wavelet": "db2"}, marks=_needs_pywt),
]


@pytest.fixture
def segment(ecg):
    x, _ = ecg
    return x[:3000]


def test_dct_is_orthonormal(segment):
    c = dct(segment, type=2, norm="ortho")
    assert np.isclose(np.sum(c**2), np.sum(segment**2))


@pytest.mark.parametrize("name,kw", CASES)
def test_meets_prd_target(segment, name, kw):
    r = sparse_approximation(segment, name, target_prd=7.0, **kw)
    assert r.prd <= 7.0 + 0.01
    assert r.nnz < len(segment)
    assert r.compression > 1.0
    assert 0 < r.energy_pct <= 100.0


@pytest.mark.parametrize("name,kw", CASES)
def test_tighter_target_needs_more_coefficients(segment, name, kw):
    loose = sparse_approximation(segment, name, target_prd=7.0, **kw)
    tight = sparse_approximation(segment, name, target_prd=3.0, **kw)
    assert tight.nnz >= loose.nnz
    assert tight.prd <= loose.prd


@pytest.mark.parametrize("name,kw", CASES)
def test_snr_target(segment, name, kw):
    r = sparse_approximation(segment, name, target_snr=20.0, **kw)
    assert r.snr_db >= 20.0


@pytest.mark.parametrize("name,kw", CASES)
def test_near_perfect_reconstruction_when_target_is_zero(segment, name, kw):
    r = sparse_approximation(segment, name, target_prd=0.0, tol=0.01, **kw)
    assert r.prd < 0.011


def test_dft_selection_is_hermitian_symmetric(segment):
    """Every kept bin k must have its conjugate bin n-k kept too (real reconstruction)."""
    n = len(segment)
    basis = make_basis("dft", n)
    c = basis.forward(segment)
    ranked = basis.ranked_units(c)
    for k_units in (1, 10, 500, len(ranked)):
        mask = np.zeros(n, dtype=bool)
        mask[basis.always_keep()] = True
        mask[basis.unit_indices(ranked[:k_units])] = True
        assert np.array_equal(mask, mask[(-np.arange(n)) % n])


def test_metrics_basic():
    x = np.array([1.0, 2.0, 3.0])
    assert prd(x, x) == 0.0
    assert snr_db(x, x) == float("inf")
    assert np.isclose(prd(x, np.zeros(3)), 100.0)


def test_needs_exactly_one_target(segment):
    with pytest.raises(ValueError):
        sparse_approximation(segment, "dct")
    with pytest.raises(ValueError):
        sparse_approximation(segment, "dct", target_prd=5, target_snr=20)


@pytest.mark.parametrize("name,kw", CASES)
def test_strict_is_minimal_and_meets_target(segment, name, kw):
    from ecg_anomaly.sparsity.transforms import _reconstruct, make_basis as mk

    r = sparse_approximation(segment, name, target_prd=7.0, mode="strict", **kw)
    assert r.prd <= 7.0
    basis = mk(name, len(segment), kw.get("wavelet", "haar"), 5)
    c = basis.forward(segment)
    ranked = basis.ranked_units(c)
    if r.k_units > 0:
        _, xh = _reconstruct(basis, c, ranked, r.k_units - 1)
        assert prd(segment, xh) > 7.0


@pytest.mark.parametrize("name,kw", CASES)
def test_modes_are_close(segment, name, kw):
    s = sparse_approximation(segment, name, target_prd=7.0, mode="strict", **kw)
    m = sparse_approximation(segment, name, target_prd=7.0, mode="matlab", **kw)
    assert s.mode == "strict" and m.mode == "matlab"
    assert abs(s.nnz - m.nnz) <= 40


def test_invalid_mode(segment):
    with pytest.raises(ValueError):
        sparse_approximation(segment, "dct", target_prd=7.0, mode="fast")
