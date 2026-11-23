"""Parity with the MATLAB thesis app.

Copy ``matDBs/6.mat`` from https://github.com/akill6/signal-sparsity-dot to
``tests/data/6.mat`` to enable. Reference numbers come from the README (signal 6.mat,
lead 4, first 3000 samples, target PRD = 7 %).
"""

from pathlib import Path

import pytest

from ecg_anomaly.sparsity import sparse_approximation

MAT = Path(__file__).parent / "data" / "6.mat"
EXPECTED = {  # transform -> (nonzero coefficients, PRD [%])
    ("dft", "haar"): (1043, 7.0093),
    ("dct", "haar"): (861, 7.0066),
    ("dwt", "haar"): (337, 7.0109),
}


@pytest.mark.skipif(not MAT.exists(), reason="tests/data/6.mat not present")
@pytest.mark.parametrize("key", list(EXPECTED))
def test_matches_matlab_readme_table(key):
    from scipy.io import loadmat

    name, wavelet = key
    if name == "dwt":
        pytest.importorskip("pywt")
    x = loadmat(MAT)["ecgSignal"][:3000, 3]  # lead 4 -> column index 3
    r = sparse_approximation(x, name, target_prd=7.0, wavelet=wavelet, mode="matlab")
    nnz, prd_ref = EXPECTED[key]
    assert r.prd == pytest.approx(prd_ref, abs=5e-3)
    assert abs(r.nnz - nnz) <= 2


@pytest.mark.skipif(not MAT.exists(), reason="tests/data/6.mat not present")
@pytest.mark.parametrize("name", ["dft", "dct", "dwt"])
def test_strict_vs_matlab_on_real_signal(name):
    """strict guarantees PRD <= 7 and stays close to the MATLAB-mode coefficient count."""
    from scipy.io import loadmat

    if name == "dwt":
        pytest.importorskip("pywt")
    x = loadmat(MAT)["ecgSignal"][:3000, 3]
    s = sparse_approximation(x, name, target_prd=7.0, mode="strict")
    m = sparse_approximation(x, name, target_prd=7.0, mode="matlab")
    assert s.prd <= 7.0
    assert abs(s.nnz - m.nnz) <= 40
