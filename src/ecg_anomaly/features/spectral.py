"""Sparsity features of a beat in four orthonormal bases: DCT, DFT, Haar and Daubechies-2 DWT.

For an orthonormal transform the error after keeping the largest coefficients is
``PRD = 100 * sqrt(discarded energy / total energy)``, so the number of coefficients ``k`` needed
for a target PRD can be computed for all beats at once. A beat that is well described by few
coefficients is "sparse" in that basis; an unusual beat (different shape, width, extra waves)
needs more. The bases differ in what they find compact: DCT / DFT - smooth oscillations,
Haar / db2 - sharp, localised changes such as the QRS complex.

k counts REAL parameters: a DFT conjugate pair (two real numbers) costs 2, other units cost 1,
so the four bases are compared on equal terms. The DWT is written in plain NumPy (periodic
extension, beats zero-padded to a multiple of ``2**level``), so PyWavelets is not required.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

SQ2 = np.sqrt(2.0)
WAVELETS = {
    "haar": np.array([1.0, 1.0]) / SQ2,
    "db2": np.array([0.48296291314453414, 0.83651630373780794, 0.22414386804201339,
                     -0.12940952255126037]),
}
TRANSFORMS = ("dct", "dft", "haar", "db2")
LABELS = {"dct": "DCT", "dft": "DFT", "haar": "DWT Haar", "db2": "DWT db2"}


def suffix(transform: str) -> str:
    """Column suffix; DCT keeps the original names (``k_prd5``) for backward compatibility."""
    _check(transform)
    return "" if transform == "dct" else f"_{transform}"


def sparsity_cols(transform: str, targets=(5, 10)) -> list[str]:
    s = suffix(transform)
    return [f"k_prd{int(t)}{s}" for t in targets] + [f"top5_energy{s}"]


def _check(transform: str) -> None:
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}; choose from {TRANSFORMS}")


def _pywt_filters(wavelet: str):
    """Decomposition filters exactly as PyWavelets builds them for the ``db`` family.

    ``dec_lo[i] = c[F-1-i]``, ``dec_hi[i] = (-1)**(F-1-i) * c[i]`` (``haar`` is ``db1``).
    """
    c = WAVELETS[wavelet]
    f = len(c)
    dec_lo = c[::-1]
    dec_hi = (-1.0) ** (f - 1 - np.arange(f)) * c
    return dec_lo, dec_hi


def dwt_coeffs(x, wavelet: str = "haar", level: int = 5) -> np.ndarray:
    """Orthonormal multi-level DWT of every row of ``x``, same convention as
    ``pywt.wavedec(..., mode="periodization")``.

    Rows are zero-padded to a multiple of ``2**level``. Returns the coefficients of every level
    (``[cA_n, cD_n, ..., cD_1]`` as in PyWavelets) in one array of shape ``(n, padded_length)``;
    the transform preserves energy (Parseval).
    """
    if wavelet not in WAVELETS:
        raise ValueError(f"unknown wavelet {wavelet!r}; choose from {tuple(WAVELETS)}")
    dec_lo, dec_hi = _pywt_filters(wavelet)
    f = len(dec_lo)
    x = np.asarray(x, dtype=float)
    pad = (-x.shape[1]) % (2**level)
    a = np.pad(x, ((0, 0), (0, pad)))
    out = []
    for _ in range(level):
        length = a.shape[1]
        # PyWavelets (periodization): out[o] = sum_j filter[j] * a[(F/2 + 2o - j) mod length]
        idx = (f // 2 + 2 * np.arange(length // 2)[:, None] - np.arange(f)[None, :]) % length
        w = a[:, idx]  # (n, length/2, f)
        out.append(w @ dec_hi)
        a = w @ dec_lo
    out.append(a)
    return np.concatenate(out[::-1], axis=1)


def _unit_energies(beats, transform: str, level: int):
    """Energy and cost (real parameters) of every coefficient "unit" of the chosen basis."""
    x = np.asarray(beats, dtype=float)
    if transform == "dct":
        e = dct(x, norm="ortho", axis=1) ** 2
        return e, np.ones(e.shape[1])
    if transform == "dft":
        n = x.shape[1]
        e = np.abs(np.fft.rfft(x, axis=1, norm="ortho")) ** 2
        cost = np.full(e.shape[1], 2.0)  # bin k and its conjugate n-k
        cost[0] = 1.0
        if n % 2 == 0:
            cost[-1] = 1.0  # Nyquist bin has no partner
        return e * cost, cost
    c = dwt_coeffs(x, transform, level)
    return c**2, np.ones(c.shape[1])


def sparsity_features(beats_z, transform: str = "dct", targets_prd=(5.0, 10.0), top: int = 5,
                      level: int = 5) -> dict[str, np.ndarray]:
    """``k_prd5``, ``k_prd10`` and ``top5_energy`` of each beat in the chosen basis.

    ``k_prdX``: real coefficients needed to reach PRD <= X %. ``top5_energy``: share of the
    energy in the 5 largest real coefficients. Column names carry the basis suffix
    (see :func:`suffix`).
    """
    _check(transform)
    e, cost = _unit_energies(beats_z, transform, level)
    order = np.argsort(-e / cost, axis=1, kind="stable")  # largest energy per parameter first
    es = np.take_along_axis(e, order, axis=1)
    cs = cost[order]
    cum_cost = np.cumsum(cs, axis=1)
    total = e.sum(axis=1, keepdims=True) + 1e-12
    prd = 100.0 * np.sqrt(np.maximum(total - np.cumsum(es, axis=1), 0.0) / total)
    rows = np.arange(len(es))
    s = suffix(transform)
    out = {}
    for t in targets_prd:
        first = np.argmax(prd <= t, axis=1)  # PRD reaches 0 with all units, so one always exists
        out[f"k_prd{int(t)}{s}"] = cum_cost[rows, first]
    out[f"top5_energy{s}"] = (es * (cum_cost <= top)).sum(axis=1) / total[:, 0]
    return out


def all_sparsity_features(beats_z, transforms=TRANSFORMS, **kw) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for t in transforms:
        out.update(sparsity_features(beats_z, t, **kw))
    return out


def feature_set_for(transform: str = "dct", scope: str = "all") -> str:
    """Name of the feature set for a chosen basis: the single place to pick a transform.

    ``scope="all"``: the full model with that basis (``all`` / ``all_dft`` / ``all_haar`` /
    ``all_db2``); ``scope="sparsity"``: only the three sparsity features of that basis.
    ``transform="multi"`` (with ``scope="all"``) uses all four bases together (``all_multi``).
    Use the result as the ``features`` name in the experiments, e.g.
    ``("patient-abs", "iforest", feature_set_for("haar"))``.
    """
    if transform == "multi":
        if scope != "all":
            raise ValueError("'multi' is available for scope='all' only; use 'sparsity_all'")
        return "all_multi"
    _check(transform)
    if scope not in ("all", "sparsity"):
        raise ValueError("scope must be 'all' or 'sparsity'")
    return scope if transform == "dct" else f"{scope}_{transform}"
