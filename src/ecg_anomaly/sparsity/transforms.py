"""Hard-threshold sparse approximation in DFT / DCT / DWT domains.

Python port of the MATLAB thesis app (signal-sparsity-dot). For a target PRD (or SNR) the
smallest number of retained coefficients is found by binary search, using the same rules:

* coefficients are ranked by absolute value;
* DFT: DC is always kept, only positive frequencies are ranked, and a kept frequency
  keeps its complex-conjugate bin too (so the reconstruction stays real);
* DCT: orthonormal 1D DCT-II, the first (DC) coefficient is always kept;
* DWT: multilevel decomposition (5 levels by default), all coefficients thresholded together;
* two search modes (see ``sparse_approximation``): ``strict`` (default, smallest k meeting the
  target) and ``matlab`` (exact replica of the MATLAB app's search, used for parity checks).

NOTE on DFT: for even N the Nyquist bin is self-conjugate; here it is ranked together with the
other positive bins and kept alone. If the MATLAB code treats it differently, nonzero counts
can differ by one; check this against ``number_2.mlapp`` if exact parity matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct

from .metrics import prd, snr_db


class _Basis:
    """A transform plus the rules for ranking and selecting its coefficients."""

    name: str

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, c: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def always_keep(self) -> np.ndarray:
        return np.empty(0, dtype=int)

    def ranked_units(self, c: np.ndarray) -> np.ndarray:
        """Selectable units (coefficient ids), most important first."""
        raise NotImplementedError

    def unit_indices(self, units: np.ndarray) -> np.ndarray:
        """Coefficient indices that a set of units occupies (DFT: bin and its conjugate)."""
        return units


class DCTBasis(_Basis):
    name = "dct"

    def forward(self, x):
        return dct(x, type=2, norm="ortho")

    def inverse(self, c):
        return idct(c, type=2, norm="ortho")

    def always_keep(self):
        return np.array([0])

    def ranked_units(self, c):
        # all N coefficients compete (as in MATLAB); DC is additionally forced on
        return np.argsort(-np.abs(c), kind="stable")


class DFTBasis(_Basis):
    name = "dft"

    def __init__(self, n: int):
        self.n = n
        # positive-frequency bins 1..n//2; for even n the last one is the Nyquist bin,
        # which is its own conjugate (kept alone, counted once)
        self._half = n // 2

    def forward(self, x):
        return np.fft.fft(x)

    def inverse(self, c):
        return np.fft.ifft(c).real

    def always_keep(self):
        return np.array([0])

    def ranked_units(self, c):
        pos = np.arange(1, self._half + 1)
        return pos[np.argsort(-np.abs(c[pos]), kind="stable")]

    def unit_indices(self, units):
        # conjugate bin n-k; for the Nyquist bin (k = n/2) this is k itself (harmless duplicate)
        return np.concatenate([units, self.n - units])


class DWTBasis(_Basis):
    def __init__(self, wavelet: str = "haar", level: int = 5):
        self.wavelet = wavelet
        self.level = level
        self.name = f"dwt-{wavelet}"
        self._slices = None
        self._n = 0

    def forward(self, x):
        import pywt  # imported lazily so DFT/DCT work without PyWavelets installed

        self._n = len(x)
        coeffs = pywt.wavedec(x, self.wavelet, level=self.level)
        arr, self._slices = pywt.coeffs_to_array(coeffs)
        return arr

    def inverse(self, c):
        import pywt

        coeffs = pywt.array_to_coeffs(c, self._slices, output_format="wavedec")
        return pywt.waverec(coeffs, self.wavelet)[: self._n]

    def ranked_units(self, c):
        return np.argsort(-np.abs(c), kind="stable")


def make_basis(transform: str, n: int, wavelet: str = "haar", level: int = 5) -> _Basis:
    transform = transform.lower()
    if transform == "dct":
        return DCTBasis()
    if transform == "dft":
        return DFTBasis(n)
    if transform == "dwt":
        return DWTBasis(wavelet, level)
    raise ValueError(f"unknown transform {transform!r} (expected 'dft', 'dct' or 'dwt')")


@dataclass
class SparseResult:
    basis: str
    n: int
    k_units: int
    nnz: int
    prd: float
    snr_db: float
    energy_pct: float
    x_hat: np.ndarray
    mode: str = "strict"

    @property
    def compression(self) -> float:
        return self.n / self.nnz if self.nnz else float("inf")

    @property
    def sparsity(self) -> int:
        return self.n - self.nnz


def _reconstruct(basis: _Basis, c: np.ndarray, ranked: np.ndarray, k: int):
    mask = np.zeros(c.shape, dtype=bool)
    mask[basis.always_keep()] = True
    if k > 0:
        mask[basis.unit_indices(ranked[:k])] = True
    c_hat = np.where(mask, c, 0)
    return c_hat, basis.inverse(c_hat)


def _search_matlab(basis, c, ranked, x, target_prd, target_snr, tol, fallback_fraction):
    """Bisection exactly as in the MATLAB app.

    Search range k in [1, n_units]; among all visited k the one whose metric is CLOSEST to the
    target is kept (ties: higher SNR / lower PRD). The chosen PRD can therefore end up slightly
    above target + tol (e.g. 7.0109 for DWT-Haar on 6.mat).
    """
    use_prd = target_prd is not None
    left, right = 1, len(ranked)
    best_k, best_prd, best_snr = None, np.inf, -np.inf
    while left <= right:
        mid = (left + right) // 2
        _, xh = _reconstruct(basis, c, ranked, mid)
        cur_prd, cur_snr = prd(x, xh), snr_db(x, xh)
        if use_prd:
            d, d_best = abs(cur_prd - target_prd), abs(best_prd - target_prd)
            better = d < d_best or (d == d_best and cur_snr > best_snr)
            ok = cur_prd <= target_prd + tol
        else:
            d, d_best = abs(cur_snr - target_snr), abs(best_snr - target_snr)
            better = d < d_best or (d == d_best and cur_prd < best_prd)
            ok = cur_snr >= target_snr
        if better:
            best_k, best_prd, best_snr = mid, cur_prd, cur_snr
        if ok:
            right = mid - 1
        else:
            left = mid + 1
    return best_k if best_k is not None else int(round(fallback_fraction * len(ranked)))


def _search_strict(basis, c, ranked, x, target_prd, target_snr):
    """Smallest k that meets the target (PRD <= target or SNR >= target); no tolerance.

    Error is non-increasing in k for these orthonormal transforms, so plain bisection is exact.
    If the target cannot be met even with all coefficients, all of them are kept.
    """

    def ok(k: int) -> bool:
        _, xh = _reconstruct(basis, c, ranked, k)
        if target_prd is not None:
            return prd(x, xh) <= target_prd
        return snr_db(x, xh) >= target_snr

    hi = len(ranked)
    if not ok(hi):
        return hi
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def sparse_approximation(
    x: np.ndarray,
    transform: str = "dct",
    *,
    target_prd: float | None = None,
    target_snr: float | None = None,
    mode: str = "strict",
    wavelet: str = "haar",
    level: int = 5,
    tol: float = 0.01,
    fallback_fraction: float = 0.9,
) -> SparseResult:
    """Hard-thresholded approximation that meets a PRD or SNR target.

    mode="strict" (default): smallest k with PRD <= target (or SNR >= target). Guaranteed bound,
        reproducible, comparable across transforms.
    mode="matlab": reproduces the MATLAB thesis app (visited k closest to the target;
        ``tol`` and ``fallback_fraction`` apply only here). Use for parity checks.
    """
    if (target_prd is None) == (target_snr is None):
        raise ValueError("give exactly one of target_prd or target_snr")
    if mode not in ("strict", "matlab"):
        raise ValueError("mode must be 'strict' or 'matlab'")
    x = np.asarray(x, dtype=float).ravel()
    basis = make_basis(transform, len(x), wavelet, level)
    c = basis.forward(x)
    ranked = basis.ranked_units(c)

    if mode == "matlab":
        k = _search_matlab(basis, c, ranked, x, target_prd, target_snr, tol, fallback_fraction)
    else:
        k = _search_strict(basis, c, ranked, x, target_prd, target_snr)

    c_hat, x_hat = _reconstruct(basis, c, ranked, k)
    total_energy = float(np.sum(np.abs(c) ** 2))
    return SparseResult(
        basis=basis.name,
        n=len(x),
        k_units=k,
        nnz=int(np.count_nonzero(c_hat)),
        prd=prd(x, x_hat),
        snr_db=snr_db(x, x_hat),
        energy_pct=100.0 * float(np.sum(np.abs(c_hat) ** 2)) / total_energy,
        x_hat=x_hat,
        mode=mode,
    )
