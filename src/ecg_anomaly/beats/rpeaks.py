"""R-peak detection (Pan-Tompkins-style) and evaluation against reference annotations."""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..preprocessing.filters import bandpass


def detect_rpeaks(
    x,
    fs: float,
    refractory_s: float = 0.2,
    block_s: float = 10.0,
    rel_threshold: float = 0.25,
    band: tuple[float, float] = (3.0, 25.0),
    mwi_s: float = 0.15,
    searchback: bool = True,
    searchback_ratio: float = 1.66,
    searchback_rel: float = 0.5,
) -> np.ndarray:
    """Return sorted sample indices of R peaks.

    Pipeline: band-pass ``band`` -> derivative -> squaring -> moving-window integration
    (``mwi_s``) -> peak picking with a per-block adaptive threshold (``rel_threshold`` times the
    90th percentile of candidate heights in the block) -> refinement on the 5-25 Hz signal
    (largest absolute deflection, so inverted beats are handled).

    With ``searchback=True``, a gap longer than ``searchback_ratio`` times the recent median RR
    is searched for a rejected candidate whose height is at least
    ``searchback_rel * rel_threshold`` of the block reference; the highest one is accepted.
    This is meant to recover premature / low-amplitude beats.

    Defaults (threshold 0.25, band 3-25 Hz, search-back on) were tuned on MIT-BIH train+val
    records (see ``scripts/phase1_tune.py``). The original settings were ``rel_threshold=0.35,
    band=(5, 15), searchback=False``: they missed about 21 % of V beats.
    """
    x = np.asarray(x, dtype=float)
    lo_hz, hi_hz = band
    bp = bandpass(x, fs, lo_hz, min(hi_hz, 0.45 * fs))
    energy = np.gradient(bp) ** 2
    win = max(int(mwi_s * fs), 1)
    mwi = np.convolve(energy, np.ones(win) / win, mode="same")

    dist = max(int(refractory_s * fs), 1)
    cand, _ = signal.find_peaks(mwi, distance=dist)
    if cand.size == 0:
        return np.empty(0, dtype=int)
    heights = mwi[cand]

    block = max(int(block_s * fs), 1)
    global_ref = np.percentile(heights, 90)
    keep = np.zeros(cand.size, dtype=bool)
    ref_c = np.full(cand.size, global_ref)
    for start in range(0, len(x), block):
        in_block = (cand >= start) & (cand < start + block)
        if not in_block.any():
            continue
        ref = np.percentile(heights[in_block], 90) if in_block.sum() >= 3 else global_ref
        ref_c[in_block] = ref
        keep |= in_block & (heights >= rel_threshold * ref)

    if searchback and keep.sum() >= 3:
        keep = _searchback(cand, heights, ref_c, keep, rel_threshold, searchback_rel,
                           searchback_ratio, dist)

    cand = cand[keep]
    if cand.size == 0:
        return np.empty(0, dtype=int)

    fine = np.abs(bandpass(x, fs, 5.0, min(25.0, 0.45 * fs)))
    half = max(int(0.1 * fs), 1)
    refined = []
    for p in cand:
        lo, hi_i = max(p - half, 0), min(p + half + 1, len(x))
        refined.append(lo + int(np.argmax(fine[lo:hi_i])))
    refined = np.unique(np.array(refined, dtype=int))

    # enforce refractory period, keeping the stronger deflection
    out: list[int] = []
    for p in refined:
        if out and p - out[-1] < dist:
            if fine[p] > fine[out[-1]]:
                out[-1] = int(p)
        else:
            out.append(int(p))
    return np.array(out, dtype=int)


def _searchback(cand, heights, ref_c, keep, rel_threshold, searchback_rel, ratio, dist):
    """Add rejected candidates inside abnormally long gaps between accepted peaks."""
    keep = keep.copy()
    acc = np.flatnonzero(keep)
    recent: list[float] = []
    for a, b in zip(acc[:-1], acc[1:]):
        gap = cand[b] - cand[a]
        if len(recent) >= 3 and gap > ratio * np.median(recent[-8:]):
            idx = np.arange(a + 1, b)
            ok = (
                (heights[idx] >= searchback_rel * rel_threshold * ref_c[idx])
                & (cand[idx] - cand[a] >= dist)
                & (cand[b] - cand[idx] >= dist)
            )
            if ok.any():
                best = idx[ok][np.argmax(heights[idx[ok]])]
                keep[best] = True
                recent.append(cand[best] - cand[a])
                recent.append(cand[b] - cand[best])
                continue
        recent.append(gap)
    return keep


def matched_flags(ref, det, tol: float) -> np.ndarray:
    """For every reference peak: True if a detection lies within ``tol`` samples.

    One-to-one greedy matching (a detection is used for at most one reference peak).
    """
    ref = np.asarray(ref)
    det = np.sort(np.asarray(det))
    used = np.zeros(len(det), dtype=bool)
    flags = np.zeros(len(ref), dtype=bool)
    for i, r in enumerate(ref):
        if len(det) == 0:
            break
        j = int(np.searchsorted(det, r))
        cands = [k for k in (j - 1, j) if 0 <= k < len(det) and not used[k] and abs(det[k] - r) <= tol]
        if cands:
            k = min(cands, key=lambda k: abs(det[k] - r))
            used[k] = True
            flags[i] = True
    return flags


def match_peaks(ref, det, tol: float) -> tuple[int, int, int]:
    """One-to-one greedy matching within ``tol`` samples. Returns (tp, fp, fn)."""
    tp = int(matched_flags(ref, det, tol).sum())
    return tp, len(det) - tp, len(ref) - tp


def evaluate_detection(ref, det, fs: float, tol_s: float = 0.1) -> dict:
    tp, fp, fn = match_peaks(ref, det, tol=tol_s * fs)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "sensitivity": sens, "ppv": ppv}
