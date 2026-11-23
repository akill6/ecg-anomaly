"""Fuse several specialised unsupervised detectors into one anomaly score."""

from __future__ import annotations

import numpy as np

from .baselines import BaselineScorer


class FusionScorer:
    """One detector per group of features; scores are put on a common scale and combined.

    Different anomalies live in different feature groups (rhythm for premature beats, shape for
    ventricular beats, ...). One detector on all features dilutes them; here each group is
    scored on its own. A group's score is rescaled with train statistics only,
    ``(s - median) / (90th percentile - median)``, and the final score is the ``max`` (an
    anomaly in any group is enough) or the ``mean`` over groups. No labels are used.
    """

    def __init__(self, groups: dict[str, list[str]], kind: str = "iforest", combine: str = "max",
                 seed: int = 0, n_estimators: int = 200):
        if combine not in {"max", "mean"}:
            raise ValueError("combine must be 'max' or 'mean'")
        self.groups, self.kind, self.combine, self.seed = groups, kind, combine, seed
        self.n_estimators = n_estimators

    def fit(self, df) -> "FusionScorer":
        self.scorers_, self.scale_ = {}, {}
        for name, cols in self.groups.items():
            sc = BaselineScorer(self.kind, seed=self.seed, n_estimators=self.n_estimators).fit(df[cols].to_numpy())
            s = sc.score(df[cols].to_numpy())
            med, q90 = np.median(s), np.quantile(s, 0.9)
            self.scorers_[name] = sc
            self.scale_[name] = (med, max(q90 - med, 1e-9))
        return self

    def group_scores(self, df) -> dict[str, np.ndarray]:
        out = {}
        for name, cols in self.groups.items():
            med, sc = self.scale_[name]
            out[name] = (self.scorers_[name].score(df[cols].to_numpy()) - med) / sc
        return out

    def score(self, df) -> np.ndarray:
        stack = np.vstack(list(self.group_scores(df).values()))
        return stack.max(axis=0) if self.combine == "max" else stack.mean(axis=0)

