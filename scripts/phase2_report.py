"""Phase 2c: figure and table from the saved baseline results.

    python scripts/phase2_report.py            # -> results/phase2_report.png / .md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_anomaly.evaluation.experiment import dominated_by
from ecg_anomaly.evaluation.plots import make_report_figure, results_markdown
from ecg_anomaly.features import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed")
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="iforest")
    ap.add_argument("--feature-set", default="fusion-max")
    ap.add_argument("--fp-target", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.results)
    res = pd.read_csv(out / "phase2_baseline.csv")
    z = np.load(out / "phase2_scores.npz")
    scores = {tuple(k.split("|")): z[k] for k in z.files}
    table, _, _ = load_dataset(args.features)
    te = table[table.split == "test"].reset_index(drop=True)
    res.attrs["s_dominated_records"] = dominated_by(te, "S")

    png = make_report_figure(res, scores, te, args.model, args.feature_set,
                             out / "phase2_report.png", args.fp_target)
    md = "\n\n".join(results_markdown(res, m, args.fp_target) for m in res.model.unique())
    (out / "phase2_report.md").write_text(md + "\n", encoding="utf8")
    print(md)
    print(f"\nFigure: {png}")


if __name__ == "__main__":
    main()
