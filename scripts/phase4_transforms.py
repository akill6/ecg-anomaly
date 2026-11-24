"""Compare the sparsity bases (DCT, DFT, DWT Haar, DWT db2) on development records (DS1 CV).

    python scripts/phase4_transforms.py --features data/processed

Two views: (a) the sparsity features ALONE, which shows how informative each basis is on its own;
(b) the full model with that basis, which shows what it adds to the other features. The test split
and external data are NOT touched. Writes results/transforms_folds.csv, transforms_summary.csv
and transforms_report.png. Pick the winner by ``auprc`` (mean over folds) and look at its
standard error: differences smaller than one SE are noise.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecg_anomaly.evaluation import cross_validate, summarize_cv
from ecg_anomaly.features import TRANSFORMS, load_dataset
from ecg_anomaly.features.spectral import LABELS

ALONE = {"dct": "sparsity", "dft": "sparsity_dft", "haar": "sparsity_haar", "db2": "sparsity_db2"}
FULL = {"dct": "all", "dft": "all_dft", "haar": "all_haar", "db2": "all_db2"}
COL = {"dct": "#3b6fb6", "dft": "#e08a2c", "haar": "#3a9d78", "db2": "#9a5fb4"}


def figure(summ, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=False)
    metrics = [("auprc", "auprc_se", "AUPRC (all anomalies)"), ("auprc_S", None, "AUPRC, S beats"),
               ("auroc", "auroc_se", "AUROC (all anomalies)")]
    rows = summ.set_index("features")
    for ax, (m, se, title) in zip(axes, metrics):
        x = np.arange(2)
        w = 0.19
        for i, t in enumerate(TRANSFORMS):
            v = [rows.loc[ALONE[t], m], rows.loc[FULL[t], m]]
            e = [rows.loc[ALONE[t], se], rows.loc[FULL[t], se]] if se else None
            ax.bar(x + (i - 1.5) * w, v, w * 0.92, yerr=e, color=COL[t], label=LABELS[t],
                   capsize=3, error_kw={"lw": 1, "ecolor": "#444"})
        ax.set_xticks(x, ["sparsity features\nalone", "full model\n(all features)"])
        ax.set_title(title, fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Which sparsity basis fits ECG beats best? Patient-wise 5-fold CV on DS1 "
                 "(error bars: standard error over folds)", fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    table, _, _ = load_dataset(args.features)
    dev = table[table.split.isin(["train", "val"])].reset_index(drop=True)
    print(f"{dev.record.nunique()} development records, {len(dev)} beats; test not used\n")
    sets = list(ALONE.values()) + list(FULL.values()) + ["sparsity_all", "all_multi"]
    cv = cross_validate(dev, [("patient-abs", "iforest", s) for s in sets], k=args.k,
                        n_estimators=args.n_estimators)
    summ = summarize_cv(cv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv.to_csv(out / "transforms_folds.csv", index=False)
    summ.to_csv(out / "transforms_summary.csv", index=False)
    figure(summ, out / "transforms_report.png")
    cols = ["features", "n_features", "auprc", "auprc_se", "auroc", "auroc_V", "auroc_S", "auprc_S"]
    print("\n", summ[cols].round(3).to_string(index=False))
    print(f"\nSaved to {out}/ (transforms_summary.csv, transforms_report.png)")


if __name__ == "__main__":
    main()
