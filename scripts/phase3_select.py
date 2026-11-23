"""Phase 3a: choose the configuration by patient-wise cross-validation (development records).

    python scripts/phase3_select.py --features data/processed

Uses only train + val records (DS1); the test split and external data are NOT touched.
Takes about 15-25 minutes with the full grid (use --quick for a smaller one).
Writes results/cv_folds.csv, cv_summary.csv and selected.json.
"""

import argparse
import json
from pathlib import Path

from ecg_anomaly.evaluation import (
    cross_validate,
    default_configs,
    select_best,
    summarize_cv,
)
from ecg_anomaly.features import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed")
    ap.add_argument("--k", type=int, default=5, help="number of patient-wise folds")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="smaller grid for a fast look")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    table, _, _ = load_dataset(args.features)
    dev = table[table.split.isin(["train", "val"])].reset_index(drop=True)
    print(f"{dev.record.nunique()} development records, {len(dev)} beats; test split not used\n")

    if args.quick:
        configs = default_configs(("global", "patient-abs"), ("iforest",),
                                  ["rhythm", "morph", "template", "all_v1", "all", "fusion-mean"])
    else:
        configs = default_configs()
    cv = cross_validate(dev, configs, k=args.k, n_estimators=args.n_estimators)
    summary = summarize_cv(cv)
    sel = select_best(summary)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv.to_csv(out / "cv_folds.csv", index=False)
    summary.to_csv(out / "cv_summary.csv", index=False)
    (out / "selected.json").write_text(json.dumps(sel, indent=2), encoding="utf8")

    cols = ["norm", "model", "features", "auprc", "auprc_se", "auroc", "auroc_V", "auroc_S",
            "auprc_min"]
    print("\nTop 15 by mean AUPRC over folds:")
    print(summary[cols].head(15).round(3).to_string(index=False))
    print("\nSelected:")
    print(json.dumps(sel, indent=2))
    print(f"\nSaved to {out}/ . Next: python scripts/phase3_final.py")


if __name__ == "__main__":
    main()
