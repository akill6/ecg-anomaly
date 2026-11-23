"""Phase 2b: unsupervised baselines on hand-made features.

    python scripts/phase2_baseline.py --features data/processed
    python scripts/phase2_report.py

Models are fitted on train beats WITHOUT labels, with three feature normalisations
(global / per patient / per patient with a live window). Anomaly = S, V or F beat vs N beat;
Q and false detections are ignored in the metrics. Saves results/phase2_baseline.csv and the
test scores (results/phase2_scores.npz) used by phase2_report.py.
"""

import argparse
from pathlib import Path

import numpy as np

from ecg_anomaly.evaluation import ALL_SETS, NORMS, run_baselines
from ecg_anomaly.features import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed")
    ap.add_argument("--models", nargs="*", default=["iforest", "gmm"])
    ap.add_argument("--sets", nargs="*", default=list(ALL_SETS))
    ap.add_argument("--norms", nargs="*", default=list(NORMS))
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    table, _, durations = load_dataset(args.features)
    res, scores = run_baselines(table, durations, args.models, args.sets, args.norms,
                                n_boot=args.n_boot)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "phase2_baseline.csv", index=False)
    np.savez_compressed(out / "phase2_scores.npz",
                        **{"|".join(k): v for k, v in scores.items()})
    print(f"\nS-dominated test records (excluded from auroc_S_adj): "
          f"{res.attrs['s_dominated_records']}")
    print(f"Saved {out / 'phase2_baseline.csv'}; now run: python scripts/phase2_report.py")


if __name__ == "__main__":
    main()
