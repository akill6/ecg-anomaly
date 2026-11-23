"""Phase 1 check: preprocess MIT-BIH records, detect R peaks, compare with annotations.

    python scripts/phase1_check.py --data-dir data/raw/mitdb --fs 250 --out results/phase1_rpeaks.csv

Exit criterion for phase 1: sensitivity and PPV clearly above 99 % on most records
(the hardest MIT-BIH records, e.g. 108, 203, 207, 208, 228, will be lower).
"""

import argparse
from pathlib import Path

import pandas as pd

from ecg_anomaly.beats import detect_rpeaks, evaluate_detection
from ecg_anomaly.data import PACED, load_record, split_records
from ecg_anomaly.preprocessing import bad_segment_mask, preprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/mitdb")
    ap.add_argument("--fs", type=float, default=250.0, help="common sampling rate after resampling")
    ap.add_argument("--tol", type=float, default=0.1, help="matching tolerance [s]")
    ap.add_argument("--out", default="results/phase1_rpeaks.csv")
    args = ap.parse_args()

    splits = split_records()
    rows = []
    for split, names in splits.items():
        for name in names:
            if name in PACED:
                continue
            rec = load_record(name, args.data_dir, target_fs=args.fs)
            x = preprocess(rec.signal, rec.fs)
            peaks = detect_rpeaks(x, rec.fs)
            res = evaluate_detection(rec.ann_samples, peaks, rec.fs, tol_s=args.tol)
            bad = float(bad_segment_mask(x, rec.fs).mean())
            rows.append({"split": split, "record": name, "lead": rec.lead,
                         "bad_fraction": round(bad, 4), **res})
            print(f"{split:5s} {name}: sens={res['sensitivity']:.4f} ppv={res['ppv']:.4f}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nMean by split:")
    print(df.groupby("split")[["sensitivity", "ppv", "bad_fraction"]].mean().round(4))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
