"""How many annotated beats does the R-peak detector find in ANY database, and which classes does
it lose? Needed because every anomaly metric is computed on detected beats only.

    python scripts/detector_recall.py --data-dir data/raw/svdb --fs 250 --out results/svdb_detector.csv

Reports overall sensitivity / PPV, sensitivity per AAMI class (N, S, V, F, Q) and the records
with the most missed beats. Detection counts as correct within +-0.1 s of the annotation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_anomaly.beats import detect_rpeaks
from ecg_anomaly.beats.tuning import ALL_CLASSES, record_detection_summary
from ecg_anomaly.data import PACED, load_record
from ecg_anomaly.features import list_records
from ecg_anomaly.preprocessing import bad_segment_mask, preprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/svdb")
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--lead", default=None)
    ap.add_argument("--tol", type=float, default=0.1, help="matching tolerance [s]")
    ap.add_argument("--out", default="results/detector_recall.csv")
    args = ap.parse_args()

    rows = []
    for name in list_records(args.data_dir):
        if name in PACED:
            continue
        rec = load_record(name, args.data_dir, target_fs=args.fs, lead=args.lead)
        x = preprocess(rec.signal, rec.fs)
        peaks = detect_rpeaks(x, rec.fs)
        row = {"record": name, "lead": rec.lead,
               "bad_fraction": round(float(bad_segment_mask(x, rec.fs).mean()), 4)}
        row.update(record_detection_summary(peaks, rec.ann_samples, rec.ann_classes, rec.fs,
                                            args.tol))
        rows.append(row)
        print(f"{name}: annotated {row['annotated']}, missed {row['fn']}, false {row['fp']}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    tot = df[["annotated", "detected", "tp", "fp", "fn"]].sum()
    print(f"\n{len(df)} records, {int(tot.annotated)} annotated beats")
    print(f"Overall sensitivity {tot.tp / tot.annotated:.4f}   PPV {tot.tp / tot.detected:.4f}")
    t = pd.DataFrame({"n": [int(df[f"n_{c}"].sum()) for c in ALL_CLASSES],
                      "missed": [int(df[f"missed_{c}"].sum()) for c in ALL_CLASSES]},
                     index=list(ALL_CLASSES))
    t["sensitivity"] = np.where(t.n > 0, 1 - t.missed / t.n.clip(lower=1), np.nan).round(4)
    t["share_of_misses_%"] = (100 * t.missed / max(t.missed.sum(), 1)).round(1)
    print("\nPer AAMI class:")
    print(t.to_string())
    df["sens"] = (1 - df.fn / df.annotated).round(4)
    print("\nRecords with the lowest sensitivity:")
    print(df.sort_values("sens").head(10)[["record", "annotated", "fn", "fp", "sens",
                                           "bad_fraction"]].to_string(index=False))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
