"""Phase 1b: which beat classes does the R-peak detector miss?

    python scripts/phase1_by_class.py --data-dir data/raw/mitdb --fs 250

For every annotated beat we check whether the detector found it, then report sensitivity
per AAMI class (N normal, S supraventricular, V ventricular, F fusion, Q other/paced) and the
share of all missed beats that belongs to each class. If V dominates the misses, the detector
is losing the beats we care about most.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_anomaly.beats import detect_rpeaks, matched_flags
from ecg_anomaly.data import PACED, load_record, split_records
from ecg_anomaly.preprocessing import preprocess

CLASSES = ["N", "S", "V", "F", "Q"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/mitdb")
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--tol", type=float, default=0.1, help="matching tolerance [s]")
    ap.add_argument("--out", default="results/phase1_by_class.csv")
    args = ap.parse_args()

    rows = []
    for split, names in split_records().items():
        for name in names:
            if name in PACED:
                continue
            rec = load_record(name, args.data_dir, target_fs=args.fs)
            peaks = detect_rpeaks(preprocess(rec.signal, rec.fs), rec.fs)
            found = matched_flags(rec.ann_samples, peaks, tol=args.tol * rec.fs)
            classes = rec.ann_classes
            for cls in CLASSES:
                m = classes == cls
                if m.any():
                    rows.append({"split": split, "record": name, "cls": cls,
                                 "n": int(m.sum()), "missed": int((~found[m]).sum())})
            print(f"{split:5s} {name}: done")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    for label, g in [("ALL", df)] + [(s, df[df.split == s]) for s in ("train", "val", "test")]:
        t = g.groupby("cls")[["n", "missed"]].sum().reindex(CLASSES).dropna(how="all").fillna(0)
        t["sensitivity"] = (1 - t.missed / t.n).round(4)
        t["share_of_misses_%"] = (100 * t.missed / max(t.missed.sum(), 1)).round(1)
        print(f"\n=== {label} ===")
        print(t.astype({"n": int, "missed": int}).to_string())

    v = df[df.cls == "V"].copy()
    v = v[v.n >= 20]
    v["sens_V"] = (1 - v.missed / v.n).round(3)
    print("\nWorst records by V-beat sensitivity (>= 20 V beats):")
    print(v.sort_values("sens_V").head(10)[["split", "record", "n", "missed", "sens_V"]]
          .to_string(index=False))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
