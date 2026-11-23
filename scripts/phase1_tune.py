"""Phase 1c: tune the R-peak detector so that it stops losing premature (V) beats.

    python scripts/phase1_tune.py --data-dir data/raw/mitdb --fs 250

Tries a small grid of detector settings (threshold, band-pass, search-back) on the TRAIN and
VAL records only -- the test split is deliberately not used for tuning. Prints pooled
sensitivity (overall and per class) and PPV for each setting, and recommends the one with the
best overall sensitivity among settings with PPV >= --min-ppv.
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd

from ecg_anomaly.beats.tuning import evaluate_config
from ecg_anomaly.data import PACED, load_record, split_records
from ecg_anomaly.preprocessing import preprocess

BASELINE = {"rel_threshold": 0.35, "band": (5.0, 15.0), "searchback": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/mitdb")
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="splits used for tuning (keep 'test' out of it)")
    ap.add_argument("--max-records", type=int, default=None, help="quick run on fewer records")
    ap.add_argument("--min-ppv", type=float, default=0.99)
    ap.add_argument("--out", default="results/phase1_tune.csv")
    args = ap.parse_args()

    if "test" in args.splits:
        print("WARNING: tuning on the test split makes the final test results optimistic.")

    splits = split_records()
    names = [n for s in args.splits for n in splits[s] if n not in PACED]
    if args.max_records:
        names = names[: args.max_records]

    records = []
    for name in names:
        rec = load_record(name, args.data_dir, target_fs=args.fs)
        records.append({"name": name, "fs": rec.fs, "x": preprocess(rec.signal, rec.fs),
                        "ann_samples": rec.ann_samples, "ann_classes": rec.ann_classes})
    print(f"Loaded {len(records)} records: {names}\n")

    grid = itertools.product([0.35, 0.25, 0.15], [(5.0, 15.0), (3.0, 25.0)], [False, True])
    rows = []
    for rel, band, sb in grid:
        cfg = {"rel_threshold": rel, "band": band, "searchback": sb}
        res = evaluate_config(records, cfg)
        rows.append({"rel_threshold": rel, "band": f"{band[0]:g}-{band[1]:g}", "searchback": sb,
                     "baseline": cfg == BASELINE, **res})
        r = rows[-1]
        print(f"rel={rel:<4} band={r['band']:<5} sb={sb!s:<5} sensV={r['sens_V']:.4f} "
              f"sensN={r['sens_N']:.4f} sens={r['sens_all']:.4f} ppv={r['ppv']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    cols = ["rel_threshold", "band", "searchback", "sens_V", "sens_N", "sens_S", "sens_all",
            "ppv", "baseline"]
    print("\nSorted by sensitivity on V beats:")
    print(df.sort_values("sens_V", ascending=False)[cols].round(4).to_string(index=False))

    ok = df[df.ppv >= args.min_ppv]
    if ok.empty:
        print(f"\nNo setting reaches PPV >= {args.min_ppv}.")
    else:
        best = ok.sort_values(["sens_all", "ppv"], ascending=False).iloc[0]
        print(f"\nRecommended (best overall sensitivity with PPV >= {args.min_ppv}):")
        print(best[cols].to_string())
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
