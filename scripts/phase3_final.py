"""Phase 3b: ONE final evaluation of the configuration chosen by cross-validation.

    python scripts/phase3_final.py --features data/processed --ext data/processed_svdb

Fits on the development records (train + val), then scores MIT-BIH DS2 and (if given) an external
database, comparing the chosen configuration with a fixed reference (phase-2a features, per-patient
normalisation, Isolation Forest). Confidence intervals and paired differences are bootstrapped over
patients. A lock file stops accidental repeated looks at these data; re-running needs --force.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_anomaly.evaluation import evaluate_final
from ecg_anomaly.evaluation.plots import final_markdown, make_final_figure
from ecg_anomaly.features import load_dataset

REFERENCE = ("patient", "iforest", "all_v1")   # fixed in advance (best phase-2a configuration)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed")
    ap.add_argument("--ext", default=None, help="folder with the external feature table (SVDB)")
    ap.add_argument("--selected", default=None, help="default: <out-dir>/selected.json")
    ap.add_argument("--which", choices=["best", "one_se"], default="best")
    ap.add_argument("--include-one-se", action="store_true",
                    help="also evaluate the simplest-within-1-SE choice")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--force", action="store_true", help="evaluate again although already done")
    args = ap.parse_args()

    out = Path(args.out_dir)
    lock = out / "final_done.lock"
    if lock.exists() and not args.force:
        raise SystemExit(f"Final evaluation was already done ({lock.read_text(encoding='utf8')}).\n"
                         "Looking again turns the test data into a tuning set. Use --force only "
                         "if you accept that.")

    sel = json.loads(Path(args.selected or out / "selected.json").read_text(encoding="utf8"))
    pick = sel[args.which]
    configs = {"baseline": REFERENCE,
               "selected": (pick["norm"], pick["model"], pick["features"])}
    if args.include_one_se and sel["one_se"] != pick:
        o = sel["one_se"]
        configs["one_se"] = (o["norm"], o["model"], o["features"])

    table, _, durations = load_dataset(args.features)
    dev = table[table.split.isin(["train", "val"])].reset_index(drop=True)
    test = table[table.split == "test"].reset_index(drop=True)
    eval_sets = {"MIT-BIH DS2": (test, durations)}
    if args.ext:
        ext, _, ext_dur = load_dataset(args.ext)
        eval_sets["SVDB (external)"] = (ext.reset_index(drop=True), ext_dur)

    res, scores = evaluate_final(dev, eval_sets, configs, "baseline", n_boot=args.n_boot)

    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "final_results.csv", index=False)
    np.savez_compressed(out / "final_scores.npz", **{f"{d}|{c}": v for (d, c), v in scores.items()})
    md = final_markdown(res)
    (out / "final_report.md").write_text(md + "\n", encoding="utf8")
    cv_summary = pd.read_csv(out / "cv_summary.csv")
    png = make_final_figure(res, scores, eval_sets, cv_summary, configs, out / "final_report.png")
    lock.write_text(f"{time.strftime('%Y-%m-%d %H:%M')} configs={configs}", encoding="utf8")
    print(md)
    print(f"\nFigure: {png}\nLocked: {lock}")


if __name__ == "__main__":
    main()
