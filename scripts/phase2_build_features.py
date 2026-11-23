"""Phase 2a: detect beats in every record and save the feature table and z-scored beats.

    # MIT-BIH Arrhythmia DB (train / val / test split of de Chazal)
    python scripts/phase2_build_features.py --data-dir data/raw/mitdb --fs 250

    # external database, never used for development (e.g. SVDB)
    python scripts/download_data.py --db svdb --dest data/raw/svdb
    python scripts/phase2_build_features.py --data-dir data/raw/svdb --fs 250 --split ext \
        --out data/processed_svdb

Writes features.csv.gz, beats.npz, durations.csv into --out (takes a few minutes).
"""

import argparse

from ecg_anomaly.features import build_dataset, save_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/mitdb")
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--split", default=None,
                    help="give every record this split name (external database); default: "
                         "MIT-BIH train/val/test split")
    ap.add_argument("--lead", default=None, help="lead name (default: MLII, else first channel)")
    args = ap.parse_args()
    table, beats, durations = build_dataset(args.data_dir, args.fs, split=args.split,
                                            lead=args.lead)
    save_dataset(table, beats, durations, args.out)
    print(f"\n{len(table)} beats, saved to {args.out}")
    print(table.groupby(["split", "cls"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
