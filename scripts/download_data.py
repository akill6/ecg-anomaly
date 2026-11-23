"""Download a PhysioNet database, e.g.:

    python scripts/download_data.py --db mitdb --dest data/raw/mitdb
    python scripts/download_data.py --db nsrdb --dest data/raw/nsrdb
"""

import argparse

from ecg_anomaly.data import download_database


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mitdb", help="PhysioNet database id")
    ap.add_argument("--dest", default="data/raw/mitdb")
    ap.add_argument("--records", nargs="*", default=None, help="subset of records (default: all)")
    args = ap.parse_args()
    failed = download_database(args.db, args.dest, args.records)
    if failed:
        print(f"\nNot downloaded: {failed}. Run the same command again to retry only these.")
    else:
        print(f"Done: {args.db} -> {args.dest}")


if __name__ == "__main__":
    main()
