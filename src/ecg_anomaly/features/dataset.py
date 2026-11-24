"""Build the beat table (features, labels, split) for a whole database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..beats import detect_rpeaks, extract_beats
from ..data import PACED, label_detected_peaks, load_record, split_records
from ..preprocessing import preprocess
from .beat_features import beat_feature_table
from .spectral import TRANSFORMS, all_sparsity_features, sparsity_cols

PRE_S, POST_S = 0.25, 0.45


def record_table(rec, pre_s: float = PRE_S, post_s: float = POST_S):
    """Detect beats in one record and return ``(table, beats_z)``.

    ``cls`` is the AAMI class of the nearest annotation (labels are for evaluation only);
    detections without an annotation within 0.15 s get class ``X`` (false detections).
    """
    y = preprocess(rec.signal, rec.fs)
    peaks = detect_rpeaks(y, rec.fs)
    beats_raw, kept = extract_beats(y, peaks, rec.fs, pre_s, post_s, normalize="none")
    beats_z, _ = extract_beats(y, peaks, rec.fs, pre_s, post_s, normalize="zscore")
    pre = int(round(pre_s * rec.fs))
    tab = beat_feature_table(beats_raw, beats_z, kept, rec.fs, pre)
    labels = label_detected_peaks(kept, rec)
    tab.insert(0, "cls", np.where(labels == "", "X", labels))
    tab.insert(0, "time_s", kept / rec.fs)
    tab.insert(0, "sample", kept)
    tab.insert(0, "record", rec.name)
    tab.attrs["duration_s"] = len(rec.signal) / rec.fs
    return tab, beats_z


def list_records(data_dir) -> list[str]:
    """Record names found in a folder (every ``<name>.hea``), sorted."""
    return sorted(p.stem for p in Path(data_dir).glob("*.hea"))


def build_dataset(data_dir, fs: float = 250.0, records: list[str] | None = None, log=print,
                  split: str | None = None, lead: str | None = None):
    """Feature table + z-scored beats for all (non-paced) records, with a ``split`` column.

    ``split=None``: MIT-BIH Arrhythmia DB with the train / val / test split of de Chazal.
    ``split="ext"`` (any name): another database (e.g. SVDB); every record found in
    ``data_dir`` gets that split name and ``lead`` selects the channel (default: first).
    Returns ``(table, beats, durations)``; ``durations`` maps record -> length in seconds.
    """
    if split is None:
        splits = split_records()
        split_of = {r: s for s, names in splits.items() for r in names}
        names = [r for r in (records or list(split_of)) if r not in PACED]
    else:
        names = list(records) if records else list_records(data_dir)
        split_of = dict.fromkeys(names, split)
    tabs, beats, durations = [], [], {}
    for name in names:
        rec = load_record(name, data_dir, target_fs=fs, lead=lead)
        tab, bz = record_table(rec)
        tab.insert(1, "split", split_of[name])
        tabs.append(tab)
        beats.append(bz.astype(np.float32))
        durations[name] = tab.attrs["duration_s"]
        log(f"{split_of[name]:5s} {name}: {len(tab)} beats")
    return pd.concat(tabs, ignore_index=True), np.concatenate(beats), durations


def save_dataset(table, beats, durations, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "features.csv.gz", index=False)
    np.savez_compressed(out / "beats.npz", beats=beats)
    pd.Series(durations, name="duration_s").rename_axis("record").to_csv(out / "durations.csv")


def ensure_sparsity_columns(table: pd.DataFrame, beats) -> pd.DataFrame:
    """Add sparsity columns of any basis missing in an older table (computed from the beats)."""
    missing = [t for t in TRANSFORMS if not set(sparsity_cols(t)) <= set(table.columns)]
    if not missing:
        return table
    new = pd.DataFrame(all_sparsity_features(beats, missing), index=table.index)
    return pd.concat([table, new[[c for c in new.columns if c not in table.columns]]], axis=1)


def load_dataset(out_dir):
    out = Path(out_dir)
    table = pd.read_csv(out / "features.csv.gz", dtype={"record": str})
    beats = np.load(out / "beats.npz")["beats"]
    table = ensure_sparsity_columns(table, beats)
    durations = pd.read_csv(out / "durations.csv", dtype={"record": str}).set_index("record")["duration_s"]
    return table, beats, durations.to_dict()
