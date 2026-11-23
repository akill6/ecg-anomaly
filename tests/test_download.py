import sys
import types

import pytest

from ecg_anomaly.data import download_database, load_record


def _fake_wfdb(calls):
    mod = types.ModuleType("wfdb")
    mod.get_record_list = lambda db: ["a", "b"]

    def dl_database(db, dl_dir, records):
        for r in records:
            calls.append(r)
            for ext in ("hea", "dat", "atr"):
                (type(dl_dir) is str and __import__("pathlib").Path(dl_dir) / f"{r}.{ext}").write_text("x")

    mod.dl_database = dl_database
    return mod


def test_incomplete_record_is_downloaded_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "wfdb", _fake_wfdb(calls))
    for ext in ("hea", "dat"):                      # 'a' stopped before the annotations
        (tmp_path / f"a.{ext}").write_text("x")
    for ext in ("hea", "dat", "atr"):               # 'b' is complete
        (tmp_path / f"b.{ext}").write_text("x")
    failed = download_database("svdb", tmp_path)
    assert failed == [] and calls == ["a"]
    assert (tmp_path / "a.atr").exists()


def test_load_record_explains_missing_annotation_file(tmp_path):
    for ext in ("hea", "dat"):
        (tmp_path / f"844.{ext}").write_text("x")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_record("844", tmp_path)
