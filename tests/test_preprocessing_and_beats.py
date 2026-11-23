import numpy as np

from ecg_anomaly.beats import (
    detect_rpeaks,
    evaluate_detection,
    extract_beats,
    match_peaks,
    matched_flags,
    rr_features,
)
from ecg_anomaly.preprocessing import bad_segment_mask, highpass, preprocess

FS = 250


def test_highpass_removes_baseline_wander():
    t = np.arange(0, 30, 1 / FS)
    drift = np.sin(2 * np.pi * 0.1 * t)
    y = highpass(drift, FS, 0.5)
    assert np.std(y[FS * 5 : -FS * 5]) < 0.1 * np.std(drift)


def test_bad_segments_flag_flatline_and_artifact(ecg):
    x, _ = ecg
    x = x.copy()
    x[10 * FS : 14 * FS] = 0.0           # lead off
    x[30 * FS : 32 * FS] *= 40.0         # saturation / motion artifact
    mask = bad_segment_mask(x, FS)
    assert mask[11 * FS : 13 * FS].all()
    assert mask[30 * FS : 32 * FS].all()
    assert mask[:8 * FS].sum() == 0


def test_rpeak_detection_on_synthetic_ecg(ecg):
    x, truth = ecg
    det = detect_rpeaks(preprocess(x, FS), FS)
    res = evaluate_detection(truth, det, FS, tol_s=0.05)
    assert res["sensitivity"] > 0.98
    assert res["ppv"] > 0.98


def test_rpeak_detection_handles_inverted_polarity(ecg):
    x, truth = ecg
    det = detect_rpeaks(preprocess(-x, FS), FS)
    res = evaluate_detection(truth, det, FS, tol_s=0.05)
    assert res["sensitivity"] > 0.95 and res["ppv"] > 0.95


def test_match_peaks_counts():
    tp, fp, fn = match_peaks([100, 200, 300], [101, 205, 400], tol=10)
    assert (tp, fp, fn) == (2, 1, 1)


def test_matched_flags_per_reference_peak():
    flags = matched_flags([100, 200, 300], [101, 205, 400], tol=10)
    assert flags.tolist() == [True, True, False]


def test_extract_beats_and_rr_features(ecg):
    x, truth = ecg
    beats, peaks = extract_beats(x, truth, FS)
    assert beats.shape == (len(peaks), round(0.25 * FS) + round(0.45 * FS))
    assert np.allclose(beats.mean(axis=1), 0, atol=1e-8)
    f = rr_features(peaks, FS)
    assert len(f["rr_prev"]) == len(peaks)
    assert np.all(f["ratio_prev"] > 0)


def test_searchback_recovers_a_weak_beat():
    from conftest import synthetic_ecg

    x, truth = synthetic_ecg(noise=0.0, drift=0.0, seed=3)
    x = x.copy()
    p = truth[len(truth) // 2]
    w = int(0.12 * FS)
    x[p - w : p + w] *= 0.5  # a low-amplitude (premature-like) beat
    y = preprocess(x, FS)
    original = dict(rel_threshold=0.35, band=(5.0, 15.0))
    plain = detect_rpeaks(y, FS, searchback=False, **original)
    sb = detect_rpeaks(y, FS, searchback=True, **original)
    near = lambda det: np.any(np.abs(det - p) <= int(0.05 * FS))
    assert not near(plain)
    assert near(sb)
    res = evaluate_detection(truth, sb, FS, tol_s=0.05)
    assert res["ppv"] > 0.98


def test_detector_defaults_are_the_tuned_settings(ecg):
    x, _ = ecg
    y = preprocess(x, FS)
    explicit = detect_rpeaks(y, FS, rel_threshold=0.25, band=(3.0, 25.0), mwi_s=0.15,
                             searchback=True)
    assert np.array_equal(detect_rpeaks(y, FS), explicit)


def test_evaluate_config_per_class(ecg):
    from ecg_anomaly.beats.tuning import evaluate_config

    x, truth = ecg
    rec = {"x": preprocess(x, FS), "fs": FS, "ann_samples": truth,
           "ann_classes": np.array(["N"] * len(truth))}
    res = evaluate_config([rec], {"rel_threshold": 0.35}, tol_s=0.05)
    assert res["beats"] == len(truth)
    assert res["sens_N"] > 0.98 and res["ppv"] > 0.98
    assert np.isnan(res["sens_V"])


def test_record_detection_summary_counts_per_class():
    from ecg_anomaly.beats.tuning import record_detection_summary

    ann = np.array([100, 200, 300, 400])
    cls = np.array(["N", "S", "V", "N"])
    det = np.array([101, 299, 405, 500])       # misses the S beat at 200; 500 is a false detection
    s = record_detection_summary(det, ann, cls, fs=100, tol_s=0.1)
    assert (s["annotated"], s["tp"], s["fp"], s["fn"]) == (4, 3, 1, 1)
    assert (s["n_N"], s["missed_N"], s["n_S"], s["missed_S"], s["missed_V"]) == (2, 0, 1, 1, 0)
