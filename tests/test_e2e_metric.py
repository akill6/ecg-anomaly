import numpy as np

from ecg_anomaly.evaluation.metrics import sens_at_fp_rate


def test_e2e_false_alarms_count_false_detections():
    cls = np.array(["N"] * 1000 + ["X"] * 10 + ["V"] * 10)
    score = np.concatenate([np.random.default_rng(0).random(1000), np.full(10, 2.0), np.full(10, 1.5)])
    a = sens_at_fp_rate(cls, score, 1.0, 5)
    b = sens_at_fp_rate(cls, score, 1.0, 5, fp_classes=("N", "X"))
    assert a["sens_anomaly"] > b["sens_anomaly"]
