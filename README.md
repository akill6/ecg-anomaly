# ecg-anomaly

Unsupervised detection of anomalous beats and rhythm segments in long ECG recordings.
Continuation of my bachelor's thesis
[signal-sparsity-dot](https://github.com/akill6/signal-sparsity-dot) (MATLAB, sparsity of ECG in
DFT / DCT / DWT domains), now in Python / PyTorch.

**Research question.** Can anomalous beats be found without labels, and what works better:
fixed sparse bases (DCT / DWT from the thesis), learned representations (autoencoders), or
simple hand-made features?

> **Disclaimer.** This is a research / educational project, not a medical device. It does not
> diagnose arrhythmias; it only highlights atypical beats and segments for review by an expert.
> Evaluation uses a small set of public databases and says nothing about real-world performance.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # add ",dl" later for PyTorch
pytest
```

## Phase 1 quickstart

```bash
python scripts/download_data.py --db mitdb --dest data/raw/mitdb
python scripts/phase1_check.py --data-dir data/raw/mitdb --fs 250
```

Exit criterion: R-peak sensitivity and PPV above 99 % on most records.

To verify the port of the thesis code, copy `matDBs/6.mat` from the MATLAB repository to
`tests/data/6.mat`; `pytest` then compares nonzero-coefficient counts and PRD with the README
table of the MATLAB app (DFT 1043, DCT 861, DWT-Haar 337 at PRD = 7 %).

## Evaluation protocol (fixed before modelling)

* Splits are by patient: train = DS1 minus validation records, validation = 4 DS1 records,
  test = DS2 (de Chazal et al.). Paced records 102, 104, 107, 217 are excluded. Records 201 and
  202 come from the same patient, a known small leak of that split.
* Labels (AAMI classes N, S, V, F, Q) are used **only** to compute metrics, never for training.
* Thresholds are chosen on validation data only; the test split is touched once at the end.
* Report AUPRC, sensitivity at a fixed false-alarm rate, and bootstrap confidence intervals
  over patients.

## Layout

```
src/ecg_anomaly/
  data/           MIT-BIH loading, AAMI labels, patient-wise splits
  preprocessing/  filters, bad-segment detection
  beats/          R-peak detection, beat windows, RR features
  sparsity/       DFT / DCT / DWT sparse approximation (port of the MATLAB thesis app)
scripts/          download_data.py, phase1_check.py
tests/            synthetic-signal tests + MATLAB parity test
```

## Data and licences

MIT-BIH Arrhythmia Database (Moody & Mark, 2001) from PhysioNet; please cite it and the
PhysioNet platform paper when using results. LUDB (ODC-By) is kept only as an optional
secondary dataset.
