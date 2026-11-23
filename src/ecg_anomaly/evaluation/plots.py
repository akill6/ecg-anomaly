"""Report figure and markdown table for the phase-2 baselines."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, roc_curve

from .metrics import ANOMALY_CLASSES, eval_mask, per_record_auroc

# Reference palette (validated): blue / orange / aqua; neutral inks on a light surface.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
COLORS = {"global": "#eb6834", "patient": "#2a78d6", "patient-abs": "#6250d6",
          "patient-abs-causal": "#1baf7a", "patient-causal": "#1baf7a"}
LABELS = {"global": "global (one norm for everyone)", "patient": "per patient (all features)",
          "patient-abs": "per patient (absolute features only)",
          "patient-abs-causal": "per patient, abs. only (live, 300 beats)",
          "patient-causal": "per patient (live, 300 beats)"}
SET_LABELS = {"rr": "RR", "rhythm": "rhythm", "morph": "shape", "template": "tem-\nplate",
              "sparsity": "DCT", "all_v1": "all\nv1", "all": "all\nv2",
              "fusion-max": "fusion\nmax", "fusion-mean": "fusion\nmean"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "legend.frameon": False,
    })


def _ygrid(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, res, model, metric, norms, sets, ref=None, ylim=None):
    sub = res[res.model == model]
    w = 0.78 / len(norms) - 0.03
    x0 = np.arange(len(sets))
    offs = (np.arange(len(norms)) - (len(norms) - 1) / 2) * (w + 0.03)
    for off, n in zip(offs, norms):
        d = sub[sub.norm == n].set_index("features").reindex(sets)
        y = d[metric].to_numpy()
        err = np.vstack([y - d[f"{metric}_lo"], d[f"{metric}_hi"] - y])
        ax.bar(x0 + off, y, w, color=COLORS[n], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.errorbar(x0 + off, y, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=0,
                    alpha=0.7, zorder=4)
    if ref is not None:
        ax.axhline(ref, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
    ax.set_xticks(x0, [SET_LABELS[s] for s in sets])
    ax.tick_params(axis="x", length=0, labelsize=8.5)
    if ylim:
        ax.set_ylim(*ylim)
    _ygrid(ax)


def _curve_data(te, score):
    m = eval_mask(te.cls)
    y = np.isin(te.cls.to_numpy()[m], ANOMALY_CLASSES).astype(int)
    return y, np.asarray(score)[m]


def make_report_figure(res, scores, te, model="iforest", features="all", out="report.png",
                       fp_target=100, dpi=200):
    """Dashboard: feature sets, normalisation effect, curves, per-patient view.

    ``te`` is the test part of the beat table (same row order as the stored scores). The
    configuration shown in the curve / per-patient panels is fixed in advance
    (``model`` / ``features``), not picked by test performance.
    """
    _style()
    norms = [n for n in ("global", "patient", "patient-abs", "patient-abs-causal", "patient-causal")
             if n in set(res.norm)]
    sets = [s for s in SET_LABELS if s in set(res.features)]
    prev = float(res.prevalence.iloc[0])

    fig = plt.figure(figsize=(16, 10.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.865), w_pad=0.03, h_pad=0.04)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    fig.text(0.012, 0.972, "Unsupervised arrhythmia detection on MIT-BIH: hand-made features + "
             f"{'Isolation Forest' if model == 'iforest' else 'GMM'}",
             ha="left", fontsize=16, fontweight="bold")
    n_test = te.record.nunique()
    fig.text(0.012, 0.942,
             f"Fitted on train patients without labels · evaluated on {n_test} unseen test patients · "
             "anomaly = S, V or F beat vs normal beat · bars show 95 % CI over patients",
             color=INK2, fontsize=10, ha="left")
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=9, color=COLORS[n],
                      label=LABELS[n]) for n in norms]
    handles.append(Line2D([0], [0], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2,
                          label="random scorer"))
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.006, 0.925),
               fontsize=10, labelcolor=INK2, handlelength=1.4, columnspacing=2.2)

    ax = fig.add_subplot(gs[0, 0])
    _bars(ax, res, model, "auroc", norms, sets, ref=0.5, ylim=(0.4, 1.0))
    ax.set_title("AUROC by feature set")
    ax = fig.add_subplot(gs[0, 1])
    _bars(ax, res, model, "auprc", norms, sets, ref=prev, ylim=(0, 1.0))
    ax.set_title("AUPRC by feature set")

    # sensitivity per class at fixed false-alarm rate
    ax = fig.add_subplot(gs[0, 2])
    classes = [("V", "ventricular (V)"), ("S", "supraventricular (S)*"), ("F", "fusion (F)")]
    w = 0.78 / len(norms) - 0.03
    offs = (np.arange(len(norms)) - (len(norms) - 1) / 2) * (w + 0.03)
    for off, n in zip(offs, norms):
        r = res[(res.model == model) & (res.features == features) & (res.norm == n)].iloc[0]
        vals = [r[f"sens_{c}_adj@{fp_target}" if c == "S" else f"sens_{c}@{fp_target}"]
                for c, _ in classes]
        ax.bar(np.arange(3) + off, vals, w, color=COLORS[n], edgecolor=SURFACE, linewidth=1.5,
               zorder=3)
        for xi, v in zip(np.arange(3) + off, vals):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(3), [lbl for _, lbl in classes])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _ygrid(ax)
    ax.set_title(f"Beats caught at {fp_target} false alarms / hour")

    # curves for global vs per-patient, fixed config
    cmp_norms = [n for n in ("global", "patient") if n in norms]
    ax_roc, ax_pr = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for n in cmp_norms:
        y, s = _curve_data(te, scores[(n, model, features)])
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        r = res[(res.model == model) & (res.features == features) & (res.norm == n)].iloc[0]
        ax_roc.plot(fpr, tpr, color=COLORS[n], linewidth=2, solid_capstyle="round")
        ax_pr.plot(rc, pr, color=COLORS[n], linewidth=2, solid_capstyle="round")
        ax_roc.text(0.98, 0.20 if n == "patient" else 0.10, f"AUROC {r.auroc:.3f}", color=COLORS[n],
                    ha="right", fontweight="bold", transform=ax_roc.transAxes)
        ax_pr.text(0.98, 0.92 if n == "patient" else 0.82, f"AUPRC {r.auprc:.3f}", color=COLORS[n],
                   ha="right", fontweight="bold", transform=ax_pr.transAxes)
    ax_roc.plot([0, 1], [0, 1], color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax_pr.axhline(prev, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax_roc.set(xlabel="false-alarm rate (normal beats flagged)", ylabel="sensitivity",
               xlim=(0, 1), ylim=(0, 1.01))
    ax_pr.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.set_title("ROC curve")
    ax_pr.set_title("Precision-recall curve")
    for a in (ax_roc, ax_pr):
        _ygrid(a)

    # per-patient dumbbell
    ax = fig.add_subplot(gs[1, 2])
    pa = {}
    for n in cmp_norms:
        d = te[["record", "cls"]].copy()
        d["score"] = scores[(n, model, features)]
        pa[n] = per_record_auroc(d).set_index("record")
    if len(cmp_norms) == 2 and len(pa["patient"]):
        both = pa["patient"].join(pa["global"][["auroc"]].rename(columns={"auroc": "auroc_global"})).sort_values("auroc")
        ypos = np.arange(len(both))
        ax.hlines(ypos, both.auroc_global, both.auroc, color=GRID, linewidth=3, zorder=2)
        sz = 30 + 90 * np.sqrt(both.n_anomalous / both.n_anomalous.max())
        ax.scatter(both.auroc_global, ypos, s=sz, color=COLORS["global"], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
        ax.scatter(both.auroc, ypos, s=sz, color=COLORS["patient"], edgecolor=SURFACE,
                   linewidth=1.5, zorder=4)
        ax.set_yticks(ypos, both.index)
        ax.axvline(0.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
        lo = min(both.auroc.min(), both.auroc_global.min())
        ax.set_xlim(0.0 if lo < 0.2 else 0.2, 1.02)
        ax.set_xlabel("AUROC inside the patient's own recording")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelsize=8.5)
        ax.spines["left"].set_visible(False)
    ax.set_title("Every patient on its own (dot size = # anomalies)")

    fig.text(0.012, 0.008,
             "Research / educational project, not a medical device. "
             "Test split = DS2 of the de Chazal inter-patient protocol.",
             color=MUTED, fontsize=8.5)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def results_markdown(res, model="iforest", fp_target=100) -> str:  # noqa: C901
    """Compact results table (one model) with confidence intervals."""
    d = res[res.model == model]
    lines = [
        f"### {model}: test results (AUROC / AUPRC with 95 % CI over patients)", "",
        f"| normalisation | features | AUROC | AUPRC | AUROC V | AUROC S* | median patient AUROC | "
        f"sens. @{fp_target} FA/h | threshold transfer |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in d.iterrows():
        lines.append(
            f"| {r.norm} | {r.features} | {r.auroc:.3f} ({r.auroc_lo:.2f}-{r.auroc_hi:.2f}) | "
            f"{r.auprc:.3f} ({r.auprc_lo:.2f}-{r.auprc_hi:.2f}) | {r.auroc_V:.3f} | "
            f"{r.auroc_S_adj:.3f} | {r.auroc_patient_median:.3f} | "
            f"{r[f'sens_all@{fp_target}']:.0%} / {r[f'sens_V@{fp_target}']:.0%} / "
            f"{r[f'sens_S_adj@{fp_target}']:.0%} / {r[f'sens_F@{fp_target}']:.0%} | {r.transfer:.2f} |")
    dom = ", ".join(res.attrs.get("s_dominated_records", [])) or "none"
    lines += ["", f"\\* S (AUROC and sensitivity) computed without records where S outnumbers N ({dom}). "
              f"Random scorer: AUROC 0.5, AUPRC {d.prevalence.iloc[0]:.3f}. "
              "Threshold transfer = share of test beats above the train-fixed threshold divided "
              "by the intended share (1.0 is ideal)."]
    return "\n".join(lines)


# ---- final report: CV leaderboard + evaluation on data not used for selection ---------------

CFG_COLORS = {"baseline": "#eb6834", "selected": "#2a78d6", "one_se": "#1baf7a"}
CFG_NAMES = {"baseline": "reference (phase 2a features)", "selected": "chosen by cross-validation",
             "one_se": "simplest within 1 SE of the best"}


def _cfg_text(norm, model, fs):
    return f"{model} · {norm} · {fs}"


def _cfg_short(norm, model, fs):
    return f"{'IF' if model == 'iforest' else model.upper()} · {norm} · {fs}"


def make_final_figure(res, scores, eval_sets, cv_summary, selected, out="final_report.png",
                      dpi=200, top=12, fp_target=100):
    """Leaderboard of the cross-validation and results on the held-out datasets.

    ``selected``: ``label -> (norm, model, features)`` for the configurations shown.
    ``eval_sets``: ``name -> (table, durations)``; the LAST one is used for the curves and the
    per-patient panel (it should be the external database).
    """
    _style()
    labels = [c for c in ("baseline", "selected", "one_se") if c in selected]
    datasets = list(eval_sets)
    ext = datasets[-1]

    fig = plt.figure(figsize=(16, 10.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.865), w_pad=0.03, h_pad=0.04)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    fig.text(0.012, 0.972, "Chosen by cross-validation, tested once on data it never saw",
             ha="left", fontsize=16, fontweight="bold")
    fig.text(0.012, 0.942,
             "Unsupervised beat-anomaly detection · configuration picked by leave-patients-out CV on "
             "the development records · anomaly = S, V or F beat vs normal beat · bars show 95 % CI "
             "over patients", color=INK2, fontsize=10, ha="left")
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=9, color=CFG_COLORS[c],
                      label=f"{CFG_NAMES[c]}: {_cfg_text(*selected[c])}") for c in labels]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.006, 0.925),
               fontsize=9, labelcolor=INK2, handlelength=1.2, columnspacing=1.6)

    # A: CV leaderboard
    ax = fig.add_subplot(gs[0, 0])
    top_df = cv_summary.head(top).iloc[::-1]
    sel_keys = {tuple(v): k for k, v in selected.items()}
    colors = []
    for _, r in top_df.iterrows():
        k = sel_keys.get((r.norm, r.model, r.features))
        colors.append(CFG_COLORS[k] if k else "#b9b8b2")
    y = np.arange(len(top_df))
    ax.barh(y, top_df.auprc, height=0.62, color=colors, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.errorbar(top_df.auprc, y, xerr=top_df.auprc_se, fmt="none", ecolor=INK2, elinewidth=1,
                capsize=0, alpha=0.7, zorder=4)
    ax.set_yticks(y, [_cfg_short(r.norm, r.model, r.features) for _, r in top_df.iterrows()])
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    for yi, v, e in zip(y, top_df.auprc, top_df.auprc_se.fillna(0)):
        ax.text(v + e + 0.012, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, max(0.7, float(top_df.auprc.max()) + 0.12))
    ax.set_xlabel("AUPRC, mean over folds (± SE)  ·  IF = Isolation Forest")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("Cross-validation leaderboard")

    # B, C: AUROC / AUPRC with CI per dataset
    for col, metric, title in ((1, "auroc", "AUROC"), (2, "auprc", "AUPRC")):
        ax = fig.add_subplot(gs[0, col])
        w = 0.78 / len(labels) - 0.03
        offs = (np.arange(len(labels)) - (len(labels) - 1) / 2) * (w + 0.03)
        for off, c in zip(offs, labels):
            d = res[res.config == c].set_index("dataset").reindex(datasets)
            yv = d[metric].to_numpy()
            err = np.vstack([yv - d[f"{metric}_lo"], d[f"{metric}_hi"] - yv])
            xs = np.arange(len(datasets)) + off
            ax.bar(xs, yv, w, color=CFG_COLORS[c], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.errorbar(xs, yv, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=0,
                        alpha=0.7, zorder=4)
            for xi, v in zip(xs, yv):
                ax.text(xi, 0.02 if metric == "auprc" else 0.42, f"{v:.3f}", ha="center",
                        fontsize=9, color="white", fontweight="bold", zorder=5)
        prev = float(res[res.dataset == datasets[0]].prevalence.iloc[0])
        ax.axhline(0.5 if metric == "auroc" else prev, color=MUTED, linestyle=(0, (4, 3)),
                   linewidth=1, zorder=2)
        ax.set_xticks(np.arange(len(datasets)), datasets)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim((0.4, 1.0) if metric == "auroc" else (0, 1.0))
        _ygrid(ax)
        ax.set_title(title)

    # D, E: curves on the external set
    te_ext = eval_sets[ext][0].reset_index(drop=True)
    ax_roc, ax_pr = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    prev_ext = float(res[res.dataset == ext].prevalence.iloc[0])
    for k, c in enumerate(labels):
        yy, ss = _curve_data(te_ext, scores[(ext, c)])
        fpr, tpr, _ = roc_curve(yy, ss)
        pr, rc, _ = precision_recall_curve(yy, ss)
        r = res[(res.dataset == ext) & (res.config == c)].iloc[0]
        ax_roc.plot(fpr, tpr, color=CFG_COLORS[c], linewidth=2, solid_capstyle="round")
        ax_pr.plot(rc, pr, color=CFG_COLORS[c], linewidth=2, solid_capstyle="round")
        ax_roc.text(0.98, 0.08 + 0.09 * k, f"AUROC {r.auroc:.3f}", color=CFG_COLORS[c], ha="right",
                    fontweight="bold", transform=ax_roc.transAxes)
        ax_pr.text(0.98, 0.94 - 0.09 * k, f"AUPRC {r.auprc:.3f}", color=CFG_COLORS[c], ha="right",
                   fontweight="bold", transform=ax_pr.transAxes)
    ax_roc.plot([0, 1], [0, 1], color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax_pr.axhline(prev_ext, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax_roc.set(xlabel="false-alarm rate (normal beats flagged)", ylabel="sensitivity",
               xlim=(0, 1), ylim=(0, 1.01))
    ax_pr.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.set_title(f"ROC curve · {ext}")
    ax_pr.set_title(f"Precision-recall curve · {ext}")
    for a in (ax_roc, ax_pr):
        _ygrid(a)

    # F: per-patient dumbbell on the external set (reference -> chosen)
    ax = fig.add_subplot(gs[1, 2])
    pa = {}
    for c in ("baseline", "selected"):
        if c in labels:
            d = te_ext[["record", "cls"]].copy()
            d["score"] = scores[(ext, c)]
            pa[c] = per_record_auroc(d).set_index("record")
    if len(pa) == 2 and len(pa["selected"]):
        both = pa["selected"].join(pa["baseline"][["auroc"]].rename(columns={"auroc": "auroc_b"}))
        both = both.sort_values("n_anomalous", ascending=False).head(28).sort_values("auroc")
        ypos = np.arange(len(both))
        ax.hlines(ypos, both.auroc_b, both.auroc, color=GRID, linewidth=3, zorder=2)
        sz = 22 + 70 * np.sqrt(both.n_anomalous / both.n_anomalous.max())
        ax.scatter(both.auroc_b, ypos, s=sz, color=CFG_COLORS["baseline"], edgecolor=SURFACE,
                   linewidth=1.2, zorder=3)
        ax.scatter(both.auroc, ypos, s=sz, color=CFG_COLORS["selected"], edgecolor=SURFACE,
                   linewidth=1.2, zorder=4)
        ax.set_yticks(ypos, both.index)
        ax.axvline(0.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
        lo = min(both.auroc.min(), both.auroc_b.min())
        ax.set_xlim(0.0 if lo < 0.2 else 0.2, 1.02)
        ax.set_xlabel("AUROC inside the patient's own recording")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelsize=7.5)
        ax.spines["left"].set_visible(False)
    ax.set_title("Patients with most anomalies")

    fig.text(0.012, 0.008,
             "Research / educational project, not a medical device. Models never saw these "
             "patients or their labels; the configuration was fixed before they were scored.",
             color=MUTED, fontsize=8.5)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def final_markdown(res, fp_target=100) -> str:
    """Final results table: metrics with CIs and paired differences against the reference."""
    lines = [
        "### Final evaluation (configurations fixed by cross-validation beforehand)", "",
        f"| dataset | configuration | AUROC | AUPRC | Δ AUPRC vs reference | AUROC V | AUROC S* | "
        f"sens. @{fp_target} FA/h (all / V / S* / F) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        delta = ""
        if "d_auprc" in r and pd.notna(r.get("d_auprc")):
            delta = (f"{r.d_auprc:+.3f} ({r.d_auprc_lo:+.2f} to {r.d_auprc_hi:+.2f}); "
                     f"better in {r.p_better_auprc:.0%} of resamples")
        lines.append(
            f"| {r.dataset} ({r.patients} patients) | {r.config}: {r.model} · {r.norm} · "
            f"{r.features} | {r.auroc:.3f} ({r.auroc_lo:.2f}-{r.auroc_hi:.2f}) | "
            f"{r.auprc:.3f} ({r.auprc_lo:.2f}-{r.auprc_hi:.2f}) | {delta} | {r.auroc_V:.3f} | "
            f"{r.auroc_S_adj:.3f} | {r[f'sens_all@{fp_target}']:.0%} / {r[f'sens_V@{fp_target}']:.0%}"
            f" / {r[f'sens_S_adj@{fp_target}']:.0%} / {r[f'sens_F@{fp_target}']:.0%} |")
    dom = "; ".join(f"{d}: {s or 'none'}" for d, s in
                    res.drop_duplicates("dataset")[["dataset", "s_dominated"]].itertuples(index=False))
    lines += ["", f"\\* S computed without records where S outnumbers N ({dom}). "
              "Δ: paired patient bootstrap (same patients resampled for both configurations)."]
    return "\n".join(lines)
