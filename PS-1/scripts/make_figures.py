"""Figures for reports/report.tex (vector PDF, LaTeX fonts).

Usage: python scripts/make_figures.py
Needs: models/detector.joblib, data/trace.csv and the tables written by the experiment scripts.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hpcdetect import HPCBackdoorDetector  # noqa: E402

warnings.filterwarnings("ignore")
FIG, TAB = ROOT / "reports/figures", ROOT / "reports/tables"

plt.rcParams.update({
    # Computer Modern (ships with matplotlib) so the figures match the LaTeX body text
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "pdf.fonttype": 42,
    "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
GREY, DARK = "0.62", "0.1"


def data_figure(det: HPCBackdoorDetector, df: pd.DataFrame) -> None:
    flagged = det.predict(df).astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.55))
    slope = np.polyfit(df["cache-references"], df["LLC-loads"], 1)
    panels = [("LLC-loads", 1e6, "LLC-loads (millions)"), ("cycles", 1e9, "cycles (billions)")]
    for ax, (ycol, ys, ylab) in zip(axes, panels):
        x = df["cache-references"] / 1e6
        y = df[ycol] / ys
        xs = np.linspace(x.min() - 3, x.max() + 3, 300)
        pad = (y.max() - y.min()) * 0.12
        yy = np.linspace(y.min() - pad, y.max() + pad, 300)
        XX, YY = np.meshgrid(xs, yy)
        grid = pd.DataFrame({c: np.full(XX.size, df[c].median()) for c in df.columns})
        grid["cache-references"] = XX.ravel() * 1e6
        grid[ycol] = YY.ravel() * ys
        if ycol == "cycles":
            grid["LLC-loads"] = np.polyval(slope, grid["cache-references"])
        S = det.score_samples(grid).reshape(XX.shape)
        ax.contour(XX, YY, S, levels=[det.threshold_], colors=DARK, linewidths=0.8)
        ax.scatter(x[~flagged], y[~flagged], s=4, c=GREY, linewidths=0, label="accepted")
        ax.scatter(x[flagged], y[flagged], s=9, facecolors="none", edgecolors=DARK, linewidths=0.6,
                   label="flagged")
        ax.set_xlabel("cache-references (millions)")
        ax.set_ylabel(ylab)
    axes[0].text(177, 6.62, "228 traces", fontsize=7.5, color="0.35", ha="right")
    axes[0].text(181.5, 6.95, "572 traces", fontsize=7.5, color="0.35", ha="right")
    i = int(df["cycles"].idxmax())
    axes[1].annotate(f"row {i}", (df.at[i, "cache-references"] / 1e6, df.at[i, "cycles"] / 1e9),
                     xytext=(6, -2), textcoords="offset points", fontsize=7, color="0.35", va="center")
    axes[0].legend(loc="lower right", handletextpad=0.2, markerscale=1.4)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "fig_data.pdf")
    plt.close(fig)


def auroc_figure() -> None:
    p = pd.read_csv(TAB / "reference_proxy.csv")
    p = p[(p.setting == "pooled") & p.features.str.contains(r"\+")]
    c = pd.read_csv(TAB / "challengers.csv")
    names = {("zscore", "{'combine': 'max'}"): "z-score (max)", ("ugmm", "{'combine': 'max'}"): "per-counter GMM",
             ("mahal", "{'robust': True}"): "Mahalanobis (MCD)", ("pca", "{'n_keep': 2}"): "PCA residual",
             ("iforest", "{}"): "isolation forest", ("ocsvm", "{'nu': 0.05}"): "one-class SVM",
             ("mgmm", "{'k': 1}"): "GMM, $K=1$", ("mgmm", "{'k': 2}"): "GMM, $K=2$",
             ("mgmm", "{'k': 3}"): "GMM, $K=3$"}
    rows = [(names[(r.scorer, r.params)], r.auroc) for r in p.itertuples() if (r.scorer, r.params) in names]
    extra = {"kde": "kernel density", "bgmm": "bagged GMM, $K=3$"}
    for r in c.itertuples():
        if r.scorer in extra:
            rows.append((extra[r.scorer], r.proxy_auroc))
        elif r.scorer == "knn" and "'k': 5" in r.params:
            rows.append(("5-NN distance", r.proxy_auroc))
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(3.15, 2.6))
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.17, top=0.95)
    yy = np.arange(len(rows))
    for y, (lab, v) in zip(yy, rows):
        final = lab == "GMM, $K=3$"
        ax.plot([0.4, v], [y, y], color="0.85", lw=0.6, zorder=1)
        ax.scatter(v, y, s=16 if final else 10, color=DARK if final else "0.45", zorder=2)
    ax.axvline(0.5, color="0.4", lw=0.6, ls=(0, (3, 2)))
    ax.text(0.507, -0.15, "chance", fontsize=7, color="0.35", va="bottom")
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.4, 0.9)
    ax.set_xlabel("AUROC on real backdoor traces")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIG / "fig_auroc.pdf", bbox_inches=None)
    plt.close(fig)


def alpha_figure() -> None:
    prox = pd.read_csv(TAB / "alpha_sweep_proxy.csv").groupby(["k", "alpha"]).mean(numeric_only=True).loc[3]
    syn = pd.read_csv(TAB / "alpha_sweep_synthetic.csv").groupby(["k", "alpha"]).mean(numeric_only=True).loc[3]
    fig, ax = plt.subplots(figsize=(3.15, 2.6))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.17, top=0.95)
    for d, ls, src in ((prox, "-", "real"), (syn, "--", "synthetic")):
        ax.plot(d.index, d.acc, ls, color=DARK, lw=1.0, marker="o", ms=2.2, label=f"accuracy ({src})")
        ax.plot(d.index, d.f1, ls, color="0.55", lw=1.0, marker="o", ms=2.2, label=f"F1 ({src})")
    ax.axvline(0.10, color="0.4", lw=0.6, ls=(0, (1, 2)))
    ax.set_xlabel(r"target false-positive rate $\alpha$")
    ax.set_ylim(0.2, 0.85)
    ax.legend(loc="lower right", handlelength=2.2)
    fig.savefig(FIG / "fig_alpha.pdf", bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    det = HPCBackdoorDetector.load(ROOT / "models/detector.joblib")
    df = pd.read_csv(ROOT / "data/trace.csv")
    data_figure(det, df)
    auroc_figure()
    alpha_figure()
    print("figures written to", FIG)
