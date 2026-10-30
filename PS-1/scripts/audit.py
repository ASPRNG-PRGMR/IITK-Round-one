"""Reproducible data audit of the clean HPC trace CSV.

Usage:  python scripts/audit.py data/trace.csv
Writes: reports/tables/audit_*.csv|json, reports/figures/audit_*.png
The raw CSV is never modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hpcdetect.schema import FEATURES  # noqa: E402
from hpcdetect.scorers import select_gmm_by_bic  # noqa: E402

FIG, TAB = ROOT / "reports/figures", ROOT / "reports/tables"
UNIT = {"cache-references": 1e6, "cycles": 1e9, "LLC-loads": 1e6}
ULAB = {"cache-references": "millions", "cycles": "billions", "LLC-loads": "millions"}


def main(path: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(path)
    rep: dict = {"file": str(path), "shape": list(raw.shape), "columns": list(raw.columns),
                 "dtypes": raw.dtypes.astype(str).to_dict(),
                 "missing": raw.isna().sum().to_dict(),
                 "duplicate_rows": int(raw.duplicated().sum())}
    df = raw[list(FEATURES)].astype(float)
    rep["nonfinite"] = int((~np.isfinite(df.to_numpy())).sum())
    rep["near_constant"] = {c: float(df[c].std() / abs(df[c].mean())) for c in FEATURES}  # CV
    desc = df.describe(percentiles=[.001, .01, .05, .25, .5, .75, .95, .99, .999]).T
    desc["skew"] = df.skew()
    desc["excess_kurtosis"] = df.kurt()
    desc.to_csv(TAB / "audit_summary.csv")
    df.corr("pearson").to_csv(TAB / "audit_corr_pearson.csv")
    df.corr("spearman").to_csv(TAB / "audit_corr_spearman.csv")

    # normality tests per feature
    rep["normality"] = {c: {"shapiro_p": float(stats.shapiro(df[c]).pvalue),
                            "dagostino_p": float(stats.normaltest(df[c]).pvalue)} for c in FEATURES}

    # extremes: top/bottom 5 per feature with robust z
    ext = []
    for c in FEATURES:
        med, mad = df[c].median(), stats.median_abs_deviation(df[c], scale="normal")
        for side, idx in (("low", df[c].nsmallest(5).index), ("high", df[c].nlargest(5).index)):
            for i in idx:
                ext.append({"feature": c, "side": side, "row": int(i), "value": df.at[i, c],
                            "robust_z": (df.at[i, c] - med) / mad,
                            **{f"row_{f}": df.at[i, f] for f in FEATURES}})
    pd.DataFrame(ext).to_csv(TAB / "audit_extremes.csv", index=False)

    # highest-cycles row in context: its other counters' percentiles, Mahalanobis rank
    Z = StandardScaler().fit_transform(df)
    i_max = int(df["cycles"].idxmax())
    d2 = ((Z - Z.mean(0)) @ np.linalg.inv(np.cov(Z.T)) * (Z - Z.mean(0))).sum(1)
    gap = df["cycles"].sort_values().diff().iloc[-1] / df["cycles"].std()
    rep["max_cycles_row"] = {
        "row": i_max, "values": df.loc[i_max].to_dict(),
        "percentile_of_other_counters": {c: float(stats.percentileofscore(df[c], df.at[i_max, c]))
                                         for c in FEATURES},
        "mahalanobis_sq": float(d2[i_max]), "mahalanobis_rank": int((d2 >= d2[i_max]).sum()),
        "chi2_3_tail_p": float(stats.chi2.sf(d2[i_max], 3)),
        "gap_to_next_highest_in_sd": float(gap),
        "robust_z_cycles": float((df.at[i_max, "cycles"] - df["cycles"].median())
                                 / stats.median_abs_deviation(df["cycles"], scale="normal")),
    }
    rep["n_rows_mahal_chi2_p_below_0.001"] = int((stats.chi2.sf(d2, 3) < 1e-3).sum())

    # row-order drift (a proxy for collection-session effects, if rows are in time order)
    blocks = np.array_split(np.arange(len(df)), 8)
    drift = pd.DataFrame([(df.iloc[b].mean() - df.mean()) / df.std() for b in blocks])
    drift.to_csv(TAB / "audit_row_order_drift.csv")
    rep["row_order_spearman_with_index"] = {c: float(stats.spearmanr(np.arange(len(df)), df[c])[0])
                                            for c in FEATURES}

    # PCA
    pca = PCA().fit(Z)
    rep["pca"] = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                  "loadings": pd.DataFrame(pca.components_, columns=FEATURES,
                                           index=["PC1", "PC2", "PC3"]).round(4).to_dict("index")}

    # univariate GMM BIC per feature
    rep["univariate_gmm_bic"] = {}
    for c in FEATURES:
        g, bics = select_gmm_by_bic(StandardScaler().fit_transform(df[[c]]), 6, seed=0)
        rep["univariate_gmm_bic"][c] = {"best_k": g.n_components, "bic": bics}

    # multivariate GMM: BIC, occupancy, stability (ARI across bootstrap refits)
    mv = {}
    rng = np.random.default_rng(0)
    for cov in ("full", "diag"):
        for k in range(1, 7):
            g = GaussianMixture(k, covariance_type=cov, n_init=10, random_state=0).fit(Z)
            lab = g.predict(Z)
            aris = []
            for b in range(20):
                idx = rng.choice(len(Z), len(Z), replace=True)
                gb = GaussianMixture(k, covariance_type=cov, n_init=3, random_state=b).fit(Z[idx])
                aris.append(adjusted_rand_score(lab, gb.predict(Z)) if k > 1 else 1.0)
            mv[f"{cov}_k{k}"] = {"bic": float(g.bic(Z)), "occupancy": np.bincount(lab, minlength=k).tolist(),
                                 "min_component_size": int(np.bincount(lab, minlength=k).min()),
                                 "bootstrap_ari_mean": float(np.mean(aris)),
                                 "bootstrap_ari_min": float(np.min(aris))}
    rep["multivariate_gmm"] = mv
    g2 = GaussianMixture(2, n_init=10, random_state=0).fit(Z)
    lab2 = g2.predict(Z)
    rep["k2_component_means"] = df.groupby(lab2).mean().to_dict("index")
    rep["k2_component_sizes"] = np.bincount(lab2).tolist()
    rep["k2_label_vs_row_index_spearman"] = float(stats.spearmanr(np.arange(len(df)), lab2)[0])

    # ratios
    rep["ratios"] = {
        "llc/cacheref": {"mean": float((df["LLC-loads"] / df["cache-references"]).mean()),
                         "sd": float((df["LLC-loads"] / df["cache-references"]).std())},
        "llc/cycles": {"mean": float((df["LLC-loads"] / df["cycles"]).mean()),
                       "sd": float((df["LLC-loads"] / df["cycles"]).std())},
        "cacheref/cycles": {"mean": float((df["cache-references"] / df["cycles"]).mean()),
                            "sd": float((df["cache-references"] / df["cycles"]).std())},
    }
    (TAB / "audit_report.json").write_text(json.dumps(rep, indent=2, default=float))

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, c in zip(ax, FEATURES):
        a.hist(df[c] / UNIT[c], bins=50, color="#4C72B0", alpha=.85)
        a.set_xlabel(f"{c} ({ULAB[c]})"); a.set_ylabel("traces")
    fig.suptitle("Clean HPC trace distributions (n=%d)" % len(df)); fig.tight_layout()
    fig.savefig(FIG / "audit_histograms.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (x, y) in zip(ax, [(0, 2), (0, 1), (1, 2)]):
        cx, cy = FEATURES[x], FEATURES[y]
        a.scatter(df[cx] / UNIT[cx], df[cy] / UNIT[cy], c=lab2, s=7, cmap="coolwarm", alpha=.7)
        a.scatter(df.at[i_max, cx] / UNIT[cx], df.at[i_max, cy] / UNIT[cy], marker="x", c="k", s=60)
        a.set_xlabel(f"{cx} ({ULAB[cx]})"); a.set_ylabel(f"{cy} ({ULAB[cy]})")
    fig.suptitle("Pairwise scatter, colored by exploratory 2-component GMM; x = max-cycles row")
    fig.tight_layout(); fig.savefig(FIG / "audit_pairs.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    for a, c in zip(ax, FEATURES):
        a.plot(df[c].to_numpy() / UNIT[c], lw=.6); a.set_ylabel(f"{c}\n({ULAB[c]})", fontsize=8)
    ax[-1].set_xlabel("row index (file order)")
    fig.suptitle("Counters in file order (checks for drift / session blocks)")
    fig.tight_layout(); fig.savefig(FIG / "audit_row_order.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for cov, mk in (("full", "o-"), ("diag", "s--")):
        ax.plot(range(1, 7), [mv[f"{cov}_k{k}"]["bic"] for k in range(1, 7)], mk, label=cov)
    ax.set_xlabel("components K"); ax.set_ylabel("BIC (standardized features)"); ax.legend()
    ax.set_title("Multivariate GMM model selection"); fig.tight_layout()
    fig.savefig(FIG / "audit_gmm_bic.png", dpi=150); plt.close(fig)

    print(json.dumps({k: rep[k] for k in ("shape", "missing", "duplicate_rows", "max_cycles_row",
                                          "row_order_spearman_with_index", "k2_component_sizes",
                                          "k2_label_vs_row_index_spearman")}, indent=2, default=float))
    print("\nMultivariate GMM:")
    for k, v in mv.items():
        print(f"  {k:8s} BIC={v['bic']:9.1f} sizes={v['occupancy']} ARI={v['bootstrap_ari_mean']:.2f}")
    print("\nUnivariate best K:", {c: v["best_k"] for c, v in rep["univariate_gmm_bic"].items()})
    print("PCA EVR:", np.round(pca.explained_variance_ratio_, 4), "\nPC3:", rep["pca"]["loadings"]["PC3"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/trace.csv"))
