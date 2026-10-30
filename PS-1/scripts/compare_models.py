"""Compare candidate detectors on the clean CSV under one protocol.

For every (representation, scorer, params) candidate and each of N repeats:
  60% fit / 20% calibrate (conformal threshold, alpha) / 20% clean test,
then measure clean test FPR and sensitivity to synthetic shifts of the test rows
(see hpcdetect.evaluation). Also: held-out clean log-likelihood for GMM K choice.

Usage: python scripts/compare_models.py data/trace.csv [--repeats 20] [--alpha 0.05]
Writes reports/tables/model_comparison.csv, gmm_k_selection.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hpcdetect.evaluation import DELTAS, PERTURBATIONS, evaluate_candidate  # noqa: E402
from hpcdetect.features import make_features  # noqa: E402
from hpcdetect.schema import load_traces  # noqa: E402

warnings.filterwarnings("ignore")

CANDIDATES = [
    ("zscore", {"combine": "max"}), ("zscore", {"combine": "sumsq"}),
    ("ugmm", {"combine": "max"}), ("ugmm", {"combine": "mean"}), ("ugmm", {"combine": "rawsum"}),
    ("mgmm", {"k": 1}), ("mgmm", {"k": 2}), ("mgmm", {"k": 3}), ("mgmm", {"k": 4}),
    ("mgmm", {"k": 2, "covariance_type": "diag"}),
    ("mahal", {"robust": False}), ("mahal", {"robust": True}),
    ("pca", {"n_keep": 1}), ("pca", {"n_keep": 2}),
    ("iforest", {}), ("ocsvm", {"nu": 0.05}),
    ("ensemble", {"members": [("mgmm", {"k": 2}), ("pca", {"n_keep": 2})], "combine": "max"}),
    ("ensemble", {"members": [("mgmm", {"k": 2}), ("pca", {"n_keep": 2})], "combine": "mean"}),
]
REPRESENTATIONS = ("raw", "log", "ratio")


def gmm_k_selection(df: pd.DataFrame, reps=("raw", "log", "ratio")) -> pd.DataFrame:
    """Held-out mean log-likelihood per K (5-fold x 4 repeats) + component-size checks."""
    rows = []
    for rep in reps:
        X = make_features(df, rep).to_numpy()
        for cov in ("full", "diag"):
            for k in range(1, 7):
                ll, minsz = [], []
                for i, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=4, random_state=0).split(X)):
                    sc = StandardScaler().fit(X[tr])
                    g = GaussianMixture(k, covariance_type=cov, n_init=5, random_state=i).fit(sc.transform(X[tr]))
                    ll.append(g.score(sc.transform(X[te])))
                    minsz.append(np.bincount(g.predict(sc.transform(X[tr])), minlength=k).min())
                rows.append({"representation": rep, "cov": cov, "k": k, "heldout_ll_mean": np.mean(ll),
                             "heldout_ll_sd": np.std(ll), "min_component_size_median": np.median(minsz),
                             "min_component_size_min": int(np.min(minsz))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(ROOT / "data/trace.csv"))
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    df = load_traces(a.csv)
    out = ROOT / "reports/tables"
    out.mkdir(parents=True, exist_ok=True)

    ks = gmm_k_selection(df)
    ks.to_csv(out / "gmm_k_selection.csv", index=False)
    print(ks.round(3).to_string(index=False))

    rows, t0 = [], time.time()
    for rep in REPRESENTATIONS:
        for name, p in CANDIDATES:
            if name == "pca" and rep == "ratio":
                pass  # 4-D: keeping 1 or 2 PCs still leaves a residual; allowed
            rows.append(evaluate_candidate(df, rep, name, p, alpha=a.alpha, n_repeats=a.repeats))
            r = rows[-1]
            print(f"[{time.time()-t0:6.0f}s] {rep:5s} {name:8s} {str(p)[:60]:60s} "
                  f"FPR={r['clean_fpr_mean']:.3f}±{r['clean_fpr_sd']:.3f} "
                  f"AUC(all_up,1)={r['auc_all_up_1.0']:.2f} AUC(cyc,2)={r['auc_cycles_up_2.0']:.2f} "
                  f"AUC(rel,1)={r['auc_relational_1.0']:.2f}", flush=True)
    res = pd.DataFrame(rows)
    # summary: mean AUROC and TPR over all synthetic perturbations and deltas
    auc_cols = [f"auc_{k}_{d}" for k in PERTURBATIONS for d in DELTAS]
    tpr_cols = [f"tpr_{k}_{d}" for k in PERTURBATIONS for d in DELTAS]
    res["synthetic_auc_mean"] = res[auc_cols].mean(axis=1)
    res["synthetic_auc_worst_family"] = np.min(
        [res[[f"auc_{k}_{d}" for d in DELTAS]].mean(axis=1) for k in PERTURBATIONS], axis=0)
    res["synthetic_tpr_mean"] = res[tpr_cols].mean(axis=1)
    res.to_csv(out / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
