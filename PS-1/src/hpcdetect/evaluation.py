"""Clean-only validation and synthetic sensitivity analysis.

What can honestly be measured without attack labels:

* Clean FPR at the calibrated threshold, on clean traces used neither for
  fitting nor for calibration (does the alpha guarantee hold in practice?).
* Stability of scores / decisions across resamples.
* Sensitivity to *hypothetical* deviations injected into held-out clean traces.
  These perturbations are assumptions about what an attack might do to the
  counters. TPR/AUROC computed on them describe the detector's geometry, NOT its
  performance on real backdoor inferences. They must never be reported as such.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .calibration import conformal_threshold
from .features import make_features
from .schema import FEATURES
from .scorers import build_scorer

DELTAS = (0.5, 1.0, 2.0, 3.0)


def perturb(df: pd.DataFrame, kind: str, delta: float, ref: pd.DataFrame) -> pd.DataFrame:
    """Shift clean rows by `delta` clean standard deviations (computed on `ref`)."""
    out = df.copy()
    sd = ref[list(FEATURES)].std()
    if kind in ("cycles_up", "cacheref_up", "llc_up"):
        col = {"cycles_up": "cycles", "cacheref_up": "cache-references", "llc_up": "LLC-loads"}[kind]
        out[col] = out[col] + delta * sd[col]
    elif kind == "all_up":
        for col in FEATURES:
            out[col] = out[col] + delta * sd[col]
    elif kind == "relational":
        # move along the least-variance standardised direction of the clean data
        Z = (ref[list(FEATURES)] - ref[list(FEATURES)].mean()) / sd
        w = np.linalg.eigh(np.cov(Z.to_numpy().T))[1][:, 0]  # smallest-eigenvalue eigenvector
        for j, col in enumerate(FEATURES):
            out[col] = out[col] + delta * w[j] * sd[col]
    else:
        raise ValueError(kind)
    return out


PERTURBATIONS = ("cycles_up", "cacheref_up", "llc_up", "all_up", "relational")


def evaluate_candidate(df: pd.DataFrame, representation: str, scorer: str, params: dict,
                       alpha: float = 0.05, n_repeats: int = 20, fit_frac: float = 0.6,
                       cal_frac: float = 0.2, ratios=("llc_per_cacheref",), seed: int = 0) -> dict:
    """Repeated fit / calibrate / test split evaluation for one candidate."""
    fprs, sens = [], {(k, d): [] for k in PERTURBATIONS for d in DELTAS}
    aucs = {(k, d): [] for k in PERTURBATIONS for d in DELTAS}
    diags = []
    for r in range(n_repeats):
        rs = seed + r
        fit_df, rest = train_test_split(df, train_size=fit_frac, random_state=rs)
        cal_df, test_df = train_test_split(rest, train_size=cal_frac / (1 - fit_frac), random_state=rs)
        f = lambda d: make_features(d, representation, ratios).to_numpy()
        p = dict(params)
        sc = build_scorer(scorer, p)
        if "seed" in getattr(sc, "__dataclass_fields__", {}) and "seed" not in p:
            sc = build_scorer(scorer, {**p, "seed": rs})
        sc.fit(f(fit_df))
        diags.append(sc.diagnostics())
        t = conformal_threshold(sc.score(f(cal_df)), alpha)
        s_clean = sc.score(f(test_df))
        fprs.append(float((s_clean > t).mean()))
        for k in PERTURBATIONS:
            for d in DELTAS:
                s_att = sc.score(f(perturb(test_df, k, d, fit_df)))
                sens[(k, d)].append(float((s_att > t).mean()))
                y = np.r_[np.zeros(len(s_clean)), np.ones(len(s_att))]
                aucs[(k, d)].append(roc_auc_score(y, np.r_[s_clean, s_att]))
    row = {"representation": representation, "scorer": scorer, "params": str(params), "alpha": alpha,
           "clean_fpr_mean": float(np.mean(fprs)), "clean_fpr_sd": float(np.std(fprs)),
           "clean_fpr_max": float(np.max(fprs))}
    for (k, d), v in sens.items():
        row[f"tpr_{k}_{d}"] = float(np.mean(v))
        row[f"auc_{k}_{d}"] = float(np.mean(aucs[(k, d)]))
    row["_diagnostics_first"] = str(diags[0])
    return row
