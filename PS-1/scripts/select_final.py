"""Paired K comparison on identical splits, and alpha (operating point) analysis.

Usage: python scripts/select_final.py data/trace.csv [path/to/DATE24 repo]
Writes reports/tables/paired_k.csv, alpha_sweep_proxy.csv, alpha_sweep_synthetic.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from hpcdetect.calibration import conformal_threshold
from hpcdetect.evaluation import perturb, PERTURBATIONS, DELTAS
from hpcdetect.features import make_features
from hpcdetect.schema import load_traces
from hpcdetect.scorers import build_scorer
warnings.filterwarnings("ignore")
TAB = ROOT / "reports/tables"; ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)

def paired_k(df, reps=40):
    rows = []
    for r in range(reps):
        fit, rest = train_test_split(df, train_size=0.6, random_state=r)
        cal, test = train_test_split(rest, train_size=0.5, random_state=r)
        X = lambda d: make_features(d, "raw").to_numpy()
        for k in (1, 2, 3):
            sc = build_scorer("mgmm", {"k": k, "seed": r}).fit(X(fit))
            t = conformal_threshold(sc.score(X(cal)), 0.05); s0 = sc.score(X(test))
            row = {"rep": r, "k": k, "fpr": (s0 > t).mean(), "heldout_nll": s0.mean(),
                   "max_cycles_rows_flagged": int((sc.score(X(df.iloc[[668, 724, 294]])) > t).sum())}
            for f in PERTURBATIONS:
                a = [roc_auc_score(np.r_[np.zeros(len(s0)), np.ones(len(s0))],
                                   np.r_[s0, sc.score(X(perturb(test, f, d, fit)))]) for d in DELTAS]
                row["auc_" + f] = np.mean(a)
            rows.append(row)
    return pd.DataFrame(rows)

def alpha_sweep_synthetic(df, k, reps=20):
    rows = []
    for r in range(reps):
        fit, rest = train_test_split(df, train_size=0.6, random_state=r)
        cal, test = train_test_split(rest, train_size=0.5, random_state=r)
        X = lambda d: make_features(d, "raw").to_numpy()
        sc = build_scorer("mgmm", {"k": k, "seed": r}).fit(X(fit)); sc_cal = sc.score(X(cal)); s0 = sc.score(X(test))
        for f in PERTURBATIONS:
            for d in DELTAS:
                s1 = sc.score(X(perturb(test, f, d, fit))); y = np.r_[np.zeros(len(s0)), np.ones(len(s1))]
                for a in ALPHAS:
                    yh = (np.r_[s0, s1] > conformal_threshold(sc_cal, a)).astype(int)
                    rows.append({"k": k, "family": f, "delta": d, "alpha": a, "f1": f1_score(y, yh),
                                 "acc": (yh == y).mean(), "tpr": yh[y == 1].mean(), "fpr": yh[y == 0].mean()})
    return pd.DataFrame(rows)

def alpha_sweep_proxy(repo, reps=10):
    from reference_proxy import parse, EVENTS
    ben, bd = parse(repo / "hpc_data/perf_benign.log"), parse(repo / "hpc_data/perf_backdoor.log")
    rows = []
    for k in (2, 3):
        for r in range(reps):
            rng = np.random.default_rng(r); p = rng.permutation(len(ben))
            fit, cal, test = p[:480], p[480:640], p[640:1640]; att = rng.choice(len(bd), 1000, replace=False)
            sc = build_scorer("mgmm", {"k": k, "seed": r}).fit(ben.iloc[fit][EVENTS].to_numpy())
            c = sc.score(ben.iloc[cal][EVENTS].to_numpy())
            s = np.r_[sc.score(ben.iloc[test][EVENTS].to_numpy()), sc.score(bd.iloc[att][EVENTS].to_numpy())]
            y = np.r_[np.zeros(1000), np.ones(1000)]
            for a in ALPHAS:
                yh = (s > conformal_threshold(c, a)).astype(int)
                rows.append({"k": k, "alpha": a, "auroc": roc_auc_score(y, s), "f1": f1_score(y, yh),
                             "acc": (yh == y).mean(), "tpr": yh[y == 1].mean(), "fpr": yh[y == 0].mean()})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = load_traces(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/trace.csv")
    pk = paired_k(df); pk.to_csv(TAB / "paired_k.csv", index=False)
    print(pk.groupby("k").mean().drop(columns="rep").round(4).T)
    w = pk.pivot(index="rep", columns="k")
    for col in ["heldout_nll"] + ["auc_" + f for f in PERTURBATIONS]:
        d = w[col][3] - w[col][2]
        print(f"K3-K2 {col:18s} mean={d.mean():+.4f}  sd={d.std():.4f}  K3 better in {(d < 0).mean() if col=='heldout_nll' else (d > 0).mean():.0%} of splits")
    s = pd.concat([alpha_sweep_synthetic(df, 2), alpha_sweep_synthetic(df, 3)]); s.to_csv(TAB / "alpha_sweep_synthetic.csv", index=False)
    print("\nSynthetic (all families, deltas 0.5-3 SD) by alpha:\n", s.groupby(["k", "alpha"])[["f1", "acc", "tpr", "fpr"]].mean().round(3))
    repo = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / ".cache/dnn-backdoor-detection-DATE24"
    if repo.exists():
        p = alpha_sweep_proxy(repo); p.to_csv(TAB / "alpha_sweep_proxy.csv", index=False)
        print("\nDATE'24 labelled proxy (5 counters, n_fit=480, n_cal=160) by alpha:\n",
              p.groupby(["k", "alpha"]).mean().round(3))
