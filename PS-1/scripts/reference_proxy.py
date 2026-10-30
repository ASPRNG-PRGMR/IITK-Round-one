"""Labelled proxy experiment on the public DATE'24 HPC logs.

The DATE'24 repository (momalab/dnn-backdoor-detection-DATE24, MIT licence)
ships perf logs for 10,000 benign and 10,000 backdoor CIFAR-10/ResNet18
inferences on an Intel i7-9700, with five counters (branches, branch-misses,
cache-references, cache-misses, instructions). Only cache-references overlaps
with the challenge features, and the hardware, model and trigger differ.

Purpose: check, on REAL attack labels, how one-class scorers + conformal
calibration behave when trained on ~800 clean traces. Results transfer as
evidence about methods, not as a prediction of challenge scores.

Usage: python scripts/reference_proxy.py [path/to/dnn-backdoor-detection-DATE24]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hpcdetect.calibration import conformal_threshold  # noqa: E402
from hpcdetect.scorers import build_scorer  # noqa: E402

EVENTS = ["branches", "branch-misses", "cache-references", "cache-misses", "instructions"]
URL = "https://github.com/momalab/dnn-backdoor-detection-DATE24"


def parse(path: Path) -> pd.DataFrame:
    d = {e: [] for e in EVENTS}
    for line in path.read_text().splitlines():
        s = line.split()
        if len(s) >= 2 and s[1] in d and re.fullmatch(r"[\d,]+", s[0]):
            d[s[1]].append(float(s[0].replace(",", "")))
    return pd.DataFrame(d)


CANDIDATES = [
    ("zscore", {"combine": "max"}), ("zscore", {"combine": "sumsq"}),
    ("ugmm", {"combine": "max"}), ("ugmm", {"combine": "mean"}),
    ("mgmm", {"k": 1}), ("mgmm", {"k": 2}), ("mgmm", {"k": 3}),
    ("mahal", {"robust": True}), ("pca", {"n_keep": 2}),
    ("iforest", {}), ("ocsvm", {"nu": 0.05}),
]


def run(ben: pd.DataFrame, bd: pd.DataFrame, cols, n_fit=640, n_cal=160, reps=10, alpha=0.05,
        setting="pooled", block=None):
    rows = []
    for name, p in CANDIDATES:
        if name == "pca" and len(cols) < 3:
            continue
        aucs, tprs, fprs, f1s = [], [], [], []
        for r in range(reps):
            rng = np.random.default_rng(r)
            if setting == "pooled":
                b_idx, a_idx = np.arange(len(ben)), np.arange(len(bd))
            else:  # within one collection block (one class, contiguous in time)
                b_idx = np.arange(block * 1000, (block + 1) * 1000)
                a_idx = np.arange(block * 1000, (block + 1) * 1000)
            perm = rng.permutation(b_idx)
            fit, cal, test = perm[:n_fit], perm[n_fit:n_fit + n_cal], perm[n_fit + n_cal:]
            att = rng.choice(a_idx, len(test), replace=False)
            X = lambda df, i: df.iloc[i][cols].to_numpy()
            sc = build_scorer(name, p).fit(X(ben, fit))
            t = conformal_threshold(sc.score(X(ben, cal)), alpha)
            s0, s1 = sc.score(X(ben, test)), sc.score(X(bd, att))
            y = np.r_[np.zeros(len(s0)), np.ones(len(s1))]
            yhat = (np.r_[s0, s1] > t).astype(int)
            aucs.append(roc_auc_score(y, np.r_[s0, s1]))
            tprs.append(yhat[y == 1].mean()); fprs.append(yhat[y == 0].mean())
            f1s.append(f1_score(y, yhat, zero_division=0))
        rows.append({"setting": setting if block is None else f"block{block}", "features": "+".join(cols),
                     "scorer": name, "params": str(p), "auroc": np.mean(aucs), "auroc_sd": np.std(aucs),
                     "tpr": np.mean(tprs), "fpr": np.mean(fprs), "f1": np.mean(f1s)})
    return rows


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache/dnn-backdoor-detection-DATE24"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", URL, str(repo)], check=True)
    ben = parse(repo / "hpc_data/perf_benign.log")
    bd = parse(repo / "hpc_data/perf_backdoor.log")
    assert len(ben) == len(bd) == 10000, (len(ben), len(bd))
    rows = []
    rows += run(ben, bd, EVENTS)                         # all 5 counters, pooled across sessions
    rows += run(ben, bd, ["cache-references"])           # the one overlapping counter
    for b in (0, 5, 9):                                  # within a single collection block
        rows += run(ben, bd, EVENTS, setting="block", block=b)
    out = pd.DataFrame(rows)
    (ROOT / "reports/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "reports/tables/reference_proxy.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
