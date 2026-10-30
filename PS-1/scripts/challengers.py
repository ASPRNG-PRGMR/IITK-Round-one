"""Second-round check: does any untested model family beat the chosen GMM K=3?

Same protocols as before, same seeds:
  * our CSV: 20 x (60 % fit / 20 % calibrate / 20 % test), clean FPR + synthetic shifts
  * DATE'24 logs: real backdoor labels, 640 fit / 160 calibrate, 10 repeats, pooled sessions

Usage: python scripts/challengers.py data/trace.csv [path/to/DATE24 repo]
Writes reports/tables/challengers.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hpcdetect.evaluation import DELTAS, PERTURBATIONS, evaluate_candidate  # noqa: E402
from hpcdetect.schema import load_traces  # noqa: E402
import reference_proxy as rp  # noqa: E402

warnings.filterwarnings("ignore")

CANDIDATES = [
    ("mgmm", {"k": 3}),
    ("bgmm", {"k": 3, "n_bags": 10}),
    ("kde", {}),
    ("knn", {"k": 5}),
    ("knn", {"k": 15}),
]


def main():
    df = load_traces(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/trace.csv")
    rows = []
    for name, p in CANDIDATES:
        r = evaluate_candidate(df, "raw", name, p, alpha=0.05, n_repeats=20)
        fam = {f: np.mean([r[f"auc_{f}_{d}"] for d in DELTAS]) for f in PERTURBATIONS}
        rows.append({"scorer": name, "params": str(p), "clean_fpr": r["clean_fpr_mean"],
                     "synthetic_auc_mean": np.mean(list(fam.values())),
                     "synthetic_auc_worst": min(fam.values()), **{f"auc_{f}": v for f, v in fam.items()}})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)

    repo = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / ".cache/dnn-backdoor-detection-DATE24"
    if repo.exists():
        ben = rp.parse(repo / "hpc_data/perf_benign.log")
        bd = rp.parse(repo / "hpc_data/perf_backdoor.log")
        rp.CANDIDATES[:] = CANDIDATES
        prox = pd.DataFrame(rp.run(ben, bd, rp.EVENTS))[["scorer", "params", "auroc", "auroc_sd", "f1", "fpr"]]
        prox.columns = ["scorer", "params", "proxy_auroc", "proxy_auroc_sd", "proxy_f1_a05", "proxy_fpr_a05"]
        out = out.merge(prox, on=["scorer", "params"])
    out.to_csv(ROOT / "reports/tables/challengers.csv", index=False)
    with pd.option_context("display.width", 220):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
