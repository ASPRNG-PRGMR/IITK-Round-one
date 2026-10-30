"""Refit the final config with 8 seeds and check how much the detector moves.

Usage: python scripts/seed_stability.py data/trace.csv
Writes reports/tables/seed_stability.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hpcdetect import DetectorConfig, HPCBackdoorDetector  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/trace.csv")
    cfg = json.loads((ROOT / "configs/final.json").read_text())
    cfg["ratios"] = tuple(cfg["ratios"])
    # probe: the clean rows plus two sets of shifted rows, so decisions near the threshold get exercised
    probe = pd.concat([df,
                       df.sample(200, random_state=1).assign(cycles=lambda d: d.cycles + df.cycles.std()),
                       df.sample(200, random_state=2) + df.std()], ignore_index=True)
    S, F, T, occ = [], [], [], []
    for seed in range(8):
        det = HPCBackdoorDetector(DetectorConfig(**{**cfg, "seed": seed})).fit(df)
        S.append(det.score_samples(probe)); F.append(det.predict(probe)); T.append(det.threshold_)
        occ.append(sorted(det.scorer_.occupancy_.tolist(), reverse=True))
    S, F = np.array(S), np.array(F)
    out = {"n_seeds": 8, "threshold_mean": float(np.mean(T)), "threshold_sd": float(np.std(T)),
           "score_spearman_vs_seed0": float(np.mean([spearmanr(S[0], S[i])[0] for i in range(1, 8)])),
           "flag_agreement_vs_seed0": float(np.mean([(F[0] == F[i]).mean() for i in range(1, 8)])),
           "rows_with_any_flag_change": int((F.min(0) != F.max(0)).sum()), "n_probe_rows": len(probe),
           "component_sizes_per_seed": occ}
    (ROOT / "reports/tables/seed_stability.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
