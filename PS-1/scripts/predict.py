"""Score new traces with a saved detector.

Usage:  python scripts/predict.py input.csv [--model models/detector.joblib] [--out predictions.csv]

Output columns: all input columns, plus
  anomaly_score  continuous score (higher = less consistent with clean data; use for AUROC)
  p_value        conformal p-value vs clean out-of-fold scores (small = unusual)
  prediction     1 = flagged as possible backdoor inference, 0 = clean
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hpcdetect import HPCBackdoorDetector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default=str(ROOT / "models/detector.joblib"))
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()
    det = HPCBackdoorDetector.load(a.model)
    df = pd.read_csv(a.csv)
    out = df.copy()
    out["anomaly_score"] = det.score_samples(df)
    out["p_value"] = det.p_values(df)
    out["prediction"] = det.predict(df)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} rows, {int(out.prediction.sum())} flagged "
          f"(threshold={det.threshold_:.4g}, target clean FPR={det.config.alpha})")


if __name__ == "__main__":
    main()
