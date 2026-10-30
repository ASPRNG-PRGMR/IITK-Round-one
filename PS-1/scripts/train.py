"""Train the final detector on all clean traces and save it.

Usage:  python scripts/train.py data/trace.csv [--config configs/final.json] [--out models/detector.joblib]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hpcdetect import DetectorConfig, HPCBackdoorDetector, load_traces  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--config", default=str(ROOT / "configs/final.json"))
    ap.add_argument("--out", default=str(ROOT / "models/detector.joblib"))
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg["ratios"] = tuple(cfg.get("ratios", ()))
    det = HPCBackdoorDetector(DetectorConfig(**cfg)).fit(load_traces(a.csv))
    det.save(a.out)
    print(json.dumps(det.card(), indent=2))


if __name__ == "__main__":
    main()
