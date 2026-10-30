#!/usr/bin/env bash
# Re-run the whole study from the raw CSV. Takes ~10 minutes on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
CSV=${1:-data/trace.csv}
python -m pytest -q
python scripts/audit.py "$CSV"
python scripts/compare_models.py "$CSV" --repeats 20
python scripts/reference_proxy.py
python scripts/select_final.py "$CSV" .cache/dnn-backdoor-detection-DATE24
python scripts/challengers.py "$CSV" .cache/dnn-backdoor-detection-DATE24
python scripts/train.py "$CSV" --config configs/final.json --out models/detector.joblib
python scripts/seed_stability.py "$CSV"
python scripts/make_figures.py
python scripts/build_report.py
