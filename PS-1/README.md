# HPC-based detection of neural-network backdoor inferences

We only get clean HPC traces, so the detector learns what clean inferences look like and flags traces that
don't fit as possible backdoor inferences. The write-up is [`reports/report.pdf`](reports/report.pdf)
(LaTeX source: `reports/report.tex`).

## Final detector

| | |
|---|---|
| Features | raw `cache-references`, `cycles`, `LLC-loads` (standardised inside the model) |
| Model | full-covariance Gaussian mixture, K = 3 (n_init 10, reg_covar 1e-6, seed 0) |
| Anomaly score | `-log p(x)` under the clean mixture (higher = more anomalous) |
| Threshold | split-conformal quantile of 10-fold out-of-fold clean scores at target clean FPR α = 0.10 → **4.664** |
| Training data | all 800 clean traces (no rows removed) |
| Files | `configs/final.json`, `models/detector.joblib`, model card `models/detector.json` |

**Why this model** (details in the report and `PROGRESS.md`):
- The clean data has two groups along the cache-references/LLC-loads line. On the public DATE'24 traces, which have
  real backdoor labels and also several clean groups, z-score, Mahalanobis, PCA residual, one-class SVM and a
  one-component GMM all scored 0.43 to 0.52 AUROC. The joint GMM scored 0.87.
- K = 3 beat K = 2 on held-out clean likelihood in 90 % of paired splits, at the same clean FPR. K = 4 had the best
  BIC but was rejected: its extra component held only the 3 cycle-spike traces.
- A second pass tried kernel density, k-nearest-neighbour distance and 10 bagged GMMs. None beat K = 3, and
  KDE/kNN dropped to 0.76 to 0.79 on the DATE'24 traces. Refitting with 8 seeds gave the same components each time.
- α = 0.10 gave the best accuracy and a higher F1 than α = 0.05 on both the synthetic and the DATE'24 sweeps. It only
  beats α = 0.05 on accuracy if at least 30 to 40 % of the private set is backdoored. If you expect a mostly clean
  set, set `alpha` to 0.05 in `configs/final.json` and retrain.

## Input schema
CSV with columns `cache-references`, `cycles`, `LLC-loads` (any order; extra columns such as an id or label are
ignored; names are matched case-insensitively, `_` = `-`). Missing columns, non-numeric, NaN/inf or negative
values raise `SchemaError`.

## Installation
Python ≥ 3.10. Rebuilding the PDF report also needs LaTeX with `latexmk` (TeX Live), or upload
`reports/report.tex` and `reports/figures/` to Overleaf.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                     # 30 tests
```

## Inference
```bash
python scripts/predict.py private_traces.csv --model models/detector.joblib --out predictions.csv
```
Adds three columns to the input:
- `anomaly_score`: continuous score; **use this for AUROC**.
- `p_value`: conformal p-value against clean traces (small = unusual).
- `prediction`: 1 = flagged as a possible backdoor inference, 0 = clean; use for Accuracy / TPR / FPR / F1.

Python API:
```python
import pandas as pd
from hpcdetect import HPCBackdoorDetector
det = HPCBackdoorDetector.load("models/detector.joblib")
df = pd.read_csv("private_traces.csv")
scores, flags = det.score_samples(df), det.predict(df)
```
The saved model was built with scikit-learn 1.9.1. If your version differs and loading warns or fails, retrain
with the command below. It takes a few seconds and gives the same model every time.

## Training
```bash
python scripts/train.py data/trace.csv --config configs/final.json --out models/detector.joblib
```

## Full reproduction
```bash
bash scripts/reproduce.sh        # audit, comparison, selection, DATE'24 check, train, figures, report
```
| Script | Output |
|---|---|
| `scripts/audit.py` | `reports/tables/audit_*`, `reports/figures/audit_*` |
| `scripts/compare_models.py` | `model_comparison.csv` (54 candidates × 20 repeats), `gmm_k_selection.csv` |
| `scripts/select_final.py` | `paired_k.csv`, `alpha_sweep_synthetic.csv`, `alpha_sweep_proxy.csv` |
| `scripts/reference_proxy.py` | `reference_proxy.csv`; clones the public DATE'24 repo into `.cache/` |
| `scripts/challengers.py` | `challengers.csv` (KDE, kNN, bagged GMM against the final model) |
| `scripts/seed_stability.py` | `seed_stability.json` (final config refit with 8 seeds) |
| `scripts/train.py` | `models/detector.joblib`, `models/detector.json` |
| `scripts/make_figures.py` | `reports/figures/fig_*.pdf` (the three report figures) |
| `scripts/build_report.py` | compiles `reports/report.tex` into `reports/report.pdf` |

## Layout
```
src/hpcdetect/   schema.py  features.py  scorers.py  calibration.py  detector.py  evaluation.py
scripts/         audit, compare_models, select_final, reference_proxy, challengers, seed_stability,
                 train, predict, make_figures, build_report, reproduce.sh
configs/final.json   models/   reports/{report.tex,report.pdf,figures,tables}   tests/   data/trace.csv
```

## What the evaluation can and cannot show
With only clean traces we can't compute real-attack TPR, F1 or AUROC locally. What we can report:
1. held-out clean FPR;
2. stability across resamples;
3. sensitivity to *synthetic* counter shifts, labelled as such;
4. a labelled proxy on the public DATE'24 logs, which use different counters and hardware.

None of these predicts the private-set score. A backdoor that moves a trace from one clean group to the other, or
along a direction where clean traces already vary, will look normal to any detector trained on clean data.

## Reference
M. Alam, Y. Wang, M. Maniatakos, "Detecting Backdoor Attacks in Black-Box Neural Networks through Hardware
Performance Counters", DATE 2024. doi:10.23919/DATE58400.2024.10546739.
Public code: github.com/momalab/dnn-backdoor-detection-DATE24 (MIT).
