# PROGRESS — handoff log

Read this first when resuming. Each part ends with a zip of the project.

## Status (Part 5 — 2026-10-30)

**Report redone as a plain LaTeX article** (`reports/report.tex` -> `reports/report.pdf`, 4 pages, Computer Modern,
black text, booktabs table). Figures replaced by three simple ones from `scripts/make_figures.py`
(`fig_data.pdf`, `fig_auroc.pdf`, `fig_alpha.pdf`). `scripts/build_report.py` now just runs latexmk.
Author line in report.tex says "Nishanth D'souza" — change it there if the team name should go instead.
Delivered as a standalone `PS-1/` folder zip for the user to add to the IITK-Round-one repo themselves
(pushing from the cloud session was refused: no GitHub App access for that repo).

## Part 4 status (kept for history)

**Review pass + repo integration done.** Final model unchanged (GMM K=3, alpha 0.10).
- Second-round challengers (`scripts/challengers.py`, `reports/tables/challengers.csv`): KDE with CV bandwidth,
  kNN distance (k=5, 15), 10 bagged GMMs. Synthetic AUROC all within 0.02 of K=3; DATE'24 real-label AUROC:
  K=3 0.865, bagged 0.861, KDE 0.793, kNN 0.792 / 0.761. Nothing beats K=3.
- Seed stability (`scripts/seed_stability.py`): 8 seeds, identical components 559/219/22, threshold sd 0.055,
  98.8 % flag agreement on a 1,200-row probe set.
- Alpha break-even: alpha 0.10 beats 0.05 on accuracy only if >= ~30 % (DATE'24) / ~40 % (synthetic) of the private
  set is backdoored. Now stated in the report.
- Report rewritten in plain language (no-ai-slop pass), 3 pages, every number read from reports/tables/.
- Project copied into the IITK-Round-one repo as `PS-1/`; root README updated. `data/trace.csv` is gitignored
  like PS-2's organiser inputs. Nothing committed or pushed.

## Part 3 status (kept for history)

**Part 3 done — submission-ready.** Added `scripts/make_figures.py`, `scripts/build_report.py` ->
`reports/report.pdf` (3 pages), final `README.md`, `scripts/reproduce.sh`, 3 artefact tests (27 passing),
verified `pip install -e .` + model load from a clean venv.
**Only Part 4 remains:** integrate into the GitHub repo once its URL (or a zip of it) is provided — inspect its
structure first and merge; do not overwrite existing files blindly.
Optional before submitting: decide whether alpha 0.10 suits the expected class balance of the private set.

## Part 2 status (kept for history)

**Part 2 done.** CSV received (`data/trace.csv`, sha256 67f73f7a…7d3c). Final detector trained:
`configs/final.json` -> `models/detector.joblib` (+ `.json` card). Remaining: Part 3 (PDF report, final README,
report figures), Part 4 (repo integration — needs the repo URL or a zip of it).

### Part 2 findings (all scripts reproducible; tables in reports/tables/)
- Audit (`scripts/audit.py`): every figure in the original handoff note reproduces exactly (stats, Pearson
  0.920/0.529/0.540, skew, kurtosis 6.81, 2-comp 572/228, PCA 78.15/19.18/2.68 %, univariate K = 2/3/2,
  best BIC at full K=4 whose tiny component is 3 rows: 668, 724, 294).
- No file-order drift: |Spearman(row index, counter)| < 0.04; regime label vs index -0.05. If rows are in
  collection order, regimes are not session blocks (cannot be proven: order may have been shuffled).
- Max-cycles row 668 (0-based; CSV line 670): 4.996 B cycles, 3.0 SD gap to next value, robust z 9.1, AND
  the lowest cache-references in the file (0.25th pct) -> Mahalanobis rank 1, chi2 p 4e-34. Rows 724 (4.916 B)
  and 294 (4.857 B) are the next two. Kept in training (no basis for deletion); final model still flags all 3.
- Held-out clean log-likelihood (raw, full cov, 5x4 CV): K1 -3.197, K2 -3.004, K3 -2.927, K4 -2.902 (min
  component ~3 rows -> overfit). log == raw. Ratio adds nothing to full-cov GMM (already models the linear
  cache/LLC relation) but lifts z-score/ugmm on relational shifts 0.71 -> 0.95 AUROC.
- Model comparison (`compare_models.py`, 54 candidates x 20 repeats): conformal clean FPR 0.038–0.058 for
  alpha 0.05 everywhere. Top by mean synthetic AUROC: mgmm K4 0.861, K3 0.858, K2 0.853, OCSVM 0.853.
  PCA-2 residual blind to all_up (0.50); robust MCD worse than empirical covariance on multi-regime data.
- Paired K test (`select_final.py`, 40 identical splits): K3 beats K2 on held-out NLL in 90 % of splits,
  >= K2 on every synthetic family; FPR 0.051 vs 0.048. DATE'24 labelled proxy: K3 AUROC 0.865 vs K2 0.811.
- Operating point: accuracy peaks near alpha 0.10 (synthetic 0.795, proxy 0.797); F1 0.67->0.72 (synthetic),
  0.69->0.78 (proxy) vs alpha 0.05. Assumes a roughly balanced private set (unknown) -> judgment call.

### FINAL MODEL (decided)
raw counters -> StandardScaler -> full-covariance GMM K=3 (n_init 10, reg 1e-6, seed 0) -> score = -log p(x);
threshold = conformal quantile of 10-fold out-of-fold clean scores at alpha 0.10 (threshold 4.664);
refit on all 800. OOF flag rate 0.099. Components: 559 / 219 / 22 rows (two regimes + broad tail component).

## Part 1 status (kept for history)

**Blocked on input:** the challenge CSV (`trace(1).csv`) was NOT attached in the Part 1 session —
only the handoff note was, and that note is truncated mid-section 14 ("...calibration requireme").
To resume: put the CSV at `data/trace.csv` and paste the rest of the handoff (sections 14+).

### Done
- Package `src/hpcdetect/`:
  - `schema.py` — 3-feature schema, tolerant column matching, hard failures on missing/NaN/negative/non-numeric.
  - `features.py` — representations `raw`, `log`, `ratio` (raw + log-ratios; default ratio LLC/cache-ref).
  - `scorers.py` — z-score (max / sum-sq), univariate GMM per feature (BIC-selected K, robust-standardised
    out-of-fold NLLs before combining; `rawsum` kept as documented pitfall), multivariate GMM, Mahalanobis
    (empirical / MCD), PCA residual (SPE), Isolation Forest, One-Class SVM, ensemble.
    All scaling fitted inside `fit` only (no leakage).
  - `calibration.py` — split-conformal threshold + p-values (bounds clean FPR, nothing else).
  - `detector.py` — `HPCBackdoorDetector`: K-fold cross-fit OOF scores -> conformal threshold -> refit on all
    data -> joblib save + JSON model card.
  - `evaluation.py` — repeated 60/20/20 fit/cal/test clean-only protocol + synthetic perturbation map
    (cycles_up, cacheref_up, llc_up, all_up, relational) at 0.5–3 SD. **Synthetic, never report as attack TPR.**
- `scripts/audit.py` (full data audit + figures), `train.py`, `predict.py`, `reference_proxy.py`.
- `tests/test_pipeline.py` — 24 tests passing (synthetic data only).
- Smoke-tested audit/train/predict end to end on synthetic data.

### Research findings so far
1. **DATE'24 reference implementation (read in full):** uses 5 different counters (branches, branch-misses,
   cache-references, cache-misses, instructions) — only `cache-references` overlaps with ours. Per-feature GMM,
   BIC over K=1..10 on just 100 samples, threshold = mean + 3*sd of NLL **on the same 100 training samples**
   (in-sample, optimistic), 3-sigma outlier removal applied to the **test** chunks too (drops hard cases),
   each counter evaluated separately (no fusion). We deliberately do not copy these choices.
2. **Reference data (10k benign + 10k backdoor) — backdoor effect is a small mean shift**, best single-counter
   pooled AUROC ≈ 0.69 (cache-references, +0.73 SD). Not a gross outlier.
3. **Collection-session confound:** reference traces were collected sequentially by class, benign session then
   backdoor session. Benign "regimes" track collection block; within-block separability is far higher than
   pooled. => Our 572/228 regimes may be session/drift effects rather than inference behaviour. The audit's
   row-order checks (`audit_row_order.png`, `row_order_spearman_with_index`) test this on our CSV.
4. **Labelled proxy experiment** (`reports/tables/reference_proxy.csv`; train 640, calibrate 160, alpha 0.05,
   10 repeats; all 5 reference counters):
   | scorer | pooled AUROC | pooled F1 | within-block AUROC (blocks 0/5/9) |
   |---|---|---|---|
   | z-score max | 0.46 | 0.15 | 0.99 / 0.98 / 0.87 |
   | univariate GMM, max | 0.74 | 0.21 | 1.00 / 1.00 / 0.88 |
   | multivariate GMM K=1 | 0.50 | 0.13 | 0.99 / 0.98 / 0.87 |
   | multivariate GMM K=2 | 0.81 | 0.45 | 1.00 / 0.99 / 0.89 |
   | **multivariate GMM K=3** | **0.87** | **0.69** | 1.00 / 1.00 / 0.88 |
   | robust Mahalanobis (MCD) | 0.52 | 0.10 | 1.00 / 0.99 / 0.89 |
   | PCA residual | 0.50 | 0.09 | 1.00 / 0.86 / 0.87 |
   | Isolation Forest | 0.59 | 0.14 | 0.96 / 0.96 / 0.83 |
   | One-Class SVM | 0.48 | 0.16 | 1.00 / 1.00 / 0.87 |
   Readings: (a) with multi-regime clean data, single-ellipsoid methods collapse to chance and multi-component
   GMM wins; (b) in homogeneous data almost everything works; (c) conformal FPR stayed 0.04–0.07 at alpha 0.05.
   Caveat: pooled backdoor traces are from a different session, so part of the "signal" may be session drift.
5. PCA residual is provably blind to shifts along the clean correlation structure (`all_up`); never use alone
   (encoded as a test).

### Working hypothesis for the final model (to be confirmed on our CSV in Part 2)
Multivariate GMM (full covariance) on standardised raw or log counters, K chosen by **held-out clean
log-likelihood + bootstrap stability + min component size**, not BIC alone; conformal threshold from cross-fit
OOF scores. Possible addition: ensemble with the relational (log LLC/cache-ref) signal only if it improves the
synthetic `relational` sensitivity without raising clean FPR. Alpha (operating point) still to decide — it
trades FPR against TPR and the organisers score Accuracy/F1 at our threshold but AUROC on our scores.

## Plan for remaining parts
- ~~Part 2~~ done. Was: `python scripts/audit.py data/trace.csv`; verify every number in the handoff note;
  investigate max-cycles row and row-order drift; write `scripts/compare_models.py` (loop
  `evaluation.evaluate_candidate` over representations x scorers x K, 20 repeats) -> `reports/tables/model_comparison.csv`.
- ~~Part 3~~ done. Was: pick final config with written justification -> `configs/final.json`; `train.py` ->
  `models/detector.joblib` + `.json`; final figures; README completed; 2–4 page PDF report (`reports/report.pdf`).
- **Part 4 (needs repo URL / files):** inspect existing repo structure and merge, don't overwrite.

## Resume commands
```
pip install -e ".[dev]"
pytest -q
python scripts/audit.py data/trace.csv
python scripts/reference_proxy.py      # clones the DATE'24 repo into .cache/
```
