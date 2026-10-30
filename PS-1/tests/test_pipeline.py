"""Tests run on synthetic data shaped like the challenge CSV, so they do not
depend on the private dataset. They check contracts, not detection quality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpcdetect import DetectorConfig, HPCBackdoorDetector
from hpcdetect.calibration import conformal_pvalues, conformal_threshold
from hpcdetect.features import make_features
from hpcdetect.schema import FEATURES, SchemaError, validate_frame
from hpcdetect.scorers import SCORERS, build_scorer


def synthetic_clean(n=400, seed=0):
    """Two correlated regimes, roughly matching the scale of the real counters."""
    rng = np.random.default_rng(seed)
    k = rng.random(n) < 0.7
    mu = np.where(k[:, None], [185.1e6, 4.781e9, 6.726e6], [175.6e6, 4.750e9, 6.260e6])
    cov = np.array([[1, .5, .9], [.5, 1, .5], [.9, .5, 1]])
    z = rng.multivariate_normal(np.zeros(3), cov, n)
    X = mu + z * [3e6, 2e7, 1.5e5]
    return pd.DataFrame(X, columns=list(FEATURES))


@pytest.fixture(scope="module")
def clean():
    return synthetic_clean()


# ----------------------------------------------------------------- schema
def test_schema_reorders_and_tolerates_names(clean):
    messy = clean.rename(columns={"cache-references": " Cache_References "})[["cycles", "LLC-loads", " Cache_References "]]
    messy["id"] = range(len(messy))
    out = validate_frame(messy)
    assert list(out.columns) == list(FEATURES)
    np.testing.assert_allclose(out.to_numpy(), clean.to_numpy())


@pytest.mark.parametrize("mutate", [
    lambda d: d.drop(columns="cycles"),
    lambda d: d.assign(cycles=np.nan),
    lambda d: d.assign(cycles=-1.0),
    lambda d: d.assign(cycles="abc"),
    lambda d: d.iloc[:0],
])
def test_schema_rejects_bad_input(clean, mutate):
    with pytest.raises(SchemaError):
        validate_frame(mutate(clean.copy()))


# ----------------------------------------------------------------- features
@pytest.mark.parametrize("rep,width", [("raw", 3), ("log", 3), ("ratio", 4)])
def test_feature_shapes(clean, rep, width):
    assert make_features(clean, rep).shape == (len(clean), width)


def test_features_are_stateless(clean):
    a = make_features(clean.iloc[:10], "ratio")
    b = make_features(clean, "ratio").iloc[:10]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


# ----------------------------------------------------------------- calibration
def test_conformal_threshold_controls_fpr():
    rng = np.random.default_rng(1)
    fprs = []
    for _ in range(300):
        cal, test = rng.normal(size=200), rng.normal(size=2000)
        fprs.append((test > conformal_threshold(cal, 0.05)).mean())
    assert np.mean(fprs) <= 0.055


def test_pvalues_bounds_and_monotone():
    cal = np.arange(100.0)
    p = conformal_pvalues(cal, [-1, 50, 1000])
    assert p[0] == 1.0 and p[2] == pytest.approx(1 / 101) and p[0] > p[1] > p[2]


def test_threshold_infinite_when_alpha_too_small():
    assert conformal_threshold(np.arange(10.0), 0.01) == float("inf")


# ----------------------------------------------------------------- scorers
@pytest.mark.parametrize("name", [n for n in SCORERS if n not in ("ensemble", "pca")])
def test_scorers_flag_gross_outlier(clean, name):
    X = make_features(clean, "raw").to_numpy()
    s = build_scorer(name).fit(X)
    far = X.mean(0) + 8 * X.std(0)
    assert s.score(far[None, :])[0] > np.quantile(s.score(X), 0.99)


def test_pca_residual_is_blind_to_coherent_shift(clean):
    """Documented limitation: a shift along the clean correlation structure
    leaves the PCA residual small, so PCA must never be the sole detector."""
    X = make_features(clean, "raw").to_numpy()
    s = build_scorer("pca", {"n_keep": 2}).fit(X)
    far = X.mean(0) + 8 * X.std(0)
    assert s.score(far[None, :])[0] < np.quantile(s.score(X), 0.99)


def test_pca_scorer_flags_relationship_break(clean):
    X = make_features(clean, "raw").to_numpy()
    s = build_scorer("pca", {"n_keep": 2}).fit(X)
    x = X.mean(0).copy()
    x[0] += 2.5 * X[:, 0].std()  # cache-refs up, LLC unchanged: individually plausible
    x[2] -= 2.5 * X[:, 2].std()
    assert s.score(x[None, :])[0] > np.quantile(s.score(X), 0.99)


def test_ensemble_runs(clean):
    X = make_features(clean, "raw").to_numpy()
    e = build_scorer("ensemble", {"members": [("mgmm", {"k": 2}), ("pca", {"n_keep": 2})]}).fit(X)
    assert e.score(X).shape == (len(X),)


# ----------------------------------------------------------------- detector
def test_detector_fit_predict_save_load(tmp_path, clean):
    det = HPCBackdoorDetector(DetectorConfig(scorer="mgmm", scorer_params={"k": 2}, alpha=0.05)).fit(clean)
    test = synthetic_clean(n=2000, seed=7)
    assert test.pipe(det.predict).mean() < 0.09  # clean FPR near alpha
    det.save(tmp_path / "det.joblib")
    det2 = HPCBackdoorDetector.load(tmp_path / "det.joblib")
    np.testing.assert_allclose(det.score_samples(test), det2.score_samples(test))
    assert (tmp_path / "det.json").exists()


def test_detector_is_deterministic(clean):
    cfg = DetectorConfig(scorer="mgmm", scorer_params={"k": 2}, seed=3)
    a = HPCBackdoorDetector(cfg).fit(clean)
    b = HPCBackdoorDetector(cfg).fit(clean)
    assert a.threshold_ == b.threshold_


# ----------------------------------------------------------------- shipped artefacts
from pathlib import Path  # noqa: E402
import json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_final_config_is_valid():
    cfg = json.loads((ROOT / "configs/final.json").read_text())
    cfg["ratios"] = tuple(cfg["ratios"])
    det = HPCBackdoorDetector(DetectorConfig(**cfg)).fit(synthetic_clean(300))
    assert det.config.scorer == "mgmm" and det.config.scorer_params["k"] == 3


@pytest.mark.skipif(not (ROOT / "models/detector.joblib").exists(), reason="model not trained")
def test_saved_model_loads_and_scores_schema_variants():
    det = HPCBackdoorDetector.load(ROOT / "models/detector.joblib")
    x = synthetic_clean(20, seed=5)
    shuffled = x[["LLC-loads", "cycles", "cache-references"]].assign(label=0)
    np.testing.assert_allclose(det.score_samples(x), det.score_samples(shuffled))
    assert set(np.unique(det.predict(x))) <= {0, 1}


@pytest.mark.skipif(not (ROOT / "data/trace.csv").exists(), reason="challenge CSV not present")
def test_saved_model_matches_retrain():
    import pandas as pd
    cfg = json.loads((ROOT / "configs/final.json").read_text()); cfg["ratios"] = tuple(cfg["ratios"])
    det = HPCBackdoorDetector(DetectorConfig(**cfg)).fit(pd.read_csv(ROOT / "data/trace.csv"))
    saved = HPCBackdoorDetector.load(ROOT / "models/detector.joblib")
    assert det.threshold_ == pytest.approx(saved.threshold_)
