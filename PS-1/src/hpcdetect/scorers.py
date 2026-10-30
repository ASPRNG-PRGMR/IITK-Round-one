"""One-class anomaly scorers.

Every scorer follows the same contract:

    s = Scorer(**params).fit(X_train)     # X_train: (n, d) clean data only
    s.score(X)                            # (m,) anomaly scores, higher = more anomalous

All learned preprocessing (standardisation, PCA, covariance) lives inside the
scorer and is fitted in `fit` only, so a scorer fitted on a training split never
sees held-out rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from scipy.special import logsumexp


def _as2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(-1, 1) if X.ndim == 1 else X


class BaseScorer:
    name = "base"

    def fit(self, X):  # pragma: no cover - interface
        raise NotImplementedError

    def score(self, X) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}


# --------------------------------------------------------------------------- baselines
@dataclass
class ZScoreScorer(BaseScorer):
    """Per-feature standardisation. combine='max' -> max|z|, 'sumsq' -> sum z^2."""

    combine: str = "max"
    name = "zscore"

    def fit(self, X):
        self.scaler_ = StandardScaler().fit(_as2d(X))
        return self

    def score(self, X):
        z = self.scaler_.transform(_as2d(X))
        if self.combine == "max":
            return np.abs(z).max(axis=1)
        if self.combine == "sumsq":
            return (z**2).sum(axis=1)
        raise ValueError(self.combine)


# --------------------------------------------------------------------------- robust standardisation helper
def _robust_center_scale(ref: np.ndarray) -> tuple[float, float]:
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med))) * 1.4826
    if mad <= 0:
        mad = float(np.std(ref)) or 1.0
    return med, mad


def _oof_scores(make, X: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Out-of-fold scores of X under scorers built by `make()` (internal cross-fit)."""
    oof = np.empty(len(X))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        oof[te] = make().fit(X[tr]).score(X[te])
    return oof


# --------------------------------------------------------------------------- univariate GMM (DATE'24 idea)
def select_gmm_by_bic(x: np.ndarray, k_max: int, seed: int, n_init: int = 5,
                      covariance_type: str = "full", reg_covar: float = 1e-6):
    """Fit GMMs with 1..k_max components; return (best_model, {k: bic})."""
    x = _as2d(x)
    bics, best, best_bic = {}, None, np.inf
    for k in range(1, k_max + 1):
        g = GaussianMixture(k, covariance_type=covariance_type, n_init=n_init,
                            reg_covar=reg_covar, max_iter=1000, random_state=seed).fit(x)
        b = float(g.bic(x))
        bics[k] = b
        if b < best_bic:
            best, best_bic = g, b
    return best, bics


class _SingleFeatureGMM(BaseScorer):
    def __init__(self, k: int, seed: int):
        self.k, self.seed = k, seed

    def fit(self, x):
        self.scaler_ = StandardScaler().fit(_as2d(x))
        self.gmm_ = GaussianMixture(self.k, n_init=5, max_iter=1000, reg_covar=1e-6,
                                    random_state=self.seed).fit(self.scaler_.transform(_as2d(x)))
        return self

    def score(self, x):
        return -self.gmm_.score_samples(self.scaler_.transform(_as2d(x)))


@dataclass
class UnivariateGMMScorer(BaseScorer):
    """Independent GMM per feature; per-feature NLLs are put on a common scale
    before combining, using robust (median/MAD) standardisation estimated from
    *out-of-fold* NLLs of the training data. Raw NLLs are not added directly.

    combine: 'max'  -> the single most surprising feature (conservative OR rule)
             'mean' -> average evidence across features
             'rawsum' -> naive sum of raw NLLs (kept only as a documented pitfall)
    k: int to fix the component count for all features, or None to pick by BIC.
    """

    k: int | None = None
    k_max: int = 5
    combine: str = "max"
    seed: int = 0
    n_splits: int = 5
    name = "ugmm"

    def fit(self, X):
        X = _as2d(X)
        self.ks_, self.bics_, self.models_, self.ref_ = [], [], [], []
        for j in range(X.shape[1]):
            xj = X[:, j]
            z = StandardScaler().fit_transform(xj.reshape(-1, 1))
            if self.k is None:
                g, bics = select_gmm_by_bic(z, self.k_max, self.seed)
                k = g.n_components
            else:
                k, bics = self.k, {}
            self.ks_.append(k)
            self.bics_.append(bics)
            self.models_.append(_SingleFeatureGMM(k, self.seed).fit(xj))
            oof = _oof_scores(lambda k=k: _SingleFeatureGMM(k, self.seed), xj, self.n_splits, self.seed)
            self.ref_.append(_robust_center_scale(oof))
        return self

    def feature_scores(self, X) -> np.ndarray:
        X = _as2d(X)
        cols = []
        for j, m in enumerate(self.models_):
            nll = m.score(X[:, j])
            if self.combine == "rawsum":
                cols.append(nll)
            else:
                med, sc = self.ref_[j]
                cols.append((nll - med) / sc)
        return np.column_stack(cols)

    def score(self, X):
        f = self.feature_scores(X)
        if self.combine == "max":
            return f.max(axis=1)
        if self.combine in ("mean", "rawsum"):
            return f.sum(axis=1) if self.combine == "rawsum" else f.mean(axis=1)
        raise ValueError(self.combine)

    def diagnostics(self):
        return {"components_per_feature": self.ks_, "bic_per_feature": self.bics_}


# --------------------------------------------------------------------------- multivariate GMM
@dataclass
class MultivariateGMMScorer(BaseScorer):
    """Joint density: score = -log p(x) under a K-component GMM on standardised features."""

    k: int = 2
    covariance_type: str = "full"
    reg_covar: float = 1e-6
    n_init: int = 10
    seed: int = 0
    name = "mgmm"

    def fit(self, X):
        X = _as2d(X)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.gmm_ = GaussianMixture(self.k, covariance_type=self.covariance_type,
                                    reg_covar=self.reg_covar, n_init=self.n_init,
                                    max_iter=1000, random_state=self.seed).fit(Z)
        self.bic_ = float(self.gmm_.bic(Z))
        self.occupancy_ = np.bincount(self.gmm_.predict(Z), minlength=self.k)
        return self

    def score(self, X):
        return -self.gmm_.score_samples(self.scaler_.transform(_as2d(X)))

    def diagnostics(self):
        return {"bic": self.bic_, "occupancy": self.occupancy_.tolist(),
                "weights": self.gmm_.weights_.round(4).tolist(),
                "converged": bool(self.gmm_.converged_)}


# --------------------------------------------------------------------------- Mahalanobis
@dataclass
class MahalanobisScorer(BaseScorer):
    """Squared Mahalanobis distance; robust=True uses Minimum Covariance Determinant."""

    robust: bool = False
    support_fraction: float | None = None
    seed: int = 0
    name = "mahal"

    def fit(self, X):
        X = _as2d(X)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.cov_ = (MinCovDet(support_fraction=self.support_fraction, random_state=self.seed)
                     if self.robust else EmpiricalCovariance()).fit(Z)
        return self

    def score(self, X):
        return self.cov_.mahalanobis(self.scaler_.transform(_as2d(X)))


# --------------------------------------------------------------------------- PCA residual
@dataclass
class PCAResidualScorer(BaseScorer):
    """Squared reconstruction error in standardised feature space after keeping
    `n_keep` principal components (the 'Q' / SPE statistic)."""

    n_keep: int = 2
    name = "pca"

    def fit(self, X):
        X = _as2d(X)
        self.scaler_ = StandardScaler().fit(X)
        self.pca_ = PCA(self.n_keep).fit(self.scaler_.transform(X))
        return self

    def score(self, X):
        Z = self.scaler_.transform(_as2d(X))
        R = Z - self.pca_.inverse_transform(self.pca_.transform(Z))
        return (R**2).sum(axis=1)

    def diagnostics(self):
        return {"explained_variance_ratio": self.pca_.explained_variance_ratio_.round(4).tolist(),
                "components": self.pca_.components_.round(4).tolist()}


# --------------------------------------------------------------------------- nonlinear challengers
@dataclass
class IsolationForestScorer(BaseScorer):
    n_estimators: int = 300
    max_samples: str | int = "auto"
    seed: int = 0
    name = "iforest"

    def fit(self, X):
        X = _as2d(X)
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                      random_state=self.seed).fit(self.scaler_.transform(X))
        return self

    def score(self, X):
        return -self.model_.score_samples(self.scaler_.transform(_as2d(X)))


@dataclass
class OCSVMScorer(BaseScorer):
    nu: float = 0.05
    gamma: str | float = "scale"
    name = "ocsvm"

    def fit(self, X):
        X = _as2d(X)
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = OneClassSVM(nu=self.nu, gamma=self.gamma).fit(self.scaler_.transform(X))
        return self

    def score(self, X):
        return -self.model_.decision_function(self.scaler_.transform(_as2d(X)))


# --------------------------------------------------------------------------- ensemble
@dataclass
class EnsembleScorer(BaseScorer):
    """Combine member scorers after robust standardisation of each member's
    out-of-fold training scores. combine='max' or 'mean'."""

    members: list = field(default_factory=list)  # list of (scorer_name, params)
    combine: str = "mean"
    seed: int = 0
    n_splits: int = 5
    name = "ensemble"

    def fit(self, X):
        X = _as2d(X)
        self.fitted_, self.ref_ = [], []
        for nm, params in self.members:
            make = lambda nm=nm, params=params: build_scorer(nm, params)
            self.fitted_.append(make().fit(X))
            self.ref_.append(_robust_center_scale(_oof_scores(make, X, self.n_splits, self.seed)))
        return self

    def member_scores(self, X):
        return np.column_stack([(m.score(X) - c) / s for m, (c, s) in zip(self.fitted_, self.ref_)])

    def score(self, X):
        S = self.member_scores(X)
        return S.max(axis=1) if self.combine == "max" else S.mean(axis=1)


# --------------------------------------------------------------------------- second-round challengers
@dataclass
class KDEScorer(BaseScorer):
    """Gaussian kernel density on standardised features. bandwidth=None picks it
    by 5-fold held-out log-likelihood on the training split only."""

    bandwidth: float | None = None
    seed: int = 0
    name = "kde"

    def fit(self, X):
        X = _as2d(X)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        bw = self.bandwidth
        if bw is None:
            gs = GridSearchCV(KernelDensity(), {"bandwidth": np.geomspace(0.08, 1.5, 16)},
                              cv=KFold(5, shuffle=True, random_state=self.seed)).fit(Z)
            bw = float(gs.best_params_["bandwidth"])
        self.bandwidth_ = bw
        self.kde_ = KernelDensity(bandwidth=bw).fit(Z)
        return self

    def score(self, X):
        return -self.kde_.score_samples(self.scaler_.transform(_as2d(X)))

    def diagnostics(self):
        return {"bandwidth": self.bandwidth_}


@dataclass
class KNNScorer(BaseScorer):
    """Mean distance to the k nearest clean training traces (standardised)."""

    k: int = 10
    name = "knn"

    def fit(self, X):
        X = _as2d(X)
        self.scaler_ = StandardScaler().fit(X)
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(self.scaler_.transform(X))
        return self

    def score(self, X):
        d, _ = self.nn_.kneighbors(self.scaler_.transform(_as2d(X)))
        return d.mean(axis=1)


@dataclass
class BaggedGMMScorer(BaseScorer):
    """-log of the average density of `n_bags` GMMs, each fitted on a bootstrap
    resample. Smooths the run-to-run variance of a single EM fit."""

    k: int = 3
    n_bags: int = 10
    seed: int = 0
    name = "bgmm"

    def fit(self, X):
        X = _as2d(X)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        rng = np.random.default_rng(self.seed)
        self.gmms_ = [GaussianMixture(self.k, n_init=3, reg_covar=1e-6, max_iter=1000,
                                      random_state=self.seed + b).fit(Z[rng.integers(0, len(Z), len(Z))])
                      for b in range(self.n_bags)]
        return self

    def score(self, X):
        Z = self.scaler_.transform(_as2d(X))
        ll = np.column_stack([g.score_samples(Z) for g in self.gmms_])
        return -(logsumexp(ll, axis=1) - np.log(len(self.gmms_)))


SCORERS = {
    "kde": KDEScorer,
    "knn": KNNScorer,
    "bgmm": BaggedGMMScorer,
    "zscore": ZScoreScorer,
    "ugmm": UnivariateGMMScorer,
    "mgmm": MultivariateGMMScorer,
    "mahal": MahalanobisScorer,
    "pca": PCAResidualScorer,
    "iforest": IsolationForestScorer,
    "ocsvm": OCSVMScorer,
    "ensemble": EnsembleScorer,
}


def build_scorer(name: str, params: dict | None = None) -> BaseScorer:
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r}; choose from {sorted(SCORERS)}")
    return SCORERS[name](**(params or {}))
