"""The deployable detector: schema validation -> features -> scorer -> calibrated threshold.

Training uses cross-fitting so all 800 clean traces contribute to both the
model and the threshold without any trace calibrating a model that saw it:

  1. Split the clean data into K folds.
  2. For each fold, fit the full pipeline on the other K-1 folds and score the
     held-out fold -> one out-of-fold (OOF) score per training trace.
  3. Threshold = conformal quantile of the OOF scores at the target FPR alpha.
  4. Refit the pipeline on all clean data; this is the model that is saved.

Step 4 makes the deployed model very slightly tighter than the fold models, so
the realised clean FPR can sit marginally above alpha; the clean-only
validation protocol (evaluation.py) measures this directly on held-out data.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold

from . import __version__
from .calibration import conformal_pvalues, conformal_threshold
from .features import make_features
from .schema import FEATURES, validate_frame
from .scorers import build_scorer


@dataclass
class DetectorConfig:
    representation: str = "raw"
    ratios: tuple[str, ...] = ("llc_per_cacheref",)
    scorer: str = "mgmm"
    scorer_params: dict = field(default_factory=dict)
    alpha: float = 0.05          # target clean false-positive rate
    n_folds: int = 5
    seed: int = 0


class HPCBackdoorDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

    # ------------------------------------------------------------------ internals
    def _X(self, df: pd.DataFrame) -> np.ndarray:
        c = self.config
        return make_features(validate_frame(df), c.representation, tuple(c.ratios)).to_numpy()

    def _new_scorer(self):
        params = dict(self.config.scorer_params)
        if "seed" in build_scorer(self.config.scorer, params).__dataclass_fields__ and "seed" not in params:
            params["seed"] = self.config.seed
        return build_scorer(self.config.scorer, params)

    # ------------------------------------------------------------------ API
    def fit(self, df: pd.DataFrame) -> "HPCBackdoorDetector":
        clean = validate_frame(df)
        X = self._X(clean)
        c = self.config
        oof = np.empty(len(X))
        for tr, te in KFold(c.n_folds, shuffle=True, random_state=c.seed).split(X):
            oof[te] = self._new_scorer().fit(X[tr]).score(X[te])
        self.oof_scores_ = oof
        self.threshold_ = conformal_threshold(oof, c.alpha)
        self.scorer_ = self._new_scorer().fit(X)
        self.n_train_ = len(X)
        self.train_hash_ = hashlib.sha256(clean.to_numpy().tobytes()).hexdigest()[:16]
        return self

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """Continuous anomaly score (higher = less consistent with clean behaviour). Use for AUROC."""
        return self.scorer_.score(self._X(df))

    def p_values(self, df: pd.DataFrame) -> np.ndarray:
        """Conformal p-value against out-of-fold clean scores (small = unusual)."""
        return conformal_pvalues(self.oof_scores_, self.score_samples(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """1 = flagged as suspicious (possible backdoor), 0 = consistent with clean."""
        return (self.score_samples(df) > self.threshold_).astype(int)

    # ------------------------------------------------------------------ persistence
    def card(self) -> dict:
        return {
            "package_version": __version__,
            "features": list(FEATURES),
            "config": {**asdict(self.config), "ratios": list(self.config.ratios)},
            "threshold": self.threshold_,
            "n_train": self.n_train_,
            "train_sha256_16": self.train_hash_,
            "scorer_diagnostics": _jsonable(self.scorer_.diagnostics()),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".json").write_text(json.dumps(self.card(), indent=2))

    @staticmethod
    def load(path: str | Path) -> "HPCBackdoorDetector":
        det = joblib.load(path)
        if not isinstance(det, HPCBackdoorDetector):
            raise TypeError(f"{path} does not contain an HPCBackdoorDetector")
        return det


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
