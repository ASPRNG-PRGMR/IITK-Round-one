"""Feature representations compared in the study.

A  "raw"   : the three counters as measured.
B  "log"   : natural log of each counter (counts are strictly positive in practice).
C  "ratio" : raw counters plus log-ratios that describe relationships between counters.

Ratios are taken in log space (log a - log b) so they are symmetric, additive and
less noise-amplifying than plain quotients. All *learned* scaling happens later
inside each scorer, fitted on the training split only; nothing here is learned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURES

RATIOS: dict[str, tuple[str, str]] = {
    "llc_per_cacheref": ("LLC-loads", "cache-references"),
    "llc_per_cycle": ("LLC-loads", "cycles"),
    "cacheref_per_cycle": ("cache-references", "cycles"),
}

REPRESENTATIONS = ("raw", "log", "ratio")


def make_features(
    df: pd.DataFrame,
    representation: str = "raw",
    ratios: tuple[str, ...] = ("llc_per_cacheref",),
) -> pd.DataFrame:
    """Build the model input matrix for a validated trace frame."""
    base = df.loc[:, list(FEATURES)].astype("float64")
    if representation == "raw":
        return base.copy()
    if representation == "log":
        return np.log(base.clip(lower=1.0)).add_prefix("log_")
    if representation == "ratio":
        out = base.copy()
        logs = np.log(base.clip(lower=1.0))
        for name in ratios:
            num, den = RATIOS[name]
            out[f"logratio_{name}"] = logs[num] - logs[den]
        return out
    raise ValueError(f"unknown representation {representation!r}; choose from {REPRESENTATIONS}")
