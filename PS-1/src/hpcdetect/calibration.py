"""Threshold calibration from held-out clean scores.

Given n clean calibration scores that were NOT used to fit the scorer, the
split-conformal threshold

    t = the ceil((n + 1)(1 - alpha))-th smallest calibration score

guarantees P(score(x_new) > t) <= alpha for a new clean trace exchangeable with
the calibration set. That is the only guarantee available without attack labels:
it bounds the clean false-positive rate, not the detection rate.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_threshold(cal_scores, alpha: float) -> float:
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(s)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:  # too few calibration points for this alpha: never flag on calibration evidence alone
        return float("inf")
    return float(s[rank - 1])


def conformal_pvalues(cal_scores, scores) -> np.ndarray:
    """p = (1 + #{cal >= s}) / (n + 1). Small p = unusual relative to clean data."""
    cal = np.sort(np.asarray(cal_scores, dtype=np.float64))
    s = np.asarray(scores, dtype=np.float64)
    n_ge = len(cal) - np.searchsorted(cal, s, side="left")
    return (1.0 + n_ge) / (len(cal) + 1.0)
