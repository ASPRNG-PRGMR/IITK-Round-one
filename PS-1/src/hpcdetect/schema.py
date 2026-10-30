"""Feature schema and input validation.

The detector is trained on exactly these three HPC counters. Inference input is
validated against this schema so that a mis-ordered, renamed, or corrupted CSV
fails loudly instead of producing meaningless scores.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES: tuple[str, ...] = ("cache-references", "cycles", "LLC-loads")


class SchemaError(ValueError):
    """Raised when input data does not match the expected HPC schema."""


def _normalise(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def validate_frame(df: pd.DataFrame, features: tuple[str, ...] = FEATURES) -> pd.DataFrame:
    """Return a float64 frame with exactly `features`, in canonical order.

    Column matching tolerates case, surrounding whitespace and `_` vs `-`.
    Extra columns (e.g. an id or label column) are dropped. Missing columns,
    non-numeric values, non-finite values and negative counts raise SchemaError.
    """
    lookup = {_normalise(c): c for c in df.columns}
    missing = [f for f in features if _normalise(f) not in lookup]
    if missing:
        raise SchemaError(f"missing required HPC column(s): {missing}; got {list(df.columns)}")
    out = pd.DataFrame({f: df[lookup[_normalise(f)]] for f in features})
    try:
        out = out.apply(pd.to_numeric, errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"non-numeric value in HPC columns: {exc}") from exc
    values = out.to_numpy()
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).any(axis=1).sum())
        raise SchemaError(f"{bad} row(s) contain NaN or infinite counter values")
    if (values < 0).any():
        raise SchemaError("negative counter values are not valid HPC counts")
    if len(out) == 0:
        raise SchemaError("input contains no rows")
    return out.reset_index(drop=True)


def load_traces(path: str | Path, features: tuple[str, ...] = FEATURES) -> pd.DataFrame:
    """Load and validate an HPC trace CSV."""
    return validate_frame(pd.read_csv(path), features)
