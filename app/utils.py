from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

def to_series(dates: Optional[List[str]], values: List[float]) -> pd.Series:
    y = pd.Series(values, dtype="float64")
    if dates is None:
        y.index = pd.RangeIndex(start=0, stop=len(y), step=1)
        return y

    dt = pd.to_datetime(pd.Series(dates), errors="coerce", utc=False)
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Invalid dates: {bad} items could not be parsed.")
    y.index = pd.DatetimeIndex(dt.values)
    y = y.sort_index()
    return y

def infer_freq_safe(idx: pd.Index) -> Optional[str]:
    if not isinstance(idx, pd.DatetimeIndex):
        return None
    try:
        f = pd.infer_freq(idx)
        return f
    except Exception:
        return None

def default_seasonal_period(freq: Optional[str]) -> int:
    # Heuristic defaults; override from request if you know better.
    if not freq:
        return 12
    f = freq.upper()
    if f.startswith("H"):
        return 24
    if f.startswith("D"):
        return 7
    if f.startswith("W"):
        return 52
    if f.startswith("M"):
        return 12
    if f.startswith("Q"):
        return 4
    if f.startswith("Y") or f.startswith("A"):
        return 1
    return 12

def resample_if_datetime(y: pd.Series, resample_freq: Optional[str]) -> Tuple[pd.Series, List[str]]:
    warnings = []
    if resample_freq is None:
        return y, warnings
    if not isinstance(y.index, pd.DatetimeIndex):
        warnings.append("resample_freq ignored because series has no datetime index.")
        return y, warnings

    y2 = y.resample(resample_freq).mean()
    if len(y2) < 5:
        warnings.append("Resampling produced very short series; check resample_freq.")
    return y2, warnings

def fill_missing(y: pd.Series, method: str) -> Tuple[pd.Series, List[str]]:
    warnings = []
    if method == "none":
        return y, warnings
    if y.isna().sum() == 0:
        return y, warnings

    if method == "ffill":
        return y.ffill(), warnings
    if method == "bfill":
        return y.bfill(), warnings
    if method == "interpolate":
        # works for numeric; for datetime index uses index order
        return y.interpolate(limit_direction="both"), warnings
    warnings.append(f"Unknown fill_method={method}, no filling applied.")
    return y, warnings

def clip_values(y: pd.Series, clip_min: Optional[float], clip_max: Optional[float]) -> pd.Series:
    if clip_min is None and clip_max is None:
        return y
    return y.clip(lower=clip_min, upper=clip_max)

def to_jsonable_series(y: pd.Series) -> dict:
    if isinstance(y.index, pd.DatetimeIndex):
        x = [t.isoformat() for t in y.index.to_pydatetime()]
    else:
        x = y.index.astype(int).tolist()
    return {"x": x, "y": [None if pd.isna(v) else float(v) for v in y.values]}

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None
