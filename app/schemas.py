from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

# ---------------------------------
# Shared input/output schemas
# ---------------------------------

ModelName = Literal["naive", "seasonal_naive", "arima", "ets", "theta"]
FillMethod = Literal["none", "ffill", "bfill", "interpolate"]
OutlierMethod = Literal["iqr", "zscore"]
ADFRegression = Literal["c", "ct", "ctt", "n"]
KPSSRegression = Literal["c", "ct"]


class SeriesIn(BaseModel):
    """Time series input.

    - dates: ISO-like strings (YYYY-MM-DD recommended). If omitted, backend may infer an index.
    - values: numeric values; use null for missing values.
    """
    dates: Optional[List[str]] = Field(default=None, description="ISO dates like 2020-01-01")
    values: List[Optional[float]] = Field(..., description="Values; use null for missing")
    freq: Optional[str] = Field(default=None, description="Optional frequency hint (e.g., 'D', 'M')")


class PreprocessIn(BaseModel):
    series: SeriesIn
    resample_freq: Optional[str] = None
    fill_method: FillMethod = "interpolate"
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


class SummaryIn(BaseModel):
    series: SeriesIn
    params: Dict[str, Any] = Field(default_factory=dict)


class OutliersIn(BaseModel):
    series: SeriesIn
    method: OutlierMethod = "iqr"
    z_thresh: float = 3.5
    iqr_k: float = 1.5


class DecomposeIn(BaseModel):
    series: SeriesIn
    period: Optional[int] = None
    robust: bool = True


class StrengthIn(BaseModel):
    series: SeriesIn
    period: Optional[int] = None


class StationarityIn(BaseModel):
    series: SeriesIn
    regression: ADFRegression = "c"
    kpss_regression: KPSSRegression = "c"


class AutocorrIn(BaseModel):
    series: SeriesIn
    nlags: int = 14


class SpectrumIn(BaseModel):
    series: SeriesIn
    detrend: bool = True


class XCorrIn(BaseModel):
    x: SeriesIn
    y: SeriesIn
    max_lag: int = 60


class ForecastIn(BaseModel):
    series: SeriesIn
    horizon: int = 30
    seasonal_period: Optional[int] = None

    auto: bool = True
    max_p: int = 3
    max_d: int = 2
    max_q: int = 3

    seasonal: bool = True
    max_P: int = 1
    max_D: int = 1
    max_Q: int = 1


class ETSIn(BaseModel):
    series: SeriesIn
    horizon: int = 30
    seasonal_period: Optional[int] = None
    seasonal: bool = True


class ThetaIn(BaseModel):
    series: SeriesIn
    horizon: int = 30
    seasonal_period: Optional[int] = None


class BacktestIn(BaseModel):
    series: SeriesIn
    model: ModelName = "arima"
    horizon: int = 14
    initial_train_size: Optional[int] = None
    step: int = 1
    seasonal_period: Optional[int] = None


class WaveletIn(BaseModel):
    series: SeriesIn
    params: Dict[str, Any] = Field(default_factory=dict)

    wavelet: Optional[str] = None
    min_period: Optional[float] = None
    max_period: Optional[float] = None
    num_scales: Optional[int] = None
    detrend: Optional[bool] = None
    normalize: Optional[bool] = None
    max_time_points: Optional[int] = None


class ApiOut(BaseModel):
    ok: bool
    result: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
