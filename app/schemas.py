from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

ModelName = Literal["naive", "seasonal_naive", "arima", "ets", "theta"]

class SeriesIn(BaseModel):
    dates: Optional[list[str]] = Field(
        default=None,
        description="ISO dates like 2020-01-01"
    )
    values: list[float | None] = Field(
        ...,
        description="Values; null allowed for missing"
    )
    freq: Optional[str] = None


class PreprocessIn(BaseModel):
    series: SeriesIn
    resample_freq: Optional[str] = None
    fill_method: Literal["none", "ffill", "bfill", "interpolate"] = "interpolate"
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

class OutliersIn(BaseModel):
    series: SeriesIn
    method: Literal["iqr", "zscore"] = "iqr"
    z_thresh: float = 3.5
    iqr_k: float = 1.5

class DecomposeIn(BaseModel):
    series: SeriesIn
    period: Optional[int] = None
    robust: bool = True

class StationarityIn(BaseModel):
    series: SeriesIn
    regression: Literal["c", "ct", "ctt", "n"] = "c"  # ADF regression
    kpss_regression: Literal["c", "ct"] = "c"

class AutocorrIn(BaseModel):
    series: SeriesIn
    nlags: int = 40

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

    # ARIMA/SARIMA controls
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
    trend: Literal["add", "mul", "none"] = "add"
    seasonal: Literal["add", "mul", "none"] = "add"
    damped_trend: bool = False

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

class ApiOut(BaseModel):
    ok: bool
    result: Dict[str, Any]
    warnings: List[str] = []
