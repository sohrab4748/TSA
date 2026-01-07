
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import APIRouter

from scipy import signal
from scipy.stats import zscore

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, q_stat
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel

from app.schemas import (
    ApiOut,
    PreprocessIn, OutliersIn, DecomposeIn, StationarityIn, AutocorrIn,
    SpectrumIn, XCorrIn, ForecastIn, ETSIn, ThetaIn, BacktestIn
)
from app.utils import (
    to_series, infer_freq_safe, default_seasonal_period,
    resample_if_datetime, fill_missing, clip_values, to_jsonable_series, safe_float
)

router = APIRouter()

# ---------------------------
# Helpers
# ---------------------------
from typing import Optional
from fastapi import Body, Query, HTTPException
import json
from pydantic import BaseModel


from typing import Any

class SeriesIn(BaseModel):
    dates: list[str]
    values: list[float]

class SummaryIn(BaseModel):
    series: SeriesIn
    params: dict[str, Any] = {}



def _prep_series(series_in) -> Tuple[pd.Series, List[str], Optional[str]]:
    warnings: List[str] = []
    y = to_series(series_in.dates, series_in.values)

    # use provided freq if any; otherwise infer
    freq = getattr(series_in, 'freq', None)
    if freq is None:
        freq = infer_freq_safe(y.index)

    # basic type/length checks
    if len(y) < 5:
        warnings.append("Series length < 5; many analyses may be unreliable.")
    if y.isna().mean() > 0.4:
        warnings.append("More than 40% missing values; consider resampling/filling.")
    return y, warnings, freq

def _seasonal_period_from_inputs(freq: Optional[str], seasonal_period: Optional[int]) -> int:
    if seasonal_period is not None:
        return int(seasonal_period)
    return default_seasonal_period(freq)

def _as_float_list(x: np.ndarray) -> List[float]:
    return [float(v) for v in np.asarray(x).ravel()]

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"mae": None, "rmse": None, "mape": None, "smape": None}

    yt = y_true[mask]
    yp = y_pred[mask]

    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))

    # MAPE: avoid divide by 0
    denom = np.where(np.abs(yt) < 1e-12, np.nan, np.abs(yt))
    mape = np.nanmean(np.abs((yt - yp) / denom)) * 100.0

    smape = np.mean(2.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-12)) * 100.0

    return {"mae": safe_float(mae), "rmse": safe_float(rmse), "mape": safe_float(mape), "smape": safe_float(smape)}

def _fit_arima_grid(y: pd.Series, seasonal_period: int, seasonal: bool,
                    max_p: int, max_d: int, max_q: int,
                    max_P: int, max_D: int, max_Q: int) -> Tuple[Tuple[int,int,int], Tuple[int,int,int,int], List[str]]:
    warnings: List[str] = []
    best_aic = np.inf
    best_order = (0, 0, 0)
    best_sorder = (0, 0, 0, 0)

    # Small safe search space
    p_range = range(0, max_p + 1)
    d_range = range(0, max_d + 1)
    q_range = range(0, max_q + 1)

    if seasonal and seasonal_period >= 2:
        P_range = range(0, max_P + 1)
        D_range = range(0, max_D + 1)
        Q_range = range(0, max_Q + 1)
    else:
        P_range = [0]
        D_range = [0]
        Q_range = [0]

    y2 = y.dropna()
    if len(y2) < 20:
        warnings.append("Short series for auto-ARIMA grid search; results may be unstable.")

    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            sorder = (P, D, Q, seasonal_period if (seasonal and seasonal_period >= 2) else 0)
                            try:
                                model = SARIMAX(
                                    y2,
                                    order=(p, d, q),
                                    seasonal_order=sorder,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                res = model.fit(disp=False)
                                aic = res.aic
                                if np.isfinite(aic) and aic < best_aic:
                                    best_aic = aic
                                    best_order = (p, d, q)
                                    best_sorder = sorder
                            except Exception:
                                continue

    if not np.isfinite(best_aic):
        warnings.append("Auto-ARIMA failed to find a valid model; defaulting to (1,0,0).")
        best_order = (1, 0, 0)
        best_sorder = (0, 0, 0, 0)

    return best_order, best_sorder, warnings

def _forecast_index(y: pd.Series, h: int) -> List[Any]:
    if isinstance(y.index, pd.DatetimeIndex):
        # try to use index freq; fallback to step-by-step days
        freq = infer_freq_safe(y.index)
        if freq:
            start = y.index[-1]
            idx = pd.date_range(start=start, periods=h+1, freq=freq)[1:]
        else:
            idx = pd.date_range(start=y.index[-1], periods=h+1, freq="D")[1:]
        return [t.isoformat() for t in idx.to_pydatetime()]
    else:
        last = int(y.index[-1])
        return list(range(last + 1, last + 1 + h))

# ---------------------------
# A) Preprocess / clean
# ---------------------------

@router.post("/tsa/A_preprocess", response_model=ApiOut)
def tsa_a_preprocess(payload: PreprocessIn):
    y, warnings, _freq = _prep_series(payload.series)

    y, w1 = resample_if_datetime(y, payload.resample_freq)
    warnings += w1

    y, w2 = fill_missing(y, payload.fill_method)
    warnings += w2

    y = clip_values(y, payload.clip_min, payload.clip_max)

    result = {
        "series": to_jsonable_series(y),
        "missing_count": int(pd.isna(y).sum()),
        "n": int(len(y)),
        "freq_inferred": infer_freq_safe(y.index),
    }
    return ApiOut(ok=True, result=result, warnings=warnings)

# ---------------------------
# B) Summary stats
# ---------------------------

@router.post("/tsa/B_summary", response_model=ApiOut)
def tsa_b_summary(
    req: Optional[SummaryIn] = Body(None),
    payload: Optional[str] = Query(None),
):
    """
    Summary statistics.

    Accepts either:
      - JSON body: { "series": { "dates": [...], "values": [...] }, "params": {...} }
      - Query payload: ?payload=<json string>  (backward compatible)
    """
    if req is None:
        if payload is None:
            raise HTTPException(status_code=422, detail="Provide JSON body or ?payload=<json>")
        try:
            req = SummaryIn(**json.loads(payload))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid payload JSON: {e}")

    y, warnings, freq = _prep_series(req.series)
    y2 = y.dropna()

    if len(y2) == 0:
        return ApiOut(ok=False, result={"error": "All values are missing."}, warnings=warnings)

    desc = y2.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    result = {
        "n": int(len(y)),
        "n_valid": int(len(y2)),
        "freq_inferred": freq,
        "min": safe_float(desc.get("min")),
        "max": safe_float(desc.get("max")),
        "mean": safe_float(desc.get("mean")),
        "std": safe_float(desc.get("std")),
        "p05": safe_float(desc.get("5%")),
        "p25": safe_float(desc.get("25%")),
        "p50": safe_float(desc.get("50%")),
        "p75": safe_float(desc.get("75%")),
        "p95": safe_float(desc.get("95%")),
    }
    return ApiOut(ok=True, result=result, warnings=warnings)
# ---------------------------
# C) Outliers
# ---------------------------

@router.post("/tsa/C_outliers", response_model=ApiOut)
def tsa_c_outliers(payload: OutliersIn):
    y, warnings, _freq = _prep_series(payload.series)
    y2 = y.dropna()
    if len(y2) < 10:
        warnings.append("Very short series for outlier detection.")

    flags = pd.Series(False, index=y.index)

    if payload.method == "iqr":
        q1 = y2.quantile(0.25)
        q3 = y2.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - payload.iqr_k * iqr
        hi = q3 + payload.iqr_k * iqr
        flags.loc[y.index] = (y < lo) | (y > hi)
        method_info = {"method": "iqr", "q1": float(q1), "q3": float(q3), "lo": float(lo), "hi": float(hi)}
    else:
        z = zscore(y2.values, nan_policy="omit")
        z_idx = y2.index
        z_flags = np.abs(z) > payload.z_thresh
        flags.loc[z_idx] = z_flags
        method_info = {"method": "zscore", "z_thresh": float(payload.z_thresh)}

    outlier_points = []
    for idx, is_out in flags.items():
        if bool(is_out) and pd.notna(y.loc[idx]):
            outlier_points.append({
                "x": idx.isoformat() if isinstance(y.index, pd.DatetimeIndex) else int(idx),
                "y": float(y.loc[idx])
            })

    result = {
        "series": to_jsonable_series(y),
        "outliers": outlier_points,
        "outlier_count": int(flags.sum()),
        "method_info": method_info,
    }
    return ApiOut(ok=True, result=result, warnings=warnings)

# ---------------------------
# D) Decomposition (STL)
# ---------------------------

@router.post("/tsa/D_decompose", response_model=ApiOut)
def tsa_d_decompose(payload: DecomposeIn):
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()

    if len(y2) < 3:
        return ApiOut(ok=False, result={"error": "Not enough data."}, warnings=warnings)

    period = payload.period
    if period is None:
        period = default_seasonal_period(freq)

    # STL requires 2 <= period <= len/2
    if period < 2:
        period = 2
    if period > max(2, len(y2) // 2):
        period = max(2, len(y2) // 2)
        warnings.append("period adjusted to fit STL constraints.")

    try:
        stl = STL(y2, period=period, robust=payload.robust).fit()
        trend = stl.trend.reindex(y.index)
        seasonal = stl.seasonal.reindex(y.index)
        resid = stl.resid.reindex(y.index)

        result = {
            "period_used": int(period),
            "series": to_jsonable_series(y),
            "trend": to_jsonable_series(trend),
            "seasonal": to_jsonable_series(seasonal),
            "resid": to_jsonable_series(resid),
        }
        return ApiOut(ok=True, result=result, warnings=warnings)
    except Exception as e:
        return ApiOut(ok=False, result={"error": f"STL failed: {str(e)}"}, warnings=warnings)

# ---------------------------
# E) Seasonality / trend strength (from STL)
# ---------------------------

@router.post("/tsa/E_strength", response_model=ApiOut)
def tsa_e_strength(payload: DecomposeIn):
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()

    if len(y2) < 20:
        warnings.append("Short series; strength metrics may be noisy.")

    period = payload.period or default_seasonal_period(freq)
    period = max(2, min(period, max(2, len(y2) // 2)))

    try:
        stl = STL(y2, period=period, robust=True).fit()
        resid = stl.resid
        trend = stl.trend
        seas = stl.seasonal

        # Hyndman-style strength measures
        var_r = np.var(resid, ddof=1) if len(resid) > 1 else np.nan
        var_tr = np.var((trend + resid), ddof=1) if len(resid) > 1 else np.nan
        var_sr = np.var((seas + resid), ddof=1) if len(resid) > 1 else np.nan

        trend_strength = 1.0 - (var_r / var_tr) if np.isfinite(var_r) and np.isfinite(var_tr) and var_tr > 0 else None
        seasonal_strength = 1.0 - (var_r / var_sr) if np.isfinite(var_r) and np.isfinite(var_sr) and var_sr > 0 else None

        result = {
            "period_used": int(period),
            "trend_strength": safe_float(trend_strength),
            "seasonal_strength": safe_float(seasonal_strength),
        }
        return ApiOut(ok=True, result=result, warnings=warnings)
    except Exception as e:
        return ApiOut(ok=False, result={"error": f"Strength calc failed: {str(e)}"}, warnings=warnings)

# ---------------------------
# F) Stationarity tests (ADF, KPSS)
# ---------------------------

@router.post("/tsa/F_stationarity", response_model=ApiOut)
def tsa_f_stationarity(payload: StationarityIn):
    y, warnings, _freq = _prep_series(payload.series)
    y2 = y.dropna()

    if len(y2) < 12:
        warnings.append("Short series; stationarity test p-values may be unreliable.")

    out: Dict[str, Any] = {}
    try:
        adf = adfuller(y2.values, regression=payload.regression, autolag="AIC")
        out["adf"] = {
            "stat": float(adf[0]),
            "pvalue": float(adf[1]),
            "usedlag": int(adf[2]),
            "nobs": int(adf[3]),
            "crit": {k: float(v) for k, v in adf[4].items()},
        }
    except Exception as e:
        out["adf_error"] = str(e)

    try:
        k = kpss(y2.values, regression=payload.kpss_regression, nlags="auto")
        out["kpss"] = {
            "stat": float(k[0]),
            "pvalue": float(k[1]),
            "nlags": int(k[2]),
            "crit": {k: float(v) for k, v in k[3].items()},
        }
    except Exception as e:
        out["kpss_error"] = str(e)

    return ApiOut(ok=True, result=out, warnings=warnings)

# ---------------------------
# G) ACF/PACF + Ljung-Box
# ---------------------------

@router.post("/tsa/G_autocorr", response_model=ApiOut)
def tsa_g_autocorr(payload: AutocorrIn):
    y, warnings, _freq = _prep_series(payload.series)
    y2 = y.dropna()

    nlags = int(max(1, payload.nlags))
    nlags = min(nlags, max(1, len(y2) - 2))

    ac = acf(y2.values, nlags=nlags, fft=True)
    pc = pacf(y2.values, nlags=nlags, method="ywm")

    # Ljung-Box
    try:
        lb = acorr_ljungbox(y2.values, lags=[min(10, nlags), min(20, nlags)], return_df=True)
        lj = [{"lag": int(i), "stat": float(r["lb_stat"]), "pvalue": float(r["lb_pvalue"])} for i, r in lb.iterrows()]
    except Exception:
        lj = []

    result = {
        "nlags": int(nlags),
        "acf": _as_float_list(ac),
        "pacf": _as_float_list(pc),
        "ljung_box": lj,
    }
    return ApiOut(ok=True, result=result, warnings=warnings)

# ---------------------------
# H) Spectrum / Periodogram
# ---------------------------

@router.post("/tsa/H_spectrum", response_model=ApiOut)
def tsa_h_spectrum(payload: SpectrumIn):
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna().values.astype("float64")

    if len(y2) < 8:
        return ApiOut(ok=False, result={"error": "Not enough data for spectrum."}, warnings=warnings)

    if payload.detrend:
        y2 = signal.detrend(y2)

    f, pxx = signal.periodogram(y2, scaling="density")
    # remove zero frequency for display
    f = f[1:]
    pxx = pxx[1:]

    result = {
        "freq": _as_float_list(f),
        "power": _as_float_list(pxx),
        "note": "freq is in cycles/sample (convert using your sampling interval).",
        "freq_inferred": freq,
    }
    return ApiOut(ok=True, result=result, warnings=warnings)

# ---------------------------
# I) Cross-correlation (two series)
# ---------------------------

@router.post("/tsa/I_xcorr", response_model=ApiOut)
def tsa_i_xcorr(payload: XCorrIn):
    x, w1, _fx = _prep_series(payload.x)
    y, w2, _fy = _prep_series(payload.y)
    warnings = w1 + w2

    # align on index intersection if datetime
    if isinstance(x.index, pd.DatetimeIndex) and isinstance(y.index, pd.DatetimeIndex):
        df = pd.DataFrame({"x": x, "y": y}).dropna()
        xs = df["x"].values
        ys = df["y"].values
    else:
        n = min(len(x), len(y))
        xs = x.values[:n]
        ys = y.values[:n]
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]

    if len(xs) < 10:
        return ApiOut(ok=False, result={"error": "Not enough overlapping data for xcorr."}, warnings=warnings)

    max_lag = int(max(1, payload.max_lag))
    max_lag = min(max_lag, len(xs) - 2)

    xs = (xs - xs.mean()) / (xs.std() + 1e-12)
    ys = (ys - ys.mean()) / (ys.std() + 1e-12)

    corr_full = np.correlate(xs, ys, mode="full") / len(xs)
    mid = len(corr_full) // 2
    lags = np.arange(-max_lag, max_lag + 1)
    corr = corr_full[mid - max_lag: mid + max_lag + 1]

    result = {
        "lags": lags.astype(int).tolist(),
        "xcorr": _as_float_list(corr),
        "definition": "xcorr(lag) = corr(x[t], y[t+lag]) on standardized series",
    }
    return ApiOut(ok=True, result=result, warnings=warnings)

# ---------------------------
# J) ARIMA/SARIMA forecast
# ---------------------------

@router.post("/tsa/J_arima_forecast", response_model=ApiOut)
def tsa_j_arima(payload: ForecastIn):
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    h = int(max(1, payload.horizon))

    sp = _seasonal_period_from_inputs(freq, payload.seasonal_period)

    if payload.auto:
        order, sorder, w = _fit_arima_grid(
            y2, seasonal_period=sp, seasonal=payload.seasonal,
            max_p=payload.max_p, max_d=payload.max_d, max_q=payload.max_q,
            max_P=payload.max_P, max_D=payload.max_D, max_Q=payload.max_Q
        )
        warnings += w
    else:
        order = (1, 0, 0)
        sorder = (0, 0, 0, 0)

    try:
        model = SARIMAX(
            y2,
            order=order,
            seasonal_order=sorder,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=h)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)

        x_fc = _forecast_index(y2, h)
        result = {
            "order": {"p": order[0], "d": order[1], "q": order[2]},
            "seasonal_order": {"P": sorder[0], "D": sorder[1], "Q": sorder[2], "s": sorder[3]},
            "aic": safe_float(res.aic),
            "forecast": {"x": x_fc, "y": _as_float_list(mean.values)},
            "conf_int_95": {
                "lower": _as_float_list(ci.iloc[:, 0].values),
                "upper": _as_float_list(ci.iloc[:, 1].values),
            },
        }
        return ApiOut(ok=True, result=result, warnings=warnings)
    except Exception as e:
        return ApiOut(ok=False, result={"error": f"ARIMA fit/forecast failed: {str(e)}"}, warnings=warnings)

# ---------------------------
# K) ETS (Holt-Winters) forecast
# ---------------------------

@router.post("/tsa/K_ets_forecast", response_model=ApiOut)
def tsa_k_ets(payload: ETSIn):
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    h = int(max(1, payload.horizon))
    sp = _seasonal_period_from_inputs(freq, payload.seasonal_period)

    if len(y2) < max(10, 2 * sp):
        warnings.append("Short series for ETS with seasonality; consider disabling seasonal.")

    trend = None if payload.trend == "none" else payload.trend
    seasonal = None if payload.seasonal == "none" else payload.seasonal

    try:
        model = ExponentialSmoothing(
            y2,
            trend=trend,
            damped_trend=payload.damped_trend if trend is not None else False,
            seasonal=seasonal,
            seasonal_periods=sp if seasonal is not None else None
        )
        res = model.fit(optimized=True)
        mean = res.forecast(h)

        x_fc = _forecast_index(y2, h)
        result = {
            "params": {k: (float(v) if np.isfinite(v) else None) for k, v in res.params.items() if isinstance(v, (int, float, np.floating))},
            "forecast": {"x": x_fc, "y": _as_float_list(mean.values)},
            "seasonal_period_used": int(sp) if seasonal is not None else None,
            "trend": payload.trend,
            "seasonal": payload.seasonal,
            "damped_trend": payload.damped_trend,
        }
        return ApiOut(ok=True, result=result, warnings=warnings)
    except Exception as e:
        return ApiOut(ok=False, result={"error": f"ETS failed: {str(e)}"}, warnings=warnings)

# ---------------------------
# L) Theta forecast
# ---------------------------

@router.post("/tsa/L_theta_forecast", response_model=ApiOut)
def tsa_l_theta(payload: ThetaIn):
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    h = int(max(1, payload.horizon))
    sp = _seasonal_period_from_inputs(freq, payload.seasonal_period)

    try:
        tm = ThetaModel(y2, period=sp)
        res = tm.fit()
        mean = res.forecast(h)

        x_fc = _forecast_index(y2, h)
        result = {
            "seasonal_period_used": int(sp),
            "forecast": {"x": x_fc, "y": _as_float_list(mean.values)},
        }
        return ApiOut(ok=True, result=result, warnings=warnings)
    except Exception as e:
        return ApiOut(ok=False, result={"error": f"Theta failed: {str(e)}"}, warnings=warnings)

# ---------------------------
# M) Backtest + metrics
# ---------------------------

def _fit_and_forecast(model_name: str, y_train: pd.Series, h: int, sp: int) -> np.ndarray:
    if model_name == "naive":
        return np.full(h, y_train.iloc[-1], dtype="float64")

    if model_name == "seasonal_naive":
        if sp < 1 or len(y_train) < sp:
            return np.full(h, y_train.iloc[-1], dtype="float64")
        last_season = y_train.iloc[-sp:].values
        reps = int(np.ceil(h / sp))
        return np.tile(last_season, reps)[:h].astype("float64")

    if model_name == "theta":
        res = ThetaModel(y_train, period=max(1, sp)).fit()
        return res.forecast(h).values.astype("float64")

    if model_name == "ets":
        # safe default
        seasonal = "add" if (sp >= 2 and len(y_train) >= 2 * sp) else None
        model = ExponentialSmoothing(
            y_train, trend="add", seasonal=seasonal, seasonal_periods=(sp if seasonal else None)
        )
        res = model.fit(optimized=True)
        return res.forecast(h).values.astype("float64")

    # default: arima (small fixed to be fast for backtest)
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, sp) if (sp >= 2 and len(y_train) >= 3 * sp) else (0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)
    return res.get_forecast(steps=h).predicted_mean.values.astype("float64")

@router.post("/tsa/M_backtest", response_model=ApiOut)
def tsa_m_backtest(payload: BacktestIn):
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()

    h = int(max(1, payload.horizon))
    sp = _seasonal_period_from_inputs(freq, payload.seasonal_period)

    n = len(y2)
    if n < (h + 10):
        return ApiOut(ok=False, result={"error": "Series too short for backtest."}, warnings=warnings)

    if payload.initial_train_size is None:
        initial = max(20, n // 2)
    else:
        initial = int(payload.initial_train_size)

    initial = max(initial, 10)
    initial = min(initial, n - h)

    step = int(max(1, payload.step))

    y_true_all = []
    y_pred_all = []
    cut_points = []

    for start in range(initial, n - h + 1, step):
        y_train = y2.iloc[:start]
        y_test = y2.iloc[start:start + h]

        try:
            yhat = _fit_and_forecast(payload.model, y_train, h, sp)
        except Exception as e:
            warnings.append(f"Backtest step failed at t={start}: {str(e)}")
            continue

        y_true_all.append(y_test.values.astype("float64"))
        y_pred_all.append(yhat)
        cut_points.append(start)

    if len(y_true_all) == 0:
        return ApiOut(ok=False, result={"error": "Backtest failed for all windows."}, warnings=warnings)

    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    met = _metrics(yt, yp)

    result = {
        "model": payload.model,
        "horizon": h,
        "initial_train_size": initial,
        "step": step,
        "n_folds": int(len(cut_points)),
        "metrics": met,
    }
    return ApiOut(ok=True, result=result, warnings=warnings)
