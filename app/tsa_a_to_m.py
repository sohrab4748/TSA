
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Query, HTTPException
import os
from datetime import datetime
from pydantic import BaseModel, Field

import pywt
from scipy.signal import detrend as sp_detrend

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
    SeriesIn,
    PreprocessIn, SummaryIn, OutliersIn, DecomposeIn, StrengthIn,
    StationarityIn, AutocorrIn, SpectrumIn, XCorrIn, ForecastIn,
    ETSIn, ThetaIn, BacktestIn, WaveletIn,
)
from app.utils import (
    to_series, infer_freq_safe, default_seasonal_period,
    resample_if_datetime, fill_missing, clip_values, to_jsonable_series, safe_float
)

router = APIRouter()

# ---------------------------
# Helpers
# ---------------------------
import json

def _prep_series(series):
    import pandas as pd

    # Convert date/value arrays to a pandas Series with datetime index
    x = pd.to_datetime(series.dates)
    y = pd.to_numeric(pd.Series(series.values), errors="coerce")
    y.index = x
    y = y.sort_index()

    # Simple cleaning and metadata
    warnings = []
    freq = pd.infer_freq(y.index)
    return y, warnings, freq

def _as_float_list(arr: Any) -> List[Optional[float]]:
    """Convert numpy/pandas arrays to a JSON-safe list of floats (NaN/Inf -> None)."""
    if arr is None:
        return []
    try:
        if isinstance(arr, (list, tuple)):
            return [safe_float(v) for v in arr]
        if hasattr(arr, "tolist"):
            arr = arr.tolist()
            if isinstance(arr, (list, tuple)):
                return [safe_float(v) for v in arr]
            return [safe_float(arr)]
        return [safe_float(arr)]
    except Exception:
        try:
            return [safe_float(v) for v in list(arr)]
        except Exception:
            return []

def _seasonal_period_from_inputs(freq_inferred: Optional[str], seasonal_period: Optional[int]) -> int:
    """Choose a seasonal period from explicit input or inferred frequency."""
    if seasonal_period is not None:
        try:
            sp = int(seasonal_period)
            return sp if sp > 0 else 0
        except Exception:
            return 0
    try:
        sp = int(default_seasonal_period(freq_inferred))
        return sp if sp > 0 else 0
    except Exception:
        return 0

def _forecast_index(y: pd.Series, h: int) -> List[Any]:
    """Build forecast x-axis values aligned with the input series index."""
    h = int(max(1, h))
    idx = y.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) >= 1:
        last = idx[-1]
        # Try inferred freq first
        freq = infer_freq_safe(idx)
        if freq:
            try:
                rng = pd.date_range(start=last, periods=h + 1, freq=freq)
                return [d.isoformat() for d in rng[1:]]
            except Exception:
                pass
        # Fallback to median delta (or 1 day)
        delta = pd.Timedelta(days=1)
        try:
            if len(idx) >= 2:
                d = idx.to_series().diff().dropna()
                if len(d) > 0:
                    md = d.median()
                    if isinstance(md, pd.Timedelta) and md > pd.Timedelta(0):
                        delta = md
        except Exception:
            pass
        return [(last + delta * (i + 1)).isoformat() for i in range(h)]

    # Non-datetime index: use integer steps if possible
    try:
        last_val = idx[-1]
        last_num = float(last_val)
        return [last_num + (i + 1) for i in range(h)]
    except Exception:
        return list(range(1, h + 1))

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute common forecast metrics on finite pairs."""
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return {"n": 0}
    yt = yt[mask]
    yp = yp[mask]
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # MAPE / sMAPE
    denom = np.where(np.abs(yt) > 1e-12, np.abs(yt), np.nan)
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)

    smape_d = (np.abs(yt) + np.abs(yp))
    smape = float(np.nanmean(2.0 * np.abs(err) / np.where(smape_d > 1e-12, smape_d, np.nan)) * 100.0)

    return {
        "n": int(len(yt)),
        "mae": safe_float(mae),
        "rmse": safe_float(rmse),
        "mape_pct": safe_float(mape),
        "smape_pct": safe_float(smape),
    }

def _fit_arima_grid(
    y: pd.Series,
    seasonal_period: int,
    seasonal: bool,
    max_p: int, max_d: int, max_q: int,
    max_P: int, max_D: int, max_Q: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], List[str]]:
    """Fast, bounded AIC search for ARIMA/SARIMA.

    Notes:
    - Render/low-CPU environments will time out if we do a large SARIMAX grid.
    - We therefore cap the sample size, cap iterations, and try a small candidate set.
    """
    warnings: List[str] = []

    yv = pd.to_numeric(y, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(yv) < 8:
        warnings.append("Series too short for auto-ARIMA; using ARIMA(0,1,0).")
        return (0, 1, 0), (0, 0, 0, 0), warnings

    # Speed guard: fit on the most recent window only
    MAX_N = 2000
    if len(yv) > MAX_N:
        warnings.append(f"Auto-ARIMA: using last {MAX_N} points (n={len(yv)}) for speed.")
        yv = yv.iloc[-MAX_N:]

    # Prefer non-seasonal unless explicitly requested AND period is sensible
    seasonal_period = int(seasonal_period or 0)
    use_seasonal = bool(seasonal) and seasonal_period >= 2 and seasonal_period <= 366

    # Candidate sets (small but practical)
    p_cand = [0, 1, 2]
    q_cand = [0, 1, 2]
    d_cand = [0, 1] if int(max_d) >= 1 else [0]

    # Keep seasonal search extremely small
    seasonal_candidates = [(0, 0, 0, 0)]
    if use_seasonal:
        seasonal_candidates.append((1, 0, 1, seasonal_period))

    best_aic = np.inf
    best_order = (0, 1, 0)
    best_sorder = (0, 0, 0, 0)

    tried = 0
    max_tries = 24  # hard cap

    for d in d_cand:
        for p in p_cand:
            for q in q_cand:
                if p == 0 and d == 0 and q == 0:
                    continue
                for sorder in seasonal_candidates:
                    tried += 1
                    if tried > max_tries:
                        warnings.append("Auto-ARIMA candidate cap reached; using best found so far.")
                        return best_order, best_sorder, warnings
                    try:
                        D = int(sorder[1])
                        simple_diff = (int(d) > 0) or (D > 0)
                        model = SARIMAX(
                            yv,
                            order=(int(p), int(d), int(q)),
                            seasonal_order=tuple(map(int, sorder)),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            simple_differencing=simple_diff,
                            low_memory=True,
                        )
                        res = model.fit(disp=False, maxiter=60)
                        aic = float(res.aic) if np.isfinite(res.aic) else np.inf
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (int(p), int(d), int(q))
                            best_sorder = tuple(map(int, sorder))
                    except Exception:
                        continue

    if not np.isfinite(best_aic):
        warnings.append("Auto-ARIMA could not fit any candidate; using ARIMA(0,1,0).")
        return (0, 1, 0), (0, 0, 0, 0), warnings

    return best_order, best_sorder, warnings

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
    
@router.post("/tsa/N_wavelet_cwt", response_model=ApiOut)
def tsa_n_wavelet_cwt(payload: WaveletIn):
    y, warnings, _freq = _prep_series(payload.series)
    y2 = y.dropna()

    n = int(len(y2))
    if n < 8:
        return ApiOut(ok=False, result={"error": "Wavelet needs at least ~8 points."}, warnings=warnings)

    # helper to read param from root OR payload.params
    def P(name, default):
        v = getattr(payload, name, None)
        if v is not None:
            return v
        if isinstance(payload.params, dict) and name in payload.params:
            return payload.params.get(name)
        return default

    wavelet = str(P("wavelet", "morl"))
    min_period = float(P("min_period", 2.0))
    max_period = P("max_period", None)
    max_period = float(max_period) if max_period not in (None, "", 0) else float(max(4.0, n / 2))
    num_scales = int(P("num_scales", 64))
    do_detrend = bool(P("detrend", True))
    normalize = bool(P("normalize", True))
    max_time_points = int(P("max_time_points", 500))

    # estimate dt (in days) from datetime index; fallback to 1
    dt = 1.0
    try:
        idx = y2.index
        if len(idx) >= 2:
            d = np.diff(idx.values.astype("datetime64[ns]"))
            dt_days = np.median(d) / np.timedelta64(1, "D")
            if np.isfinite(dt_days) and dt_days > 0:
                dt = float(dt_days)
    except Exception:
        pass

    # prepare values
    vals = y2.values.astype(float)
    if do_detrend:
        vals = sp_detrend(vals, type="linear")
    else:
        vals = vals - np.nanmean(vals)

    if normalize:
        s = np.nanstd(vals)
        if np.isfinite(s) and s > 0:
            vals = vals / s

    # choose periods (log-spaced)
    max_period = max(max_period, min_period * 1.01)
    num_scales = max(8, min(num_scales, 256))
    periods = np.geomspace(min_period, max_period, num_scales)

    # scales -> use central frequency
    try:
        cf = pywt.central_frequency(wavelet)
    except Exception:
        cf = pywt.central_frequency("morl")
        warnings.append(f"Unknown wavelet '{wavelet}', using 'morl' instead.")
        wavelet = "morl"

    # IMPORTANT:
    # - Users specify min/max period in *time steps* (e.g., 2..16 months for monthly series).
    # - We keep `sampling_period=dt` so returned freqs/periods can be expressed in days,
    #   but compute scales in *steps* so we don't create extremely small scales for coarse
    #   (monthly/yearly) series.
    #
    # Relationship (PyWavelets): freq = central_frequency / (scale * sampling_period)
    # => period_days = (scale * sampling_period) / central_frequency
    # If period is provided in steps, period_days = period_steps * dt_days
    # => scale = period_steps * central_frequency
    scales = periods * cf

    # compute CWT (catch scale errors and return a clean API error instead of a hard 500)
    try:
        coef, freqs = pywt.cwt(vals, scales, wavelet, sampling_period=dt)
    except Exception as e:
        return ApiOut(ok=False, result={"error": f"Wavelet CWT failed: {e}"}, warnings=warnings)
    power = np.abs(coef) ** 2

    # periods implied by returned freqs
    # - freqs: cycles per day
    # - periods_out_days: days
    # - periods_out_steps: time-steps (matches user inputs)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods_out_days = np.where(freqs > 0, 1.0 / freqs, np.nan)
        periods_out_steps = np.where(np.isfinite(dt) and dt > 0, periods_out_days / dt, periods_out_days)

    # downsample time if large
    x = y2.index.astype(str).tolist()
    if n > max_time_points:
        keep = np.linspace(0, n - 1, max_time_points).astype(int)
        x = [x[i] for i in keep]
        power = power[:, keep]

    # global wavelet spectrum
    gws = np.nanmean(power, axis=1)

    # ensure JSON-safe (no NaN/Inf). Power is large; convert efficiently.
    power_list = power.tolist()
    for i in range(len(power_list)):
        row = power_list[i]
        for j in range(len(row)):
            v = row[j]
            # python float('nan') can break ORJSON; normalize to None
            if v is None:
                continue
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    row[j] = None
            except Exception:
                row[j] = None

    result = {
        "wavelet": wavelet,
        "dt_days": dt,
        "n": int(n),
        "time": x,
        "period_unit": "steps",
        "periods": [safe_float(p) for p in periods_out_steps.tolist()],
        "periods_days": [safe_float(p) for p in periods_out_days.tolist()],
        "power": power_list,             # shape: [n_scales][n_time]
        "global_spectrum": [safe_float(v) for v in gws.tolist()],
        "notes": {
            "suggest_plot": "Use log10(1+power) for heatmap; 'periods' are in time-steps; use periods_days for physical time."
        }
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
        kpss_res = kpss(y2.values, regression=payload.kpss_regression, nlags="auto")
        out["kpss"] = {
            "stat": float(kpss_res[0]),
            "pvalue": float(kpss_res[1]),
            "nlags": int(kpss_res[2]),
            "crit": {kk: float(vv) for kk, vv in kpss_res[3].items()},
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
    n = int(len(y2))

    if n < 3:
        return ApiOut(ok=False, result={"error": "Need at least 3 observations for ACF/PACF."}, warnings=warnings)

    # Accept nlags either as payload.nlags OR payload.params["nlags"]
    nlags_in = getattr(payload, "nlags", None)
    if nlags_in is None and hasattr(payload, "params") and isinstance(payload.params, dict):
        nlags_in = payload.params.get("nlags", None)

    nlags_req = int(max(1, nlags_in or 1))

    # ACF can go up to n-2 (safe), PACF has stricter constraint (< n/2 for common methods)
    nlags = min(nlags_req, max(1, n - 2))

    nlags_pacf_max = max(1, (n // 2) - 1)  # ensures nlags < n/2
    if nlags > nlags_pacf_max:
        warnings.append(f"nlags reduced from {nlags} to {nlags_pacf_max} (PACF requires nlags < n/2).")
        nlags = nlags_pacf_max

    try:
        ac = acf(y2.values, nlags=nlags, fft=True)
        pc = pacf(y2.values, nlags=nlags, method="ywm")

        # optional Ljung-Box
        lj = []
        try:
            lags = sorted(set([min(10, nlags), min(20, nlags)]))
            lb = acorr_ljungbox(y2.values, lags=lags, return_df=True)
            for idx, row in lb.iterrows():
                lj.append({
                    "lag": int(idx),
                    "stat": safe_float(row.get("lb_stat")),
                    "pvalue": safe_float(row.get("lb_pvalue")),
                })
        except Exception:
            pass

        return ApiOut(
            ok=True,
            result={
                "nlags": int(nlags),
                "acf": _as_float_list(ac),
                "pacf": _as_float_list(pc),
                "ljung_box": lj,
            },
            warnings=warnings,
        )

    except Exception as e:
        warnings.append(f"G_autocorr failed: {str(e)}")
        return ApiOut(ok=False, result={"error": str(e)}, warnings=warnings)



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

# J - ARIMA forecast (SARIMAX)
@router.post("/tsa/J_arima_forecast", response_model=ApiOut)
def tsa_j_arima(payload: ForecastIn):
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    # Speed guard: ARIMA on very long series is slow on low-CPU servers
    if y2.shape[0] > 2000:
        warnings.append(f"ARIMA: input series is long (n={int(y2.shape[0])}); fitting on last 2000 points for speed.")
        y2 = y2.iloc[-2000:]
    n = int(y2.shape[0])
    h = int(max(1, payload.horizon))

    # seasonal period inferred / provided
    sp = _seasonal_period_from_inputs(freq, payload.seasonal_period)

    # --- Safety rules for short series (prevents unstable models) ---
    # With very short series, auto-ARIMA can easily pick unstable AR terms.
    # Restrict the search hard and (optionally) disable seasonal.
    seasonal = bool(getattr(payload, "seasonal", False))
    if n < 20:
        warnings.append(f"Very short series (n={n}); restricting ARIMA search to avoid unstable fits.")
        max_p = min(int(getattr(payload, "max_p", 1)), 1)
        max_d = min(int(getattr(payload, "max_d", 1)), 1)
        max_q = min(int(getattr(payload, "max_q", 1)), 1)
        max_P = max_D = max_Q = 0
        seasonal = False
        sp = 0
    else:
        max_p = int(getattr(payload, "max_p", 3))
        max_d = int(getattr(payload, "max_d", 1))
        max_q = int(getattr(payload, "max_q", 3))
        max_P = int(getattr(payload, "max_P", 1))
        max_D = int(getattr(payload, "max_D", 1))
        max_Q = int(getattr(payload, "max_Q", 1))

    # seasonal_order must be (P,D,Q,s) with s>=2 to matter
    def normalize_sorder(sorder):
        P, D, Q, s = sorder
        if (not seasonal) or (s is None) or (int(s) < 2) or (P == 0 and D == 0 and Q == 0):
            return (0, 0, 0, 0)
        return (int(P), int(D), int(Q), int(s))

    # --- Choose order / seasonal_order ---
    if getattr(payload, "auto", True):
        order, sorder, w = _fit_arima_grid(
            y2,
            seasonal_period=sp,
            seasonal=seasonal,
            max_p=max_p, max_d=max_d, max_q=max_q,
            max_P=max_P, max_D=max_D, max_Q=max_Q
        )
        warnings += w
    else:
        # If you later add manual order fields to ForecastIn, use them here.
        order = (1, 0, 0)
        sorder = (0, 0, 0, 0)

    sorder = normalize_sorder(sorder)

    def fit_and_forecast(ord_, sord_):
        d = int(ord_[1])
        D = int(sord_[1]) if sord_ else 0
        simple_diff = (d > 0) or (D > 0)

        model = SARIMAX(
            y2,
            order=tuple(map(int, ord_)),
            seasonal_order=tuple(map(int, sord_)),
            enforce_stationarity=False,
            enforce_invertibility=False,
            simple_differencing=simple_diff,
            low_memory=True,
        )
        res = model.fit(disp=False, maxiter=60)
        fc = res.get_forecast(steps=h)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        return res, mean, ci

    try:
        res, mean, ci = fit_and_forecast(order, sorder)

        # --- Explosion guard ---
        # If forecast magnitude explodes relative to observed scale, fall back to a safe baseline.
        y_scale = float(np.nanmax(np.abs(y2.values))) if n > 0 else 1.0
        y_scale = max(y_scale, 1e-9)
        mmax = float(np.nanmax(np.abs(mean.values))) if len(mean.values) else 0.0

        if (not np.isfinite(mmax)) or (mmax / y_scale > 1e6):
            warnings.append(
                "ARIMA forecast magnitude exploded (unstable fit). "
                "Falling back to ARIMA(0,1,0). Provide more data or reduce max_p/max_q."
            )
            order = (0, 1, 0)
            sorder = (0, 0, 0, 0)
            res, mean, ci = fit_and_forecast(order, sorder)

        x_fc = _forecast_index(y2, h)

        result = {
            "order": {"p": int(order[0]), "d": int(order[1]), "q": int(order[2])},
            "seasonal_order": {"P": int(sorder[0]), "D": int(sorder[1]), "Q": int(sorder[2]), "s": int(sorder[3])},
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
# K) ETS (Exponential Smoothing) forecast
# ---------------------------

@router.post("/tsa/K_ets_forecast", response_model=ApiOut)
def tsa_k_ets(payload: ETSIn):
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    n = int(y2.shape[0])

    # Frontend may send steps; schema may use horizon
    h = int(max(1, int(getattr(payload, "horizon", getattr(payload, "steps", 30)))))

    sp = _seasonal_period_from_inputs(freq, getattr(payload, "seasonal_period", None))

    # Options (accept common names safely)
    trend = getattr(payload, "trend", None)
    seasonal = getattr(payload, "seasonal", None)
    damped_trend = bool(getattr(payload, "damped_trend", getattr(payload, "damped", False)))

    # Normalize option values
    trend = trend if trend in ("add", "mul") else None
    seasonal = seasonal if seasonal in ("add", "mul") else None

    # Safety: don't allow seasonal if series too short
    if seasonal is not None:
        if sp is None or int(sp) < 2:
            warnings.append("ETS seasonal requested but seasonal_period is invalid; disabling seasonal component.")
            seasonal = None
        elif n < 2 * int(sp):
            warnings.append(f"ETS seasonal requested but series is too short for seasonal_period={int(sp)} (n={n}); disabling seasonal component.")
            seasonal = None

    # Safety: multiplicative needs strictly positive data
    if (trend == "mul" or seasonal == "mul") and (y2.min() <= 0):
        warnings.append("ETS multiplicative components require all values > 0; switching 'mul' components to 'add'.")
        if trend == "mul":
            trend = "add"
        if seasonal == "mul":
            seasonal = "add"

    try:
        model = ExponentialSmoothing(
            y2,
            trend=trend,
            damped_trend=(damped_trend if trend else False),
            seasonal=seasonal,
            seasonal_periods=(int(sp) if seasonal else None),
            initialization_method="estimated",
        )

        # Keep it fast for web use
        res = model.fit(optimized=True)

        mean = res.forecast(h)
        x_fc = _forecast_index(y2, h)

        # Approximate 95% CI using residual std (simple + fast).
        fitted = getattr(res, "fittedvalues", None)
        sigma = float("nan")
        if fitted is not None:
            resid = (y2 - fitted).dropna()
            if resid.shape[0] >= 2:
                sigma = float(resid.std(ddof=1))

        z = 1.96
        lower = mean - (z * sigma) if np.isfinite(sigma) else None
        upper = mean + (z * sigma) if np.isfinite(sigma) else None

        result = {
            "model": {
                "type": "ETS",
                "trend": trend,
                "seasonal": seasonal,
                "damped_trend": bool(damped_trend if trend else False),
                "seasonal_period_used": int(sp) if seasonal else None,
            },
            "horizon": h,
            "forecast": {"x": x_fc, "y": _as_float_list(mean.values)},
        }
        if lower is not None and upper is not None:
            result["conf_int_95"] = {
                "lower": _as_float_list(lower.values if hasattr(lower, "values") else lower),
                "upper": _as_float_list(upper.values if hasattr(upper, "values") else upper),
            }

        return ApiOut(ok=True, result=result, warnings=warnings)

    except Exception as e:
        return ApiOut(ok=False, result={"error": f"ETS fit/forecast failed: {str(e)}"}, warnings=warnings)

# ---------------------------
# L) Theta forecast
# ---------------------------

@router.post("/tsa/L_theta_forecast", response_model=ApiOut)
def tsa_l_theta(payload: ThetaIn):
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()

    # Accept either `horizon` or `steps` (frontend uses "Forecast steps")
    _h = getattr(payload, "horizon", None)
    if _h is None:
        _h = getattr(payload, "steps", None)
    h = int(max(1, _h if _h is not None else 12))

    sp = _seasonal_period_from_inputs(freq, getattr(payload, "seasonal_period", None))
    sp = int(max(1, sp))

    try:
        tm = ThetaModel(y2, period=sp)
        res = tm.fit()

        mean = res.forecast(h)
        x_fc = _forecast_index(y2, h)

        result = {
            "model": {
                "type": "Theta",
                "seasonal_period_used": int(sp),
            },
            "horizon": int(h),
            "forecast": {"x": x_fc, "y": _as_float_list(mean.values)},
        }

        # 95% prediction intervals (when available in statsmodels ThetaModelResults)
        try:
            pi = res.prediction_intervals(steps=h, alpha=0.05)
            if pi is not None and "lower" in pi.columns and "upper" in pi.columns:
                result["conf_int_95"] = {
                    "lower": _as_float_list(pi["lower"].values),
                    "upper": _as_float_list(pi["upper"].values),
                }
        except Exception:
            # Not fatal: keep forecast only
            warnings.append("Theta: prediction intervals not available; returned forecast only.")

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


# ---------------------------
# AI Interpretation (Gemini): /analysis/ai/interpret (mounted via /analysis prefix in main)
# ---------------------------

class AIInterpretIn(BaseModel):
    provider: str = Field(default="gemini")
    style: str = Field(default="technical")  # technical | simple | executive
    extra_instruction: str = Field(default="")
    series: SeriesIn
    series_meta: Dict[str, Any] = Field(default_factory=dict)
    analyses: List[Dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = Field(default=None)  # optional override, e.g. "gemini-2.0-flash"
    debug: bool = Field(default=False)

def _compact_series_for_ai(y: pd.Series, max_points: int = 300) -> Dict[str, Any]:
    # Return a compact representation of the series to keep prompts small
    y2 = y.copy()
    y2 = pd.to_numeric(y2, errors="coerce").dropna()
    if len(y2) == 0:
        return {"n": 0, "preview": []}
    if len(y2) > max_points:
        y2 = y2.iloc[-max_points:]

    rows = []
    for idx, val in y2.items():
        if isinstance(idx, (pd.Timestamp, datetime)):
            t = pd.Timestamp(idx).isoformat()
        else:
            t = str(idx)
        rows.append([t, safe_float(val)])
    return {"n": int(len(y2)), "preview": rows}

def _build_ai_prompt(payload: AIInterpretIn, y: pd.Series, freq: Optional[str], warnings: List[str]):
    # Returns (system_instruction, user_prompt, compact_context)
    y_clean = pd.to_numeric(y, errors="coerce")
    stats = {
        "n": int(len(y_clean)),
        "start": y_clean.index[0].isoformat() if isinstance(y_clean.index, pd.DatetimeIndex) and len(y_clean) else None,
        "end": y_clean.index[-1].isoformat() if isinstance(y_clean.index, pd.DatetimeIndex) and len(y_clean) else None,
        "freq_inferred": freq,
        "min": safe_float(np.nanmin(y_clean.values)) if len(y_clean) else None,
        "max": safe_float(np.nanmax(y_clean.values)) if len(y_clean) else None,
        "mean": safe_float(np.nanmean(y_clean.values)) if len(y_clean) else None,
        "std": safe_float(np.nanstd(y_clean.values)) if len(y_clean) else None,
    }

    compact = {
        "series_meta": payload.series_meta or {},
        "series_stats": stats,
        "series_preview": _compact_series_for_ai(y_clean),
        "selected_analyses": [],
        "warnings": warnings or [],
    }

    for a in (payload.analyses or []):
        compact["selected_analyses"].append({
            "analysis_id": a.get("analysis_id"),
            "analysis_label": a.get("analysis_label"),
            "ran_at": a.get("ran_at"),
            "inputs": a.get("inputs") or {},
            "summary": a.get("summary"),
        })

    style = (payload.style or "technical").strip().lower()
    if style not in ("technical", "simple", "executive"):
        style = "technical"

    system = (
        "You are a time series analysis expert. "
        "Use ONLY the provided context and numbers. "
        "If something is not provided, say 'not provided'. "
        "Do not invent data."
    )
    if style == "simple":
        system += " Write for a non-technical audience using plain language."
    elif style == "executive":
        system += " Write a short executive summary with clear bullets and recommendations."
    else:
        system += " Write a technical interpretation with concise explanations."

    if payload.extra_instruction:
        system += f" Additional instruction: {payload.extra_instruction.strip()}"

    user = (
        "Interpret the following time-series results from a web dashboard.\n"
        "Tasks:\n"
        "1) Briefly describe the series behavior (trend/seasonality/volatility) using the provided stats and preview.\n"
        "2) For each selected analysis, explain what it means and what it suggests.\n"
        "3) Give 3–5 practical next steps (e.g., preprocessing, model choice, parameter checks).\n\n"
        "Context (JSON):\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
    )

    return system, user, compact

def _call_gemini_text(system_instruction: str, user_prompt: str, model: str) -> str:
    # Calls Gemini using google-genai if available, else REST fallback.
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable on the server.")

    # Prefer the new Google GenAI SDK
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=900,
            ),
        )
        return (resp.text or "").strip()
    except ImportError:
        pass
    except Exception:
        # fall back to REST if SDK fails
        pass

    import urllib.request
import urllib.error

url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
body = {
    "systemInstruction": {"parts": [{"text": system_instruction}]},
    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900},
}

req = urllib.request.Request(
    url,
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        status = getattr(resp, "status", 200)
except urllib.error.HTTPError as e:
    raw = e.read().decode("utf-8", errors="ignore")
    raise RuntimeError(f"Gemini REST error {e.code}: {raw[:500]}")
except Exception as e:
    raise RuntimeError(f"Gemini REST request failed: {e}")

if status >= 400:
    raise RuntimeError(f"Gemini REST error {status}: {raw[:500]}")

data = json.loads(raw)

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return json.dumps(data)[:2000]

@router.post("/ai/interpret", response_model=ApiOut)
def ai_interpret(payload: AIInterpretIn = Body(...)):
    # Server-side AI interpretation: build prompt in code, call Gemini, return text
    try:
        y, warnings, freq = _prep_series(payload.series)
        system, user, compact = _build_ai_prompt(payload, y, freq, warnings)
        model = payload.model or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"

        text = _call_gemini_text(system, user, model=model)
        result = {"text": text, "model": model, "style": payload.style}

        if payload.debug:
            result["context"] = compact
            result["system_instruction"] = system

        return ApiOut(ok=True, result=result, warnings=warnings)
    except Exception as e:
        return ApiOut(ok=False, result={"error": str(e)}, warnings=[])
