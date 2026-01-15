
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Query, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
from typing import Optional
from fastapi.security import HTTPAuthorizationCredentials

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
# Auth0 JWT protection (for premium endpoints like AI interpretation)
# ---------------------------
# Required Render env vars:
#   AUTH0_DOMAIN   = agrimetsoft.us.auth0.com
#   AUTH0_AUDIENCE = https://tsa-api
#   AUTH0_ISSUER   = https://agrimetsoft.us.auth0.com/
#
# Required dependency:
#   python-jose[cryptography]
#
import time
import urllib.request
import urllib.error
try:
    from jose import jwt  # type: ignore
    from jose.exceptions import JWTError, ExpiredSignatureError  # type: ignore
    _JOSE_OK = True
except Exception:  # pragma: no cover
    jwt = None  # type: ignore
    JWTError = Exception  # type: ignore
    ExpiredSignatureError = Exception  # type: ignore
    _JOSE_OK = False

_AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "").strip()
_AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "").strip()
_AUTH0_ISSUER = os.getenv("AUTH0_ISSUER", "").strip()
# If AUTH0_ISSUER isn't set, derive it from AUTH0_DOMAIN (recommended by Auth0: https://<domain>/)
if not _AUTH0_ISSUER and _AUTH0_DOMAIN:
    _AUTH0_ISSUER = f"https://{_AUTH0_DOMAIN}/"
_AUTH0_ALGOS = ["RS256"]

_auth_bearer = HTTPBearer(auto_error=False)
_jwks_cache: Dict[str, Any] = {"ts": 0.0, "jwks": None}
_JWKS_TTL_SECONDS = 3600


def _get_jwks() -> Dict[str, Any]:
    if not _AUTH0_DOMAIN:
        raise RuntimeError("Missing AUTH0_DOMAIN environment variable.")
    now = time.time()
    if _jwks_cache["jwks"] is not None and (now - float(_jwks_cache["ts"])) < _JWKS_TTL_SECONDS:
        return _jwks_cache["jwks"]

    url = f"https://{_AUTH0_DOMAIN}/.well-known/jwks.json"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))
    _jwks_cache["jwks"] = data
    _jwks_cache["ts"] = now
    return data


def _get_rsa_key(token: str) -> Optional[Dict[str, Any]]:
    if not _JOSE_OK or jwt is None:
        return None
    try:
        headers = jwt.get_unverified_header(token)
        kid = headers.get("kid")
        if not kid:
            return None
        jwks = _get_jwks()
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key
        return None
    except Exception:
        return None


def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_auth_bearer)) -> Dict[str, Any]:
    """FastAPI dependency: validates Auth0 JWT and returns decoded claims."""
    if not _JOSE_OK or jwt is None:
        raise HTTPException(status_code=500, detail="Auth dependencies not installed. Add python-jose[cryptography] to requirements.")
    if not _AUTH0_DOMAIN or not _AUTH0_AUDIENCE or not _AUTH0_ISSUER:
        missing = []
        if not _AUTH0_DOMAIN: missing.append("AUTH0_DOMAIN")
        if not _AUTH0_AUDIENCE: missing.append("AUTH0_AUDIENCE")
        if not _AUTH0_ISSUER: missing.append("AUTH0_ISSUER")
        raise HTTPException(
            status_code=500,
            detail="Auth is not configured on the server (missing: " + ", ".join(missing) + "). "
                   "Set these Render environment variables so the API can validate Auth0 JWTs."
        )

    if credentials is None or (credentials.scheme or "").lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing Bearer token.")

    token = credentials.credentials
    rsa_key = _get_rsa_key(token)
    if not rsa_key:
        raise HTTPException(status_code=401, detail="Invalid token (no matching JWKS key).")

    try:
        decoded = jwt.decode(
            token,
            rsa_key,
            algorithms=_AUTH0_ALGOS,
            audience=_AUTH0_AUDIENCE,
            issuer=_AUTH0_ISSUER,
        )
        return decoded
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token.")

from typing import Optional
from fastapi.security import HTTPAuthorizationCredentials
from fastapi import APIRouter, Request

router = APIRouter()

@router.post("/billing/fastspring/webhook")
async def fastspring_webhook(request: Request):
    payload = await request.json()

    # Step 1: log and acknowledge
    # (Render logs will show it)
    print("FASTSPRING_WEBHOOK:", payload)

    return {"ok": True}

@router.get("/account/me")
def account_me(
    user_claims=Depends(require_auth),
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_auth_bearer),
):
    # token used to call /userinfo
    token = creds.credentials if creds else None

    email = user_claims.get("email")
    email_verified = user_claims.get("email_verified")

    # If email isn't in the JWT, fetch it from Auth0 /userinfo
    if token and not email:
        try:
            req = urllib.request.Request(
                f"https://{_AUTH0_DOMAIN}/userinfo",
                headers={"Authorization": f"Bearer {token}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                u = json.loads(r.read().decode("utf-8"))
            email = u.get("email")
            email_verified = u.get("email_verified")
        except Exception:
            pass

        plan = (get_plan(email) or "free") if email else "free"
        return {
            "ok": True,
            "user": {
                "sub": user_claims.get("sub"),
                "email": email,
                "email_verified": email_verified,
                "plan": plan,
            },
        }

_PAID_EMAILS = {
    e.strip().lower()
    for e in (os.getenv("PAID_EMAILS", "")).split(",")
    if e.strip()
}


def require_pro(email: str | None):
    plan = (get_plan(email) or "free") if email else "free"
    if plan != "pro":
        raise HTTPException(
            status_code=403,
            detail="Upgrade required: AI Interpretation is available on the Pro plan."
        )

def get_user_plan(email: str | None) -> str:
    if not email:
        return "free"
    return "pro" if email.strip().lower() in _PAID_EMAILS else "free"

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
    model: Optional[str] = Field(default=None)  # optional override, e.g. "gemini-2.5-flash"
    debug: bool = Field(default=False)


def _compact_series_for_ai(y: pd.Series, max_points: int = 160) -> Dict[str, Any]:
    # Compact series preview for LLM context
    y2 = pd.to_numeric(y, errors="coerce").dropna()
    n = int(len(y2))
    if n == 0:
        return {"n": 0, "preview": []}

    # Take last max_points (keeps recent behavior for forecasts)
    if n > max_points:
        y2 = y2.iloc[-max_points:]
        n = int(len(y2))

    rows: List[List[Any]] = []
    for idx, val in y2.items():
        if isinstance(idx, (pd.Timestamp, datetime)):
            t = pd.Timestamp(idx).date().isoformat()
        else:
            t = str(idx)
        rows.append([t, safe_float(val)])

    return {"n": n, "preview": rows}


def _trim_obj(obj, max_list: int = 80, max_str: int = 2500, depth: int = 3):
    # Trim nested dict/list/strings to avoid huge prompts
    if depth <= 0:
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _trim_obj(v, max_list=max_list, max_str=max_str, depth=depth - 1)
        return out
    if isinstance(obj, list):
        if len(obj) <= max_list:
            return [_trim_obj(v, max_list=max_list, max_str=max_str, depth=depth - 1) for v in obj]
        head = obj[: max(10, max_list - 10)]
        tail = obj[-5:]
        return {
            "n": len(obj),
            "head": [_trim_obj(v, max_list=max_list, max_str=max_str, depth=depth - 1) for v in head],
            "tail": [_trim_obj(v, max_list=max_list, max_str=max_str, depth=depth - 1) for v in tail],
        }
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else (obj[:max_str] + "…(truncated)")
    return obj


def _build_ai_prompt(payload: AIInterpretIn, y: pd.Series, freq: Optional[str], warnings: List[str]):
    # Returns (system_instruction, user_prompt, compact_context)
    y_clean = pd.to_numeric(y, errors="coerce")

    stats = {
        "n": int(len(y_clean)),
        "start": y_clean.index[0].date().isoformat() if isinstance(y_clean.index, pd.DatetimeIndex) and len(y_clean) else None,
        "end": y_clean.index[-1].date().isoformat() if isinstance(y_clean.index, pd.DatetimeIndex) and len(y_clean) else None,
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
        compact["selected_analyses"].append(
            {
                "analysis_id": a.get("analysis_id"),
                "analysis_label": a.get("analysis_label"),
                "ran_at": a.get("ran_at"),
                "inputs": a.get("inputs") or {},
                "summary": a.get("summary"),
            }
        )

    compact = _trim_obj(compact)

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

    extra = (payload.extra_instruction or "").strip()
    if extra:
        system += " Extra instruction: " + extra

    user = (
        "Interpret the selected time-series analyses.\n"
        "1) Summarize the series behavior.\n"
        "2) For each selected analysis, explain what it means and what it suggests.\n"
        "3) Give 3–5 practical next steps (preprocessing, parameter checks, model choices).\n\n"
        "Context (JSON):\n"
        f"{json.dumps(compact, ensure_ascii=False, separators=(',', ':'), default=str)}\n"
    )

    return system, user, compact


def _call_gemini_text(system_instruction: str, user_prompt: str, model: str) -> str:
    """
    Call Gemini and return plain text.

    Notes:
    - Prefer the official `google-genai` SDK when available.
    - Some Gemini responses may split text across multiple `parts`; we therefore
      concatenate all text parts (SDK + REST) instead of only the first one.
    - Includes basic retry/backoff for transient 429/503.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable on the server.")

    def _join_text_parts(parts) -> str:
        try:
            out = []
            for p in (parts or []):
                if isinstance(p, dict):
                    t = p.get("text")
                else:
                    t = getattr(p, "text", None)
                if t:
                    out.append(str(t))
            return "\n".join(out).strip()
        except Exception:
            return ""

    # 1) Prefer the official Google GenAI SDK if available
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=1600,
            ),
        )

        # Robust extraction: concatenate all text parts from the top candidate
        try:
            cands = getattr(resp, "candidates", None) or []
            if cands:
                cand0 = cands[0]
                content = getattr(cand0, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                txt = _join_text_parts(parts)
                if txt:
                    return txt
        except Exception:
            pass

        # Fallback: resp.text (may already be concatenated, but sometimes isn't)
        txt = getattr(resp, "text", None)
        if txt:
            return str(txt).strip()
    except ImportError:
        # SDK not installed; fall back to REST
        pass
    except Exception:
        # If SDK fails, try REST
        pass

    # 2) REST fallback (no extra deps)
    import urllib.request
    import urllib.error
    import time
    import random

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1600},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200) or 200
                raw = resp.read().decode("utf-8", errors="ignore")
                if status >= 400:
                    raise RuntimeError(f"Gemini REST error {status}: {raw[:800]}")
                data = json.loads(raw)

                # Concatenate all text parts from the first candidate
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    txt = _join_text_parts(parts)
                    if txt:
                        return txt
                except Exception:
                    pass

                # As a last resort, return the raw JSON (trimmed)
                return json.dumps(data)[:2000]

        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="ignore")
            msg = f"Gemini REST error {e.code}: {raw[:1200]}"
            last_err = RuntimeError(msg)
            if e.code in (429, 503) and attempt < 2:
                time.sleep((2 ** attempt) + random.random() * 0.25)
                continue
            raise last_err

        except Exception as e:
            last_err = e
            s = str(e).lower()
            if (("429" in s) or ("too many" in s) or ("rate" in s) or ("quota" in s) or ("503" in s) or ("service unavailable" in s)) and attempt < 2:
                time.sleep((2 ** attempt) + random.random() * 0.25)
                continue
            raise RuntimeError(f"Gemini REST request failed: {e}")

    raise RuntimeError(f"Gemini request failed: {last_err}")

@router.post("/ai/interpret", response_model=ApiOut)
def ai_interpret(payload: AIInterpretIn = Body(...), user_claims: Dict[str, Any] = Depends(require_auth)):
    # Server-side AI interpretation: build prompt in code, call Gemini, return text.
    try:
        y, warnings, freq = _prep_series(payload.series)
        system, user, compact = _build_ai_prompt(payload, y, freq, warnings)
        model = payload.model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

        text = _call_gemini_text(system, user, model=model)
        # If the model stops early (e.g., only section 1), ask once more to continue.
        def _needs_continue(t: str) -> bool:
            if not t:
                return True
            t_low = t.lower()
            # Expect at least sections 1,2,3 or an explicit 'next steps' section.
            has2 = ('### 2' in t_low) or ('## 2' in t_low) or ('2)' in t_low)
            has3 = ('### 3' in t_low) or ('## 3' in t_low) or ('3)' in t_low) or ('next steps' in t_low)
            # Also treat a mid-sentence ending as incomplete.
            ends_ok = t.strip().endswith(('.', '!', '?', ':'))
            too_short = len(t.strip()) < 400
            return (not (has2 and has3)) or (too_short and not ends_ok)

        if _needs_continue(text):
            tail = text[-1200:] if text else ''
            user2 = (
                'Continue the interpretation from where you stopped.\n'
                'Do NOT repeat section 1. Provide sections 2 and 3 clearly.\n\n'
                'Previous text (tail):\n' + tail + '\n\n'
                'Context (same JSON, abbreviated):\n' + json.dumps(compact, ensure_ascii=False, separators=(",",":"), default=str) + '\n'
            )
            try:
                text2 = _call_gemini_text(system, user2, model=model)
                if text2 and text2.strip() and text2.strip() not in text:
                    text = (text.rstrip() + '\n\n' + text2.strip()).strip()
            except Exception:
                pass
        result = {"text": text, "model": model, "style": payload.style}

        if payload.debug:
            result["context"] = compact
            result["system_instruction"] = system

        return ApiOut(ok=True, result=result, warnings=warnings)
    except Exception as e:
        return ApiOut(ok=False, result={"error": str(e)}, warnings=[])




# ---------------------------
# Compatibility endpoints: SARIMA / SARIMAX wrappers
# These keep your HTML stable even if it calls explicit SARIMA/SARIMAX routes.
# ---------------------------

@router.post("/tsa/J_sarima_forecast", response_model=ApiOut)
def tsa_j_sarima_forecast(payload: ForecastIn):
    # Force seasonal = True; otherwise use the same logic as ARIMA endpoint.
    try:
        p2 = payload.model_copy(deep=True)  # pydantic v2
    except Exception:
        p2 = payload  # fallback
    try:
        setattr(p2, "seasonal", True)
    except Exception:
        pass
    return tsa_j_arima_forecast(p2)


class SARIMAXIn(BaseModel):
    series: SeriesIn
    horizon: int = 30
    # If provided, use these; otherwise auto-fit like Auto-ARIMA
    order: Optional[Tuple[int, int, int]] = None
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
    seasonal_period: Optional[int] = None
    auto: bool = True
    # Optional exogenous regressors:
    # - exog: length n (aligned with series)
    # - exog_future: length horizon (future values)
    exog: Optional[List[float]] = None
    exog_future: Optional[List[float]] = None


@router.post("/tsa/J_sarimax_forecast", response_model=ApiOut)
def tsa_j_sarimax_forecast(payload: SARIMAXIn):
    """
    SARIMAX forecast (supports optional exogenous regressor arrays).
    If no exog is supplied, behaves like SARIMA.
    """
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    h = int(max(1, payload.horizon))

    sp = payload.seasonal_period
    if sp is None:
        sp = _seasonal_period_from_inputs(freq, None)
    sp = int(sp or 0)

    exog = None
    exog_future = None
    if payload.exog is not None:
        try:
            exog = np.asarray(payload.exog, dtype="float64")
            if exog.shape[0] != len(y2):
                warnings.append("exog length does not match series length; ignoring exog.")
                exog = None
        except Exception:
            exog = None
    if payload.exog_future is not None:
        try:
            exog_future = np.asarray(payload.exog_future, dtype="float64")
            if exog_future.shape[0] != h:
                warnings.append("exog_future length does not match horizon; ignoring exog_future.")
                exog_future = None
        except Exception:
            exog_future = None

    # Choose orders
    order = payload.order
    sorder = payload.seasonal_order
    if payload.auto or (order is None):
        o, so, w = _fit_arima_grid(
            y2,
            seasonal_period=sp,
            seasonal=(sp >= 2),
            max_p=3, max_d=1, max_q=3,
            max_P=1, max_D=1, max_Q=1
        )
        warnings += w
        order = tuple(map(int, o))
        sorder = tuple(map(int, so))
    else:
        order = tuple(map(int, order))
        sorder = tuple(map(int, sorder or (0, 0, 0, 0)))

    # Normalize seasonal if not meaningful
    if not sorder or int(sorder[3]) < 2:
        sorder = (0, 0, 0, 0)

    try:
        d = int(order[1])
        D = int(sorder[1]) if sorder else 0
        simple_diff = (d > 0) or (D > 0)

        model = SARIMAX(
            y2,
            exog=exog,
            order=order,
            seasonal_order=sorder,
            enforce_stationarity=False,
            enforce_invertibility=False,
            simple_differencing=simple_diff,
            low_memory=True,
        )
        res = model.fit(disp=False, maxiter=80)
        fc = res.get_forecast(steps=h, exog=exog_future)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        x_fc = _forecast_index(y2, h)

        return ApiOut(
            ok=True,
            result={
                "order": {"p": int(order[0]), "d": int(order[1]), "q": int(order[2])},
                "seasonal_order": {"P": int(sorder[0]), "D": int(sorder[1]), "Q": int(sorder[2]), "s": int(sorder[3])},
                "used_exog": bool(exog is not None and exog_future is not None),
                "aic": safe_float(res.aic),
                "forecast": {"x": x_fc, "y": _as_float_list(mean.values)},
                "conf_int_95": {"lower": _as_float_list(ci.iloc[:, 0].values), "upper": _as_float_list(ci.iloc[:, 1].values)},
            },
            warnings=warnings,
        )
    except Exception as e:
        return ApiOut(ok=False, result={"error": f"SARIMAX failed: {str(e)}"}, warnings=warnings)

# ============================
# Extra TSA methods (MSTL, changepoints, fingerprint, auto models, conformal, and optional deep models)
# These are added in a way that does NOT break existing endpoints.
# Some "deep" methods require optional dependencies; if missing, the endpoint returns HTTP 501 with install guidance.
# ============================

class MSTLIn(BaseModel):
    series: SeriesIn
    periods: Optional[List[int]] = None
    robust: bool = True


def _default_mstl_periods(freq: Optional[str]) -> List[int]:
    """Pick a practical set of seasonal periods from inferred pandas freq string."""
    f = (freq or "").upper()
    # common pandas freq strings: 'D', 'H', 'M', 'MS', 'W', 'Q', 'A', 'AS', 'B', etc.
    if f.startswith("H"):
        return [24, 24 * 7]          # daily + weekly
    if f.startswith("D") or f.startswith("B"):
        return [7, 365]              # weekly + yearly
    if f.startswith("W"):
        return [52]                  # yearly (weekly data)
    if f.startswith("M"):
        return [12]                  # yearly (monthly data)
    if f.startswith("Q"):
        return [4]                   # yearly (quarterly data)
    if f.startswith("A") or f.startswith("Y"):
        return []                    # annual data: no meaningful seasonality
    # fallback: try the default period from your helper, and also a weekly component
    try:
        sp = int(default_seasonal_period(freq))
        return [sp] if sp >= 2 else []
    except Exception:
        return []


@router.post("/tsa/O_mstl_decompose", response_model=ApiOut)
def tsa_o_mstl_decompose(payload: MSTLIn):
    """
    MSTL decomposition: trend + multiple seasonalities + residual.
    Returns each seasonal component separately plus the combined seasonal.
    """
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()

    if len(y2) < 10:
        return ApiOut(ok=False, result={"error": "Not enough data for MSTL (need ~10+ points)."}, warnings=warnings)

    periods = payload.periods or _default_mstl_periods(freq)
    periods = [int(p) for p in periods if p and int(p) >= 2]
    # MSTL needs at least one period
    if not periods:
        return ApiOut(ok=False, result={"error": "No valid seasonal periods for MSTL. Provide 'periods' (e.g., [7,365] or [12])."}, warnings=warnings)

    # Guard periods relative to series length
    max_allowed = max(2, len(y2) // 2)
    periods2 = []
    for p in periods:
        if p > max_allowed:
            warnings.append(f"MSTL: period {p} too large for series length; adjusted to {max_allowed}.")
            p = max_allowed
        if p >= 2:
            periods2.append(int(p))
    periods = sorted(list(dict.fromkeys(periods2)))
    if not periods:
        return ApiOut(ok=False, result={"error": "After adjustment, no valid MSTL periods remained."}, warnings=warnings)

    try:
        from statsmodels.tsa.seasonal import MSTL  # type: ignore
    except Exception:
        return ApiOut(ok=False, result={"error": "MSTL not available in your statsmodels version. Upgrade statsmodels (>=0.14)."}, warnings=warnings)

    try:
        stl_kwargs = {"robust": bool(payload.robust)}
        mstl = MSTL(y2, periods=periods, stl_kwargs=stl_kwargs).fit()
        trend = mstl.trend.reindex(y.index)
        resid = mstl.resid.reindex(y.index)

        # seasonal can be DataFrame (one column per period) or Series depending on version
        seasonals: Dict[str, Any] = {}
        combined = None

        seas = getattr(mstl, "seasonal", None)
        if seas is None:
            seas = getattr(mstl, "seasonal_", None)

        if seas is not None:
            if hasattr(seas, "columns"):
                for col in list(seas.columns):
                    s = seas[col].reindex(y.index)
                    seasonals[str(col)] = to_jsonable_series(s)
                try:
                    combined = seas.sum(axis=1).reindex(y.index)
                except Exception:
                    combined = None
            else:
                combined = seas.reindex(y.index)

        result = {
            "periods_used": periods,
            "freq_inferred": freq,
            "series": to_jsonable_series(y),
            "trend": to_jsonable_series(trend),
            "resid": to_jsonable_series(resid),
            "seasonal_components": seasonals,
        }
        if combined is not None:
            result["seasonal"] = to_jsonable_series(combined)

        return ApiOut(ok=True, result=result, warnings=warnings)
    except Exception as e:
        return ApiOut(ok=False, result={"error": f"MSTL failed: {str(e)}"}, warnings=warnings)


class ChangepointIn(BaseModel):
    series: SeriesIn
    n_bkps: Optional[int] = Field(default=5, description="Target number of change points (approx).")
    penalty: Optional[float] = Field(default=None, description="Penalty for PELT (if using ruptures).")
    min_size: Optional[int] = Field(default=5, description="Minimum segment length.")
    model: Optional[str] = Field(default="l2", description="ruptures cost model, e.g., 'l2'.")


def _simple_changepoints_fallback(yv: np.ndarray, k: int, min_gap: int) -> List[int]:
    """Dependency-free fallback: detect large changes in rolling mean."""
    n = int(len(yv))
    if n < 2 * max(3, min_gap):
        return []
    w = max(3, min_gap)
    s = pd.Series(yv).rolling(w, center=True, min_periods=w).mean().values
    d = np.abs(np.diff(s))
    # pick top-k peaks separated by min_gap
    idx = np.argsort(-np.nan_to_num(d, nan=-np.inf))
    chosen: List[int] = []
    for i in idx:
        cp = int(i + 1)  # diff index -> split index
        if cp < w or cp > (n - w):
            continue
        if all(abs(cp - c) >= min_gap for c in chosen):
            chosen.append(cp)
        if len(chosen) >= k:
            break
    return sorted(chosen)


@router.post("/tsa/P_changepoints", response_model=ApiOut)
def tsa_p_changepoints(payload: ChangepointIn):
    """
    Changepoint detection.
    Uses 'ruptures' if installed; otherwise uses a lightweight fallback.
    """
    y, warnings, _freq = _prep_series(payload.series)
    y2 = y.dropna()
    if len(y2) < 20:
        warnings.append("Short series; changepoint results may be unstable.")

    yv = y2.values.astype("float64")
    n = int(len(yv))
    k = int(max(0, min(int(payload.n_bkps or 0), max(0, n // 5))))
    min_size = int(max(2, int(payload.min_size or 5)))

    bkps: List[int] = []
    method_used = "fallback"

    # Try ruptures (recommended)
    try:
        import ruptures as rpt  # type: ignore
        method_used = "ruptures"
        sig = yv.reshape(-1, 1)
        if k <= 0:
            # PELT with penalty
            pen = payload.penalty
            if pen is None:
                # heuristic penalty
                pen = float(np.log(max(n, 2)) * (np.nanvar(yv) + 1e-9))
            algo = rpt.Pelt(model=str(payload.model or "l2"), min_size=min_size).fit(sig)
            bkps = [int(b) for b in algo.predict(pen=float(pen)) if int(b) < n]
        else:
            algo = rpt.Binseg(model=str(payload.model or "l2"), min_size=min_size).fit(sig)
            bkps = [int(b) for b in algo.predict(n_bkps=int(k)) if int(b) < n]
    except Exception:
        # fallback
        if k <= 0:
            k = 5
        bkps = _simple_changepoints_fallback(yv, k=k, min_gap=min_size)

    # Convert to dates
    cps = []
    for b in bkps:
        if b <= 0 or b >= n:
            continue
        idx = y2.index[b - 1]  # end of segment
        cps.append({
            "index": int(b),
            "x": idx.isoformat() if isinstance(y2.index, pd.DatetimeIndex) else safe_float(b),
            "y": safe_float(y2.iloc[b - 1]),
        })

    return ApiOut(
        ok=True,
        result={
            "method": method_used,
            "n": int(n),
            "change_points": cps,
        },
        warnings=warnings,
    )


class FingerprintIn(BaseModel):
    series: SeriesIn
    period: Optional[int] = None


@router.post("/tsa/Q_fingerprint", response_model=ApiOut)
def tsa_q_fingerprint(payload: FingerprintIn):
    """
    Compact "series fingerprint" for quick diagnostics and for AI context.
    """
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    n = int(len(y2))
    if n == 0:
        return ApiOut(ok=False, result={"error": "All values are missing."}, warnings=warnings)

    period = payload.period
    if period is None:
        try:
            period = int(default_seasonal_period(freq))
        except Exception:
            period = None

    # Basic stats
    vals = y2.values.astype("float64")
    mu = float(np.nanmean(vals))
    sd = float(np.nanstd(vals, ddof=1)) if n > 1 else float("nan")
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))

    # Trend slope (simple linear fit vs. time index)
    slope = None
    try:
        t = np.arange(n, dtype="float64")
        A = np.vstack([t, np.ones_like(t)]).T
        m, c = np.linalg.lstsq(A, vals, rcond=None)[0]
        slope = float(m)
    except Exception:
        slope = None

    # Autocorr at a few lags
    def _ac(lag: int) -> Optional[float]:
        try:
            if lag <= 0 or lag >= n:
                return None
            a = np.corrcoef(vals[:-lag], vals[lag:])[0, 1]
            return safe_float(a)
        except Exception:
            return None

    ac1 = _ac(1)
    ac7 = _ac(7) if n > 8 else None
    ac_sp = _ac(int(period)) if (period and isinstance(period, int) and period > 1 and n > period + 2) else None

    # Outlier ratio (IQR)
    outlier_ratio = None
    try:
        q1, q3 = np.nanpercentile(vals, [25, 75])
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outlier_ratio = float(np.mean((vals < lo) | (vals > hi)))
    except Exception:
        outlier_ratio = None

    # Stationarity (ADF p-value) - quick, may fail on tiny series
    adf_p = None
    try:
        if n >= 12:
            adf = adfuller(vals, autolag="AIC")
            adf_p = float(adf[1])
    except Exception:
        adf_p = None

    # Seasonality & trend strength via STL (optional, fast)
    trend_strength = None
    seasonal_strength = None
    if period and isinstance(period, int) and period >= 2 and n >= 2 * period:
        try:
            stl = STL(y2, period=int(period), robust=True).fit()
            resid = stl.resid.values
            tr = stl.trend.values
            seas = stl.seasonal.values

            var_r = np.nanvar(resid, ddof=1)
            var_tr = np.nanvar(tr + resid, ddof=1)
            var_sr = np.nanvar(seas + resid, ddof=1)

            if np.isfinite(var_r) and np.isfinite(var_tr) and var_tr > 0:
                trend_strength = float(1.0 - (var_r / var_tr))
            if np.isfinite(var_r) and np.isfinite(var_sr) and var_sr > 0:
                seasonal_strength = float(1.0 - (var_r / var_sr))
        except Exception:
            pass

    # Entropy-ish measure (binned)
    entropy = None
    try:
        bins = max(8, int(np.sqrt(n)))
        hist, _ = np.histogram(vals[np.isfinite(vals)], bins=bins)
        p = hist.astype("float64")
        p = p / (p.sum() + 1e-12)
        p = p[p > 0]
        entropy = float(-(p * np.log(p)).sum())
    except Exception:
        entropy = None

    result = {
        "n": int(n),
        "freq_inferred": freq,
        "min": safe_float(vmin),
        "max": safe_float(vmax),
        "mean": safe_float(mu),
        "std": safe_float(sd),
        "trend_slope": safe_float(slope),
        "autocorr_lag1": ac1,
        "autocorr_lag7": ac7,
        "autocorr_lag_seasonal": ac_sp,
        "adf_pvalue": safe_float(adf_p),
        "trend_strength": safe_float(trend_strength),
        "seasonal_strength": safe_float(seasonal_strength),
        "outlier_ratio": safe_float(outlier_ratio),
        "entropy": safe_float(entropy),
    }
    return ApiOut(ok=True, result=result, warnings=warnings)


class AutoETSIn(BaseModel):
    series: SeriesIn
    horizon: int = 30
    seasonal_period: Optional[int] = None


@router.post("/tsa/R_auto_ets_forecast", response_model=ApiOut)
def tsa_r_auto_ets_forecast(payload: AutoETSIn):
    """
    Auto-ETS: tries a small set of ETS configurations and picks the best (AIC / SSE).
    Uses statsmodels ExponentialSmoothing.
    """
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    n = int(len(y2))
    h = int(max(1, payload.horizon))

    sp = payload.seasonal_period
    if sp is None:
        sp = _seasonal_period_from_inputs(freq, None)
    sp = int(sp or 0)

    if n < 8:
        return ApiOut(ok=False, result={"error": "Series too short for Auto-ETS."}, warnings=warnings)

    # Candidate config list (kept small for web performance)
    candidates = []
    candidates.append((None, None, False))
    candidates.append(("add", None, False))
    candidates.append(("add", None, True))
    # seasonal only when enough data
    if sp >= 2 and n >= 2 * sp:
        candidates.append((None, "add", False))
        candidates.append(("add", "add", False))
        candidates.append(("add", "add", True))

    # Allow multiplicative only if strictly positive
    if y2.min() > 0 and sp >= 2 and n >= 2 * sp:
        candidates.append((None, "mul", False))
        candidates.append(("add", "mul", False))
        candidates.append(("add", "mul", True))

    best = None
    best_score = np.inf
    best_res = None

    for trend, seas, damped in candidates:
        try:
            model = ExponentialSmoothing(
                y2,
                trend=trend,
                damped_trend=(bool(damped) if trend else False),
                seasonal=seas,
                seasonal_periods=(sp if seas else None),
                initialization_method="estimated",
            )
            res = model.fit(optimized=True)
            # Prefer AIC if available, else SSE
            score = getattr(res, "aic", None)
            if score is None or not np.isfinite(score):
                fitted = getattr(res, "fittedvalues", None)
                if fitted is None:
                    continue
                resid = (y2 - fitted).dropna()
                score = float(np.sum(np.square(resid.values))) if len(resid) else np.inf
            score = float(score)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best = (trend, seas, damped)
                best_res = res
        except Exception:
            continue

    if best_res is None or best is None:
        return ApiOut(ok=False, result={"error": "Auto-ETS could not fit any candidate."}, warnings=warnings)

    mean = best_res.forecast(h)
    x_fc = _forecast_index(y2, h)

    # Simple residual-based interval
    lower = upper = None
    try:
        fitted = getattr(best_res, "fittedvalues", None)
        if fitted is not None:
            resid = (y2 - fitted).dropna()
            if len(resid) >= 8:
                q = float(np.nanquantile(np.abs(resid.values), 0.95))
                lower = mean - q
                upper = mean + q
    except Exception:
        pass

    trend, seas, damped = best
    result = {
        "best_model": {
            "type": "AutoETS",
            "trend": trend,
            "seasonal": seas,
            "damped_trend": bool(damped if trend else False),
            "seasonal_period_used": int(sp) if seas else None,
            "score": safe_float(best_score),
            "score_type": "aic_or_sse",
        },
        "horizon": int(h),
        "forecast": {"x": x_fc, "y": _as_float_list(mean.values)},
    }
    if lower is not None and upper is not None:
        result["conf_int_approx"] = {
            "lower": _as_float_list(lower.values),
            "upper": _as_float_list(upper.values),
            "note": "Approx interval from residual absolute-quantile (quick, not exact).",
        }

    return ApiOut(ok=True, result=result, warnings=warnings)


class AutoARIMAIn(BaseModel):
    series: SeriesIn
    horizon: int = 30
    seasonal: bool = False
    seasonal_period: Optional[int] = None


@router.post("/tsa/S_auto_arima_forecast", response_model=ApiOut)
def tsa_s_auto_arima_forecast(payload: AutoARIMAIn):
    """
    Auto-ARIMA/SARIMA forecast. Thin wrapper around the same grid logic used in J_arima_forecast.
    """
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    if len(y2) < 12:
        warnings.append("Short series; Auto-ARIMA may be unstable.")

    h = int(max(1, payload.horizon))
    sp = payload.seasonal_period
    if sp is None:
        sp = _seasonal_period_from_inputs(freq, None)

    order, sorder, w = _fit_arima_grid(
        y2,
        seasonal_period=int(sp or 0),
        seasonal=bool(payload.seasonal),
        max_p=3, max_d=1, max_q=3,
        max_P=1, max_D=1, max_Q=1
    )
    warnings += w

    # Normalize seasonal order
    P, D, Q, s = sorder
    if (not payload.seasonal) or (s is None) or (int(s) < 2) or (P == 0 and D == 0 and Q == 0):
        sorder = (0, 0, 0, 0)

    try:
        d = int(order[1])
        D = int(sorder[1]) if sorder else 0
        simple_diff = (d > 0) or (D > 0)

        model = SARIMAX(
            y2,
            order=tuple(map(int, order)),
            seasonal_order=tuple(map(int, sorder)),
            enforce_stationarity=False,
            enforce_invertibility=False,
            simple_differencing=simple_diff,
            low_memory=True,
        )
        res = model.fit(disp=False, maxiter=60)
        fc = res.get_forecast(steps=h)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        x_fc = _forecast_index(y2, h)

        return ApiOut(
            ok=True,
            result={
                "order": {"p": int(order[0]), "d": int(order[1]), "q": int(order[2])},
                "seasonal_order": {"P": int(sorder[0]), "D": int(sorder[1]), "Q": int(sorder[2]), "s": int(sorder[3])},
                "aic": safe_float(res.aic),
                "forecast": {"x": x_fc, "y": _as_float_list(mean.values)},
                "conf_int_95": {"lower": _as_float_list(ci.iloc[:, 0].values), "upper": _as_float_list(ci.iloc[:, 1].values)},
            },
            warnings=warnings,
        )
    except Exception as e:
        return ApiOut(ok=False, result={"error": f"Auto-ARIMA failed: {str(e)}"}, warnings=warnings)


class STLForecastIn(BaseModel):
    series: SeriesIn
    horizon: int = 30
    period: Optional[int] = None
    robust: bool = True
    remainder_model: str = "theta"  # 'theta' or 'arima010'


@router.post("/tsa/T_stl_forecast", response_model=ApiOut)
def tsa_t_stl_forecast(payload: STLForecastIn):
    """
    STL + forecast:
      - Decompose with STL
      - Seasonal forecast by repeating last season
      - Remainder/trend forecast with a lightweight model
    """
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    n = int(len(y2))
    if n < 12:
        return ApiOut(ok=False, result={"error": "Not enough data for STL forecast (need ~12+ points)."}, warnings=warnings)

    h = int(max(1, payload.horizon))
    period = payload.period
    if period is None:
        period = int(default_seasonal_period(freq))
    period = int(max(2, min(int(period), max(2, n // 2))))

    try:
        stl = STL(y2, period=period, robust=bool(payload.robust)).fit()
        trend = stl.trend
        seas = stl.seasonal
        resid = stl.resid

        # Seasonal forecast: repeat last period of seasonal component
        last_seas = seas.iloc[-period:].values
        reps = int(np.ceil(h / period))
        seas_fc = np.tile(last_seas, reps)[:h].astype("float64")

        # Remainder series to forecast (trend + resid)
        rem = (trend + resid).astype("float64")

        base = payload.remainder_model.lower().strip()
        if base == "arima010":
            # very fast baseline (random-walk)
            model = SARIMAX(rem, order=(0, 1, 0), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=40)
            mean_rem = res.get_forecast(steps=h).predicted_mean.values.astype("float64")
        else:
            # theta is usually stable
            sp = max(1, period)
            res = ThetaModel(rem, period=sp).fit()
            mean_rem = res.forecast(h).values.astype("float64")

        mean = mean_rem + seas_fc
        x_fc = _forecast_index(y2, h)

        result = {
            "period_used": int(period),
            "remainder_model": base,
            "components": {
                "trend": to_jsonable_series(trend.reindex(y.index)),
                "seasonal": to_jsonable_series(seas.reindex(y.index)),
                "resid": to_jsonable_series(resid.reindex(y.index)),
            },
            "forecast": {"x": x_fc, "y": _as_float_list(mean)},
        }
        return ApiOut(ok=True, result=result, warnings=warnings)
    except Exception as e:
        return ApiOut(ok=False, result={"error": f"STL forecast failed: {str(e)}"}, warnings=warnings)


class ConformalIn(BaseModel):
    series: SeriesIn
    horizon: int = 30
    base_model: str = "arima"   # arima|ets|theta|naive|seasonal_naive
    alpha: float = 0.1          # 0.1 => 90% interval (approx)
    seasonal_period: Optional[int] = None


@router.post("/tsa/U_conformal_forecast", response_model=ApiOut)
def tsa_u_conformal_forecast(payload: ConformalIn):
    """
    Conformal-style intervals (fast approximation):
      - Fit a base model
      - Forecast mean
      - Interval width from residual absolute quantile (1-alpha) on fitted residuals
    This is *approximate* but very practical for web dashboards.
    """
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    n = int(len(y2))
    if n < 12:
        warnings.append("Short series; conformal intervals may be noisy.")

    h = int(max(1, payload.horizon))
    alpha = float(payload.alpha)
    alpha = min(max(alpha, 0.01), 0.5)

    sp = payload.seasonal_period
    if sp is None:
        sp = _seasonal_period_from_inputs(freq, None)
    sp = int(sp or 1)

    base = payload.base_model.lower().strip()
    mean = None
    resid_abs = None

    try:
        if base in ("naive", "seasonal_naive", "theta", "ets"):
            # use existing helper for mean forecast
            mean = _fit_and_forecast(base, y2, h, sp)
            # residuals: one-step fitted values (rough)
            if base == "naive":
                fitted = y2.shift(1)
                resid_abs = np.abs((y2 - fitted).dropna().values)
            elif base == "seasonal_naive":
                fitted = y2.shift(sp)
                resid_abs = np.abs((y2 - fitted).dropna().values)
            elif base == "theta":
                res = ThetaModel(y2, period=max(1, sp)).fit()
                fitted = getattr(res, "fittedvalues", None)
                if fitted is not None:
                    resid_abs = np.abs((y2 - fitted).dropna().values)
            elif base == "ets":
                seasonal = "add" if (sp >= 2 and len(y2) >= 2 * sp) else None
                model = ExponentialSmoothing(y2, trend="add", seasonal=seasonal, seasonal_periods=(sp if seasonal else None))
                res = model.fit(optimized=True)
                fitted = getattr(res, "fittedvalues", None)
                if fitted is not None:
                    resid_abs = np.abs((y2 - fitted).dropna().values)
        else:
            # arima (auto, small grid)
            order, sorder, w = _fit_arima_grid(
                y2,
                seasonal_period=sp,
                seasonal=(sp >= 2),
                max_p=3, max_d=1, max_q=3,
                max_P=1, max_D=1, max_Q=1
            )
            warnings += w
            d = int(order[1])
            D = int(sorder[1]) if sorder else 0
            simple_diff = (d > 0) or (D > 0)
            model = SARIMAX(
                y2,
                order=tuple(map(int, order)),
                seasonal_order=tuple(map(int, sorder)),
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=simple_diff,
                low_memory=True,
            )
            res = model.fit(disp=False, maxiter=60)
            fc = res.get_forecast(steps=h)
            mean = fc.predicted_mean.values.astype("float64")
            fitted = getattr(res, "fittedvalues", None)
            if fitted is not None:
                resid_abs = np.abs((y2 - fitted).dropna().values)

        if mean is None:
            raise RuntimeError("Base model did not return a forecast.")

        # interval half-width
        q = None
        if resid_abs is not None and len(resid_abs) >= 8:
            q = float(np.nanquantile(resid_abs, 1.0 - alpha))
        else:
            # fallback using sd
            q = float(np.nanstd(y2.values, ddof=1) if n > 1 else 0.0)

        lower = mean - q
        upper = mean + q
        x_fc = _forecast_index(y2, h)

        result = {
            "base_model": base,
            "alpha": alpha,
            "interval_note": "Approx residual-quantile interval (practical conformal-style).",
            "forecast": {"x": x_fc, "y": _as_float_list(mean)},
            "conf_int": {"lower": _as_float_list(lower), "upper": _as_float_list(upper)},
        }
        return ApiOut(ok=True, result=result, warnings=warnings)

    except Exception as e:
        return ApiOut(ok=False, result={"error": f"Conformal forecast failed: {str(e)}"}, warnings=warnings)


# ---------------------------
# Optional "Deep/Modern" forecasting methods (placeholders unless dependencies are installed)
# ---------------------------


class DeepForecastIn(BaseModel):
    series: SeriesIn
    horizon: int = 30
    seasonal_period: Optional[int] = None
    max_train_points: int = 1000
    epochs: int = 5
    max_steps: Optional[int] = None  # caps training steps for NeuralForecast models
    model_name: Optional[str] = None  # used for transformer-family chooser


def _require_optional(dep_name: str, hint: str) -> None:
    """Raise a consistent 501 when an optional dependency isn't installed."""
    raise HTTPException(status_code=501, detail=f"Optional dependency missing for {dep_name}. {hint}")


def _infer_nf_freq(dt_index: pd.DatetimeIndex) -> str:
    """Infer a NeuralForecast frequency string."""
    try:
        f = pd.infer_freq(dt_index)
    except Exception:
        f = None
    if not f:
        return "D"
    # Use the base alias (e.g., 'D', 'H', 'M', 'MS', 'W-SUN', 'QS-DEC'...)
    # NeuralForecast uses pandas offsets; we'll pass the inferred alias directly when possible.
    return f
# In-memory cache to avoid retraining deep models when the same request repeats.
# Note: Render instances can restart; this cache is best-effort per-process.
import hashlib
_DEEP_CACHE: Dict[str, Any] = {}
_DEEP_CACHE_TTL_SECONDS = int(os.getenv("DEEP_CACHE_TTL_SECONDS", "3600"))


def _deep_cache_key(model_name: str, y2: pd.Series, h: int, input_size: int, max_steps: int, extra: str = "") -> str:
    hsh = hashlib.sha1()
    # values
    hsh.update(np.asarray(y2.values, dtype=np.float64).tobytes())
    # index bounds + length (stable-ish)
    hsh.update(str(y2.index[0]).encode('utf-8'))
    hsh.update(str(y2.index[-1]).encode('utf-8'))
    hsh.update(str(len(y2)).encode('utf-8'))
    hsh.update(str(h).encode('utf-8'))
    hsh.update(str(input_size).encode('utf-8'))
    hsh.update(str(max_steps).encode('utf-8'))
    hsh.update(model_name.encode('utf-8'))
    if extra:
        hsh.update(extra.encode('utf-8'))
    return hsh.hexdigest()


def _deep_cache_get(key: str) -> Optional[Dict[str, Any]]:
    try:
        item = _DEEP_CACHE.get(key)
        if not item:
            return None
        ts = float(item.get('ts', 0))
        if (time.time() - ts) > _DEEP_CACHE_TTL_SECONDS:
            _DEEP_CACHE.pop(key, None)
            return None
        return item.get('value')
    except Exception:
        return None


def _deep_cache_set(key: str, value: Dict[str, Any]) -> None:
    try:
        _DEEP_CACHE[key] = {'ts': time.time(), 'value': value}
    except Exception:
        pass




def _nf_forecast_univariate(payload: DeepForecastIn, model_name: str, model_ctor):
    """Run a NeuralForecast model on a single time series."""
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    if len(y2) < 20:
        raise HTTPException(status_code=400, detail="Need at least 20 non-missing points for deep forecasting models.")
    # limit training length
    if len(y2) > int(payload.max_train_points):
        y2 = y2.iloc[-int(payload.max_train_points):]
        warnings.append(f"Trimmed series to last {payload.max_train_points} points for deep model training.")
    h = int(payload.horizon)
    if h < 1:
        raise HTTPException(status_code=422, detail="horizon must be >= 1")

    # input window (lags)
    # A simple robust choice: at least 24, or 2*h, but not more than ~half the data.
    input_size = max(24, 2 * h)
    input_size = min(input_size, max(8, len(y2) // 2))
    input_size = min(input_size, int(os.getenv("DEEP_MAX_INPUT_SIZE", "96")))
    # training steps (keep it small for API usage)
    if payload.max_steps is not None:
        max_steps = int(payload.max_steps)
    else:
        env_steps = os.getenv("DEEP_MAX_STEPS", "").strip()
        if env_steps:
            max_steps = int(env_steps)
        else:
            # Default: keep training very small for synchronous API usage
            max_steps = int(max(20, min(200, int(payload.epochs or 5) * 20)))
    # Hard safety cap
    max_steps = int(max(10, min(max_steps, 300)))
    # build df in NeuralForecast format
    df = pd.DataFrame({"unique_id": "ts", "ds": y2.index.to_pydatetime(), "y": y2.values.astype("float64")})
    nf_freq = _infer_nf_freq(pd.DatetimeIndex(df["ds"]))

    # cache key (best-effort)
    cache_key = _deep_cache_key(model_name, y2, h, input_size, max_steps)
    cached = _deep_cache_get(cache_key)
    if cached is not None:
        warnings.append("Deep model result served from cache (no retraining).")
        return ApiOut(ok=True, result=cached, warnings=warnings)

    try:
        from neuralforecast import NeuralForecast
    except Exception:
        _require_optional("neuralforecast", "Install neuralforecast in requirements.txt and redeploy.")

    # create model instance
    try:
        model = model_ctor(h=h, input_size=input_size, max_steps=max_steps)
    except TypeError:
        # some models use different signature; try without max_steps
        model = model_ctor(h=h, input_size=input_size)

    nf = NeuralForecast(models=[model], freq=nf_freq)
    nf.fit(df=df)
    fcst = nf.predict()
    # fcst: columns include unique_id, ds and model name (often model.__class__.__name__ or alias)
    # Find the prediction column:
    pred_col = None
    for c in fcst.columns:
        if c.lower() == model_name.lower():
            pred_col = c
            break
    if pred_col is None:
        # fallback: last column that's not unique_id/ds
        cand = [c for c in fcst.columns if c not in ("unique_id", "ds")]
        pred_col = cand[-1] if cand else None
    if pred_col is None:
        raise RuntimeError("NeuralForecast returned no prediction column.")

    yhat = fcst[pred_col].to_numpy(dtype="float64")
    x_fc = _forecast_index(y2, h)
    out = {"forecast": {"x": x_fc, "y": _as_float_list(yhat)}, "model": model_name, "horizon": h, "input_size": input_size, "max_steps": max_steps}
    _deep_cache_set(cache_key, out)
    return ApiOut(ok=True, result=out, warnings=warnings)


@router.post("/tsa/V_neuralprophet_forecast", response_model=ApiOut)
def tsa_v_neuralprophet_forecast(payload: DeepForecastIn):
    """NeuralProphet forecast (requires neuralprophet)."""
    y, warnings, freq = _prep_series(payload.series)
    y2 = y.dropna()
    if len(y2) < 20:
        raise HTTPException(status_code=400, detail="Need at least 20 non-missing points for NeuralProphet.")
    if len(y2) > int(payload.max_train_points):
        y2 = y2.iloc[-int(payload.max_train_points):]
        warnings.append(f"Trimmed series to last {payload.max_train_points} points for NeuralProphet training.")
    h = int(payload.horizon)
    if h < 1:
        raise HTTPException(status_code=422, detail="horizon must be >= 1")

    try:
        from neuralprophet import NeuralProphet
    except Exception:
        _require_optional("neuralprophet", "Add neuralprophet to requirements.txt and redeploy.")

    # Build df
    df = pd.DataFrame({"ds": y2.index.to_pydatetime(), "y": y2.values.astype("float64")})

    cache_key = _deep_cache_key('neuralprophet', y2, h, input_size=min(max(12, 2*h), max(8, len(y2)//2)), max_steps=int(os.getenv('NEURALPROPHET_EPOCHS', str(payload.epochs))))
    cached = _deep_cache_get(cache_key)
    if cached is not None:
        warnings.append("NeuralProphet result served from cache (no retraining).")
        return ApiOut(ok=True, result=cached, warnings=warnings)

    # Minimal settings for API usage
    m = NeuralProphet(
        n_lags=min(max(12, 2*h), max(8, len(y2)//2)),
        n_forecasts=h,
        epochs=int(os.getenv("NEURALPROPHET_EPOCHS", str(payload.epochs))),
        learning_rate=float(os.getenv("NEURALPROPHET_LR", "1e-2")),
    )
    m.fit(df, freq=freq if freq else None)

    future = m.make_future_dataframe(df, periods=h, n_historic_predictions=False)
    forecast = m.predict(future)
    # forecast columns: yhat1..yhatH
    col = f"yhat{h}"
    if col not in forecast.columns:
        # fallback: pick the last yhat column
        yhats = [c for c in forecast.columns if c.startswith("yhat")]
        if not yhats:
            raise RuntimeError("NeuralProphet returned no yhat columns.")
        col = yhats[-1]
    yhat = forecast[col].to_numpy(dtype="float64")
    x_fc = _forecast_index(y2, h)
    out = {"forecast": {"x": x_fc, "y": _as_float_list(yhat)}, "model": "neuralprophet"}
    _deep_cache_set(cache_key, out)
    return ApiOut(ok=True, result=out, warnings=warnings)


@router.post("/tsa/W_nhits_forecast", response_model=ApiOut)
def tsa_w_nhits_forecast(payload: DeepForecastIn):
    """NHITS forecast via NeuralForecast (requires neuralforecast)."""
    try:
        from neuralforecast.models import NHITS
    except Exception:
        _require_optional("neuralforecast", "Install neuralforecast (includes NHITS) in requirements.txt and redeploy.")
    return _nf_forecast_univariate(payload, "NHITS", NHITS)


@router.post("/tsa/X_patchtst_or_tide_forecast", response_model=ApiOut)
def tsa_x_patchtst_or_tide_forecast(payload: DeepForecastIn):
    """PatchTST or TiDE forecast via NeuralForecast (requires neuralforecast)."""
    name = (payload.model_name or "patchtst").lower().strip()
    try:
        from neuralforecast import NeuralForecast  # noqa: F401
        from neuralforecast.models import PatchTST, TiDE
    except Exception:
        _require_optional("neuralforecast", "Install neuralforecast in requirements.txt and redeploy.")

    if name == "tide":
        return _nf_forecast_univariate(payload, "TiDE", TiDE)
    else:
        return _nf_forecast_univariate(payload, "PatchTST", PatchTST)


@router.post("/tsa/Y_transformer_family_forecast", response_model=ApiOut)
def tsa_y_transformer_family_forecast(payload: DeepForecastIn):
    """TimesNet / iTransformer / SOFTS via NeuralForecast (requires neuralforecast)."""
    name = (payload.model_name or "timesnet").lower().strip()
    try:
        from neuralforecast import NeuralForecast  # noqa: F401
        from neuralforecast.models import TimesNet, iTransformer, SOFTS
    except Exception:
        _require_optional("neuralforecast", "Install neuralforecast in requirements.txt and redeploy.")

    if name == "itransformer":
        # iTransformer expects n_series; we'll run it in univariate mode with n_series=1
        def ctor(**kwargs):
            return iTransformer(n_series=1, **kwargs)
        return _nf_forecast_univariate(payload, "iTransformer", ctor)

    if name == "softs":
        def ctor(**kwargs):
            return SOFTS(n_series=1, **kwargs)
        return _nf_forecast_univariate(payload, "SOFTS", ctor)

    # default TimesNet (univariate)
    return _nf_forecast_univariate(payload, "TimesNet", TimesNet)
