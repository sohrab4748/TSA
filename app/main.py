from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.tsa_a_to_m import router as tsa_router
from app.analysis_run import router as run_router
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from fastapi import HTTPException

app = FastAPI(title="TSA Dashboard API (A–M)", version="0.1.0")

@app.get("/routes")
def routes():
    out = []
    for r in app.routes:
        if hasattr(r, "methods"):
            out.append({"path": r.path, "methods": sorted(list(r.methods))})
    return sorted(out, key=lambda x: x["path"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tsa.agrimetsoft.com"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tsa_router, prefix="/analysis", tags=["tsa"])


class SeriesIn(BaseModel):
    dates: List[str]
    values: List[float]

class AnalysisRunRequest(BaseModel):
    analysis_key: str
    series: SeriesIn
    params: Dict[str, Any] = Field(default_factory=dict)

@app.post("/analysis/run")
def analysis_run(req: AnalysisRunRequest):
    """
    Frontend-friendly single endpoint.
    Dispatches to your existing /analysis/tsa/* handlers.
    """
    key = (req.analysis_key or "").strip().lower()

    # Import your actual handler functions from tsa_a_to_m
    # IMPORTANT: change these imports/names to match your code.
    from app.tsa_a_to_m import (
        B_summary,
        F_stationarity,
        G_autocorr,
        D_decompose,
        J_arima_forecast,
    )

    if key == "summary":
        return B_summary(req)

    if key in ("adf_test", "stationarity"):
        return F_stationarity(req)

    if key in ("acf_pacf", "autocorr"):
        return G_autocorr(req)

    if key in ("stl", "decompose", "decomposition"):
        return D_decompose(req)

    if key in ("forecast_arima", "arima_forecast"):
        return J_arima_forecast(req)

    raise HTTPException(status_code=400, detail=f"Unknown analysis_key: {req.analysis_key}")

@app.get("/")
def root():
    return {
        "name": "TSA Dashboard API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok"}


