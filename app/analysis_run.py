from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

class SeriesIn(BaseModel):
    dates: List[str]
    values: List[float]

class AnalysisRunRequest(BaseModel):
    analysis_key: str
    series: SeriesIn
    params: Dict[str, Any] = Field(default_factory=dict)

@router.post("/run")
def analysis_run(req: AnalysisRunRequest):
    """
    Single entry point used by the HTML dashboard.
    Dispatch to existing endpoints/functions inside tsa_a_to_m.

    IMPORTANT: You must map analysis_key values to your real functions.
    """

    key = req.analysis_key.lower().strip()

    # Example dispatch (replace with your real functions)
    # from app.tsa_a_to_m import summary, adf_test, acf_pacf, stl, forecast_arima

    if key == "summary":
        # return summary(req.series.dates, req.series.values, req.params)
        raise HTTPException(status_code=501, detail="summary not wired yet")

    if key == "adf_test":
        raise HTTPException(status_code=501, detail="adf_test not wired yet")

    if key == "acf_pacf":
        raise HTTPException(status_code=501, detail="acf_pacf not wired yet")

    if key == "stl":
        raise HTTPException(status_code=501, detail="stl not wired yet")

    if key == "forecast_arima":
        raise HTTPException(status_code=501, detail="forecast_arima not wired yet")

    raise HTTPException(status_code=400, detail=f"Unknown analysis_key: {req.analysis_key}")
