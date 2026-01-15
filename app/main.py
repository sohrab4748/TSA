from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.tsa_a_to_m import router as tsa_router
from app.analysis_run import router as run_router

app = FastAPI(title="TSA Dashboard API (A–M)", version="0.1.0")

@app.get("/routes")
def routes():
    out = []
    for r in app.routes:
        if hasattr(r, "methods"):
            out.append({"path": r.path, "methods": sorted(list(r.methods))})
    return sorted(out, key=lambda x: x["path"])
ALLOWED_ORIGINS = [
    "https://tsa.agrimetsoft.com",
    # Local dev (optional)
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tsa_router, prefix="/analysis", tags=["tsa"])

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


