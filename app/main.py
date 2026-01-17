import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# The TSA router (your big tsa_a_to_m.py). It already includes the billing webhook.
from app.tsa_a_to_m import router as tsa_router

app = FastAPI(title="TSA Dashboard API")

# ---- CORS ----
# Override via env var (comma-separated) if needed.
# Example: CORS_ALLOW_ORIGINS=https://tsa.agrimetsoft.com,http://localhost:5500
origins_env = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "https://tsa.agrimetsoft.com,"
    "http://localhost:5500,http://127.0.0.1:5500,"
    "http://localhost:8000,http://127.0.0.1:8000",
)
allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Your routes are already defined under router paths like /analysis/tsa/...
app.include_router(tsa_router, prefix="/analysis")


@app.get("/")
def root():
    return {
        "name": "TSA Dashboard API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


