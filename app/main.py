from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.tsa_a_to_m import router as tsa_router

app = FastAPI(
    title="TSA Dashboard API (A–M)",
    version="0.1.0",
)

# CORS: adjust origins later (e.g., your frontend domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tsa_router, prefix="/analysis", tags=["tsa"])

@app.get("/health")
def health():
    return {"status": "ok"}

