import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictRequest, PredictResponse
from .model import model

app = FastAPI(title="ML API", version="0.1.0")

NETLIFY_ORIGIN = os.getenv("NETLIFY_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[NETLIFY_ORIGIN] if NETLIFY_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    y = model.predict(req.feature_1, req.feature_2)
    return PredictResponse(prediction=y, model_version=model.version)
