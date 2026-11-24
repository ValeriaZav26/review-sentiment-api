"""REST API for review sentiment classification.

Run locally:  uvicorn app.main:app --reload
Docs:         http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predictor import DEFAULT_MODEL_PATH, SentimentModel

MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = SentimentModel(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Review Sentiment API",
    description="Classifies product reviews as positive or negative (TF-IDF + Logistic Regression).",
    version="1.0.0",
    lifespan=lifespan,
)


class Review(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000, examples=["Great quality, arrived quickly. Would buy again!"])


class Reviews(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class Prediction(BaseModel):
    label: str
    positive_probability: float


@app.get("/health")
def health() -> dict:
    model = state.get("model")
    return {"status": "ok" if model else "loading", "model": model.name if model else None}


@app.get("/model-info")
def model_info() -> dict:
    model = state["model"]
    return {"model": model.name, "test_metrics": model.metrics}


@app.post("/predict", response_model=Prediction)
def predict(review: Review) -> dict:
    return state["model"].predict([review.text])[0]


@app.post("/predict-batch", response_model=list[Prediction])
def predict_batch(reviews: Reviews) -> list[dict]:
    if any(not t.strip() for t in reviews.texts):
        raise HTTPException(status_code=422, detail="Texts must not be empty")
    return state["model"].predict(reviews.texts)
