"""Thin wrapper around the saved pipeline, shared by the API and tests."""
from __future__ import annotations

from pathlib import Path

import joblib

from src.data import LABELS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


class SentimentModel:
    def __init__(self, path: str | Path = DEFAULT_MODEL_PATH):
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.name = bundle.get("model_name", "unknown")
        self.metrics = bundle.get("metrics", {})

    def predict(self, texts: list[str]) -> list[dict]:
        proba = self.pipeline.predict_proba(texts)[:, 1]
        return [
            {
                "label": LABELS[int(p >= 0.5)],
                "positive_probability": round(float(p), 4),
            }
            for p in proba
        ]
