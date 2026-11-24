"""Shared fixtures: a tiny model trained on toy data, so tests run in seconds."""
import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.train import make_vectorizer

POSITIVE = ["great product, works perfectly", "love it, excellent quality", "amazing, highly recommend",
            "very good value, happy with it", "perfect, fast delivery and great"]
NEGATIVE = ["terrible, broke after a day", "awful quality, waste of money", "stopped working, very bad",
            "do not buy, poor and cheap", "bad product, returned it"]


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    texts = (POSITIVE + NEGATIVE) * 3
    labels = ([1] * len(POSITIVE) + [0] * len(NEGATIVE)) * 3
    vec = make_vectorizer()
    vec.set_params(min_df=1, max_df=1.0)
    pipe = Pipeline([("tfidf", vec), ("clf", LogisticRegression(max_iter=1000))]).fit(texts, labels)
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump({"pipeline": pipe, "model_name": "tiny-test-model", "metrics": {"f1": 1.0}}, path)
    return path
