import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tiny_model_path, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "MODEL_PATH", str(tiny_model_path))
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "tiny-test-model"}


def test_predict(client):
    r = client.post("/predict", json={"text": "excellent quality, love it"})
    assert r.status_code == 200
    assert r.json()["label"] == "positive"


def test_predict_batch(client):
    r = client.post("/predict-batch", json={"texts": ["love it", "terrible, broke"]})
    assert r.status_code == 200
    assert [p["label"] for p in r.json()] == ["positive", "negative"]


def test_validation_error(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
