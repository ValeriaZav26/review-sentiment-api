from src.data import clean_text
from src.predictor import SentimentModel


def test_clean_text():
    assert clean_text("  Great <b>Product</b>!  See https://x.com  ") == "great product ! see"


def test_predict_format_and_direction(tiny_model_path):
    model = SentimentModel(tiny_model_path)
    pos, neg = model.predict(["excellent quality, love it", "terrible, broke and stopped working"])
    assert set(pos) == {"label", "positive_probability"}
    assert 0.0 <= pos["positive_probability"] <= 1.0
    assert pos["label"] == "positive"
    assert neg["label"] == "negative"
