"""Data loading for the review sentiment model.

Default source: the Amazon Polarity dataset from Hugging Face
(individual product reviews labelled 0 = negative, 1 = positive).
Alternatively, any CSV with `text` and `label` columns can be used.
"""
from __future__ import annotations

import re

import pandas as pd

LABELS = {0: "negative", 1: "positive"}


def clean_text(text: str) -> str:
    """Light normalisation: lowercase, strip URLs and HTML, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_amazon_polarity(n_train: int, n_test: int, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download Amazon Polarity and return random subsets of train and test."""
    from datasets import load_dataset  # imported lazily: only needed for training

    # One of four train shards (~900k reviews) is plenty for sampling and avoids a 1 GB download
    ds = load_dataset(
        "fancyzhx/amazon_polarity",
        data_files={
            "train": "amazon_polarity/train-00000-of-00004.parquet",
            "test": "amazon_polarity/test-00000-of-00001.parquet",
        },
    )
    train = ds["train"].shuffle(seed=seed).select(range(n_train)).to_pandas()
    test = ds["test"].shuffle(seed=seed).select(range(n_test)).to_pandas()
    for df in (train, test):
        df["text"] = df["title"].fillna("") + ". " + df["content"].fillna("")
    return train[["text", "label"]], test[["text", "label"]]


def load_csv(path: str, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a CSV with `text` and `label` (0/1) columns and split it with stratification."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {missing}")
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset="text")
    df["label"] = df["label"].astype(int)
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return train[["text", "label"]], test[["text", "label"]]
