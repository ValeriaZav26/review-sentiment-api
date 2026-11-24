"""Train, compare and save the sentiment model.

Usage:
    python -m src.train                       # Amazon Polarity subset (default)
    python -m src.train --n-train 100000      # bigger sample
    python -m src.train --csv data/my.csv     # your own CSV with text,label
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from src.data import LABELS, clean_text, load_amazon_polarity, load_csv

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=200_000,
        sublinear_tf=True,
    )


def evaluate(name: str, model, X_test, y_test, train_time: float) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "train_time_s": round(train_time, 1),
    }


def metrics_table(results: list[dict]) -> str:
    rows = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time, s |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for r in results:
        rows.append(f"| {r['model']} | {r['accuracy']} | {r['precision']} | {r['recall']} | "
                    f"{r['f1']} | {r['roc_auc']} | {r['train_time_s']} |")
    return "\n".join(rows)


def update_readme(table: str, n_train: int, n_test: int) -> None:
    readme = ROOT / "README.md"
    start, end = "<!-- METRICS:START -->", "<!-- METRICS:END -->"
    text = readme.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return
    block = f"{start}\nTest set: {n_test:,} reviews, trained on {n_train:,} reviews.\n\n{table}\n{end}"
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    readme.write_text(before + block + after, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="CSV with text,label columns instead of Amazon Polarity")
    parser.add_argument("--n-train", type=int, default=60_000)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-readme", action="store_true", help="do not write metrics into README.md")
    args = parser.parse_args()

    if args.csv:
        train, test = load_csv(args.csv, seed=args.seed)
    else:
        train, test = load_amazon_polarity(args.n_train, args.n_test, seed=args.seed)
    X_train, y_train = train["text"], train["label"]
    X_test, y_test = test["text"], test["label"]
    print(f"Train: {len(train):,}  Test: {len(test):,}  Positive share (train): {y_train.mean():.2%}")

    results, models = [], {}

    # 1. Baseline: TF-IDF + Complement Naive Bayes
    nb = Pipeline([("tfidf", make_vectorizer()), ("clf", ComplementNB())])
    t0 = time.time()
    nb.fit(X_train, y_train)
    results.append(evaluate("TF-IDF + Naive Bayes (baseline)", nb, X_test, y_test, time.time() - t0))
    models["nb"] = nb

    # 2. TF-IDF + Logistic Regression, C tuned with 5-fold CV on the training set only
    lr = Pipeline([
        ("tfidf", make_vectorizer()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])
    grid = GridSearchCV(lr, {"clf__C": [0.5, 1, 2, 4, 8]}, cv=5, scoring="f1", n_jobs=-1)
    t0 = time.time()
    grid.fit(X_train, y_train)
    print(f"Best C: {grid.best_params_['clf__C']}  CV F1: {grid.best_score_:.4f}")
    results.append(evaluate("TF-IDF + Logistic Regression", grid.best_estimator_, X_test, y_test, time.time() - t0))
    models["lr"] = grid.best_estimator_

    best = max(results, key=lambda r: r["f1"])
    best_model = models["lr"] if "Logistic" in best["model"] else models["nb"]

    # Overfitting check: train vs test F1 for the chosen model
    train_f1 = f1_score(y_train, best_model.predict(X_train))
    print(f"Chosen: {best['model']}  train F1 {train_f1:.4f} vs test F1 {best['f1']:.4f}")

    # Reports
    REPORTS.mkdir(exist_ok=True)
    pred = best_model.predict(X_test)
    print(classification_report(y_test, pred, target_names=[LABELS[0], LABELS[1]], digits=4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=[LABELS[0], LABELS[1]], cmap="Blues")
    plt.title(best["model"])
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=150)

    # Most informative words for the linear model
    if "Logistic" in best["model"]:
        vocab = best_model.named_steps["tfidf"].get_feature_names_out()
        coef = best_model.named_steps["clf"].coef_[0]
        order = coef.argsort()
        top = {"negative": vocab[order[:20]].tolist(), "positive": vocab[order[-20:][::-1]].tolist()}
        (REPORTS / "top_words.json").write_text(json.dumps(top, indent=2), encoding="utf-8")

    summary = {
        "n_train": len(train),
        "n_test": len(test),
        "chosen_model": best["model"],
        "train_f1": round(train_f1, 4),
        "results": results,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": best_model, "model_name": best["model"], "metrics": best}, MODEL_PATH, compress=3)
    print(f"Saved model to {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    table = metrics_table(results)
    print("\n" + table)
    if not args.no_readme:
        update_readme(table, len(train), len(test))
        print("README.md metrics updated")


if __name__ == "__main__":
    main()
