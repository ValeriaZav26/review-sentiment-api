# 💬 Review Sentiment API

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LogReg-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Train and test](https://github.com/ValeriaZav26/review-sentiment-api/actions/workflows/train.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end NLP project: a model that classifies product reviews as **positive** or **negative**, trained with scikit-learn and **deployed as a REST API** with FastAPI and Docker.

**Live demo:** _add your Render URL here after deploying_ → `/docs` opens the interactive Swagger UI.

## 🧭 What this project shows

* **NLP pipeline:** text cleaning → TF-IDF (unigrams + bigrams) → linear classifier, all inside one sklearn `Pipeline`.
* **Model comparison:** Naive Bayes baseline vs. Logistic Regression tuned with `GridSearchCV` (5-fold CV, F1).
* **Honest evaluation:** held-out test set, Precision / Recall / F1 / ROC-AUC, confusion matrix and a train-vs-test check for overfitting.
* **Deployment:** the saved model is served by a FastAPI app with input validation, packaged in Docker and covered by tests.
* **CI/CD:** a GitHub Actions workflow runs the tests, retrains the model and commits the fresh model and metrics on every code change.

## 📊 Results

<!-- METRICS:START -->
_Run `python -m src.train` — this table is filled in automatically._
<!-- METRICS:END -->

The confusion matrix is saved to `reports/confusion_matrix.png`, and the words that push predictions most strongly towards each class are in `reports/top_words.json`.

## 🗂 Data

[Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) (Hugging Face): Amazon product reviews labelled 0 = negative (1–2 stars) and 1 = positive (4–5 stars). The training script downloads one of the four train shards and the test file, then samples 60,000 reviews for training and 10,000 for testing (configurable), so training takes a few minutes.

You can also train on your own data: any CSV with `text` and `label` (0/1) columns via `--csv`.

## 🏗 Project structure

```
review-sentiment-api/
├── src/
│   ├── data.py          # loading and text cleaning
│   ├── train.py         # training, model comparison, reports, saving the model
│   └── predictor.py     # loads the saved model and makes predictions
├── app/
│   └── main.py          # FastAPI app: /health, /model-info, /predict, /predict-batch
├── tests/               # pytest: model wrapper and API
├── models/model.joblib  # trained pipeline (created by src/train.py)
├── reports/             # metrics.json, confusion_matrix.png, top_words.json
├── .github/workflows/train.yml  # CI: tests → training → commit model and metrics
├── Dockerfile
├── requirements.txt     # runtime dependencies (pinned)
└── requirements-dev.txt # + training and test dependencies
```

## 🚀 Quick start

```bash
git clone https://github.com/ValeriaZav26/review-sentiment-api.git
cd review-sentiment-api

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m src.train                # downloads data, trains, saves models/model.joblib
pytest                             # runs the tests
uvicorn app.main:app --reload      # API at http://127.0.0.1:8000/docs
```

**No local Python?** Open `train_in_colab.ipynb` in Google Colab: it installs the dependencies, trains the model, runs the tests and lets you download `model.joblib`.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Stopped working after two days, very disappointed."}'
```

```json
// example response
{"label": "negative", "positive_probability": 0.03}
```

### Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| GET | `/health` | Service status and loaded model name |
| GET | `/model-info` | Test metrics of the deployed model |
| POST | `/predict` | `{"text": "..."}` → label and probability |
| POST | `/predict-batch` | `{"texts": ["...", "..."]}` → list of predictions (up to 100) |

## 🐳 Docker

```bash
docker build -t review-sentiment-api .
docker run -p 8000:8000 review-sentiment-api
```

## ☁️ Deployment (Render, free tier)

1. Push the repository to GitHub. The **Train and test** workflow trains the model and commits `models/model.joblib`.
2. On [render.com](https://render.com): **New → Web Service**, connect this repository; Render detects the `Dockerfile` automatically.
3. Choose the free instance type and deploy. The public URL + `/docs` is your live demo.

The free instance sleeps when idle, so the first request after a pause can take up to a minute.

## ⚠️ Limitations and next steps

* The model sees words and short phrases, not meaning: sarcasm and negations spread across a sentence can fool it.
* Only English and only two classes (no neutral).
* Next step: fine-tune a small transformer (e.g. DistilBERT) and compare quality and latency with the TF-IDF model.

## 📝 License

[MIT](LICENSE)
