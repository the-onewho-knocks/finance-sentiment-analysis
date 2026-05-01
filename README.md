# Finance Sentiment Intelligence System

A machine-learning pipeline that scrapes Reddit finance communities, classifies post sentiment (positive / neutral / negative), and exposes results through a FastAPI REST API with a live HTML dashboard and full Prometheus + Grafana monitoring.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Install Dependencies](#install-dependencies)
- [Running the Pipeline](#running-the-pipeline)
- [Running the API](#running-the-api)
- [Docker & Monitoring Stack](#docker--monitoring-stack)
- [API Reference](#api-reference)
- [ML Models](#ml-models)
- [Visualizations](#visualizations)
- [Project Roadmap](#project-roadmap)

---

## Overview

The system is organized as a multi-phase pipeline:

| Phase | Script | What it does |
|-------|--------|--------------|
| 1 | `backend/run_phase1.py` | Scrape Reddit posts via PRAW |
| 2 | `backend/run_phase2.py` | Clean, wrangle & encode the raw data |
| 3 | `backend/run_phase3.py` | TF-IDF vectorization + numeric feature scaling |
| 4 | `backend/run_phase4.py` | Train & evaluate Logistic Regression, Naive Bayes, Random Forest |
| 5 | `backend/run_phase5.py` | Hyperparameter tuning (GridSearchCV) |
| — | `backend/main.py` | FastAPI server — serves predictions & analytics |

---

## Project Structure

```
finance-sentiment-intelligence/
├── data/
│   ├── raw/                  # Scraped Reddit CSVs (git-ignored)
│   ├── processed/            # Cleaned & encoded data
│   └── models/               # Saved .pkl model files
│
├── src/
│   ├── data/                 # reddit_scraper.py, data_loader.py
│   ├── preprocessing/        # cleaner.py, tokenizer.py, wrangler.py
│   ├── features/             # feature_engineer.py (TF-IDF + scaling)
│   ├── models/               # train.py, tuner.py, evaluator.py
│   ├── visualization/        # visualizer.py (10 plot types)
│   └── utils/                # config.py, logger.py
│
├── backend/
│   ├── main.py               # FastAPI app entry point
│   ├── routes/               # analyze.py, stats.py
│   ├── schemas/              # Pydantic request/response models
│   └── services/             # predictor.py (model loading & inference)
│
├── frontend/
│   ├── index.html
│   ├── dashboard.html
│   ├── css/style.css
│   └── js/app.js
│
├── tests/
│   ├── test_cleaner.py
│   ├── test_api.py
│   └── test_models.py
│
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/
│
├── .env                      # Secrets (git-ignored)
├── requirements.txt
├── Dockerfile
├── compose.yml
└── run_pipeline.py
```

---

## Tech Stack

**Data & ML**
- [PRAW](https://praw.readthedocs.io/) — Reddit API wrapper
- pandas, NumPy, scikit-learn — data processing & ML
- NLTK — text cleaning & tokenization
- WordCloud, Matplotlib, Seaborn, Plotly — visualizations

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn — REST API
- Pydantic — request/response validation
- joblib — model serialization
- aiohttp + tenacity — async scraping with retries

**Monitoring**
- Prometheus (`prometheus-client`, `prometheus-fastapi-instrumentator`)
- Grafana — dashboards at `http://localhost:3000`
- Pushgateway — metrics from short-lived pipeline jobs

**Infrastructure**
- Docker + Docker Compose

---

## Getting Started

### Prerequisites

- Python 3.10+
- A Reddit account with API credentials ([create an app](https://www.reddit.com/prefs/apps))
- Docker & Docker Compose (optional, for the monitoring stack)

### Environment Variables

Copy `.env.example` to `.env` and fill in your Reddit credentials:

```env
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=FinanceSentimentBot/1.0
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Run each phase in sequence. Each phase saves its output to disk so you can re-run individual phases without starting over.

```bash
# Phase 1 — Scrape Reddit
python -m backend.run_phase1

# Phase 2 — Preprocess & clean
python -m backend.run_phase2

# Phase 3 — Feature engineering (TF-IDF + scaling)
python -m backend.run_phase3

# Phase 4 — Train models
python -m backend.run_phase4

# Phase 5 — Hyperparameter tuning (optional, slow)
python -m backend.run_phase5
```

> **Tip:** You can also run the full pipeline end-to-end with `python run_pipeline.py` once that script is populated.

---

## Running the API

```bash
uvicorn backend.main:app --reload --port 8000
```

Interactive docs are available at:
- Swagger UI → `http://localhost:8000/docs`
- ReDoc → `http://localhost:8000/redoc`

---

## Docker & Monitoring Stack

The `compose.yml` spins up the FastAPI app, Prometheus, Grafana, and Pushgateway together:

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI app | `http://localhost:8000` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` (admin / admin) |
| Pushgateway | `http://localhost:9091` |

To run **only** the monitoring stack while running the app locally:

```bash
docker-compose up prometheus grafana pushgateway
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Analyze text sentiment |
| `GET` | `/analyze/models` | List available ML models |
| `GET` | `/stats` | Full dataset analytics report |
| `GET` | `/stats/plot/{name}` | Serve a specific visualization |
| `GET` | `/stats/plots` | List all available plots |
| `GET` | `/health` | API health check |
| `GET` | `/metrics` | Prometheus scrape endpoint |

### Example — Analyze sentiment

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Markets are rallying hard today!", "model": "best"}'
```

```json
{
  "sentiment": "positive",
  "confidence": 0.91,
  "model_used": "logistic_regression"
}
```

---

## ML Models

Three classifiers are trained on TF-IDF + numeric features and evaluated by weighted F1-score. The best-performing model is automatically saved as `best`:

| Model | Notes |
|-------|-------|
| Logistic Regression | Tuned via GridSearchCV |
| Naive Bayes | Tuned via GridSearchCV |
| Random Forest | Tuned via GridSearchCV |
| **best** | Auto-selected highest F1 model |

**Preprocessing steps applied before training:**
1. URL / HTML / emoji removal
2. Lowercasing & punctuation stripping
3. Noise word removal (`removed`, `deleted`, `amp`, etc.)
4. NLTK tokenization
5. IQR outlier removal on `score` and `num_comments`
6. Z-score outlier removal on `text_length`
7. Short-text filtering (< 3 words dropped)
8. Sentiment label encoding

**Feature matrix:** TF-IDF sparse vectors concatenated with scaled numeric features (`score`, `num_comments`, `text_length`, `word_count`, `engagement`).

---

## Visualizations

The `/stats` endpoint and `src/visualization/visualizer.py` produce the following plots, saved to `data/plots/`:

| Plot | File |
|------|------|
| Sentiment distribution (bar) | `sentiment_distribution.png` |
| Sentiment by subreddit (stacked bar) | `sentiment_by_subreddit.png` |
| Word clouds per sentiment | `wordclouds.png` |
| Sentiment over time | `sentiment_over_time.png` |
| Engagement analysis (violin / box / scatter) | `engagement_analysis.png` |
| Word count distribution | `text_length_distribution.png` |
| Top unigrams per sentiment | `top_unigrams.png` |
| Top bigrams per sentiment | `top_bigrams.png` |
| Hourly activity heatmap | `hourly_heatmap.png` |
| Sentiment metric radar chart | `radar_chart.png` |

Plots are also served statically at `/plots/<filename>` and listed via `GET /stats/plots`.

---

## Project Roadmap

- [ ] Populate `run_pipeline.py` for single-command end-to-end execution
- [ ] Add test coverage for `test_cleaner.py`, `test_api.py`, `test_models.py`
- [ ] Add VADER / FinBERT as additional model options
- [ ] Stream live Reddit posts to update model predictions in real time
- [ ] Add authentication to the API
- [ ] Publish Grafana dashboard JSON for one-click import

---

## Author
**Hardik Borse** | [LinkedIn](https://www.linkedin.com/in/hardik-borse-aa7729324/) | [Email](mailto:borsehardik@gmail.com)

## License
This project is licensed under the **Apache License 2.0**.
