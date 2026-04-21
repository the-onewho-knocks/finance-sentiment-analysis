import os
from pathlib import Path

# ── Root paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_RAW_DIR      = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR        = BASE_DIR / "data" / "models"

# Auto-create directories if they don't exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── File names ───────────────────────────────────────────────
RAW_DATA_FILE       = DATA_RAW_DIR / "reddit_posts.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "processed_posts.csv"
FEATURES_FILE       = DATA_PROCESSED_DIR / "features.pkl"
TFIDF_VECTORIZER    = MODELS_DIR / "tfidf_vectorizer.pkl"

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "logistic_regression.pkl",
    "naive_bayes":         MODELS_DIR / "naive_bayes.pkl",
    "random_forest":       MODELS_DIR / "random_forest.pkl",
}

# ── Reddit API Config (No longer requires PRAW keys) ─────────
# Scraper now uses Pullpush API which does not require authentication.

# ── Subreddits to scrape ─────────────────────────────────────
TARGET_SUBREDDITS = ["stocks", "IndianStockMarket", "investing", "wallstreetbets"]
POSTS_PER_SUBREDDIT = 100

# ── ML Config ────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000   # TF-IDF vocabulary size