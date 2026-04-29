"""
src/metrics/prometheus_metrics.py
──────────────────────────────────
Central Prometheus metrics registry.
All counters, gauges, and histograms are defined here so they are
shared across the scraper, backend, and pipeline.

Usage:
    from src.metrics.prometheus_metrics import METRICS
    METRICS.posts_scraped.labels(subreddit="stocks").inc()
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    REGISTRY,
)

# ── Scraper metrics ────────────────────────────────────────────────────────────

posts_scraped_total = Counter(
    "reddit_posts_scraped_total",
    "Total number of Reddit posts scraped",
    ["subreddit"],
)

comments_scraped_total = Counter(
    "reddit_comments_scraped_total",
    "Total number of Reddit comments scraped",
    ["subreddit"],
)

scrape_errors_total = Counter(
    "reddit_scrape_errors_total",
    "Total number of scraping errors",
    ["subreddit", "error_type"],
)

scrape_rate_limit_hits_total = Counter(
    "reddit_scrape_rate_limit_hits_total",
    "Total number of rate limit (429) responses from PullPush API",
    ["subreddit"],
)

scrape_duration_seconds = Histogram(
    "reddit_scrape_duration_seconds",
    "Time taken to scrape a single subreddit (posts + comments)",
    ["subreddit"],
    buckets=[5, 10, 30, 60, 120, 300, 600],
)

active_scrape_jobs = Gauge(
    "reddit_active_scrape_jobs",
    "Number of subreddits currently being scraped",
)

last_scrape_timestamp = Gauge(
    "reddit_last_scrape_timestamp_seconds",
    "Unix timestamp of the last successful scrape",
    ["subreddit"],
)

posts_in_raw_dataset = Gauge(
    "reddit_posts_in_raw_dataset_total",
    "Current count of rows in the raw dataset CSV",
)

# ── Sentiment metrics ──────────────────────────────────────────────────────────

sentiment_labels_total = Counter(
    "reddit_sentiment_labels_total",
    "Total sentiment labels assigned during scraping",
    ["subreddit", "sentiment"],   # sentiment: positive | negative | neutral
)

vader_compound_score = Histogram(
    "reddit_vader_compound_score",
    "Distribution of VADER compound scores",
    ["subreddit"],
    buckets=[-1.0, -0.75, -0.5, -0.25, -0.05, 0.05, 0.25, 0.5, 0.75, 1.0],
)

# ── Ticker extraction metrics ─────────────────────────────────────────────────

tickers_extracted_total = Counter(
    "reddit_tickers_extracted_total",
    "Total ticker symbols extracted from scraped text",
    ["subreddit"],
)

unique_tickers_gauge = Gauge(
    "reddit_unique_tickers_current",
    "Number of unique ticker symbols seen in the current scrape run",
)

# ── API / prediction metrics ───────────────────────────────────────────────────

api_predictions_total = Counter(
    "sentiment_api_predictions_total",
    "Total sentiment predictions made via /analyze endpoint",
    ["model", "predicted_sentiment"],
)

api_prediction_latency_seconds = Histogram(
    "sentiment_api_prediction_latency_seconds",
    "Latency of /analyze endpoint predictions",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

api_requests_total = Counter(
    "sentiment_api_requests_total",
    "Total HTTP requests to the FastAPI backend",
    ["method", "endpoint", "status_code"],
)

# ── Pipeline metrics ───────────────────────────────────────────────────────────

pipeline_stage_duration_seconds = Histogram(
    "pipeline_stage_duration_seconds",
    "Time taken for each pipeline stage to complete",
    ["stage"],   # e.g. scrape, preprocess, features, train, evaluate
    buckets=[1, 5, 15, 30, 60, 120, 300, 600, 1800],
)

pipeline_runs_total = Counter(
    "pipeline_runs_total",
    "Total number of full pipeline runs",
    ["status"],  # success | failure
)

model_training_accuracy = Gauge(
    "model_training_accuracy",
    "Final accuracy score of trained model",
    ["model_name"],
)

model_training_f1 = Gauge(
    "model_training_f1_score",
    "Final F1 score of trained model",
    ["model_name"],
)