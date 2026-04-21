"""
Generates a JSON stats report consumed by the /stats API endpoint.
Keeps visualization logic separate from API logic.
"""
import json
import pandas as pd
import numpy as np
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

STATS_FILE = MODELS_DIR.parent.parent / "data" / "processed" / "stats_report.json"


def build_stats_report(df: pd.DataFrame) -> dict:
    """
    Build a JSON-serializable stats dict from the processed DataFrame.
    This is what the /stats endpoint will return.
    """
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    total            = len(df)

    report = {
        "total_posts": total,

        "sentiment_distribution": {
            k: {"count": int(v), "percentage": round(v / total * 100, 2)}
            for k, v in sentiment_counts.items()
        },

        "subreddit_breakdown": {
            sub: df[df["subreddit"] == sub]["sentiment"]
                 .value_counts().to_dict()
            for sub in df["subreddit"].unique()
        },

        "numeric_stats": {
            col: {
                "mean":   round(df[col].mean(), 2),
                "median": round(df[col].median(), 2),
                "std":    round(df[col].std(), 2),
                "min":    round(df[col].min(), 2),
                "max":    round(df[col].max(), 2),
                "q25":    round(df[col].quantile(0.25), 2),
                "q75":    round(df[col].quantile(0.75), 2),
            }
            for col in ["score", "num_comments", "upvote_ratio",
                        "word_count", "engagement"]
            if col in df.columns
        },

        "top_words": {
            sentiment: _get_top_words(df, sentiment, n=20)
            for sentiment in ["positive", "negative", "neutral"]
        },

        "available_plots": [
            "sentiment_distribution",
            "sentiment_by_subreddit",
            "wordclouds",
            "sentiment_over_time",
            "engagement_analysis",
            "text_length_distribution",
            "top_bigrams",
            "top_unigrams",
            "hourly_heatmap",
            "radar_chart",
            "descriptive_dashboard",
            "metrics_comparison",
            "confusion_matrices_all",
            "tuning_improvement",
            "roc_curves",
        ],
    }

    return report


def _get_top_words(df: pd.DataFrame, sentiment: str, n: int = 20) -> list:
    """Return top N words for a given sentiment class."""
    texts = df[df["sentiment"] == sentiment]["cleaned_text"].dropna()
    all_words = " ".join(texts).split()
    from collections import Counter
    return [word for word, _ in Counter(all_words).most_common(n)]


def save_stats_report(df: pd.DataFrame) -> dict:
    """Build, save and return the stats report."""
    report = build_stats_report(df)
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(STATS_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"✅ Stats report saved → {STATS_FILE}")
    return report


def load_stats_report() -> dict:
    """Load cached stats report (used by FastAPI /stats endpoint)."""
    if not STATS_FILE.exists():
        raise FileNotFoundError(
            f"Stats report not found at {STATS_FILE}. Run Phase 7 first."
        )
    with open(STATS_FILE) as f:
        return json.load(f)