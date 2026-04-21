import pandas as pd
import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Outlier Detection & Removal ──────────────────────────────────────────────

def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    IQR Method: Remove rows where the value falls outside
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Best for skewed distributions (like Reddit scores/upvotes).
    """
    Q1  = df[column].quantile(0.25)
    Q3  = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    removed = before - len(df)

    logger.info(f"IQR [{column}]: removed {removed} outliers "
                f"(kept {len(df)} rows, bounds=[{lower:.1f}, {upper:.1f}])")
    return df


def remove_outliers_zscore(
    df: pd.DataFrame, column: str, threshold: float = 3.0
) -> pd.DataFrame:
    """
    Z-score Method: Remove rows where |z-score| > threshold.
    Best for roughly normal distributions.

    BUG 4 FIX: Original code used df[column].dropna() which returns a
    shorter array than df, causing a length mismatch when used as a boolean
    mask on df. Fix: fill NaNs with the column mean before computing z-scores,
    then wrap the result in a pd.Series with the original df index so the
    boolean mask always aligns correctly.
    """
    filled   = df[column].fillna(df[column].mean())
    z_scores = pd.Series(
        np.abs(stats.zscore(filled)),
        index = df.index,        # ← preserves alignment with df
    )

    before  = len(df)
    df      = df[z_scores < threshold]
    removed = before - len(df)

    logger.info(f"Z-score [{column}]: removed {removed} outliers "
                f"(threshold={threshold})")
    return df


def filter_short_texts(df: pd.DataFrame, min_words: int = 5) -> pd.DataFrame:
    """
    Logical filtering: drop posts where cleaned text is too short
    to carry meaningful sentiment signal.
    """
    before = len(df)
    df = df[df["cleaned_text"].str.split().str.len() >= min_words]
    logger.info(f"Short text filter: removed {before - len(df)} rows "
                f"(min_words={min_words})")
    return df


# ── Data Wrangling ────────────────────────────────────────────────────────────

def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived numeric columns useful for analytics and modeling.
    """
    df["text_length"]  = df["text"].str.len()
    df["word_count"]   = df["text"].str.split().str.len()
    df["clean_length"] = df["cleaned_text"].str.len()
    df["clean_wc"]     = df["cleaned_text"].str.split().str.len()

    # Engagement score: combine upvotes + comments
    df["engagement"]   = df["score"] + df["num_comments"]

    # Hour of day the post was made (for time-based analysis)
    if "created_at" in df.columns:
        df["hour_of_day"] = pd.to_datetime(df["created_at"]).dt.hour
        df["day_of_week"] = pd.to_datetime(df["created_at"]).dt.day_name()

    return df


def encode_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode sentiment labels to integers for ML models.
    negative=0, neutral=1, positive=2
    """
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["sentiment_label"] = df["sentiment"].map(label_map)
    return df


def drop_nulls_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["text", "sentiment"])
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)
    logger.info(f"Dropped nulls/duplicates: {before - len(df)} rows removed")
    return df


# ── Descriptive Statistics ────────────────────────────────────────────────────

def print_descriptive_stats(df: pd.DataFrame) -> None:
    """Print a clean summary of dataset statistics."""
    print("\n" + "="*60)
    print("📊  DESCRIPTIVE STATISTICS")
    print("="*60)

    print(f"\n🔢 Shape          : {df.shape[0]} rows × {df.shape[1]} columns")

    print(f"\n📌 Sentiment Distribution:")
    counts = df["sentiment"].value_counts()
    pcts   = df["sentiment"].value_counts(normalize=True) * 100
    for label in counts.index:
        print(f"   {label:<12}: {counts[label]:>5}  ({pcts[label]:.1f}%)")

    print(f"\n📌 Subreddit Distribution:")
    print(df["subreddit"].value_counts().to_string())

    print(f"\n📌 Numeric Summary:")
    numeric_cols = ["score", "num_comments", "upvote_ratio",
                    "text_length", "word_count", "engagement"]
    existing = [c for c in numeric_cols if c in df.columns]
    print(df[existing].describe().round(2).to_string())

    print(f"\n📌 Missing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("   None ✅")
    else:
        print(missing.to_string())

    print("="*60 + "\n")