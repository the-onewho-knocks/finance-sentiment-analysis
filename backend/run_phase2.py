"""
Run Phase 2: Data preprocessing, wrangling & descriptive stats.
Usage: python -m backend.run_phase2
"""

import pandas as pd
from src.data.data_loader import load_raw_data, save_processed_data
from src.preprocessing.cleaner import full_clean
from src.preprocessing.wrangler import (
    drop_nulls_and_duplicates,
    add_engineered_columns,
    encode_sentiment,
    remove_outliers_iqr,
    remove_outliers_zscore,
    filter_short_texts,
    print_descriptive_stats,
)
from src.preprocessing.tokenizer import get_token_stats
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_preprocessing() -> pd.DataFrame:
    """Main preprocessing pipeline"""

    # ── Step 1: Load raw data ──────────────────────────────────
    logger.info("Step 1: Loading raw data...")
    df = load_raw_data()

    # ── Step 2: Remove nulls & duplicates ──────────────────────
    logger.info("Step 2: Dropping nulls and duplicates...")
    df = drop_nulls_and_duplicates(df)

    # ── Step 3: Text cleaning ─────────────────────────────────
    logger.info("Step 3: Cleaning text (this may take a minute)...")

    # Basic cleaning
    df["cleaned_text"] = df["text"].apply(full_clean)

    # 🔥 Noise removal (FIXED & CORRECTLY PLACED)
    REMOVE_WORDS = {"removed", "deleted", "post", "amp", "http"}

    def remove_noise(text: str) -> str:
        words = text.split()
        words = [w for w in words if len(w) > 2 and w not in REMOVE_WORDS]
        return " ".join(words)

    df["cleaned_text"] = df["cleaned_text"].apply(remove_noise)

    # ── Step 4: Feature engineering ───────────────────────────
    logger.info("Step 4: Engineering wrangling columns...")
    df = add_engineered_columns(df)

    # ── Step 5: Filter short text ─────────────────────────────
    logger.info("Step 5: Filtering short texts...")
    df = filter_short_texts(df, min_words=3)

    # ── Step 6: Outlier removal ───────────────────────────────
    logger.info("Step 6: Handling outliers...")
    df = remove_outliers_iqr(df, "score")
    df = remove_outliers_iqr(df, "num_comments")
    df = remove_outliers_zscore(df, "text_length", threshold=3.0)

    # ── Step 7: Encode sentiment ─────────────────────────────
    logger.info("Step 7: Encoding sentiment labels...")
    df = encode_sentiment(df)

    # ── Step 8: Descriptive statistics ───────────────────────
    logger.info("Step 8: Computing descriptive statistics...")
    print_descriptive_stats(df)

    # ── Step 9: Token statistics ─────────────────────────────
    logger.info("Step 9: Token statistics...")
    stats = get_token_stats(df["cleaned_text"].tolist())

    print("\n📌 Token Stats:")
    print(f"   Total tokens  : {stats['total_tokens']:,}")
    print(f"   Unique tokens : {stats['unique_tokens']:,}")
    print(f"   Avg per doc   : {stats['avg_per_doc']}")
    print(f"   Top 10 tokens : {stats['top_10_tokens']}")

    # ── Step 10: Save processed data ─────────────────────────
    save_processed_data(df)

    logger.info(f"✅ Phase 2 complete — {len(df)} clean rows ready.")

    return df


if __name__ == "__main__":
    run_preprocessing()