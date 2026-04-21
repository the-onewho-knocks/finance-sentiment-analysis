"""
Run Phase 3: Feature Engineering — TF-IDF + numeric scaling + combined matrix.
Usage: python run_phase3.py
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data_loader import load_processed_data
from src.features.feature_engineer import (
    build_tfidf_matrix,
    scale_numeric_features,
    combine_features,
    get_top_terms_by_sentiment,
)
from src.utils.config import MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_feature_engineering():
    # ── Step 1: Load processed data ───────────────────────────
    logger.info("Step 1: Loading processed data...")
    df = load_processed_data()

    # Drop rows with empty cleaned text (safety check)
    df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)
    logger.info(f"  Usable rows: {len(df)}")

    # ── Step 2: Train/Test split ───────────────────────────────
    logger.info("Step 2: Splitting data...")
    X_text    = df["cleaned_text"]
    y         = df["sentiment_label"]

    (X_text_train, X_text_test,
     y_train,      y_test,
     df_train,     df_test) = train_test_split(
        X_text, y, df,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,        # keep class balance in both splits
    )

    logger.info(f"  Train: {len(X_text_train)} | Test: {len(X_text_test)}")

    # ── Step 3: TF-IDF ────────────────────────────────────────
    logger.info("Step 3: Building TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_matrix(
        X_text_train.tolist(),
        X_text_test.tolist(),
    )

    # ── Step 4: Scale numeric features ────────────────────────
    logger.info("Step 4: Scaling numeric features...")
    X_train_num, X_test_num, scaler = scale_numeric_features(df_train, df_test)

    # ── Step 5: Combine features ───────────────────────────────
    logger.info("Step 5: Combining TF-IDF + numeric features...")
    X_train_combined = combine_features(X_train_tfidf, X_train_num)
    X_test_combined  = combine_features(X_test_tfidf,  X_test_num)

    # ── Step 6: Top terms per sentiment ───────────────────────
    logger.info("Step 6: Top TF-IDF terms per sentiment class...")
    top_terms = get_top_terms_by_sentiment(df, vectorizer, n=15)

    print("\n📌 Top TF-IDF Terms by Sentiment:")
    for sentiment, terms in top_terms.items():
        print(f"   {sentiment:<10}: {', '.join(terms)}")

    # ── Step 7: Save feature matrices & labels ────────────────
    logger.info("Step 7: Saving feature matrices...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_train_combined, MODELS_DIR / "X_train.pkl")
    joblib.dump(X_test_combined,  MODELS_DIR / "X_test.pkl")
    joblib.dump(y_train,          MODELS_DIR / "y_train.pkl")
    joblib.dump(y_test,           MODELS_DIR / "y_test.pkl")

    logger.info(f"  ✅ Feature matrices saved to {MODELS_DIR}")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("✅  PHASE 3 COMPLETE — Feature Engineering Summary")
    print("="*60)
    print(f"  TF-IDF features    : {X_train_tfidf.shape[1]:,}")
    print(f"  Numeric features   : {X_train_num.shape[1]}")
    print(f"  Combined matrix    : {X_train_combined.shape}")
    print(f"  Training samples   : {X_train_combined.shape[0]}")
    print(f"  Test samples       : {X_test_combined.shape[0]}")
    print(f"  Label distribution :")
    print(f"    Train: {dict(y_train.value_counts().sort_index())}")
    print(f"    Test : {dict(y_test.value_counts().sort_index())}")
    print("="*60)   # BUG 1 FIX: was `prin` (incomplete syntax error)


if __name__ == "__main__":
    run_feature_engineering()