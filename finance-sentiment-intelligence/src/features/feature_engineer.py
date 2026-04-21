import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import (
    MAX_FEATURES,
    RANDOM_STATE,
    TFIDF_VECTORIZER,
    MODELS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Paths for saved artifacts
SCALER_PATH          = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH   = MODELS_DIR / "feature_names.pkl"


# ── TF-IDF ───────────────────────────────────────────────────────────────────

def build_tfidf_matrix(
    train_texts: list[str],
    test_texts:  list[str],
    max_features: int = MAX_FEATURES,
    ngram_range:  tuple = (1, 2),   # unigrams + bigrams
) -> tuple:
    """
    Fit TF-IDF on training texts, transform both train and test.

    Why ngram_range=(1,2)?
      Single words: "crash", "rally"
      Bigrams:      "stock crash", "bull market" → richer signal

    Returns:
        X_train_tfidf, X_test_tfidf, fitted vectorizer
    """
    logger.info(f"Building TF-IDF (max_features={max_features}, "
                f"ngrams={ngram_range})...")

    vectorizer = TfidfVectorizer(
        max_features = max_features,
        ngram_range  = ngram_range,
        sublinear_tf = True,      # apply log(1+tf) to dampen high freq terms
        min_df       = 2,         # ignore terms in fewer than 2 docs
        max_df       = 0.95,      # ignore terms in more than 95% of docs
        strip_accents = "unicode",
        analyzer     = "word",
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    logger.info(f"  TF-IDF matrix shape — train: {X_train.shape}, "
                f"test: {X_test.shape}")

    # Save vectorizer so backend can reuse it at inference time
    TFIDF_VECTORIZER.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, TFIDF_VECTORIZER)
    logger.info(f"  ✅ TF-IDF vectorizer saved → {TFIDF_VECTORIZER}")

    return X_train, X_test, vectorizer


# ── Numeric Feature Scaling ───────────────────────────────────────────────────

NUMERIC_FEATURES = ["score", "num_comments", "upvote_ratio", "engagement",
                    "word_count", "clean_wc", "text_length"]


def scale_numeric_features(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple:
    """
    Apply MinMaxScaler to numeric columns.

    Why MinMaxScaler (not StandardScaler)?
      Reddit scores & comment counts are heavily right-skewed.
      MinMax scales to [0,1] without assuming normal distribution.
      StandardScaler is better when data is roughly Gaussian.

    Returns:
        X_train_scaled (np.array), X_test_scaled (np.array), scaler
    """
    # Only keep numeric columns that actually exist in the DataFrame
    cols = [c for c in NUMERIC_FEATURES if c in train_df.columns]

    logger.info(f"Scaling numeric features: {cols}")

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(train_df[cols].fillna(0))
    X_test_scaled  = scaler.transform(test_df[cols].fillna(0))

    # Save for later use in backend
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(cols,   FEATURE_NAMES_PATH)
    logger.info(f"  ✅ Scaler saved → {SCALER_PATH}")

    return X_train_scaled, X_test_scaled, scaler


# ── Combine TF-IDF + Numeric Features ────────────────────────────────────────

def combine_features(
    X_tfidf:  object,         # sparse matrix from TF-IDF
    X_numeric: np.ndarray,    # dense array from scaler
) -> csr_matrix:
    """
    Horizontally stack TF-IDF sparse matrix with numeric features.
    Result is a single feature matrix passed to ML models.

    Why combine?
      TF-IDF captures text semantics.
      Numeric features (score, comments) capture social signals.
      Together they give the model richer context.
    """
    X_numeric_sparse = csr_matrix(X_numeric)
    combined = hstack([X_tfidf, X_numeric_sparse])
    logger.info(f"  Combined feature matrix shape: {combined.shape}")
    return combined


# ── Top TF-IDF Terms (for EDA) ────────────────────────────────────────────────

def get_top_tfidf_terms(
    vectorizer: TfidfVectorizer,
    n: int = 20,
) -> list[str]:
    """Return the top-N terms by mean TF-IDF weight."""
    feature_names = vectorizer.get_feature_names_out()
    return list(feature_names[:n])


def get_top_terms_by_sentiment(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    n: int = 15,
) -> dict[str, list[str]]:
    """
    For each sentiment class, find the top-N most important TF-IDF terms.
    Useful for wordclouds and EDA in Phase 7.
    """
    result = {}
    feature_names = vectorizer.get_feature_names_out()

    for sentiment in ["positive", "negative", "neutral"]:
        subset_texts = df.loc[
            df["sentiment"] == sentiment, "cleaned_text"
        ].tolist()

        if not subset_texts:
            result[sentiment] = []
            continue

        matrix = vectorizer.transform(subset_texts)

        # Mean TF-IDF score per term across all posts in this class
        mean_scores  = np.asarray(matrix.mean(axis=0)).flatten()
        top_indices  = mean_scores.argsort()[::-1][:n]
        top_terms    = [feature_names[i] for i in top_indices]

        result[sentiment] = top_terms
        logger.info(f"  Top terms [{sentiment}]: {top_terms[:5]}...")

    return result


# ── Load saved vectorizer (used by backend) ───────────────────────────────────

def load_vectorizer() -> TfidfVectorizer:
    if not TFIDF_VECTORIZER.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at {TFIDF_VECTORIZER}. "
            "Run Phase 3 first."
        )
    return joblib.load(TFIDF_VECTORIZER)


def load_scaler() -> MinMaxScaler:
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Run Phase 3 first."
        )
    return joblib.load(SCALER_PATH)