"""
Predictor service: loads all saved models + vectorizer,
preprocesses input text, and returns sentiment predictions.
Designed as a singleton — models are loaded once at startup.
"""
import numpy as np
import scipy.sparse as sp
import joblib
from pathlib import Path

from src.preprocessing.cleaner import full_clean
from src.utils.config import MODEL_PATHS, MODELS_DIR, TFIDF_VECTORIZER
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Label maps ────────────────────────────────────────────────────────────────
INT_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}

TUNED_PATHS = {
    "logistic_regression": MODELS_DIR / "tuned_logistic_regression.pkl",
    "naive_bayes":         MODELS_DIR / "tuned_naive_bayes.pkl",
    "random_forest":       MODELS_DIR / "tuned_random_forest.pkl",
}

SCALER_PATH        = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"


class SentimentPredictor:
    """
    Singleton class that holds all loaded models in memory.
    FastAPI loads this once at startup — every request reuses it.
    Loading from disk on every request would be very slow.
    """

    def __init__(self):
        self._models:              dict      = {}
        self._vectorizer                     = None
        self._scaler                         = None
        self._feature_names:       list[str] = []
        self._loaded_model_names:  list[str] = []
        self._load_all()

    def _load_all(self) -> None:
        """Load vectorizer, scaler, and all available models."""
        logger.info("🔄 Loading models into memory...")

        # Load TF-IDF vectorizer
        if TFIDF_VECTORIZER.exists():
            self._vectorizer = joblib.load(TFIDF_VECTORIZER)
            logger.info("  ✅ Vectorizer loaded")
        else:
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {TFIDF_VECTORIZER}. "
                "Run Phases 1-3 first."
            )

        # Load scaler (optional — graceful fallback)
        if SCALER_PATH.exists():
            self._scaler = joblib.load(SCALER_PATH)
            logger.info("  ✅ Scaler loaded")

        # Load numeric feature names so we know how many zeros to pad
        if FEATURE_NAMES_PATH.exists():
            self._feature_names = joblib.load(FEATURE_NAMES_PATH)
            logger.info(
                f"  ✅ Feature names loaded: {self._feature_names}"
            )

        # Load best model first
        best_path = MODELS_DIR / "best_model.pkl"
        if best_path.exists():
            self._models["best"] = joblib.load(best_path)
            logger.info("  ✅ best_model loaded")
            self._loaded_model_names.append("best")

        # Load tuned models (preferred), fallback to base models
        for name in ["logistic_regression", "naive_bayes", "random_forest"]:
            tuned_path = TUNED_PATHS.get(name)
            base_path  = MODEL_PATHS.get(name)

            if tuned_path and tuned_path.exists():
                self._models[name] = joblib.load(tuned_path)
                logger.info(f"  ✅ {name} (tuned) loaded")
                self._loaded_model_names.append(name)
            elif base_path and base_path.exists():
                self._models[name] = joblib.load(base_path)
                logger.info(f"  ✅ {name} (base) loaded")
                self._loaded_model_names.append(name)
            else:
                logger.warning(f"  ⚠️  {name} not found — skipping")

        logger.info(
            f"✅ Predictor ready — {len(self._models)} models loaded: "
            f"{self._loaded_model_names}"
        )

    def _preprocess(self, text: str) -> tuple[object, str]:
        """
        Clean text → TF-IDF vector → pad numeric zeros → sparse matrix
        ready for prediction.

        BUG 5 FIX: Models were trained on a COMBINED feature matrix
        (TF-IDF + scaled numeric columns). At inference we don't have a
        DataFrame with those columns, so we append a zero-filled block of
        the correct width. This keeps the feature count consistent and
        prevents a ValueError: "X has N features but model expects M".

        Returns (feature_matrix, cleaned_text)
        """
        cleaned = full_clean(text)

        if not cleaned.strip():
            cleaned = text.lower().strip()

        # TF-IDF transform
        tfidf_vec = self._vectorizer.transform([cleaned])

        # ── Pad with zeros for numeric features ───────────────
        # During training, numeric features (score, num_comments, etc.)
        # were appended after the TF-IDF columns. At inference we have no
        # such values, so we fill them with zeros.
        if self._feature_names:
            n_numeric = len(self._feature_names)
            zeros     = sp.csr_matrix(np.zeros((1, n_numeric)))
            tfidf_vec = sp.hstack([tfidf_vec, zeros])

        return tfidf_vec, cleaned

    def predict(self, text: str, model_name: str = "best") -> dict:
        """
        Run sentiment prediction on raw input text.

        Returns:
            dict with sentiment, confidence, scores, cleaned_text
        """
        if model_name not in self._models:
            available = list(self._models.keys())
            raise ValueError(
                f"Model '{model_name}' not loaded. "
                f"Available: {available}"
            )

        model      = self._models[model_name]
        X, cleaned = self._preprocess(text)

        # Naive Bayes needs non-negative values
        if "naive_bayes" in model_name or (
            model_name == "best" and
            "MultinomialNB" in type(model).__name__
        ):
            X = X.copy()
            if sp.issparse(X):
                X.data = np.clip(X.data, 0, None)

        # Predict class and probabilities
        pred_int  = model.predict(X)[0]
        sentiment = INT_TO_LABEL[pred_int]

        try:
            proba = model.predict_proba(X)[0]
        except AttributeError:
            # Fallback if model doesn't support predict_proba
            proba = np.array([0.0, 0.0, 0.0])
            proba[pred_int] = 1.0

        # Ensure proba has 3 values (negative, neutral, positive)
        if len(proba) == 3:
            neg_prob  = round(float(proba[0]), 4)
            neut_prob = round(float(proba[1]), 4)
            pos_prob  = round(float(proba[2]), 4)
        else:
            neg_prob = neut_prob = pos_prob = round(1 / 3, 4)

        confidence = round(float(proba[pred_int]), 4)

        return {
            "sentiment":    sentiment,
            "confidence":   confidence,
            "scores": {
                "positive": pos_prob,
                "neutral":  neut_prob,
                "negative": neg_prob,
            },
            "model_used":   model_name,
            "cleaned_text": cleaned,
            "word_count":   len(cleaned.split()),
        }

    @property
    def loaded_models(self) -> list[str]:
        return self._loaded_model_names


# ── Singleton instance ────────────────────────────────────────────────────────
# Instantiated once when the module is first imported
_predictor_instance: SentimentPredictor | None = None


def get_predictor() -> SentimentPredictor:
    """
    Return the global predictor singleton.
    FastAPI dependency injection calls this per request —
    but since it's a module-level singleton, models only load once.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SentimentPredictor()
    return _predictor_instance