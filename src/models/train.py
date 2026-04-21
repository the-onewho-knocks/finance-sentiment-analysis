import joblib
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from src.utils.config import MODEL_PATHS, RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Model Definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return all 3 models with sensible default hyperparameters.
    These defaults are good starting points before tuning in Phase 5.

    Why these defaults?
      LR  → C=1.0 (default regularization), max_iter=1000 (enough to converge)
      NB   → alpha=1.0 (Laplace smoothing — avoids zero probabilities)
      RF   → 100 trees, max_depth=20 (prevents overfitting on small datasets)
    """
    return {
        # BUG 2 FIX: removed `multi_class="multinomial"` — deprecated and
        # removed in scikit-learn >= 1.5. lbfgs handles multiclass natively.
        "logistic_regression": LogisticRegression(
            C            = 1.0,
            max_iter     = 1000,
            solver       = "lbfgs",
            random_state = RANDOM_STATE,
            n_jobs       = -1,
        ),
        "naive_bayes": MultinomialNB(
            alpha = 1.0,   # Laplace smoothing
        ),
        "random_forest": RandomForestClassifier(
            n_estimators      = 100,
            max_depth         = 20,
            min_samples_split = 5,
            min_samples_leaf  = 2,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
        ),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_single_model(
    model,
    model_name:   str,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """
    Train one model, evaluate it, save to disk, and return metrics.

    Returns a dict with accuracy, report, and training time.
    """
    logger.info(f"Training: {model_name}...")
    start = time.time()

    # ── Naive Bayes needs non-negative values ─────────────────
    # TF-IDF values are always >= 0 so this is fine.
    # But combined matrix may have scaled numerics slightly < 0
    # due to floating point — we clip to be safe.
    if model_name == "naive_bayes":
        import scipy.sparse as sp
        if sp.issparse(X_train):
            X_train = X_train.copy()
            X_train.data = np.clip(X_train.data, 0, None)
            X_test = X_test.copy()
            X_test.data  = np.clip(X_test.data, 0, None)

    model.fit(X_train, y_train)
    elapsed = time.time() - start

    # ── Evaluate ──────────────────────────────────────────────
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names = ["negative", "neutral", "positive"],
        output_dict  = True,
    )

    # ── Save model ────────────────────────────────────────────
    save_path = MODEL_PATHS[model_name]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    logger.info(
        f"  ✅ {model_name} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Time: {elapsed:.2f}s | "
        f"Saved → {save_path}"
    )

    return {
        "model_name":    model_name,
        "model":         model,
        "accuracy":      accuracy,
        "report":        report,
        "training_time": round(elapsed, 2),
        "y_pred":        y_pred,
    }


def train_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """
    Train all 3 models sequentially.
    Returns a dict keyed by model name, each containing metrics.
    """
    models  = get_models()
    results = {}

    print("\n" + "="*60)
    print("🤖  MODEL TRAINING — Phase 4")
    print("="*60)

    for name, model in models.items():
        result        = train_single_model(
            model, name, X_train, y_train, X_test, y_test
        )
        results[name] = result

    print("="*60)
    print("✅  All models trained and saved.")
    print("="*60)

    return results


# ── Load a saved model ────────────────────────────────────────────────────────

def load_model(model_name: str):
    """Load a saved .pkl model by name."""
    path = MODEL_PATHS.get(model_name)
    if path is None:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose from: {list(MODEL_PATHS.keys())}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run Phase 4 first."
        )
    logger.info(f"Loaded model: {model_name} ← {path}")
    return joblib.load(path)


def load_best_model():
    """
    Load the model marked as best after Phase 6 comparison.
    Falls back to logistic_regression if best_model.pkl doesn't exist.
    """
    from src.utils.config import MODELS_DIR
    best_path = MODELS_DIR / "best_model.pkl"

    if best_path.exists():
        logger.info(f"Loading best model ← {best_path}")
        return joblib.load(best_path)

    logger.warning("best_model.pkl not found — falling back to logistic_regression")
    return load_model("logistic_regression")


def load_tuned_model(model_name: str):
    """Load a tuned .pkl model by name."""
    from src.models.tuner import TUNED_PATHS
    path = TUNED_PATHS.get(model_name)
    if path is None or not path.exists():
        logger.warning(f"Tuned model not found for '{model_name}', loading base model.")
        return load_model(model_name)
    logger.info(f"Loaded tuned model: {model_name} ← {path}")
    return joblib.load(path)