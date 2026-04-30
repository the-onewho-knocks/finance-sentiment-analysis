import joblib
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from src.utils.config import MODEL_PATHS, RANDOM_STATE
from src.utils.logger import get_logger

from src.metrics.prometheus_metrics import (
    model_training_accuracy,
    model_training_f1_score
)

logger = get_logger(__name__)


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "naive_bayes": MultinomialNB(
            alpha=1.0,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_single_model(
    model,
    model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:

    logger.info(f"Training: {model_name}...")
    start = time.time()

    # Fix for Naive Bayes (non-negative values)
    if model_name == "naive_bayes":
        import scipy.sparse as sp
        if sp.issparse(X_train):
            X_train = X_train.copy()
            X_train.data = np.clip(X_train.data, 0, None)

            X_test = X_test.copy()
            X_test.data = np.clip(X_test.data, 0, None)

    # Train
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    # ── Evaluate ──────────────────────────────────────────────
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["negative", "neutral", "positive"],
        output_dict=True,
    )

    # 🔥 PROMETHEUS METRICS UPDATE
    model_training_accuracy.labels(model=model_name).set(accuracy)

    f1 = report["weighted avg"]["f1-score"]
    model_training_f1_score.labels(model=model_name).set(f1)

    # ── Save model ────────────────────────────────────────────
    save_path = MODEL_PATHS[model_name]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    logger.info(
        f"✅ {model_name} | "
        f"Accuracy: {accuracy:.4f} | "
        f"F1: {f1:.4f} | "
        f"Time: {elapsed:.2f}s | "
        f"Saved → {save_path}"
    )

    return {
        "model_name": model_name,
        "model": model,
        "accuracy": accuracy,
        "f1_score": f1,
        "report": report,
        "training_time": round(elapsed, 2),
        "y_pred": y_pred,
    }


def train_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:

    models = get_models()
    results = {}

    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING — Phase 4")
    print("=" * 60)

    for name, model in models.items():
        result = train_single_model(
            model, name, X_train, y_train, X_test, y_test
        )
        results[name] = result

    print("=" * 60)
    print("✅ All models trained and saved.")
    print("=" * 60)

    return results


# ── Load a saved model ────────────────────────────────────────────────────────

def load_model(model_name: str):
    path = MODEL_PATHS.get(model_name)

    if path is None:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose from: {list(MODEL_PATHS.keys())}"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run training first."
        )

    logger.info(f"Loaded model: {model_name} ← {path}")
    return joblib.load(path)


def load_best_model():
    from src.utils.config import MODELS_DIR
    best_path = MODELS_DIR / "best_model.pkl"

    if best_path.exists():
        logger.info(f"Loading best model ← {best_path}")
        return joblib.load(best_path)

    logger.warning("best_model.pkl not found — using logistic_regression")
    return load_model("logistic_regression")


def load_tuned_model(model_name: str):
    from src.models.tuner import TUNED_PATHS

    path = TUNED_PATHS.get(model_name)

    if path is None or not path.exists():
        logger.warning(f"Tuned model not found for '{model_name}', loading base model.")
        return load_model(model_name)

    logger.info(f"Loaded tuned model: {model_name} ← {path}")
    return joblib.load(path)