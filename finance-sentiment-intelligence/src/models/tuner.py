import joblib
import time
import numpy as np
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import classification_report, accuracy_score

from src.utils.config import MODEL_PATHS, RANDOM_STATE, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Cross-validation strategy ─────────────────────────────────────────────────
# StratifiedKFold ensures each fold has same class distribution
CV_STRATEGY = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

TUNED_PATHS = {
    "logistic_regression": MODELS_DIR / "tuned_logistic_regression.pkl",
    "naive_bayes":         MODELS_DIR / "tuned_naive_bayes.pkl",
    "random_forest":       MODELS_DIR / "tuned_random_forest.pkl",
}


# ── Parameter Grids ───────────────────────────────────────────────────────────

def get_lr_param_grid() -> dict:
    """
    Logistic Regression hyperparameters:
    C         → Regularization strength (smaller = stronger regularization)
                Too small = underfitting, too large = overfitting
    solver    → Algorithm for optimization
    max_iter  → Max iterations to converge
    """
    return {
        "C":        [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver":   ["lbfgs", "saga"],
        "max_iter": [500, 1000],
    }


def get_nb_param_grid() -> dict:
    """
    Naive Bayes hyperparameters:
    alpha → Laplace/Lidstone smoothing
            alpha=0: no smoothing (risky — zero prob for unseen words)
            alpha=1: full Laplace smoothing
            alpha>1: over-smoothing (all words treated equally)
    """
    return {
        "alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
    }


def get_rf_param_grid() -> dict:
    """
    Random Forest hyperparameters:
    n_estimators     → Number of trees (more = better but slower)
    max_depth        → Max depth of each tree (None = fully grown)
    min_samples_split→ Min samples needed to split a node
    min_samples_leaf → Min samples needed at a leaf node
    max_features     → Features to consider at each split
                       'sqrt' = sqrt(n_features) — standard for classification
    """
    return {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    }


# ── Tuning Functions ──────────────────────────────────────────────────────────

def tune_logistic_regression(X_train, y_train) -> tuple:
    """
    Use GridSearchCV — grid is small enough to try all combinations.
    5 (C) × 2 (solver) × 2 (max_iter) = 20 combinations × 5 folds = 100 fits
    """
    logger.info("Tuning Logistic Regression with GridSearchCV...")
    start = time.time()

    # BUG 3 FIX: removed `multi_class="multinomial"` — deprecated and
    # removed in scikit-learn >= 1.5. lbfgs handles multiclass natively.
    base_model = LogisticRegression(
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )

    grid_search = GridSearchCV(
        estimator  = base_model,
        param_grid = get_lr_param_grid(),
        cv         = CV_STRATEGY,
        scoring    = "f1_weighted",   # optimize for weighted F1
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,            # refit best model on full train set
    )

    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    logger.info(f"  Best params : {grid_search.best_params_}")
    logger.info(f"  Best CV F1  : {grid_search.best_score_:.4f}")
    logger.info(f"  Time taken  : {elapsed:.1f}s")

    # Save tuned model
    joblib.dump(grid_search.best_estimator_, TUNED_PATHS["logistic_regression"])
    logger.info(f"  ✅ Saved → {TUNED_PATHS['logistic_regression']}")

    return grid_search.best_estimator_, grid_search


def tune_naive_bayes(X_train, y_train) -> tuple:
    """
    Use GridSearchCV — Naive Bayes grid is tiny (6 values only).
    Must clip negative values since MultinomialNB requires X >= 0.
    """
    logger.info("Tuning Naive Bayes with GridSearchCV...")
    start = time.time()

    # Clip negatives for Naive Bayes
    X_train_nb = X_train.copy()
    if sp.issparse(X_train_nb):
        X_train_nb.data = np.clip(X_train_nb.data, 0, None)
    else:
        X_train_nb = np.clip(X_train_nb, 0, None)

    base_model = MultinomialNB()

    grid_search = GridSearchCV(
        estimator  = base_model,
        param_grid = get_nb_param_grid(),
        cv         = CV_STRATEGY,
        scoring    = "f1_weighted",
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )

    grid_search.fit(X_train_nb, y_train)
    elapsed = time.time() - start

    logger.info(f"  Best params : {grid_search.best_params_}")
    logger.info(f"  Best CV F1  : {grid_search.best_score_:.4f}")
    logger.info(f"  Time taken  : {elapsed:.1f}s")

    joblib.dump(grid_search.best_estimator_, TUNED_PATHS["naive_bayes"])
    logger.info(f"  ✅ Saved → {TUNED_PATHS['naive_bayes']}")

    return grid_search.best_estimator_, grid_search


def tune_random_forest(X_train, y_train) -> tuple:
    """
    Use RandomizedSearchCV — RF grid is huge (3×4×3×3×2 = 216 combos).
    RandomizedSearch samples n_iter=30 random combos instead of all 216.
    This saves time while still exploring the space well.
    """
    logger.info("Tuning Random Forest with RandomizedSearchCV...")
    start = time.time()

    base_model = RandomForestClassifier(
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )

    rand_search = RandomizedSearchCV(
        estimator           = base_model,
        param_distributions = get_rf_param_grid(),
        n_iter              = 30,         # try 30 random combinations
        cv                  = CV_STRATEGY,
        scoring             = "f1_weighted",
        n_jobs              = -1,
        verbose             = 1,
        random_state        = RANDOM_STATE,
        refit               = True,
    )

    rand_search.fit(X_train, y_train)
    elapsed = time.time() - start

    logger.info(f"  Best params : {rand_search.best_params_}")
    logger.info(f"  Best CV F1  : {rand_search.best_score_:.4f}")
    logger.info(f"  Time taken  : {elapsed:.1f}s")

    joblib.dump(rand_search.best_estimator_, TUNED_PATHS["random_forest"])
    logger.info(f"  ✅ Saved → {TUNED_PATHS['random_forest']}")

    return rand_search.best_estimator_, rand_search


# ── Evaluate Tuned Model ──────────────────────────────────────────────────────

def evaluate_tuned_model(
    model,
    model_name: str,
    X_test,
    y_test,
) -> dict:
    """Run the tuned model on test set and return full metrics."""

    # Clip for Naive Bayes
    if model_name == "naive_bayes":
        X_test = X_test.copy()
        if sp.issparse(X_test):
            X_test.data = np.clip(X_test.data, 0, None)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names  = ["negative", "neutral", "positive"],
        output_dict   = True,
        zero_division = 0,
    )

    logger.info(
        f"  Tuned {model_name} → "
        f"Accuracy: {accuracy:.4f} | "
        f"F1: {report['weighted avg']['f1-score']:.4f}"
    )

    return {
        "model_name":    model_name,
        "model":         model,
        "accuracy":      accuracy,
        "report":        report,
        "training_time": 0,
        "y_pred":        y_pred,
    }


# ── Compare Before vs After Tuning ───────────────────────────────────────────

def print_tuning_improvement(
    before_results: dict,
    after_results:  dict,
) -> None:
    """Print a before/after table showing improvement from tuning."""

    print("\n" + "="*70)
    print("📈  TUNING IMPROVEMENT — Before vs After")
    print("="*70)
    print(f"{'Model':<25} {'Before F1':>10} {'After F1':>10} {'Delta':>8}")
    print("-"*70)

    for name in before_results:
        before_f1 = before_results[name]["report"]["weighted avg"]["f1-score"]
        after_f1  = after_results[name]["report"]["weighted avg"]["f1-score"]
        delta     = after_f1 - before_f1
        arrow     = "⬆" if delta > 0 else "⬇" if delta < 0 else "→"

        print(
            f"{name.replace('_',' ').title():<25} "
            f"{before_f1:>10.4f} "
            f"{after_f1:>10.4f} "
            f"{arrow} {abs(delta):>6.4f}"
        )

    print("="*70)