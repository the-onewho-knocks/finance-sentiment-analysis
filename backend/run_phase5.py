"""
Run Phase 5: Hyperparameter Tuning.
Usage: python run_phase5.py
"""
import joblib
import numpy as np

from src.models.tuner import (
    tune_logistic_regression,
    tune_naive_bayes,
    tune_random_forest,
    evaluate_tuned_model,
    print_tuning_improvement,
)
from src.models.evaluator import (
    print_model_report,
    print_comparison_table,
    build_tuned_comparison_table,
)
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_tuning():
    # ── Step 1: Load features ──────────────────────────────────
    logger.info("Step 1: Loading feature matrices...")
    X_train = joblib.load(MODELS_DIR / "X_train.pkl")
    X_test  = joblib.load(MODELS_DIR / "X_test.pkl")
    y_train = joblib.load(MODELS_DIR / "y_train.pkl")
    y_test  = joblib.load(MODELS_DIR / "y_test.pkl")

    # ── Step 2: Load Phase 4 results (before tuning) ──────────
    logger.info("Step 2: Loading Phase 4 baseline results...")
    before_results = joblib.load(MODELS_DIR / "training_results.pkl")

    # ── Step 3: Tune all 3 models ──────────────────────────────
    print("\n" + "="*60)
    print("🎯  HYPERPARAMETER TUNING — Phase 5")
    print("="*60)

    # Logistic Regression — GridSearchCV
    lr_tuned, lr_search = tune_logistic_regression(X_train, y_train)

    # Naive Bayes — GridSearchCV
    nb_tuned, nb_search = tune_naive_bayes(X_train, y_train)

    # Random Forest — RandomizedSearchCV
    rf_tuned, rf_search = tune_random_forest(X_train, y_train)

    # ── Step 4: Evaluate tuned models on test set ──────────────
    logger.info("Step 4: Evaluating tuned models on test set...\n")
    after_results = {
        "logistic_regression": evaluate_tuned_model(
            lr_tuned, "logistic_regression", X_test, y_test),
        "naive_bayes":         evaluate_tuned_model(
            nb_tuned, "naive_bayes",         X_test, y_test),
        "random_forest":       evaluate_tuned_model(
            rf_tuned, "random_forest",        X_test, y_test),
    }

    # ── Step 5: Individual reports for tuned models ────────────
    logger.info("Step 5: Classification reports for tuned models...\n")
    for name, result in after_results.items():
        print_model_report(f"TUNED {name}", y_test, result["y_pred"])

    # ── Step 6: Before vs After comparison ────────────────────
    print_tuning_improvement(before_results, after_results)

    # ── Step 7: Full comparison table (tuned) ─────────────────
    comparison_df = print_comparison_table(after_results)

    # ── Step 8: Save best tuned model ─────────────────────────
    best_name = comparison_df["F1-Score"].idxmax()
    best_key  = best_name.lower().replace(" ", "_")
    best_model = after_results[best_key]["model"]

    best_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, best_path)
    logger.info(f"\n🏆 Best tuned model '{best_name}' saved → {best_path}\n")

    # ── Step 9: Save tuned results & CV search objects ─────────
    joblib.dump(after_results, MODELS_DIR / "tuned_results.pkl")
    joblib.dump({
        "logistic_regression": lr_search,
        "naive_bayes":         nb_search,
        "random_forest":       rf_search,
    }, MODELS_DIR / "cv_search_objects.pkl")

    logger.info("✅ All tuned models and results saved.")

    # ── Step 10: Print best hyperparameters summary ────────────
    print("\n" + "="*60)
    print("📋  BEST HYPERPARAMETERS SUMMARY")
    print("="*60)
    print(f"\n  Logistic Regression : {lr_search.best_params_}")
    print(f"  Naive Bayes         : {nb_search.best_params_}")
    print(f"  Random Forest       : {rf_search.best_params_}")
    print("="*60)

    return after_results


if __name__ == "__main__":
    run_tuning()