"""
Run Phase 6: Model Comparison — full evaluation, plots, final model selection.
Usage: python run_phase6.py
"""
import joblib
import pandas as pd

from src.models.evaluator import (
    print_model_report,
    print_comparison_table,
    build_tuned_comparison_table,
    plot_confusion_matrix,
    plot_all_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_tuning_improvement,
    select_best_model,
)

from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_comparison():
    # ── Step 1: Load everything ────────────────────────────────
    logger.info("Step 1: Loading models and data...")

    X_train = joblib.load(MODELS_DIR / "X_train.pkl")
    X_test  = joblib.load(MODELS_DIR / "X_test.pkl")
    y_train = joblib.load(MODELS_DIR / "y_train.pkl")
    y_test  = joblib.load(MODELS_DIR / "y_test.pkl")

    before_results = joblib.load(MODELS_DIR / "training_results.pkl")
    after_results  = joblib.load(MODELS_DIR / "tuned_results.pkl")

    # ── Step 2: Full classification reports ───────────────────
    print("\n" + "="*60)
    print("📋  DETAILED CLASSIFICATION REPORTS — Tuned Models")
    print("="*60)

    for name, result in after_results.items():
        print_model_report(name, y_test, result["y_pred"])

    # ── Step 3: Side-by-side comparison table ─────────────────
    logger.info("Step 3: Comparison table...")
    comparison_df = print_comparison_table(after_results)

    # ── Step 4: Before vs After tuning table ──────────────────
    logger.info("Step 4: Before vs After tuning summary...")
    tuning_df = build_tuned_comparison_table(before_results, after_results)

    print("\n" + "="*70)
    print("📈  TUNING IMPACT SUMMARY")
    print("="*70)
    print(tuning_df.to_string())
    print("="*70)

    # ── Step 5: Individual confusion matrices ──────────────────
    logger.info("Step 5: Plotting individual confusion matrices...")
    for name, result in after_results.items():
        plot_confusion_matrix(y_test, result["y_pred"], name, save=True)

    # ── Step 6: All confusion matrices side by side ────────────
    logger.info("Step 6: Plotting all confusion matrices together...")
    plot_all_confusion_matrices(after_results, y_test)

    # ── Step 7: Metrics bar chart comparison ──────────────────
    logger.info("Step 7: Plotting metrics bar chart...")
    plot_metrics_comparison(after_results)

    # ── Step 8: Before vs After F1 chart ──────────────────────
    logger.info("Step 8: Plotting tuning improvement chart...")
    plot_tuning_improvement(before_results, after_results)

    # ── Step 9: ROC curves ─────────────────────────────────────
    logger.info("Step 9: Plotting ROC curves...")
    plot_roc_curves(after_results, X_test, y_test)

    # ── Step 10: Select and save final best model ──────────────
    logger.info("Step 10: Selecting final production model...")
    best_key, best_model = select_best_model(after_results)

    best_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, best_path)

    # ── Step 11: Save final comparison for Phase 7 ────────────
    joblib.dump(comparison_df, MODELS_DIR / "final_comparison.pkl")
    joblib.dump(tuning_df,    MODELS_DIR / "tuning_comparison.pkl")

    # ── Final Summary ──────────────────────────────────────────
    print("\n" + "="*60)
    print("✅  PHASE 6 COMPLETE — Final Summary")
    print("="*60)
    print(f"\n  🏆 Production model : {best_key.replace('_',' ').title()}")
    print(f"  📁 Saved to         : {best_path}")
    print(f"\n  📊 Plots saved to   : data/plots/")
    print(f"     ├── confusion_matrix_logistic_regression.png")
    print(f"     ├── confusion_matrix_naive_bayes.png")
    print(f"     ├── confusion_matrix_random_forest.png")
    print(f"     ├── confusion_matrices_all.png")
    print(f"     ├── metrics_comparison.png")
    print(f"     ├── tuning_improvement.png")
    print(f"     └── roc_curves.png")
    print("="*60)

    return after_results, comparison_df


if __name__ == "__main__":
    run_comparison()