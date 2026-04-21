"""
Run Phase 4: Train all 3 ML models.
Usage: python run_phase4.py
"""
import joblib
import numpy as np
from src.models.train import train_all_models, load_model
from src.models.evaluator import print_model_report, print_comparison_table
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training():
    # ── Step 1: Load feature matrices from Phase 3 ────────────
    logger.info("Step 1: Loading feature matrices...")

    X_train = joblib.load(MODELS_DIR / "X_train.pkl")
    X_test  = joblib.load(MODELS_DIR / "X_test.pkl")
    y_train = joblib.load(MODELS_DIR / "y_train.pkl")
    y_test  = joblib.load(MODELS_DIR / "y_test.pkl")

    logger.info(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

    # ── Step 2: Train all models ───────────────────────────────
    logger.info("Step 2: Training all models...")
    results = train_all_models(X_train, y_train, X_test, y_test)

    # ── Step 3: Print individual reports ──────────────────────
    logger.info("Step 3: Individual classification reports...")
    for name, result in results.items():
        print_model_report(name, y_test, result["y_pred"])

    # ── Step 4: Print comparison table ────────────────────────
    logger.info("Step 4: Model comparison table...")
    comparison_df = print_comparison_table(results)

    # ── Step 5: Save best model by F1 ─────────────────────────
    best_model_name = comparison_df["F1-Score"].idxmax()
    best_model_key  = best_model_name.lower().replace(" ", "_")
    best_model      = results[best_model_key]["model"]

    best_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, best_path)
    logger.info(f"\n🏆 Best model '{best_model_name}' saved → {best_path}")

    # ── Step 6: Save results for Phase 6 use ──────────────────
    joblib.dump(results, MODELS_DIR / "training_results.pkl")
    logger.info("✅ Training results saved for Phase 6.")

    # ── Step 7: Quick sanity check ────────────────────────────
    print("\n── Sanity Check: predict first 3 test samples ──")
    model = load_model("logistic_regression")
    sample = X_test[:3]
    preds  = model.predict(sample)
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    for i, p in enumerate(preds):
        actual = label_map[y_test.iloc[i]]
        predicted = label_map[p]
        match = "✅" if actual == predicted else "❌"
        print(f"  Sample {i+1}: actual={actual:<10} predicted={predicted:<10} {match}")

    return results


if __name__ == "__main__":
    run_training()