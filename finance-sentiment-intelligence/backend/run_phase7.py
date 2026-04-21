"""
Run Phase 7: Full visualization suite + stats report.
Usage: python run_phase7.py
"""
import joblib
from src.data.data_loader import load_processed_data
from src.visualization.visualizer import (
    plot_sentiment_distribution,
    plot_sentiment_by_subreddit,
    plot_wordclouds,
    plot_sentiment_over_time,
    plot_engagement_analysis,
    plot_text_length_distribution,
    plot_top_ngrams,
    plot_hourly_heatmap,
    plot_radar_chart,
    plot_descriptive_dashboard,
)
from src.visualization.stats_reporter import save_stats_report
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_visualization():
    # ── Step 1: Load data ──────────────────────────────────────
    logger.info("Step 1: Loading processed data...")
    df = load_processed_data()

    # ── Step 2: Load model comparison results ─────────────────
    logger.info("Step 2: Loading model comparison data...")
    try:
        comparison_df = joblib.load(MODELS_DIR / "final_comparison.pkl")
        has_comparison = True
    except FileNotFoundError:
        logger.warning("final_comparison.pkl not found — skipping radar chart.")
        has_comparison = False

    print("\n" + "="*60)
    print("📊  VISUALIZATION SUITE — Phase 7")
    print("="*60)

    # ── Step 3: Generate all plots ─────────────────────────────

    logger.info("Plot 1/10: Sentiment distribution...")
    plot_sentiment_distribution(df)

    logger.info("Plot 2/10: Sentiment by subreddit...")
    plot_sentiment_by_subreddit(df)

    logger.info("Plot 3/10: Word clouds...")
    plot_wordclouds(df)

    logger.info("Plot 4/10: Sentiment over time...")
    plot_sentiment_over_time(df)

    logger.info("Plot 5/10: Engagement analysis...")
    plot_engagement_analysis(df)

    logger.info("Plot 6/10: Text length distribution...")
    plot_text_length_distribution(df)

    logger.info("Plot 7/10: Top unigrams...")
    plot_top_ngrams(df, n=1, top_k=15)

    logger.info("Plot 8/10: Top bigrams...")
    plot_top_ngrams(df, n=2, top_k=15)

    logger.info("Plot 9/10: Hourly heatmap...")
    plot_hourly_heatmap(df)

    if has_comparison:
        logger.info("Plot 10/10: Radar chart...")
        plot_radar_chart(comparison_df)

    logger.info("Plot 11/11: Descriptive statistics dashboard...")
    plot_descriptive_dashboard(df)

    # ── Step 4: Generate stats JSON report ─────────────────────
    logger.info("Step 4: Generating stats report (for /stats API)...")
    report = save_stats_report(df)

    # ── Final Summary ──────────────────────────────────────────
    print("\n" + "="*60)
    print("✅  PHASE 7 COMPLETE")
    print("="*60)
    print(f"\n  📁 All plots saved to: data/plots/")
    print(f"  📄 Stats JSON saved to: data/processed/stats_report.json")
    print(f"\n  Plots generated:")
    plots = [
        "sentiment_distribution.png",
        "sentiment_by_subreddit.png",
        "wordclouds.png",
        "sentiment_over_time.png",
        "engagement_analysis.png",
        "text_length_distribution.png",
        "top_unigrams.png",
        "top_bigrams.png",
        "hourly_heatmap.png",
        "radar_chart.png",
        "descriptive_dashboard.png",
        "metrics_comparison.png",
        "confusion_matrices_all.png",
        "tuning_improvement.png",
        "roc_curves.png",
    ]
    for p in plots:
        print(f"     ├── {p}")

    print(f"\n  📊 Dataset snapshot:")
    print(f"     Total posts   : {len(df):,}")
    print(f"     Positive      : {(df['sentiment']=='positive').sum():,}")
    print(f"     Neutral       : {(df['sentiment']=='neutral').sum():,}")
    print(f"     Negative      : {(df['sentiment']=='negative').sum():,}")
    print("="*60)

    return report


if __name__ == "__main__":
    run_visualization()