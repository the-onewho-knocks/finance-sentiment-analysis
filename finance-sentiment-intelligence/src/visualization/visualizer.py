import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud

from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────
PLOTS_DIR = MODELS_DIR.parent.parent / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────
PALETTE = {
    "positive": "#4CAF50",
    "neutral":  "#FF9800",
    "negative": "#F44336",
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


# ── Helpers ───────────────────────────────────
def _save(fig, filename):
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  ✅ Saved → {path}")
    return str(path)


def _ensure_engagement(df):
    if "engagement" not in df.columns:
        df["engagement"] = df["score"]
    return df


# ── 1. Sentiment Distribution ─────────────────
def plot_sentiment_distribution(df):
    counts = df["sentiment"].value_counts().reindex(
        ["positive", "neutral", "negative"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values,
           color=[PALETTE[s] for s in counts.index])
    ax.set_title("Sentiment Distribution")

    return _save(fig, "sentiment_distribution.png")


# ── 2. Sentiment by Subreddit ─────────────────
def plot_sentiment_by_subreddit(df):
    pivot = (
        df.groupby(["subreddit", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Sentiment by Subreddit")

    return _save(fig, "sentiment_by_subreddit.png")


# ── 3. Wordclouds ─────────────────────────────
def plot_wordclouds(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, sentiment in zip(axes, ["positive", "neutral", "negative"]):
        text = " ".join(
            df[df["sentiment"] == sentiment]["cleaned_text"].dropna()
        )

        if not text.strip():
            ax.set_title(f"{sentiment} (no data)")
            continue

        wc = WordCloud(width=600, height=400).generate(text)
        ax.imshow(wc)
        ax.set_title(sentiment)
        ax.axis("off")

    return _save(fig, "wordclouds.png")


# ── 4. Time Series ────────────────────────────
def plot_sentiment_over_time(df):
    if "created_at" not in df.columns:
        return ""

    df = df.copy()
    df["date"] = pd.to_datetime(df["created_at"]).dt.date

    pivot = (
        df.groupby(["date", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(ax=ax)
    ax.set_title("Sentiment Over Time")

    return _save(fig, "sentiment_over_time.png")


# ── 5. Engagement Analysis (FIXED) ────────────
def plot_engagement_analysis(df):
    df = _ensure_engagement(df)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2)

    # Score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.violinplot(data=df, x="sentiment", y="score", ax=ax1)

    # Comments
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=df, x="sentiment", y="num_comments", ax=ax2)

    # Engagement instead of upvote_ratio
    ax3 = fig.add_subplot(gs[1, 0])
    metric = "engagement"

    means = (
        df.groupby("sentiment")[metric]
        .mean()
        .reindex(["positive", "neutral", "negative"])
    )

    ax3.bar(means.index, means.values,
            color=[PALETTE[s] for s in means.index])
    ax3.set_title("Mean Engagement by Sentiment")

    # Scatter
    ax4 = fig.add_subplot(gs[1, 1])
    for s in ["positive", "neutral", "negative"]:
        sub = df[df["sentiment"] == s]
        ax4.scatter(sub["score"], sub["num_comments"],
                    label=s, alpha=0.4)

    ax4.legend()

    return _save(fig, "engagement_analysis.png")


# ── 6. Text Length ────────────────────────────
def plot_text_length_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    for s in ["positive", "neutral", "negative"]:
        sub = df[df["sentiment"] == s]
        ax.hist(sub["word_count"], alpha=0.5, label=s)

    ax.legend()
    ax.set_title("Word Count Distribution")

    return _save(fig, "text_length_distribution.png")


# ── 7. N-grams ────────────────────────────────
def plot_top_ngrams(df, n=1, top_k=15):
    """
    Top unigrams / bigrams per sentiment
    n=1 → unigrams
    n=2 → bigrams
    """
    from sklearn.feature_extraction.text import CountVectorizer

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, sentiment in zip(axes, ["positive", "neutral", "negative"]):
        texts = df[df["sentiment"] == sentiment]["cleaned_text"].dropna().tolist()

        if len(texts) < 5:
            ax.set_title(f"{sentiment} (not enough data)")
            continue

        vectorizer = CountVectorizer(ngram_range=(n, n), max_features=top_k)
        X = vectorizer.fit_transform(texts)

        freqs = np.asarray(X.sum(axis=0)).flatten()
        words = vectorizer.get_feature_names_out()

        idx = freqs.argsort()[::-1][:top_k]

        top_words = [words[i] for i in idx]
        top_freqs = [freqs[i] for i in idx]

        ax.barh(top_words[::-1], top_freqs[::-1],
                color=PALETTE[sentiment])
        ax.set_title(f"{sentiment} ({'uni' if n==1 else 'bi'}grams)")

    return _save(fig, f"top_{'unigrams' if n==1 else 'bigrams'}.png")


# ── 8. Heatmap ────────────────────────────────
def plot_hourly_heatmap(df):
    return ""


# ── 9. Radar ──────────────────────────────────
def plot_radar_chart(df):
    return ""


# ── 10. Dashboard (FIXED) ─────────────────────
def plot_descriptive_dashboard(df):
    df = _ensure_engagement(df)

    fig, ax = plt.subplots(figsize=(10, 5))

    col = "engagement"

    df[col].hist(ax=ax, bins=40)
    ax.set_title("Engagement Distribution")

    return _save(fig, "descriptive_dashboard.png")