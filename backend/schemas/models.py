from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ── /analyze ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """
    Input schema for POST /analyze.
    text: the raw finance text to analyze (required)
    model: which ML model to use (optional, defaults to best)
    """
    text: str = Field(
        ...,
        min_length = 3,
        max_length = 5000,
        description = "Finance-related text to analyze",
        examples    = ["HDFC Bank Q3 results beat expectations, stock rallied 4%"],
    )
    model: Optional[str] = Field(
        default     = "best",
        description = "Model to use: best | logistic_regression | naive_bayes | random_forest",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed = {"best", "logistic_regression", "naive_bayes", "random_forest"}
        if v not in allowed:
            raise ValueError(f"model must be one of: {allowed}")
        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be empty or whitespace only")
        return v.strip()


class SentimentScore(BaseModel):
    """Probability scores for each sentiment class."""
    positive: float
    neutral:  float
    negative: float


class AnalyzeResponse(BaseModel):
    """
    Output schema for POST /analyze.
    Returns sentiment label, confidence, scores, and cleaned text.
    """
    sentiment:    str            # "positive" | "neutral" | "negative"
    confidence:   float          # probability of predicted class
    scores:       SentimentScore # full probability breakdown
    model_used:   str
    cleaned_text: str            # text after preprocessing (for transparency)
    word_count:   int
    status:       str = "success"


# ── /stats ────────────────────────────────────────────────────────────────────

class NumericStats(BaseModel):
    mean:   float
    median: float
    std:    float
    min:    float
    max:    float
    q25:    float
    q75:    float


class SentimentCount(BaseModel):
    count:      int
    percentage: float


class StatsResponse(BaseModel):
    """Output schema for GET /stats."""
    total_posts:             int
    sentiment_distribution:  dict[str, SentimentCount]
    subreddit_breakdown:     dict[str, dict]
    numeric_stats:           dict[str, NumericStats]
    top_words:               dict[str, list[str]]
    available_plots:         list[str]
    status:                  str = "success"


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    models_loaded: list[str]
    version:       str = "1.0.0"