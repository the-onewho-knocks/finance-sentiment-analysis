"""
Standalone tokenizer utilities.
Separated from cleaner.py so other modules can import
tokenization logic independently.
"""
import nltk
from src.utils.logger import get_logger

logger = get_logger(__name__)

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


def tokenize(text: str) -> list[str]:
    """Split cleaned text into a list of word tokens."""
    if not isinstance(text, str) or not text.strip():
        return []
    return nltk.word_tokenize(text)


def get_token_stats(texts: list[str]) -> dict:
    """
    Return basic token-level stats across a list of cleaned texts.
    Useful for quick EDA.
    """
    all_tokens = [tok for text in texts for tok in tokenize(text)]
    unique     = set(all_tokens)

    return {
        "total_tokens":   len(all_tokens),
        "unique_tokens":  len(unique),
        "avg_per_doc":    round(len(all_tokens) / max(len(texts), 1), 2),
        "top_10_tokens":  nltk.FreqDist(all_tokens).most_common(10),
    }