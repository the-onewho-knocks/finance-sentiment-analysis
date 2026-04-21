import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Download required NLTK assets once
nltk.download("stopwords",       quiet=True)
nltk.download("wordnet",         quiet=True)
nltk.download("omw-1.4",        quiet=True)
nltk.download("punkt",           quiet=True)
nltk.download("punkt_tab",       quiet=True)

STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()

# Finance-specific words we want to KEEP even if they look like stopwords
FINANCE_KEEP = {
    "up", "down", "high", "low", "not", "no", "bull", "bear",
    "buy", "sell", "loss", "gain", "crash", "rally", "above", "below",
}
STOP_WORDS -= FINANCE_KEEP


def _remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", text)


def _remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _remove_reddit_artifacts(text: str) -> str:
    """Remove r/subreddit, u/username mentions."""
    text = re.sub(r"r/\w+", " ", text)
    text = re.sub(r"u/\w+", " ", text)
    return text


def _remove_special_characters(text: str) -> str:
    """Keep only letters, numbers, and spaces."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single text string.
    Order matters: URLs → HTML → Reddit refs → lowercase → special chars → whitespace.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = _remove_urls(text)
    text = _remove_html_tags(text)
    text = _remove_reddit_artifacts(text)
    text = text.lower()
    text = _remove_special_characters(text)
    text = _normalize_whitespace(text)

    return text


def tokenize_and_lemmatize(text: str) -> str:
    """
    Tokenize → remove stopwords → lemmatize → rejoin as string.
    Returns cleaned string (not a list) so TF-IDF can consume it directly.
    """
    if not text:
        return ""

    tokens = nltk.word_tokenize(text)

    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    return " ".join(tokens)


def full_clean(text: str) -> str:
    """Convenience: clean + tokenize + lemmatize in one call."""
    return tokenize_and_lemmatize(clean_text(text))