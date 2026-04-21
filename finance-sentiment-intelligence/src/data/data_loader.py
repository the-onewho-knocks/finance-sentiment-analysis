import pandas as pd
from src.utils.config import RAW_DATA_FILE, PROCESSED_DATA_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Load the raw scraped Reddit CSV."""
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_FILE}. "
            "Run Phase 1 scraper first."
        )
    df = pd.read_csv(RAW_DATA_FILE, parse_dates=["created_at"])
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_processed_data() -> pd.DataFrame:
    """Load the preprocessed/cleaned Reddit CSV."""
    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_FILE}. "
            "Run Phase 2 preprocessing first."
        )
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=["created_at"])
    logger.info(f"Loaded processed data: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def save_processed_data(df: pd.DataFrame) -> None:
    """Save cleaned/transformed DataFrame to processed CSV."""
    PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    logger.info(f"✅ Processed data saved to: {PROCESSED_DATA_FILE}")