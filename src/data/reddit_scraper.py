# import re
# import time
# import threading
# from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import requests
# import pandas as pd
# import nltk
# from dotenv import load_dotenv
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_exponential,
#     retry_if_exception_type,
#     before_sleep_log,
# )
# from prometheus_client import push_to_gateway, REGISTRY

# from src.utils.config import (
#     TARGET_SUBREDDITS,
#     POSTS_PER_SUBREDDIT,
#     RAW_DATA_FILE,
# )
# from src.utils.logger import get_logger
# from src.metrics.prometheus_metrics import (
#     posts_scraped_total,
#     comments_scraped_total,
#     scrape_errors_total,
#     scrape_rate_limit_hits_total,
#     scrape_duration_seconds,
#     active_scrape_jobs,
#     last_scrape_timestamp,
#     posts_in_raw_dataset,
#     sentiment_labels_total,
#     vader_compound_score,
#     tickers_extracted_total,
#     unique_tickers_gauge,
# )

# load_dotenv()
# logger = get_logger(__name__)
# nltk.download("vader_lexicon", quiet=True)

# # FIX: was "pushgateway:9091" (Docker hostname) — must be localhost when running locally
# PUSHGATEWAY_URL = "localhost:9091"

# # ── Constants ─────────────────────────────────────────────────

# PULLPUSH_BASE = "https://api.pullpush.io/reddit/search"
# DEFAULT_TIMEOUT = 20
# COMMENTS_PER_POST = 25
# MAX_WORKERS = 3
# REQUEST_DELAY = 1.2
# BATCH_SIZE = 100

# _TICKER_PATTERN = re.compile(r'\b\$?([A-Z]{1,5})\b')
# _COMMON_WORDS = frozenset({
#     "I", "A", "THE", "AND", "OR", "BUT", "IN", "ON", "AT", "TO", "FOR",
#     "OF", "IS", "IT", "BE", "DO", "BY", "AN", "MY", "WE", "US", "UP",
#     "DOWN", "NOT", "NO", "YES", "SO", "AS", "IF", "ALL", "ARE", "WAS",
#     "HAS", "HAD", "CAN", "WILL", "JUST", "NOW", "NEW", "OLD", "BIG",
#     "GET", "GOT", "PUT", "SET", "LET", "ETA", "TBD", "TIL", "IMO",
#     "IMHO", "TIL", "LOL", "OMG", "CEO", "CFO", "CTO", "IPO", "ETF",
#     "NFT", "GDP", "CPI", "FED", "SEC", "NYSE", "NASDAQ", "SP", "DOW",
#     "USA", "US", "UK", "EU", "DD", "YOY", "QOQ", "PE", "EPS", "ATH",
#     "YOLO", "FOMO", "FUD", "HODL", "BUY", "SELL", "CALL", "PUT", "DCA",
# })

# _local = threading.local()


# def _get_session():
#     if not hasattr(_local, "session"):
#         _local.session = requests.Session()
#         _local.session.headers.update({
#             "User-Agent": "FinanceSentimentBot/2.0 (research project)",
#         })
#     return _local.session


# # ── Background metrics pusher ─────────────────────────────────
# # Pushes metrics to Pushgateway every PUSH_INTERVAL seconds while scraping,
# # so Grafana shows live progress instead of waiting until the very end.

# PUSH_INTERVAL = 5  # seconds between pushes — tune as needed

# class _MetricsPusher:
#     """Pushes metrics to Pushgateway on a background thread at a fixed interval."""

#     def __init__(self, url: str, job: str, interval: float = PUSH_INTERVAL):
#         self._url = url
#         self._job = job
#         self._interval = interval
#         self._stop_event = threading.Event()
#         self._thread = threading.Thread(target=self._loop, daemon=True, name="metrics-pusher")

#     def start(self):
#         self._stop_event.clear()
#         self._thread.start()
#         logger.info(f"📡 Background metrics pusher started (every {self._interval}s → {self._url})")

#     def stop(self):
#         """Signal the loop to stop and do one final push."""
#         self._stop_event.set()
#         self._thread.join(timeout=self._interval + 2)
#         self._push()  # final push to capture end-of-scrape state

#     def _push(self):
#         try:
#             push_to_gateway(self._url, job=self._job, registry=REGISTRY)
#         except Exception as exc:
#             logger.warning(f"⚠️  Metrics push failed: {exc}")

#     def _loop(self):
#         while not self._stop_event.wait(timeout=self._interval):
#             self._push()


# # ── Ticker extraction ─────────────────────────────────────────

# def extract_tickers(text: str) -> list[str]:
#     if not isinstance(text, str):
#         return []
#     matches = _TICKER_PATTERN.findall(text)
#     return list({m for m in matches if m not in _COMMON_WORDS and len(m) >= 2})


# # ── Sentiment ─────────────────────────────────────────────────

# def _analyze_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[str, float]:
#     if not isinstance(text, str) or not text.strip():
#         return "neutral", 0.0
#     scores = analyzer.polarity_scores(text)
#     compound = scores["compound"]
#     if compound >= 0.05:
#         return "positive", compound
#     elif compound <= -0.05:
#         return "negative", compound
#     return "neutral", compound


# # ── HTTP retry ─────────────────────────────────────────────────

# class RateLimitError(Exception):
#     pass


# @retry(
#     retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
#     wait=wait_exponential(multiplier=1, min=2, max=30),
#     stop=stop_after_attempt(5),
#     reraise=True,
# )
# def _get_with_retry(url: str, params: dict, subreddit: str) -> dict:
#     session = _get_session()
#     response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)

#     if response.status_code == 429:
#         scrape_rate_limit_hits_total.labels(subreddit=subreddit).inc()
#         raise RateLimitError()

#     response.raise_for_status()
#     return response.json()


# # ── Fetch posts ────────────────────────────────────────────────

# def fetch_submissions(subreddit: str, limit: int) -> list[dict]:
#     url = f"{PULLPUSH_BASE}/submission/"
#     all_posts = []
#     before = int(time.time())
#     seen_ids = set()

#     while len(all_posts) < limit:
#         size = min(BATCH_SIZE, limit - len(all_posts))
#         params = {"subreddit": subreddit, "size": size, "before": before}

#         try:
#             data = _get_with_retry(url, params, subreddit)
#             batch = data.get("data", [])
#         except RateLimitError:
#             time.sleep(15)
#             continue
#         except Exception as exc:
#             scrape_errors_total.labels(
#                 subreddit=subreddit, error_type=type(exc).__name__
#             ).inc()
#             break

#         if not batch:
#             break

#         new_posts = [p for p in batch if p.get("id") not in seen_ids]
#         seen_ids.update(p.get("id") for p in new_posts)
#         all_posts.extend(new_posts)

#         before = batch[-1].get("created_utc", before - 1)
#         time.sleep(REQUEST_DELAY)

#     return all_posts


# # ── Fetch comments ─────────────────────────────────────────────

# def fetch_comments_for_post(post_id: str, subreddit: str, limit: int = COMMENTS_PER_POST) -> list[dict]:
#     url = f"{PULLPUSH_BASE}/comment/"
#     all_comments = []
#     before = int(time.time())
#     seen_ids = set()

#     while len(all_comments) < limit:
#         size = min(BATCH_SIZE, limit - len(all_comments))
#         params = {"link_id": post_id, "size": size, "before": before}

#         try:
#             data = _get_with_retry(url, params, subreddit)
#             batch = data.get("data", [])
#         except RateLimitError:
#             time.sleep(10)
#             continue
#         except Exception:
#             break

#         if not batch:
#             break

#         new_comments = [c for c in batch if c.get("id") not in seen_ids]
#         seen_ids.update(c.get("id") for c in new_comments)
#         all_comments.extend(new_comments)

#         if len(batch) < size:
#             break

#         before = batch[-1].get("created_utc", before - 1)

#     return all_comments


# # ── Scrape subreddit ───────────────────────────────────────────

# def scrape_subreddit(analyzer, subreddit_name, limit):
#     data = []
#     all_tickers = set()

#     with scrape_duration_seconds.labels(subreddit=subreddit_name).time():
#         active_scrape_jobs.inc()

#         try:
#             posts = fetch_submissions(subreddit_name, limit)

#             for post in posts:
#                 post_id = post.get("id", "")
#                 text = f"{post.get('title','')} {post.get('selftext','')}"

#                 label, compound = _analyze_sentiment(text, analyzer)
#                 tickers = extract_tickers(text)

#                 all_tickers.update(tickers)
#                 posts_scraped_total.labels(subreddit=subreddit_name).inc()
#                 sentiment_labels_total.labels(subreddit=subreddit_name, sentiment=label).inc()
#                 vader_compound_score.labels(subreddit=subreddit_name).observe(compound)

#                 if tickers:
#                     tickers_extracted_total.labels(subreddit=subreddit_name).inc(len(tickers))

#                 data.append({
#                     "id": post_id,
#                     "type": "post",
#                     "text": text,
#                     "sentiment": label,
#                     "vader_compound": compound,
#                     "tickers": ",".join(tickers),
#                     "created_utc": post.get("created_utc"),
#                 })

#                 if post.get("num_comments", 0) > 0:
#                     comments = fetch_comments_for_post(post_id, subreddit_name)

#                     for c in comments:
#                         body = c.get("body", "")
#                         if not body:
#                             continue

#                         c_label, c_comp = _analyze_sentiment(body, analyzer)
#                         comments_scraped_total.labels(subreddit=subreddit_name).inc()
#                         sentiment_labels_total.labels(subreddit=subreddit_name, sentiment=c_label).inc()

#                         data.append({
#                             "id": c.get("id"),
#                             "type": "comment",
#                             "text": body,
#                             "sentiment": c_label,
#                             "vader_compound": c_comp,
#                             "created_utc": c.get("created_utc"),
#                         })

#             unique_tickers_gauge.set(len(all_tickers))
#             last_scrape_timestamp.labels(subreddit=subreddit_name).set(time.time())

#         finally:
#             active_scrape_jobs.dec()

#     return data


# # ── Main ───────────────────────────────────────────────────────

# def collect_all_data():
#     analyzer = SentimentIntensityAnalyzer()
#     all_data = []
#     seen_ids = set()
#     lock = threading.Lock()

#     # FIX: start background pusher so Grafana gets live updates every
#     # PUSH_INTERVAL seconds instead of waiting until the full scrape finishes
#     pusher = _MetricsPusher(url=PUSHGATEWAY_URL, job="reddit_scraper")
#     pusher.start()

#     def _scrape_one(name):
#         return scrape_subreddit(analyzer, name, POSTS_PER_SUBREDDIT)

#     try:
#         with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
#             futures = {pool.submit(_scrape_one, name): name for name in TARGET_SUBREDDITS}

#             for future in as_completed(futures):
#                 items = future.result()

#                 with lock:
#                     for item in items:
#                         if item["id"] not in seen_ids:
#                             seen_ids.add(item["id"])
#                             all_data.append(item)
#     finally:
#         # Always stop the pusher — triggers one final push even if scraping crashed
#         pusher.stop()
#         logger.info("✅ Final metrics pushed to Pushgateway.")

#     df = pd.DataFrame(all_data)
#     df.drop_duplicates(subset=["id"], inplace=True)

#     RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(RAW_DATA_FILE, index=False)

#     posts_in_raw_dataset.set(len(df))

#     # Push once more so posts_in_raw_dataset reflects the final CSV row count
#     try:
#         push_to_gateway(PUSHGATEWAY_URL, job="reddit_scraper", registry=REGISTRY)
#     except Exception as exc:
#         logger.warning(f"⚠️  Could not push final dataset metric: {exc}")

#     return df


# if __name__ == "__main__":
#     collect_all_data()

# import requests
# import pandas as pd
# import time
# from dotenv import load_dotenv
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk

# from src.utils.config import (
#     TARGET_SUBREDDITS,
#     POSTS_PER_SUBREDDIT,
#     RAW_DATA_FILE,
# )
# from src.utils.logger import get_logger

# load_dotenv()
# logger = get_logger(__name__)

# # Download VADER lexicon if not already present
# nltk.download("vader_lexicon", quiet=True)


# def _label_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> str:
#     """
#     Use VADER compound score to assign a sentiment label.
#     compound >= 0.05  → positive
#     compound <= -0.05 → negative
#     else              → neutral
#     """
#     if not isinstance(text, str) or not text.strip():
#         return "neutral"
        
#     score = analyzer.polarity_scores(text)["compound"]
#     if score >= 0.05:
#         return "positive"
#     elif score <= -0.05:
#         return "negative"
#     else:
#         return "neutral"


# def fetch_submissions(subreddit: str, limit: int) -> list:
#     """Fetch posts using Pullpush API."""
#     url = "https://api.pullpush.io/reddit/search/submission/"
#     all_posts = []
#     before = int(time.time())
#     retries = 0
    
#     while len(all_posts) < limit and retries < 3:
#         size = min(100, limit - len(all_posts))
#         params = {'subreddit': subreddit, 'size': size, 'before': before}
#         try:
#             response = requests.get(url, params=params, timeout=15)
#             if response.status_code == 200:
#                 batch = response.json().get('data', [])
#                 if not batch:
#                     break
#                 all_posts.extend(batch)
#                 before = batch[-1]['created_utc']
#                 time.sleep(1.5)
#                 retries = 0
#             elif response.status_code == 429:
#                 logger.warning(f"[Posts] Rate limit exceeded. Retrying in {5 * (retries + 1)}s...")
#                 time.sleep(5 * (retries + 1))
#                 retries += 1
#                 continue
#             else:
#                 logger.error(f"[Posts] Error {response.status_code}: {response.text}")
#                 break
#         except Exception as e:
#             logger.error(f"[Posts] Connection failed: {e}")
#             break
            
#     return all_posts


# def fetch_comments_for_post(post_id: str, limit: int = 50) -> list:
#     """Fetch comments for a given post ID using Pullpush API."""
#     url = "https://api.pullpush.io/reddit/search/comment/"
#     all_comments = []
#     before = int(time.time())
#     retries = 0
    
#     while len(all_comments) < limit and retries < 3:
#         size = min(100, limit - len(all_comments))
#         params = {'link_id': post_id, 'size': size, 'before': before}
#         try:
#             response = requests.get(url, params=params, timeout=15)
#             if response.status_code == 200:
#                 batch = response.json().get('data', [])
#                 if not batch:
#                     break
#                 all_comments.extend(batch)
#                 before = batch[-1]['created_utc']
#                 if len(batch) < size:
#                     break
#                 time.sleep(1.5)
#                 retries = 0
#             elif response.status_code == 429:
#                 logger.warning(f"[Comments] Rate limit exceeded for {post_id}. Retrying in {5 * (retries + 1)}s...")
#                 time.sleep(5 * (retries + 1))
#                 retries += 1
#                 continue
#             else:
#                 logger.error(f"[Comments] Error {response.status_code}: {response.text} for post {post_id}")
#                 break
#         except Exception as e:
#             logger.error(f"[Comments] Connection failed for post {post_id}: {e}")
#             break
            
#     return all_comments


# def scrape_subreddit(
#     analyzer: SentimentIntensityAnalyzer,
#     subreddit_name: str,
#     limit: int
# ) -> list[dict]:
#     """
#     Scrape posts and their top comments from a single subreddit.
#     Returns a combined list of dictionaries containing text and sentiment.
#     """
#     data = []
#     logger.info(f"Scraping r/{subreddit_name} — target: {limit} posts + comments")

#     posts = fetch_submissions(subreddit_name, limit)
    
#     for i, post in enumerate(posts):
#         if (i + 1) % 50 == 0:
#             logger.info(f"  ... processed {i + 1}/{len(posts)} posts for r/{subreddit_name} ...")
            
#         post_id = post.get('id')
#         title = post.get('title', '')
#         selftext = post.get('selftext', '')
#         full_text = f"{title} {selftext}".strip()
        
#         data.append({
#             "id":            post_id,
#             "subreddit":     subreddit_name,
#             "title":         title,
#             "text":          full_text,
#             "score":         post.get('score', 0),
#             "num_comments":  post.get('num_comments', 0),
#             "created_utc":   post.get('created_utc', int(time.time())),
#             "url":           post.get('url', ''),
#             "sentiment":     _label_sentiment(full_text, analyzer),
#             "type":          "post"
#         })
        
#         num_expected = post.get('num_comments', 0)
        
#         # Limit comment fetching to save time, max 10 per post
#         if num_expected > 0:
#             comments = fetch_comments_for_post(post_id, limit=10)
#             for c in comments:
#                 body = c.get('body', '').strip()
#                 if not body:
#                     continue
#                 data.append({
#                     "id":            c.get('id'),
#                     "subreddit":     subreddit_name,
#                     "title":         "",  # Comments don't have titles
#                     "text":          body,
#                     "score":         c.get('score', 0),
#                     "num_comments":  0,   
#                     "created_utc":   c.get('created_utc', int(time.time())),
#                     "url":           "",
#                     "sentiment":     _label_sentiment(body, analyzer),
#                     "type":          "comment"
#                 })
                
#     logger.info(f"  [OK] Collected {len(data)} total entries (posts+comments) from r/{subreddit_name}")
#     return data


# def collect_all_data() -> pd.DataFrame:
#     """
#     Main entry point: scrapes configured subreddits,
#     combined posts and comments, calculates sentiment, and saves to CSV progressively.
#     """
#     analyzer = SentimentIntensityAnalyzer()
#     all_data = []

#     def _save_progress(data: list, final: bool = False):
#         if not data:
#             if final: logger.warning("No data collected. Check API access or subreddit list.")
#             return pd.DataFrame()
            
#         df = pd.DataFrame(data)
#         df.drop_duplicates(subset=["id"], inplace=True)
#         df.reset_index(drop=True, inplace=True)

#         # Convert Unix timestamp → readable datetime
#         df["created_at"] = pd.to_datetime(df["created_utc"], unit="s")
#         df.drop(columns=["created_utc"], inplace=True)

#         RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
#         df.to_csv(RAW_DATA_FILE, index=False)
        
#         if final:
#             logger.info(f"--- Scraping Complete ---")
#             logger.info(f"Total entries collected: {len(df)}")
#             logger.info(f"Saved to: {RAW_DATA_FILE}")
#             logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
#             logger.info(f"Type distribution:\n{df['type'].value_counts()}")
#         else:
#             logger.info(f"Progressively saved {len(df)} entries to {RAW_DATA_FILE}")
            
#         return df

#     for subreddit_name in TARGET_SUBREDDITS:
#         try:
#             items = scrape_subreddit(analyzer, subreddit_name, POSTS_PER_SUBREDDIT)
#             all_data.extend(items)
#             _save_progress(all_data, final=False)
#             time.sleep(2)
#         except Exception as e:
#             logger.error(f"Failed to scrape r/{subreddit_name}: {e}")

#     return _save_progress(all_data, final=True)


# if __name__ == "__main__":
#     collect_all_data()

"""
src/data/reddit_scraper.py
──────────────────────────
Advanced Reddit scraper using PullPush API.

Improvements over v1:
  • Concurrent subreddit scraping via ThreadPoolExecutor
  • Exponential backoff on failures (via tenacity)
  • Rich data fields: author, flair, upvote_ratio, awards, is_news, ticker symbols
  • Ticker/stock symbol extraction from post text
  • VADER compound score stored (not just label)
  • Prometheus metrics for every stage
  • Deduplicated collection with seen-id set
  • Configurable comment depth per post (default 25)
"""

import re
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import nltk
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.utils.config import (
    TARGET_SUBREDDITS,
    POSTS_PER_SUBREDDIT,
    RAW_DATA_FILE,
)
from src.utils.logger import get_logger
from src.metrics.prometheus_metrics import (
    posts_scraped_total,
    comments_scraped_total,
    scrape_errors_total,
    scrape_rate_limit_hits_total,
    scrape_duration_seconds,
    active_scrape_jobs,
    last_scrape_timestamp,
    posts_in_raw_dataset,
    sentiment_labels_total,
    vader_compound_score,
    tickers_extracted_total,
    unique_tickers_gauge,
)

from src.metrics.prometheus_metrics import (
    posts_scraped_total,
    scrape_errors_total,
    active_scrape_jobs
)

active_scrape_jobs.inc()

try:
    # your scraping logic

    posts_scraped_total.inc()

except Exception as e:
    scrape_errors_total.inc()
    raise e

finally:
    active_scrape_jobs.dec()

load_dotenv()
logger = get_logger(__name__)
nltk.download("vader_lexicon", quiet=True)

# ── Constants ──────────────────────────────────────────────────────────────────

PULLPUSH_BASE       = "https://api.pullpush.io/reddit/search"
DEFAULT_TIMEOUT     = 20          # seconds per HTTP request
COMMENTS_PER_POST   = 25          # max comments per post (was 10)
MAX_WORKERS         = 3           # parallel subreddit scrapers
REQUEST_DELAY       = 1.2         # seconds between requests to same endpoint
BATCH_SIZE          = 100         # PullPush max batch size

# Common financial ticker pattern: 1-5 uppercase letters, optionally preceded by $
# Excludes common English words that happen to be uppercase abbreviations
_TICKER_PATTERN = re.compile(r'\b\$?([A-Z]{1,5})\b')
_COMMON_WORDS = frozenset({
    "I", "A", "THE", "AND", "OR", "BUT", "IN", "ON", "AT", "TO", "FOR",
    "OF", "IS", "IT", "BE", "DO", "BY", "AN", "MY", "WE", "US", "UP",
    "DOWN", "NOT", "NO", "YES", "SO", "AS", "IF", "ALL", "ARE", "WAS",
    "HAS", "HAD", "CAN", "WILL", "JUST", "NOW", "NEW", "OLD", "BIG",
    "GET", "GOT", "PUT", "SET", "LET", "ETA", "TBD", "TIL", "IMO",
    "IMHO", "TIL", "LOL", "OMG", "CEO", "CFO", "CTO", "IPO", "ETF",
    "NFT", "GDP", "CPI", "FED", "SEC", "NYSE", "NASDAQ", "SP", "DOW",
    "USA", "US", "UK", "EU", "DD", "YOY", "QOQ", "PE", "EPS", "ATH",
    "YOLO", "FOMO", "FUD", "HODL", "BUY", "SELL", "CALL", "PUT", "DCA",
})

# Thread-local session for connection reuse
_local = threading.local()

def _get_session() -> requests.Session:
    """Return a thread-local requests.Session for connection pooling."""
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update({
            "User-Agent": "FinanceSentimentBot/2.0 (research project)",
        })
    return _local.session


# ── Ticker extraction ──────────────────────────────────────────────────────────

def extract_tickers(text: str) -> list[str]:
    """
    Extract possible stock ticker symbols from text.
    Filters out common English words and returns unique tickers.

    Examples:
        "$AAPL is going up"          → ["AAPL"]
        "TSLA and AMZN look bullish" → ["TSLA", "AMZN"]
        "I think the CEO is good"    → []
    """
    if not isinstance(text, str):
        return []
    matches = _TICKER_PATTERN.findall(text)
    return list({m for m in matches if m not in _COMMON_WORDS and len(m) >= 2})


# ── Sentiment helpers ──────────────────────────────────────────────────────────

def _analyze_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[str, float]:
    """
    Returns (label, compound_score).
    label: "positive" | "negative" | "neutral"
    """
    if not isinstance(text, str) or not text.strip():
        return "neutral", 0.0
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive", compound
    elif compound <= -0.05:
        return "negative", compound
    else:
        return "neutral", compound


# ── HTTP helpers with retry ────────────────────────────────────────────────────

class RateLimitError(Exception):
    pass


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _get_with_retry(url: str, params: dict, subreddit: str) -> dict:
    """
    GET with exponential backoff on connection errors.
    Raises RateLimitError on 429 so the caller can handle it separately.
    """
    session = _get_session()
    response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)

    if response.status_code == 429:
        scrape_rate_limit_hits_total.labels(subreddit=subreddit).inc()
        raise RateLimitError(f"Rate limited on {url}")

    response.raise_for_status()
    return response.json()


# ── Submission fetcher ─────────────────────────────────────────────────────────

def fetch_submissions(subreddit: str, limit: int) -> list[dict]:
    """
    Fetch posts from a subreddit using PullPush API.
    Pages through results using `before` cursor until `limit` is reached.
    """
    url = f"{PULLPUSH_BASE}/submission/"
    all_posts: list[dict] = []
    before = int(time.time())
    seen_ids: set[str] = set()
    consecutive_empties = 0

    logger.info(f"[{subreddit}] Fetching up to {limit} posts…")

    while len(all_posts) < limit:
        size = min(BATCH_SIZE, limit - len(all_posts))
        params = {
            "subreddit": subreddit,
            "size": size,
            "before": before,
            "sort": "desc",
            "sort_type": "created_utc",
        }

        try:
            data = _get_with_retry(url, params, subreddit)
            batch = data.get("data", [])
        except RateLimitError:
            logger.warning(f"[{subreddit}] Rate limited — sleeping 15s…")
            time.sleep(15)
            continue
        except Exception as exc:
            scrape_errors_total.labels(subreddit=subreddit, error_type=type(exc).__name__).inc()
            logger.error(f"[{subreddit}] Submission fetch failed: {exc}")
            break

        if not batch:
            consecutive_empties += 1
            if consecutive_empties >= 2:
                break
            continue

        consecutive_empties = 0
        new_posts = [p for p in batch if p.get("id") not in seen_ids]
        seen_ids.update(p.get("id") for p in new_posts)
        all_posts.extend(new_posts)

        # Move cursor back past the oldest post in this batch
        before = batch[-1].get("created_utc", before - 1)
        time.sleep(REQUEST_DELAY)

    logger.info(f"[{subreddit}] Fetched {len(all_posts)} posts")
    return all_posts


# ── Comment fetcher ────────────────────────────────────────────────────────────

def fetch_comments_for_post(post_id: str, subreddit: str, limit: int = COMMENTS_PER_POST) -> list[dict]:
    """
    Fetch top-level comments for a given post ID.
    """
    url = f"{PULLPUSH_BASE}/comment/"
    all_comments: list[dict] = []
    before = int(time.time())
    seen_ids: set[str] = set()

    while len(all_comments) < limit:
        size = min(BATCH_SIZE, limit - len(all_comments))
        params = {
            "link_id": post_id,
            "size": size,
            "before": before,
            "sort": "desc",
            "sort_type": "score",   # get highest-scored comments first
        }

        try:
            data = _get_with_retry(url, params, subreddit)
            batch = data.get("data", [])
        except RateLimitError:
            logger.warning(f"[{subreddit}] Rate limited (comments {post_id}) — sleeping 10s…")
            time.sleep(10)
            continue
        except Exception as exc:
            scrape_errors_total.labels(subreddit=subreddit, error_type=type(exc).__name__).inc()
            break

        if not batch:
            break

        new_comments = [c for c in batch if c.get("id") not in seen_ids]
        seen_ids.update(c.get("id") for c in new_comments)
        all_comments.extend(new_comments)

        if len(batch) < size:
            break   # no more comments to fetch

        before = batch[-1].get("created_utc", before - 1)
        time.sleep(REQUEST_DELAY * 0.5)   # comments endpoint is lighter

    return all_comments


# ── Subreddit scraper ─────────────────────────────────────────────────────────

def scrape_subreddit(
    analyzer: SentimentIntensityAnalyzer,
    subreddit_name: str,
    limit: int,
) -> list[dict]:
    """
    Scrape posts + their top comments from one subreddit.
    Returns enriched list of dicts.
    """
    data: list[dict] = []
    all_tickers: set[str] = set()

    with scrape_duration_seconds.labels(subreddit=subreddit_name).time():
        active_scrape_jobs.inc()
        try:
            posts = fetch_submissions(subreddit_name, limit)

            for i, post in enumerate(posts):
                if (i + 1) % 50 == 0:
                    logger.info(f"  [{subreddit_name}] processed {i + 1}/{len(posts)} posts")

                post_id         = post.get("id", "")
                title           = post.get("title", "")
                selftext        = post.get("selftext", "")
                full_text       = f"{title} {selftext}".strip()
                label, compound = _analyze_sentiment(full_text, analyzer)
                tickers         = extract_tickers(full_text)
                all_tickers.update(tickers)

                # ── Prometheus ──
                posts_scraped_total.labels(subreddit=subreddit_name).inc()
                sentiment_labels_total.labels(subreddit=subreddit_name, sentiment=label).inc()
                vader_compound_score.labels(subreddit=subreddit_name).observe(compound)
                if tickers:
                    tickers_extracted_total.labels(subreddit=subreddit_name).inc(len(tickers))

                data.append({
                    # ── Identity ──
                    "id":               post_id,
                    "type":             "post",
                    "subreddit":        subreddit_name,
                    # ── Content ──
                    "title":            title,
                    "text":             full_text,
                    "url":              post.get("url", ""),
                    # ── Engagement ──
                    "score":            post.get("score", 0),
                    "upvote_ratio":     post.get("upvote_ratio", None),
                    "num_comments":     post.get("num_comments", 0),
                    "total_awards":     post.get("total_awards_received", 0),
                    "gilded":           post.get("gilded", 0),
                    # ── Author ──
                    "author":           post.get("author", "[deleted]"),
                    "author_flair":     post.get("author_flair_text", None),
                    "is_self":          post.get("is_self", True),
                    "domain":           post.get("domain", ""),
                    # ── Classification ──
                    "link_flair":       post.get("link_flair_text", None),
                    "over_18":          post.get("over_18", False),
                    "spoiler":          post.get("spoiler", False),
                    # ── Finance-specific ──
                    "tickers":          ",".join(tickers),
                    "has_ticker":       len(tickers) > 0,
                    # ── Sentiment ──
                    "sentiment":        label,
                    "vader_compound":   round(compound, 4),
                    # ── Time ──
                    "created_utc":      post.get("created_utc", int(time.time())),
                })

                # Fetch comments for posts that have them
                if post.get("num_comments", 0) > 0:
                    comments = fetch_comments_for_post(post_id, subreddit_name)
                    for c in comments:
                        body            = c.get("body", "").strip()
                        if not body or body in ("[deleted]", "[removed]"):
                            continue
                        c_label, c_comp = _analyze_sentiment(body, analyzer)
                        c_tickers       = extract_tickers(body)
                        all_tickers.update(c_tickers)

                        # ── Prometheus ──
                        comments_scraped_total.labels(subreddit=subreddit_name).inc()
                        sentiment_labels_total.labels(subreddit=subreddit_name, sentiment=c_label).inc()
                        vader_compound_score.labels(subreddit=subreddit_name).observe(c_comp)
                        if c_tickers:
                            tickers_extracted_total.labels(subreddit=subreddit_name).inc(len(c_tickers))

                        data.append({
                            "id":               c.get("id", ""),
                            "type":             "comment",
                            "subreddit":        subreddit_name,
                            "title":            "",
                            "text":             body,
                            "url":              "",
                            "score":            c.get("score", 0),
                            "upvote_ratio":     None,
                            "num_comments":     0,
                            "total_awards":     c.get("total_awards_received", 0),
                            "gilded":           c.get("gilded", 0),
                            "author":           c.get("author", "[deleted]"),
                            "author_flair":     c.get("author_flair_text", None),
                            "is_self":          True,
                            "domain":           "",
                            "link_flair":       None,
                            "over_18":          False,
                            "spoiler":          False,
                            "tickers":          ",".join(c_tickers),
                            "has_ticker":       len(c_tickers) > 0,
                            "sentiment":        c_label,
                            "vader_compound":   round(c_comp, 4),
                            "created_utc":      c.get("created_utc", int(time.time())),
                            # Comment-only fields
                            "parent_id":        c.get("parent_id", ""),
                            "post_id":          post_id,
                        })

            # Update unique tickers gauge across all collected data
            unique_tickers_gauge.set(len(all_tickers))
            last_scrape_timestamp.labels(subreddit=subreddit_name).set(time.time())

        except Exception as exc:
            scrape_errors_total.labels(subreddit=subreddit_name, error_type=type(exc).__name__).inc()
            logger.error(f"[{subreddit_name}] Fatal scrape error: {exc}", exc_info=True)
        finally:
            active_scrape_jobs.dec()

    logger.info(f"[{subreddit_name}] ✅ {len(data)} entries (posts + comments)")
    return data


# ── Main entry point ───────────────────────────────────────────────────────────

def collect_all_data() -> pd.DataFrame:
    """
    Scrape all configured subreddits concurrently (up to MAX_WORKERS in parallel),
    deduplicate, enrich timestamps, and save to CSV.
    """
    analyzer  = SentimentIntensityAnalyzer()
    all_data: list[dict] = []
    seen_ids: set[str]   = set()
    lock = threading.Lock()

    def _scrape_one(name: str) -> list[dict]:
        return scrape_subreddit(analyzer, name, POSTS_PER_SUBREDDIT)

    logger.info(f"🚀 Starting concurrent scrape of {len(TARGET_SUBREDDITS)} subreddits "
                f"(max {MAX_WORKERS} workers)…")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_scrape_one, name): name for name in TARGET_SUBREDDITS}
        for future in as_completed(futures):
            sub = futures[future]
            try:
                items = future.result()
                with lock:
                    for item in items:
                        if item["id"] not in seen_ids:
                            seen_ids.add(item["id"])
                            all_data.append(item)
                logger.info(f"[{sub}] merged into main dataset. Total so far: {len(all_data)}")
            except Exception as exc:
                logger.error(f"[{sub}] thread failed: {exc}")

    if not all_data:
        logger.warning("⚠️  No data collected. Check API access or subreddit config.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset=["id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert Unix timestamps → readable datetime
    df["created_at"] = pd.to_datetime(df["created_utc"], unit="s")
    df.drop(columns=["created_utc"], inplace=True, errors="ignore")

    # Save to CSV
    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_FILE, index=False)

    # Update dataset size gauge
    posts_in_raw_dataset.set(len(df))

    logger.info("─" * 60)
    logger.info(f"✅ Scraping complete — {len(df)} total entries")
    logger.info(f"📁 Saved → {RAW_DATA_FILE}")
    logger.info(f"📊 Sentiment distribution:\n{df['sentiment'].value_counts()}")
    logger.info(f"📝 Type distribution:\n{df['type'].value_counts()}")
    logger.info(f"🔤 Subreddits: {df['subreddit'].unique().tolist()}")
    logger.info("─" * 60)

    return df


# if __name__ == "__main__":
#     collect_all_data()

# import re
# import time
# import threading
# from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import requests
# import pandas as pd
# import nltk
# from dotenv import load_dotenv
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_exponential,
#     retry_if_exception_type,
#     before_sleep_log,
# )

# from src.utils.config import (
#     TARGET_SUBREDDITS,
#     POSTS_PER_SUBREDDIT,
#     RAW_DATA_FILE,
# )
# from src.utils.logger import get_logger
# from src.metrics.prometheus_metrics import (
#     posts_scraped_total,
#     comments_scraped_total,
#     scrape_errors_total,
#     scrape_rate_limit_hits_total,
#     scrape_duration_seconds,
#     active_scrape_jobs,
#     last_scrape_timestamp,
#     posts_in_raw_dataset,
#     sentiment_labels_total,
#     vader_compound_score,
#     tickers_extracted_total,
#     unique_tickers_gauge,
# )

# # ─────────────────────────────────────────
# from prometheus_client import push_to_gateway, REGISTRY

# def collect_all_data():
#     # ... existing code ...
    
# load_dotenv()
# logger = get_logger(__name__)
# nltk.download("vader_lexicon", quiet=True)

# # ── Constants ─────────────────────────────────────────────────

# PULLPUSH_BASE = "https://api.pullpush.io/reddit/search"
# DEFAULT_TIMEOUT = 20
# COMMENTS_PER_POST = 25
# MAX_WORKERS = 3
# REQUEST_DELAY = 1.2
# BATCH_SIZE = 100

# _TICKER_PATTERN = re.compile(r'\b\$?([A-Z]{1,5})\b')
# _COMMON_WORDS = frozenset({
#     "I", "A", "THE", "AND", "OR", "BUT", "IN", "ON", "AT", "TO", "FOR",
#     "OF", "IS", "IT", "BE", "DO", "BY", "AN", "MY", "WE", "US", "UP",
#     "DOWN", "NOT", "NO", "YES", "SO", "AS", "IF", "ALL", "ARE", "WAS",
#     "HAS", "HAD", "CAN", "WILL", "JUST", "NOW", "NEW", "OLD", "BIG",
#     "GET", "GOT", "PUT", "SET", "LET", "ETA", "TBD", "TIL", "IMO",
#     "IMHO", "TIL", "LOL", "OMG", "CEO", "CFO", "CTO", "IPO", "ETF",
#     "NFT", "GDP", "CPI", "FED", "SEC", "NYSE", "NASDAQ", "SP", "DOW",
#     "USA", "US", "UK", "EU", "DD", "YOY", "QOQ", "PE", "EPS", "ATH",
#     "YOLO", "FOMO", "FUD", "HODL", "BUY", "SELL", "CALL", "PUT", "DCA",
# })

# _local = threading.local()

# def _get_session():
#     if not hasattr(_local, "session"):
#         _local.session = requests.Session()
#         _local.session.headers.update({
#             "User-Agent": "FinanceSentimentBot/2.0 (research project)",
#         })
#     return _local.session


# # ── Ticker extraction ─────────────────────────────────────────

# def extract_tickers(text: str) -> list[str]:
#     if not isinstance(text, str):
#         return []
#     matches = _TICKER_PATTERN.findall(text)
#     return list({m for m in matches if m not in _COMMON_WORDS and len(m) >= 2})


# # ── Sentiment ─────────────────────────────────────────

# def _analyze_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[str, float]:
#     if not isinstance(text, str) or not text.strip():
#         return "neutral", 0.0
#     scores = analyzer.polarity_scores(text)
#     compound = scores["compound"]
#     if compound >= 0.05:
#         return "positive", compound
#     elif compound <= -0.05:
#         return "negative", compound
#     return "neutral", compound


# # ── HTTP retry ─────────────────────────────────────────

# class RateLimitError(Exception):
#     pass


# @retry(
#     retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
#     wait=wait_exponential(multiplier=1, min=2, max=30),
#     stop=stop_after_attempt(5),
#     reraise=True,
# )
# def _get_with_retry(url: str, params: dict, subreddit: str) -> dict:
#     session = _get_session()
#     response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)

#     if response.status_code == 429:
#         scrape_rate_limit_hits_total.labels(subreddit=subreddit).inc()
#         raise RateLimitError()

#     response.raise_for_status()
#     return response.json()


# # ── Fetch posts ─────────────────────────────────────────

# def fetch_submissions(subreddit: str, limit: int) -> list[dict]:
#     url = f"{PULLPUSH_BASE}/submission/"
#     all_posts = []
#     before = int(time.time())
#     seen_ids = set()

#     while len(all_posts) < limit:
#         size = min(BATCH_SIZE, limit - len(all_posts))
#         params = {"subreddit": subreddit, "size": size, "before": before}

#         try:
#             data = _get_with_retry(url, params, subreddit)
#             batch = data.get("data", [])
#         except RateLimitError:
#             time.sleep(15)
#             continue
#         except Exception as exc:
#             scrape_errors_total.labels(
#                 subreddit=subreddit, error_type=type(exc).__name__
#             ).inc()
#             break

#         if not batch:
#             break

#         new_posts = [p for p in batch if p.get("id") not in seen_ids]
#         seen_ids.update(p.get("id") for p in new_posts)
#         all_posts.extend(new_posts)

#         before = batch[-1].get("created_utc", before - 1)
#         time.sleep(REQUEST_DELAY)

#     return all_posts


# # ── Fetch comments ─────────────────────────────────────────

# def fetch_comments_for_post(post_id: str, subreddit: str, limit: int = COMMENTS_PER_POST) -> list[dict]:
#     url = f"{PULLPUSH_BASE}/comment/"
#     all_comments = []
#     before = int(time.time())
#     seen_ids = set()

#     while len(all_comments) < limit:
#         size = min(BATCH_SIZE, limit - len(all_comments))
#         params = {"link_id": post_id, "size": size, "before": before}

#         try:
#             data = _get_with_retry(url, params, subreddit)
#             batch = data.get("data", [])
#         except RateLimitError:
#             time.sleep(10)
#             continue
#         except Exception:
#             break

#         if not batch:
#             break

#         new_comments = [c for c in batch if c.get("id") not in seen_ids]
#         seen_ids.update(c.get("id") for c in new_comments)
#         all_comments.extend(new_comments)

#         if len(batch) < size:
#             break

#         before = batch[-1].get("created_utc", before - 1)

#     return all_comments


# # ── Scrape subreddit ─────────────────────────────────────────

# def scrape_subreddit(analyzer, subreddit_name, limit):
#     data = []
#     all_tickers = set()

#     with scrape_duration_seconds.labels(subreddit=subreddit_name).time():
#         active_scrape_jobs.inc()

#         try:
#             posts = fetch_submissions(subreddit_name, limit)

#             for post in posts:
#                 post_id = post.get("id", "")
#                 text = f"{post.get('title','')} {post.get('selftext','')}"

#                 label, compound = _analyze_sentiment(text, analyzer)
#                 tickers = extract_tickers(text)

#                 all_tickers.update(tickers)

#                 posts_scraped_total.labels(subreddit=subreddit_name).inc()

#                 data.append({
#                     "id": post_id,
#                     "type": "post",
#                     "text": text,
#                     "sentiment": label,
#                     "vader_compound": compound,
#                     "tickers": ",".join(tickers),
#                     "created_utc": post.get("created_utc"),
#                 })

#                 if post.get("num_comments", 0) > 0:
#                     comments = fetch_comments_for_post(post_id, subreddit_name)

#                     for c in comments:
#                         body = c.get("body", "")
#                         if not body:
#                             continue

#                         c_label, c_comp = _analyze_sentiment(body, analyzer)

#                         comments_scraped_total.labels(
#                             subreddit=subreddit_name
#                         ).inc()

#                         data.append({
#                             "id": c.get("id"),
#                             "type": "comment",
#                             "text": body,
#                             "sentiment": c_label,
#                             "vader_compound": c_comp,
#                             "created_utc": c.get("created_utc"),
#                         })

#             unique_tickers_gauge.set(len(all_tickers))
#             last_scrape_timestamp.labels(subreddit=subreddit_name).set(time.time())

#         finally:
#             active_scrape_jobs.dec()

#     return data


# # ── Main ─────────────────────────────────────────

# def collect_all_data():
#     analyzer = SentimentIntensityAnalyzer()
#     all_data = []
#     seen_ids = set()
#     lock = threading.Lock()

#     def _scrape_one(name):
#         return scrape_subreddit(analyzer, name, POSTS_PER_SUBREDDIT)

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
#         futures = {pool.submit(_scrape_one, name): name for name in TARGET_SUBREDDITS}

#         for future in as_completed(futures):
#             items = future.result()

#             with lock:
#                 for item in items:
#                     if item["id"] not in seen_ids:
#                         seen_ids.add(item["id"])
#                         all_data.append(item)

#     df = pd.DataFrame(all_data)
#     df.drop_duplicates(subset=["id"], inplace=True)

#     RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(RAW_DATA_FILE, index=False)

#     posts_in_raw_dataset.set(len(df))

#     return df


# if __name__ == "__main__":
#     collect_all_data()
    
#     # Push all scraper metrics to Pushgateway before process exits
#     push_to_gateway('pushgateway:9091', job='reddit_scraper', registry=REGISTRY)
#     return df

# load_dotenv()
# logger = get_logger(__name__)
# nltk.download("vader_lexicon", quiet=True)

# # ── Constants ─────────────────────────────────────────────────

# PULLPUSH_BASE = "https://api.pullpush.io/reddit/search"
# DEFAULT_TIMEOUT = 20
# COMMENTS_PER_POST = 25
# MAX_WORKERS = 3
# REQUEST_DELAY = 1.2
# BATCH_SIZE = 100

# _TICKER_PATTERN = re.compile(r'\b\$?([A-Z]{1,5})\b')
# _COMMON_WORDS = frozenset({
#     "I", "A", "THE", "AND", "OR", "BUT", "IN", "ON", "AT", "TO", "FOR",
#     "OF", "IS", "IT", "BE", "DO", "BY", "AN", "MY", "WE", "US", "UP",
#     "DOWN", "NOT", "NO", "YES", "SO", "AS", "IF", "ALL", "ARE", "WAS",
#     "HAS", "HAD", "CAN", "WILL", "JUST", "NOW", "NEW", "OLD", "BIG",
#     "GET", "GOT", "PUT", "SET", "LET", "ETA", "TBD", "TIL", "IMO",
#     "IMHO", "TIL", "LOL", "OMG", "CEO", "CFO", "CTO", "IPO", "ETF",
#     "NFT", "GDP", "CPI", "FED", "SEC", "NYSE", "NASDAQ", "SP", "DOW",
#     "USA", "US", "UK", "EU", "DD", "YOY", "QOQ", "PE", "EPS", "ATH",
#     "YOLO", "FOMO", "FUD", "HODL", "BUY", "SELL", "CALL", "PUT", "DCA",
# })

# _local = threading.local()

# def _get_session():
#     if not hasattr(_local, "session"):
#         _local.session = requests.Session()
#         _local.session.headers.update({
#             "User-Agent": "FinanceSentimentBot/2.0 (research project)",
#         })
#     return _local.session


# # ── Ticker extraction ─────────────────────────────────────────

# def extract_tickers(text: str) -> list[str]:
#     if not isinstance(text, str):
#         return []
#     matches = _TICKER_PATTERN.findall(text)
#     return list({m for m in matches if m not in _COMMON_WORDS and len(m) >= 2})


# # ── Sentiment ─────────────────────────────────────────

# def _analyze_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[str, float]:
#     if not isinstance(text, str) or not text.strip():
#         return "neutral", 0.0
#     scores = analyzer.polarity_scores(text)
#     compound = scores["compound"]
#     if compound >= 0.05:
#         return "positive", compound
#     elif compound <= -0.05:
#         return "negative", compound
#     return "neutral", compound


# # ── HTTP retry ─────────────────────────────────────────

# class RateLimitError(Exception):
#     pass


# @retry(
#     retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
#     wait=wait_exponential(multiplier=1, min=2, max=30),
#     stop=stop_after_attempt(5),
#     reraise=True,
# )
# def _get_with_retry(url: str, params: dict, subreddit: str) -> dict:
#     session = _get_session()
#     response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)

#     if response.status_code == 429:
#         scrape_rate_limit_hits_total.labels(subreddit=subreddit).inc()
#         raise RateLimitError()

#     response.raise_for_status()
#     return response.json()


# # ── Fetch posts ─────────────────────────────────────────

# def fetch_submissions(subreddit: str, limit: int) -> list[dict]:
#     url = f"{PULLPUSH_BASE}/submission/"
#     all_posts = []
#     before = int(time.time())
#     seen_ids = set()

#     while len(all_posts) < limit:
#         size = min(BATCH_SIZE, limit - len(all_posts))
#         params = {"subreddit": subreddit, "size": size, "before": before}

#         try:
#             data = _get_with_retry(url, params, subreddit)
#             batch = data.get("data", [])
#         except RateLimitError:
#             time.sleep(15)
#             continue
#         except Exception as exc:
#             scrape_errors_total.labels(
#                 subreddit=subreddit, error_type=type(exc).__name__
#             ).inc()
#             break

#         if not batch:
#             break

#         new_posts = [p for p in batch if p.get("id") not in seen_ids]
#         seen_ids.update(p.get("id") for p in new_posts)
#         all_posts.extend(new_posts)

#         before = batch[-1].get("created_utc", before - 1)
#         time.sleep(REQUEST_DELAY)

#     return all_posts


# # ── Fetch comments ─────────────────────────────────────────

# def fetch_comments_for_post(post_id: str, subreddit: str, limit: int = COMMENTS_PER_POST) -> list[dict]:
#     url = f"{PULLPUSH_BASE}/comment/"
#     all_comments = []
#     before = int(time.time())
#     seen_ids = set()

#     while len(all_comments) < limit:
#         size = min(BATCH_SIZE, limit - len(all_comments))
#         params = {"link_id": post_id, "size": size, "before": before}

#         try:
#             data = _get_with_retry(url, params, subreddit)
#             batch = data.get("data", [])
#         except RateLimitError:
#             time.sleep(10)
#             continue
#         except Exception:
#             break

#         if not batch:
#             break

#         new_comments = [c for c in batch if c.get("id") not in seen_ids]
#         seen_ids.update(c.get("id") for c in new_comments)
#         all_comments.extend(new_comments)

#         if len(batch) < size:
#             break

#         before = batch[-1].get("created_utc", before - 1)

#     return all_comments


# # ── Scrape subreddit ─────────────────────────────────────────

# def scrape_subreddit(analyzer, subreddit_name, limit):
#     data = []
#     all_tickers = set()

#     with scrape_duration_seconds.labels(subreddit=subreddit_name).time():
#         active_scrape_jobs.inc()

#         try:
#             posts = fetch_submissions(subreddit_name, limit)

#             for post in posts:
#                 post_id = post.get("id", "")
#                 text = f"{post.get('title','')} {post.get('selftext','')}"

#                 label, compound = _analyze_sentiment(text, analyzer)
#                 tickers = extract_tickers(text)

#                 all_tickers.update(tickers)

#                 posts_scraped_total.labels(subreddit=subreddit_name).inc()

#                 data.append({
#                     "id": post_id,
#                     "type": "post",
#                     "text": text,
#                     "sentiment": label,
#                     "vader_compound": compound,
#                     "tickers": ",".join(tickers),
#                     "created_utc": post.get("created_utc"),
#                 })

#                 if post.get("num_comments", 0) > 0:
#                     comments = fetch_comments_for_post(post_id, subreddit_name)

#                     for c in comments:
#                         body = c.get("body", "")
#                         if not body:
#                             continue

#                         c_label, c_comp = _analyze_sentiment(body, analyzer)

#                         comments_scraped_total.labels(
#                             subreddit=subreddit_name
#                         ).inc()

#                         data.append({
#                             "id": c.get("id"),
#                             "type": "comment",
#                             "text": body,
#                             "sentiment": c_label,
#                             "vader_compound": c_comp,
#                             "created_utc": c.get("created_utc"),
#                         })

#             unique_tickers_gauge.set(len(all_tickers))
#             last_scrape_timestamp.labels(subreddit=subreddit_name).set(time.time())

#         finally:
#             active_scrape_jobs.dec()

#     return data


# # ── Main ─────────────────────────────────────────

# def collect_all_data():
#     analyzer = SentimentIntensityAnalyzer()
#     all_data = []
#     seen_ids = set()
#     lock = threading.Lock()

#     def _scrape_one(name):
#         return scrape_subreddit(analyzer, name, POSTS_PER_SUBREDDIT)

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
#         futures = {pool.submit(_scrape_one, name): name for name in TARGET_SUBREDDITS}

#         for future in as_completed(futures):
#             items = future.result()

#             with lock:
#                 for item in items:
#                     if item["id"] not in seen_ids:
#                         seen_ids.add(item["id"])
#                         all_data.append(item)

#     df = pd.DataFrame(all_data)
#     df.drop_duplicates(subset=["id"], inplace=True)

#     RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(RAW_DATA_FILE, index=False)

#     posts_in_raw_dataset.set(len(df))

#     return df


# if __name__ == "__main__":
#     collect_all_data()