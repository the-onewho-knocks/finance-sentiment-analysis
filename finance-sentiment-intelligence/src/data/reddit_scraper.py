import requests
import pandas as pd
import time
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

from src.utils.config import (
    TARGET_SUBREDDITS,
    POSTS_PER_SUBREDDIT,
    RAW_DATA_FILE,
)
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# Download VADER lexicon if not already present
nltk.download("vader_lexicon", quiet=True)


def _label_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> str:
    """
    Use VADER compound score to assign a sentiment label.
    compound >= 0.05  → positive
    compound <= -0.05 → negative
    else              → neutral
    """
    if not isinstance(text, str) or not text.strip():
        return "neutral"
        
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def fetch_submissions(subreddit: str, limit: int) -> list:
    """Fetch posts using Pullpush API."""
    url = "https://api.pullpush.io/reddit/search/submission/"
    all_posts = []
    before = int(time.time())
    retries = 0
    
    while len(all_posts) < limit and retries < 3:
        size = min(100, limit - len(all_posts))
        params = {'subreddit': subreddit, 'size': size, 'before': before}
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                batch = response.json().get('data', [])
                if not batch:
                    break
                all_posts.extend(batch)
                before = batch[-1]['created_utc']
                time.sleep(1.5)
                retries = 0
            elif response.status_code == 429:
                logger.warning(f"[Posts] Rate limit exceeded. Retrying in {5 * (retries + 1)}s...")
                time.sleep(5 * (retries + 1))
                retries += 1
                continue
            else:
                logger.error(f"[Posts] Error {response.status_code}: {response.text}")
                break
        except Exception as e:
            logger.error(f"[Posts] Connection failed: {e}")
            break
            
    return all_posts


def fetch_comments_for_post(post_id: str, limit: int = 50) -> list:
    """Fetch comments for a given post ID using Pullpush API."""
    url = "https://api.pullpush.io/reddit/search/comment/"
    all_comments = []
    before = int(time.time())
    retries = 0
    
    while len(all_comments) < limit and retries < 3:
        size = min(100, limit - len(all_comments))
        params = {'link_id': post_id, 'size': size, 'before': before}
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                batch = response.json().get('data', [])
                if not batch:
                    break
                all_comments.extend(batch)
                before = batch[-1]['created_utc']
                if len(batch) < size:
                    break
                time.sleep(1.5)
                retries = 0
            elif response.status_code == 429:
                logger.warning(f"[Comments] Rate limit exceeded for {post_id}. Retrying in {5 * (retries + 1)}s...")
                time.sleep(5 * (retries + 1))
                retries += 1
                continue
            else:
                logger.error(f"[Comments] Error {response.status_code}: {response.text} for post {post_id}")
                break
        except Exception as e:
            logger.error(f"[Comments] Connection failed for post {post_id}: {e}")
            break
            
    return all_comments


def scrape_subreddit(
    analyzer: SentimentIntensityAnalyzer,
    subreddit_name: str,
    limit: int
) -> list[dict]:
    """
    Scrape posts and their top comments from a single subreddit.
    Returns a combined list of dictionaries containing text and sentiment.
    """
    data = []
    logger.info(f"Scraping r/{subreddit_name} — target: {limit} posts + comments")

    posts = fetch_submissions(subreddit_name, limit)
    
    for i, post in enumerate(posts):
        if (i + 1) % 50 == 0:
            logger.info(f"  ... processed {i + 1}/{len(posts)} posts for r/{subreddit_name} ...")
            
        post_id = post.get('id')
        title = post.get('title', '')
        selftext = post.get('selftext', '')
        full_text = f"{title} {selftext}".strip()
        
        data.append({
            "id":            post_id,
            "subreddit":     subreddit_name,
            "title":         title,
            "text":          full_text,
            "score":         post.get('score', 0),
            "num_comments":  post.get('num_comments', 0),
            "created_utc":   post.get('created_utc', int(time.time())),
            "url":           post.get('url', ''),
            "sentiment":     _label_sentiment(full_text, analyzer),
            "type":          "post"
        })
        
        num_expected = post.get('num_comments', 0)
        
        # Limit comment fetching to save time, max 10 per post
        if num_expected > 0:
            comments = fetch_comments_for_post(post_id, limit=10)
            for c in comments:
                body = c.get('body', '').strip()
                if not body:
                    continue
                data.append({
                    "id":            c.get('id'),
                    "subreddit":     subreddit_name,
                    "title":         "",  # Comments don't have titles
                    "text":          body,
                    "score":         c.get('score', 0),
                    "num_comments":  0,   
                    "created_utc":   c.get('created_utc', int(time.time())),
                    "url":           "",
                    "sentiment":     _label_sentiment(body, analyzer),
                    "type":          "comment"
                })
                
    logger.info(f"  [OK] Collected {len(data)} total entries (posts+comments) from r/{subreddit_name}")
    return data


def collect_all_data() -> pd.DataFrame:
    """
    Main entry point: scrapes configured subreddits,
    combined posts and comments, calculates sentiment, and saves to CSV progressively.
    """
    analyzer = SentimentIntensityAnalyzer()
    all_data = []

    def _save_progress(data: list, final: bool = False):
        if not data:
            if final: logger.warning("No data collected. Check API access or subreddit list.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df.drop_duplicates(subset=["id"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Convert Unix timestamp → readable datetime
        df["created_at"] = pd.to_datetime(df["created_utc"], unit="s")
        df.drop(columns=["created_utc"], inplace=True)

        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_FILE, index=False)
        
        if final:
            logger.info(f"--- Scraping Complete ---")
            logger.info(f"Total entries collected: {len(df)}")
            logger.info(f"Saved to: {RAW_DATA_FILE}")
            logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
            logger.info(f"Type distribution:\n{df['type'].value_counts()}")
        else:
            logger.info(f"Progressively saved {len(df)} entries to {RAW_DATA_FILE}")
            
        return df

    for subreddit_name in TARGET_SUBREDDITS:
        try:
            items = scrape_subreddit(analyzer, subreddit_name, POSTS_PER_SUBREDDIT)
            all_data.extend(items)
            _save_progress(all_data, final=False)
            time.sleep(2)
        except Exception as e:
            logger.error(f"Failed to scrape r/{subreddit_name}: {e}")

    return _save_progress(all_data, final=True)


if __name__ == "__main__":
    collect_all_data()
