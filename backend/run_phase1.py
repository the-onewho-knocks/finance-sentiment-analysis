"""
Run Phase 1: Reddit data collection.
Usage: python run_phase1.py
"""
from dotenv import load_dotenv
load_dotenv()

from src.data.reddit_scraper import collect_all_data

if __name__ == "__main__":
    df = collect_all_data()
    print("\n── Sample Data ──")
    print(df[["subreddit", "title", "score", "sentiment"]].head(10).to_string())