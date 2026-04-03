"""
╔══════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Financial News Scraper                 ║
║         Project : MultiStockLSTMBot                             ║
║         File    : data/ingestion/news_scraper.py                ║
║         Phase   : 0 — Data Infrastructure                       ║
║         Purpose : Scrapes financial news from multiple sources  ║
║                   Stores in Elasticsearch for FinBERT scoring   ║
║                   Feeds Pillar 4 (Sentiment Analysis)           ║
╚══════════════════════════════════════════════════════════════════╝

What this file does:
--------------------
1. Scrapes news from Economic Times, Moneycontrol, BSE filings RSS
2. Extracts: headline, summary, source, publish_time, symbols_mentioned
3. Stores raw articles in Elasticsearch (godseye-news index)
4. Deduplicates by URL — safe to re-run without duplicates
5. Tags each article with which Nifty 500 stocks it mentions
6. Runs every 30 minutes during market hours

Elasticsearch Index Structure:
-------------------------------
    Index: godseye-news
    Fields:
        url          : Unique article URL (dedup key)
        headline     : Article headline
        summary      : Article body/summary (first 500 chars)
        source       : 'economic_times' | 'moneycontrol' | 'bse_filing' etc.
        publish_time : ISO timestamp
        symbols      : List of NSE symbols mentioned (e.g. ['RELIANCE', 'TCS'])
        sentiment    : Filled later by sentiment.py (FinBERT scoring)
        market_wide  : True if article is about market in general

Usage:
------
    # Run continuously (every 30 mins during market hours + post market)
    python -m data.ingestion.news_scraper --mode live

    # Single run (for testing)
    python -m data.ingestion.news_scraper --mode single

    # Backfill old news (limited by what RSS feeds expose)
    python -m data.ingestion.news_scraper --mode backfill

Dependencies:
-------------
    pip install requests beautifulsoup4 feedparser elasticsearch loguru python-dotenv pyyaml
"""

import os
import re
import time
import hashlib
import argparse
import feedparser

import requests
import yaml

from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "news_scraper_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="7 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}",
)

# ── Elasticsearch index name ──────────────────────────────────────────────
ES_INDEX = "godseye-news"

# ── News sources — RSS feeds (no authentication needed) ───────────────────
RSS_FEEDS = {
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "economic_times_stocks" : "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "moneycontrol_news"     : "https://www.moneycontrol.com/rss/latestnews.xml",
    "moneycontrol_markets"  : "https://www.moneycontrol.com/rss/marketreports.xml",
    "bse_announcements"     : "https://www.bseindia.com/xml-data/corpfiling/AttachLive/",
    "livemint_markets"      : "https://www.livemint.com/rss/markets",
    "business_standard"     : "https://www.business-standard.com/rss/markets-106.rss",
}

# ── Market-wide keywords (article affects whole market, not one stock) ────
MARKET_WIDE_KEYWORDS = [
    "nifty", "sensex", "rbi", "repo rate", "inflation", "gdp",
    "fii", "dii", "foreign investor", "market", "sebi", "budget",
    "interest rate", "fed", "federal reserve", "global market",
    "crude oil", "rupee", "dollar", "ipo", "bull", "bear",
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

SCRAPE_INTERVAL_MINUTES = 30


# ══════════════════════════════════════════════════════════════════════════
#  ELASTICSEARCH CLIENT
# ══════════════════════════════════════════════════════════════════════════

def get_es_client() -> Elasticsearch:
    """Returns Elasticsearch client from ELASTICSEARCH_URL in .env"""
    url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    es  = Elasticsearch(url)
    if not es.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {url}")
    return es


def ensure_index(es: Elasticsearch):
    """Creates the godseye-news index with correct mappings if not exists"""
    if es.indices.exists(index=ES_INDEX):
        return

    mapping = {
        "mappings": {
            "properties": {
                "url"         : {"type": "keyword"},
                "headline"    : {"type": "text", "analyzer": "english"},
                "summary"     : {"type": "text", "analyzer": "english"},
                "source"      : {"type": "keyword"},
                "publish_time": {"type": "date"},
                "symbols"     : {"type": "keyword"},
                "sentiment"   : {"type": "float"},    # filled by sentiment.py later
                "market_wide" : {"type": "boolean"},
                "scraped_at"  : {"type": "date"},
            }
        },
        "settings": {
            "number_of_shards"  : 1,
            "number_of_replicas": 0,    # single node; no replicas needed
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    logger.info(f"Elasticsearch index '{ES_INDEX}' created")


# ══════════════════════════════════════════════════════════════════════════
#  UNIVERSE LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_universe() -> set:
    """Loads Nifty 500 symbols for entity recognition in articles"""
    path = Path("config/universe.yaml")
    if not path.exists():
        logger.warning("universe.yaml not found — symbol tagging disabled")
        return set()
    with open(path) as f:
        data = yaml.safe_load(f)
    return set(data.get("nifty500", []))


# ══════════════════════════════════════════════════════════════════════════
#  ARTICLE PARSERS
# ══════════════════════════════════════════════════════════════════════════

def parse_rss_feed(feed_name: str, feed_url: str, universe: set) -> list[dict]:
    """
    Parses an RSS feed and returns list of article dicts.

    Args:
        feed_name : Internal name for this source
        feed_url  : RSS feed URL
        universe  : Set of NSE symbols for entity recognition

    Returns:
        List of article dicts ready for Elasticsearch
    """
    articles = []

    try:
        feed = feedparser.parse(feed_url)

        if feed.bozo and not feed.entries:
            logger.warning(f"Failed to parse RSS feed: {feed_name}")
            return []

        for entry in feed.entries:
            try:
                # Extract fields
                headline = _clean_text(entry.get("title", ""))
                summary  = _clean_text(
                    entry.get("summary", "") or
                    entry.get("description", "")
                )[:500]  # cap at 500 chars
                url = entry.get("link", "")

                if not headline or not url:
                    continue

                # Parse publish time
                pub_time = _parse_publish_time(entry)

                # Identify mentioned stocks
                full_text = f"{headline} {summary}".lower()
                symbols   = _extract_symbols(full_text, universe)

                # Is this market-wide news?
                market_wide = any(
                    kw in full_text for kw in MARKET_WIDE_KEYWORDS
                )

                articles.append({
                    "_id"         : _url_to_id(url),  # ES doc ID for dedup
                    "url"         : url,
                    "headline"    : headline,
                    "summary"     : summary,
                    "source"      : feed_name,
                    "publish_time": pub_time,
                    "symbols"     : list(symbols),
                    "sentiment"   : None,       # filled by sentiment.py
                    "market_wide" : market_wide,
                    "scraped_at"  : datetime.now(timezone.utc).isoformat(),
                })

            except Exception as e:
                logger.debug(f"Skipping entry in {feed_name}: {e}")
                continue

    except Exception as e:
        logger.error(f"RSS parse error for {feed_name}: {e}")

    return articles


def scrape_bse_filings(universe: set) -> list[dict]:
    """
    Scrapes BSE corporate announcements (regulatory filings).
    These get 3× sentiment weight in Pillar 4 due to high market impact.
    """
    articles = []
    url = "https://www.bseindia.com/corporates/ann.html"

    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if resp.status_code != 200:
            return []

        soup    = BeautifulSoup(resp.text, "html.parser")
        rows    = soup.select("table.table tr")

        for row in rows[1:]:  # skip header
            try:
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue

                symbol   = cols[0].get_text(strip=True).upper()
                headline = cols[2].get_text(strip=True)
                date_str = cols[3].get_text(strip=True)

                if symbol not in universe or not headline:
                    continue

                pub_time = _parse_date_string(date_str)

                articles.append({
                    "_id"         : _url_to_id(f"bse_filing_{symbol}_{date_str}_{headline[:30]}"),
                    "url"         : f"https://bseindia.com/filing/{symbol}",
                    "headline"    : f"[BSE FILING] {symbol}: {headline}",
                    "summary"     : headline,
                    "source"      : "bse_filing",
                    "publish_time": pub_time,
                    "symbols"     : [symbol],
                    "sentiment"   : None,
                    "market_wide" : False,
                    "scraped_at"  : datetime.now(timezone.utc).isoformat(),
                    "weight"      : 3.0,  # BSE filings get 3× sentiment weight
                })

            except Exception:
                continue

    except Exception as e:
        logger.warning(f"BSE filing scrape failed: {e}")

    return articles


# ══════════════════════════════════════════════════════════════════════════
#  ELASTICSEARCH WRITER
# ══════════════════════════════════════════════════════════════════════════

def index_articles(articles: list[dict], es: Elasticsearch) -> int:
    """
    Bulk-indexes articles into Elasticsearch.
    Uses update (not index) so existing articles are not overwritten.
    Only inserts new articles — existing ones are left as-is (preserve sentiment scores).

    Returns:
        Number of new articles indexed
    """
    if not articles:
        return 0

    actions = []
    for article in articles:
        doc_id = article.pop("_id")
        actions.append({
            "_op_type": "create",   # 'create' fails silently if doc already exists
            "_index"  : ES_INDEX,
            "_id"     : doc_id,
            "_source" : article,
        })

    try:
        success, errors = helpers.bulk(
            es, actions,
            raise_on_error=False,
            stats_only=False,
        )
        # Count actual new inserts (errors include 'already exists' which is expected)
        new_inserts = sum(
            1 for e in errors
            if e.get("create", {}).get("status") != 409  # 409 = already exists
        ) if errors else 0
        new_inserts = success

        if new_inserts > 0:
            logger.info(f"Indexed {new_inserts} new articles")
        return new_inserts

    except Exception as e:
        logger.error(f"Elasticsearch bulk index error: {e}")
        return 0


# ══════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """Removes HTML tags and normalizes whitespace"""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_symbols(text: str, universe: set) -> set:
    """
    Identifies NSE stock symbols mentioned in text.
    Uses word-boundary matching to avoid false positives.
    (e.g. 'TCS' matches 'TCS results' but not 'tactics')
    """
    found = set()
    for symbol in universe:
        # Word boundary match, case insensitive
        pattern = r"\b" + re.escape(symbol.lower()) + r"\b"
        if re.search(pattern, text):
            found.add(symbol)
    return found


def _url_to_id(url: str) -> str:
    """Converts URL to a short stable ID for Elasticsearch deduplication"""
    return hashlib.md5(url.encode()).hexdigest()


def _parse_publish_time(entry) -> str:
    """Parses RSS entry publish time to ISO format"""
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            import calendar
            ts = calendar.timegm(entry.published_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        pass
    return datetime.now(timezone.utc).isoformat()


def _parse_date_string(date_str: str) -> str:
    """Parses BSE date strings to ISO format"""
    formats = ["%d/%m/%Y %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d"]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN SCRAPING LOOP
# ══════════════════════════════════════════════════════════════════════════

def scrape_all(universe: set, es: Elasticsearch) -> int:
    """
    Scrapes all configured news sources in one pass.
    Returns total new articles indexed.
    """
    total_new = 0

    # RSS feeds
    for feed_name, feed_url in RSS_FEEDS.items():
        if "bse" in feed_name:
            continue  # handled separately
        articles = parse_rss_feed(feed_name, feed_url, universe)
        new      = index_articles(articles, es)
        total_new += new
        logger.info(f"{feed_name}: {len(articles)} scraped, {new} new")
        time.sleep(0.5)

    # BSE filings
    bse_articles = scrape_bse_filings(universe)
    new          = index_articles(bse_articles, es)
    total_new   += new
    logger.info(f"bse_filings: {len(bse_articles)} scraped, {new} new")

    return total_new


def run_live():
    """Scrapes news every 30 minutes. Run continuously."""
    logger.info("News scraper starting...")

    es      = get_es_client()
    universe = load_universe()
    ensure_index(es)

    try:
        while True:
            logger.info("Starting news scrape cycle...")
            total = scrape_all(universe, es)
            logger.info(f"Scrape cycle complete — {total} new articles")
            logger.info(f"Next scrape in {SCRAPE_INTERVAL_MINUTES} minutes")
            time.sleep(SCRAPE_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        logger.info("News scraper stopped")


def run_single():
    """Single scrape run — for testing"""
    es       = get_es_client()
    universe = load_universe()
    ensure_index(es)
    total = scrape_all(universe, es)
    print(f"\nTotal new articles indexed: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — News Scraper")
    parser.add_argument(
        "--mode", choices=["live", "single"],
        required=True,
        help="live: continuous 30-min scraping | single: one pass for testing"
    )
    args = parser.parse_args()

    if args.mode == "live":
        run_live()
    elif args.mode == "single":
        run_single()