"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Pillar 4: Sentiment Analysis (NLP)             ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : features/sentiment.py                                  ║
║         Phase   : 1 — Feature Engineering                               ║
║                                                                          ║
║  What this pillar learns:                                                ║
║    The sentiment embedded in financial news, BSE/NSE filings, and       ║
║    social media — at both the market level and individual stock level.  ║
║    Sentiment is a leading indicator — it often shifts BEFORE price does, ║
║    especially for event-driven moves (earnings, regulatory news,         ║
║    analyst upgrades/downgrades, management changes).                     ║
║                                                                          ║
║  Architecture — Two-Level Sentiment:                                     ║
║    Level 1 (Market): Aggregated across all Nifty 500 stocks             ║
║      → outputs market_sentiment_score [-1, +1]                          ║
║      → outputs market_fear_greed [0, 100]                               ║
║                                                                          ║
║    Level 2 (Stock): Per-stock news and filing sentiment                  ║
║      → outputs stock_sentiment_score [-1, +1] per symbol per day        ║
║      → outputs sentiment_momentum (5-day trend of sentiment)            ║
║      → outputs event_flag (earnings/regulatory/mgmt news detected)      ║
║                                                                          ║
║  Model:                                                                  ║
║    Base  : ProsusAI/finbert (pre-trained on Financial PhraseBank)       ║
║    Input : News headlines + first paragraph (max 512 tokens)            ║
║    Output: positive / negative / neutral probabilities                   ║
║    Device: CUDA (NVIDIA GPU) with automatic CPU fallback                 ║
║                                                                          ║
║  Inference pipeline:                                                     ║
║    1. Load news from Elasticsearch (news_articles index)                 ║
║    2. Batch texts through FinBERT (batch_size=16 on GPU)                ║
║    3. Aggregate per-stock scores with recency weighting                  ║
║    4. Compute market-level aggregation                                   ║
║    5. Detect event flags (3× weight multiplier)                          ║
║    6. Save to features_sentiment table                                   ║
║                                                                          ║
║  Data source fallback chain:                                             ║
║    Primary   → Elasticsearch news_articles index                         ║
║    Secondary → Hardcoded neutral (0.0) with is_estimated=True            ║
║    (Elasticsearch may be empty in early Phase 1 — handled gracefully)   ║
║                                                                          ║
║  GPU memory management:                                                  ║
║    Batch size 16 on GPU (safe for 6GB+ VRAM)                            ║
║    Batch size 4 on CPU (safe for 16GB RAM)                              ║
║    Model loaded once and cached for the session                          ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install transformers torch elasticsearch pandas numpy loguru      ║
║    Model downloads automatically on first run (~440MB from HuggingFace) ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import re
import warnings
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from datetime import date, datetime, timedelta
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

# ── Database & Search ──────────────────────────────────────────────────────
DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

# ── FinBERT model config ──────────────────────────────────────────────────
FINBERT_MODEL     = "ProsusAI/finbert"
BATCH_SIZE_GPU    = 16     # safe for 6GB+ VRAM
BATCH_SIZE_CPU    = 4      # safe for 16GB RAM
MAX_TOKEN_LENGTH  = 512    # FinBERT max input length
MODEL_CACHE_DIR   = ".model_cache"   # local cache to avoid re-downloading

# ── Sentiment computation params ──────────────────────────────────────────
RECENCY_HALF_LIFE    = 3      # days — older news decays exponentially
MOMENTUM_WINDOW      = 5      # days for sentiment momentum
EVENT_WEIGHT_MULT    = 3.0    # multiplier for regulatory/earnings news
NEWS_PER_STOCK_LIMIT = 20     # max articles per stock per day (GPU efficiency)
MIN_ARTICLE_LENGTH   = 30     # characters — skip very short snippets

# ── Event keyword detection (triggers 3× weight) ─────────────────────────
EVENT_KEYWORDS = [
    # Earnings
    "quarterly results", "q1 results", "q2 results", "q3 results", "q4 results",
    "earnings", "profit", "revenue", "ebitda", "net income", "eps",
    # Regulatory
    "sebi", "rbi", "nclt", "cbi", "ed ", "enforcement directorate",
    "show cause", "penalty", "fine", "ban", "suspension",
    # Management
    "ceo resign", "md resign", "chairman resign", "cfo resign",
    "board reshuffle", "management change", "promoter pledge",
    # Corporate actions
    "merger", "acquisition", "buyback", "rights issue", "fpo", "ipo",
    "dividend", "bonus shares", "stock split", "demerger",
    # Analyst
    "upgrade", "downgrade", "target price", "buy rating", "sell rating",
    "outperform", "underperform",
]

# ── Lazy model loading ────────────────────────────────────────────────────
# Model is loaded once on first use and cached in memory
_finbert_pipeline = None
_device           = None


def _get_device() -> str:
    """
    Detects the best available device for inference.
    Returns 'cuda' if NVIDIA GPU is available, else 'cpu'.
    """
    global _device
    if _device is None:
        try:
            import torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            if _device == "cuda":
                import torch
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU detected: {gpu_name} ({vram_gb:.1f}GB VRAM) — using CUDA")
            else:
                logger.info("No NVIDIA GPU detected — using CPU for FinBERT inference")
        except ImportError:
            _device = "cpu"
            logger.warning("PyTorch not installed — defaulting to CPU")
    return _device


def _get_batch_size() -> int:
    """Returns appropriate batch size based on device."""
    return BATCH_SIZE_GPU if _get_device() == "cuda" else BATCH_SIZE_CPU


def get_finbert_pipeline():
    """
    Loads FinBERT model and tokenizer (lazy — only on first call).
    Model is cached in .model_cache/ to avoid re-downloading.

    Returns:
        HuggingFace pipeline for text classification, or None if
        transformers library is not installed.
    """
    global _finbert_pipeline

    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        import torch

        device_str = _get_device()
        device_id  = 0 if device_str == "cuda" else -1  # HF pipeline uses -1 for CPU

        logger.info(f"Loading FinBERT model: {FINBERT_MODEL}")
        logger.info("First run: downloading ~440MB model (cached after this)...")

        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        _finbert_pipeline = pipeline(
            task              = "text-classification",
            model             = FINBERT_MODEL,
            tokenizer         = FINBERT_MODEL,
            device            = device_id,
            truncation        = True,
            max_length        = MAX_TOKEN_LENGTH,
            model_kwargs      = {"cache_dir": MODEL_CACHE_DIR},
            top_k             = None,   # return all 3 class scores
        )

        logger.success(
            f"FinBERT loaded successfully on {device_str.upper()}. "
            f"Batch size: {_get_batch_size()}"
        )
        return _finbert_pipeline

    except ImportError:
        logger.error(
            "transformers or torch not installed.\n"
            "Run: pip install transformers torch\n"
            "Sentiment will default to neutral (0.0) until installed."
        )
        return None
    except Exception as e:
        logger.error(f"Failed to load FinBERT: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_conn():
    return psycopg2.connect(DB_URL)


def _ensure_features_table(conn):
    """
    Creates features_sentiment table — both market-level and stock-level
    rows stored together (market rows have symbol = '__MARKET__').
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS features_sentiment (
                date                    DATE        NOT NULL,
                symbol                  VARCHAR(20) NOT NULL,
                -- '__MARKET__' for market-level rows

                -- Article counts
                articles_processed      SMALLINT,
                articles_positive       SMALLINT,
                articles_negative       SMALLINT,
                articles_neutral        SMALLINT,

                -- Raw FinBERT scores (weighted average probabilities)
                raw_positive            NUMERIC(5,4),
                raw_negative            NUMERIC(5,4),
                raw_neutral             NUMERIC(5,4),

                -- Derived sentiment score
                sentiment_score         NUMERIC(5,4),  -- [-1.0, +1.0]
                sentiment_momentum      NUMERIC(5,4),  -- 5-day trend [-1, +1]
                sentiment_strength      NUMERIC(5,4),  -- abs(score) [0, 1]

                -- Event detection
                event_flag              BOOLEAN,       -- high-impact event detected
                event_type              VARCHAR(50),   -- 'earnings', 'regulatory', etc.
                event_weight_applied    NUMERIC(4,2),  -- multiplier used

                -- Market-level only (NULL for stock rows)
                market_fear_greed       NUMERIC(5,1),  -- [0, 100]

                -- Data quality
                is_estimated            BOOLEAN DEFAULT FALSE,
                last_article_age_hours  SMALLINT,      -- hours since most recent article

                PRIMARY KEY (date, symbol)
            );
        """)

        cur.execute("""
            SELECT create_hypertable(
                'features_sentiment', 'date',
                if_not_exists => TRUE,
                migrate_data  => TRUE
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_sentiment_symbol
            ON features_sentiment (symbol, date DESC);
        """)

    conn.commit()
    logger.info("features_sentiment table ready.")


# ══════════════════════════════════════════════════════════════════════════
#  ELASTICSEARCH NEWS LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_news_from_elasticsearch(
    symbols: list[str],
    target_date: date,
    lookback_days: int = 3,
) -> dict[str, list[dict]]:
    """
    Loads recent news articles from Elasticsearch for given symbols.

    Query strategy:
        Fetches articles published within `lookback_days` of target_date.
        Returns dict mapping symbol → list of article dicts.
        Each article dict has: title, content, published_at, source.

    Fallback:
        If Elasticsearch is unavailable or empty, returns empty dict.
        Caller handles this gracefully with estimated neutral scores.

    Args:
        symbols      : List of NSE symbols to fetch news for
        target_date  : The date we're computing sentiment for
        lookback_days: How many days of news history to include

    Returns:
        dict[symbol → list[article_dict]]
        Plus '__MARKET__' key for market-wide articles
    """
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(ES_URL, request_timeout=10)

        if not es.ping():
            logger.warning("Elasticsearch not reachable — using estimated sentiment")
            return {}

    except ImportError:
        logger.warning("elasticsearch package not installed — using estimated sentiment")
        return {}
    except Exception as e:
        logger.warning(f"Elasticsearch connection failed: {e} — using estimated sentiment")
        return {}

    date_from = (datetime.combine(target_date, datetime.min.time())
                 - timedelta(days=lookback_days)).isoformat()
    date_to   = datetime.combine(target_date, datetime.max.time()).isoformat()

    news_by_symbol: dict[str, list[dict]] = {"__MARKET__": []}

    try:
        # ── Fetch market-wide news ─────────────────────────────────────────
        market_query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"published_at": {"gte": date_from, "lte": date_to}}},
                        {"terms": {"category": ["market", "economy", "nse", "bse",
                                                "sensex", "nifty", "rbi", "sebi"]}}
                    ]
                }
            },
            "sort": [{"published_at": "desc"}],
            "size": 50,
        }

        resp = es.search(index="news_articles", body=market_query)
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            news_by_symbol["__MARKET__"].append({
                "title"       : src.get("title", ""),
                "content"     : src.get("content", src.get("summary", "")),
                "published_at": src.get("published_at", ""),
                "source"      : src.get("source", ""),
            })

        # ── Fetch per-symbol news ─────────────────────────────────────────
        # Batch query for all symbols at once (more efficient than N queries)
        if symbols:
            stock_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"range": {"published_at": {"gte": date_from, "lte": date_to}}},
                            {"terms": {"symbols": symbols}}
                        ]
                    }
                },
                "sort": [{"published_at": "desc"}],
                "size": min(len(symbols) * NEWS_PER_STOCK_LIMIT, 500),
            }

            resp = es.search(index="news_articles", body=stock_query)
            for hit in resp["hits"]["hits"]:
                src      = hit["_source"]
                art_syms = src.get("symbols", [])
                article  = {
                    "title"       : src.get("title", ""),
                    "content"     : src.get("content", src.get("summary", "")),
                    "published_at": src.get("published_at", ""),
                    "source"      : src.get("source", ""),
                }
                for sym in art_syms:
                    if sym in symbols:
                        news_by_symbol.setdefault(sym, []).append(article)

    except Exception as e:
        logger.warning(f"Elasticsearch query failed: {e}")
        return {}

    total = sum(len(v) for v in news_by_symbol.values())
    logger.info(
        f"News loaded from ES: {total} articles for "
        f"{len([k for k in news_by_symbol if k != '__MARKET__'])} stocks + market"
    )
    return news_by_symbol


# ══════════════════════════════════════════════════════════════════════════
#  TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

def preprocess_text(title: str, content: str = "") -> str:
    """
    Prepares text for FinBERT input.

    Strategy:
        Concatenates title + first 200 chars of content.
        Title gets implicit emphasis by being placed first.
        Removes HTML tags, URLs, excessive whitespace.
        Truncation happens inside FinBERT (max 512 tokens).

    Args:
        title   : Article headline
        content : Article body (will be truncated)

    Returns:
        Clean text string ready for FinBERT tokenization.
    """
    # Clean title
    title = re.sub(r"<[^>]+>", " ", title)           # strip HTML
    title = re.sub(r"http\S+", "", title)             # strip URLs
    title = re.sub(r"\s+", " ", title).strip()

    # Clean and truncate content
    content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"http\S+", "", content)
    content = re.sub(r"\s+", " ", content).strip()
    content = content[:300]                           # keep first 300 chars

    # Combine: title has more signal than body for financial news
    text = f"{title}. {content}".strip(" .")

    # Remove very short texts (not worth processing)
    if len(text) < MIN_ARTICLE_LENGTH:
        return ""

    return text


def detect_event(title: str, content: str = "") -> tuple[bool, str]:
    """
    Detects whether an article contains a high-impact event.
    High-impact events get 3× weight in sentiment aggregation.

    Args:
        title   : Article headline
        content : Article body

    Returns:
        (is_event: bool, event_type: str)
        event_type is one of: 'earnings', 'regulatory', 'management',
                               'corporate_action', 'analyst', ''
    """
    combined = (title + " " + content[:200]).lower()

    # Check earnings keywords
    earnings_kw = ["quarterly results", "q1 result", "q2 result", "q3 result",
                   "q4 result", "earnings", "net profit", "net loss", "revenue",
                   "ebitda", "pat ", "eps "]
    if any(kw in combined for kw in earnings_kw):
        return True, "earnings"

    # Check regulatory keywords
    regulatory_kw = ["sebi", "rbi notice", "nclt", "enforcement directorate",
                     "show cause", "penalty imposed", "fine of", "ban on",
                     "suspended", "investigated"]
    if any(kw in combined for kw in regulatory_kw):
        return True, "regulatory"

    # Check management keywords
    mgmt_kw = ["ceo resigns", "md resigns", "cfo resigns", "chairman resigns",
               "board reshuffle", "promoter pledge", "management change"]
    if any(kw in combined for kw in mgmt_kw):
        return True, "management"

    # Check corporate action keywords
    corp_kw = ["merger", "acquisition", "buyback", "rights issue",
               "stock split", "bonus share", "demerger", "open offer"]
    if any(kw in combined for kw in corp_kw):
        return True, "corporate_action"

    # Check analyst keywords
    analyst_kw = ["upgrade to buy", "downgrade to sell", "target price raised",
                  "target price cut", "initiates coverage", "rating change"]
    if any(kw in combined for kw in analyst_kw):
        return True, "analyst"

    return False, ""


# ══════════════════════════════════════════════════════════════════════════
#  FINBERT INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════

def run_finbert_batch(texts: list[str]) -> list[dict]:
    """
    Runs FinBERT inference on a batch of texts.

    Args:
        texts : List of preprocessed text strings

    Returns:
        List of dicts, each with keys: positive, negative, neutral (floats)
        Returns neutral (1/3 each) for texts that fail or are empty.
    """
    if not texts:
        return []

    # Filter out empty texts
    valid_mask  = [bool(t.strip()) for t in texts]
    valid_texts = [t for t, v in zip(texts, valid_mask) if v]

    neutral_result = {"positive": 0.333, "negative": 0.333, "neutral": 0.333}

    if not valid_texts:
        return [neutral_result] * len(texts)

    pipeline = get_finbert_pipeline()

    if pipeline is None:
        # FinBERT not available — return neutral for all
        return [neutral_result] * len(texts)

    batch_size = _get_batch_size()
    all_results = []

    try:
        # Process in batches to manage memory
        for i in range(0, len(valid_texts), batch_size):
            batch      = valid_texts[i: i + batch_size]
            raw_output = pipeline(batch, batch_size=batch_size)

            for item_scores in raw_output:
                # item_scores is a list of {label, score} dicts
                # (because we set top_k=None)
                scores = {s["label"].lower(): s["score"] for s in item_scores}
                all_results.append({
                    "positive": scores.get("positive", 0.333),
                    "negative": scores.get("negative", 0.333),
                    "neutral" : scores.get("neutral",  0.333),
                })

    except Exception as e:
        logger.error(f"FinBERT inference error: {e}")
        all_results = [neutral_result] * len(valid_texts)

    # Re-insert neutral results for empty/invalid texts
    final_results = []
    valid_iter    = iter(all_results)
    for is_valid in valid_mask:
        if is_valid:
            final_results.append(next(valid_iter))
        else:
            final_results.append(neutral_result)

    return final_results


def score_from_probabilities(positive: float, negative: float) -> float:
    """
    Converts FinBERT positive/negative probabilities to a sentiment score.

    Score formula: positive - negative, range [-1.0, +1.0]
        +1.0 = perfectly positive (positive=1.0, negative=0.0)
         0.0 = neutral (equal positive and negative)
        -1.0 = perfectly negative (positive=0.0, negative=1.0)

    Args:
        positive : FinBERT positive probability [0, 1]
        negative : FinBERT negative probability [0, 1]

    Returns:
        float in [-1.0, +1.0]
    """
    return float(np.clip(positive - negative, -1.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════
#  SENTIMENT AGGREGATION
# ══════════════════════════════════════════════════════════════════════════

def aggregate_article_scores(
    articles    : list[dict],
    finbert_out : list[dict],
    target_date : date,
) -> dict:
    """
    Aggregates FinBERT scores for multiple articles into a single
    sentiment score for one stock/market on one day.

    Aggregation method:
        Each article gets a weight based on:
            1. Recency: exponential decay with half-life of 3 days
               (article from yesterday = 0.79× weight vs today)
            2. Event flag: 3× multiplier for high-impact events
        Final score = weighted average of individual article scores.

    Args:
        articles    : List of article dicts (with published_at, title, content)
        finbert_out : FinBERT output for each article (parallel list)
        target_date : The date we're computing sentiment for

    Returns:
        dict with sentiment_score, raw_positive, raw_negative, raw_neutral,
        articles_processed, articles_positive, articles_negative,
        articles_neutral, event_flag, event_type, event_weight_applied
    """
    if not articles or not finbert_out:
        return _neutral_sentiment_result(0)

    scores   = []
    weights  = []
    pos_probs= []
    neg_probs= []
    neu_probs= []

    event_detected  = False
    event_type_best = ""

    for article, fb_out in zip(articles, finbert_out):
        # ── Recency weight ─────────────────────────────────────────────
        pub_str = article.get("published_at", "")
        try:
            pub_dt  = pd.to_datetime(pub_str).date()
            age_days= (target_date - pub_dt).days
            age_days= max(0, age_days)
        except Exception:
            age_days = 1   # assume 1 day old if can't parse

        recency_weight = 2 ** (-age_days / RECENCY_HALF_LIFE)

        # ── Event weight ───────────────────────────────────────────────
        title   = article.get("title", "")
        content = article.get("content", "")
        is_event, evt_type = detect_event(title, content)

        event_weight = EVENT_WEIGHT_MULT if is_event else 1.0
        if is_event and not event_detected:
            event_detected  = True
            event_type_best = evt_type

        # ── Final article weight ───────────────────────────────────────
        total_weight = recency_weight * event_weight

        score = score_from_probabilities(fb_out["positive"], fb_out["negative"])

        scores.append(score)
        weights.append(total_weight)
        pos_probs.append(fb_out["positive"])
        neg_probs.append(fb_out["negative"])
        neu_probs.append(fb_out["neutral"])

    # ── Weighted aggregation ───────────────────────────────────────────────
    total_w = sum(weights)
    if total_w == 0:
        return _neutral_sentiment_result(len(articles))

    final_score   = sum(s * w for s, w in zip(scores, weights)) / total_w
    final_pos     = sum(p * w for p, w in zip(pos_probs, weights)) / total_w
    final_neg     = sum(n * w for n, w in zip(neg_probs, weights)) / total_w
    final_neu     = sum(n * w for n, w in zip(neu_probs, weights)) / total_w

    # Count by sentiment direction
    n_pos = sum(1 for s in scores if s >  0.1)
    n_neg = sum(1 for s in scores if s < -0.1)
    n_neu = len(scores) - n_pos - n_neg

    return {
        "sentiment_score"      : float(np.clip(final_score, -1.0, 1.0)),
        "raw_positive"         : float(final_pos),
        "raw_negative"         : float(final_neg),
        "raw_neutral"          : float(final_neu),
        "articles_processed"   : len(articles),
        "articles_positive"    : n_pos,
        "articles_negative"    : n_neg,
        "articles_neutral"     : n_neu,
        "event_flag"           : event_detected,
        "event_type"           : event_type_best,
        "event_weight_applied" : EVENT_WEIGHT_MULT if event_detected else 1.0,
        "is_estimated"         : False,
    }


def _neutral_sentiment_result(n_articles: int) -> dict:
    """Returns a neutral sentiment result when no data is available."""
    return {
        "sentiment_score"      : 0.0,
        "raw_positive"         : 0.333,
        "raw_negative"         : 0.333,
        "raw_neutral"          : 0.333,
        "articles_processed"   : n_articles,
        "articles_positive"    : 0,
        "articles_negative"    : 0,
        "articles_neutral"     : n_articles,
        "event_flag"           : False,
        "event_type"           : "",
        "event_weight_applied" : 1.0,
        "is_estimated"         : True,
    }


# ══════════════════════════════════════════════════════════════════════════
#  SENTIMENT MOMENTUM
# ══════════════════════════════════════════════════════════════════════════

def compute_sentiment_momentum(
    symbol: str,
    target_date: date,
    conn,
) -> float:
    """
    Computes 5-day sentiment momentum for a stock.

    Momentum = (current score) - (5-day avg score)
    Positive momentum = sentiment improving (often leads price up)
    Negative momentum = sentiment deteriorating (often leads price down)

    Args:
        symbol      : NSE symbol or '__MARKET__'
        target_date : Date to compute momentum for
        conn        : DB connection

    Returns:
        float in [-1.0, +1.0], or 0.0 if insufficient history
    """
    lookback_start = target_date - timedelta(days=MOMENTUM_WINDOW + 2)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT sentiment_score
            FROM features_sentiment
            WHERE symbol = %s
              AND date BETWEEN %s AND %s
            ORDER BY date ASC;
        """, (symbol, lookback_start, target_date))
        rows = cur.fetchall()

    if len(rows) < 2:
        return 0.0

    scores  = [float(r[0]) for r in rows if r[0] is not None]
    if len(scores) < 2:
        return 0.0

    current = scores[-1]
    avg     = np.mean(scores[:-1])
    momentum= float(np.clip(current - avg, -1.0, 1.0))

    return momentum


def compute_market_fear_greed(market_score: float) -> float:
    """
    Converts market-level sentiment score [-1, +1] to
    Fear & Greed index [0, 100].

    Mapping:
        score = -1.0 → Fear/Greed = 0   (extreme fear)
        score =  0.0 → Fear/Greed = 50  (neutral)
        score = +1.0 → Fear/Greed = 100 (extreme greed)

    Args:
        market_score : Market sentiment score [-1, +1]

    Returns:
        float in [0, 100]
    """
    return float(np.clip((market_score + 1.0) * 50.0, 0.0, 100.0))


# ══════════════════════════════════════════════════════════════════════════
#  MAIN EXTRACTOR CLASS
# ══════════════════════════════════════════════════════════════════════════

class SentimentExtractor:
    """
    Main interface for Pillar 4 — Sentiment Analysis.

    Usage:
        extractor = SentimentExtractor()

        # Process one date for all stocks
        extractor.process_date(date(2024, 6, 15), symbols=['RELIANCE', 'TCS'])

        # Get latest sentiment for signal engine
        score = extractor.get_latest_score('RELIANCE')
        market = extractor.get_market_sentiment(date.today())
    """

    def __init__(self):
        self.conn = _get_conn()
        _ensure_features_table(self.conn)
        # Pre-load FinBERT on init so first inference isn't slow
        logger.info("Initializing FinBERT model...")
        get_finbert_pipeline()

    def process_date(
        self,
        target_date : date,
        symbols     : list[str],
        save_to_db  : bool = True,
    ) -> dict[str, dict]:
        """
        Runs the complete sentiment pipeline for all symbols on one date.

        Pipeline:
            1. Fetch news from Elasticsearch for all symbols
            2. Preprocess and batch through FinBERT
            3. Aggregate per symbol + market level
            4. Compute sentiment momentum from DB history
            5. Save to features_sentiment

        Args:
            target_date : Date to process
            symbols     : List of NSE symbols
            save_to_db  : Whether to save results to DB

        Returns:
            dict mapping symbol → sentiment result dict
            Includes '__MARKET__' key for market-level result
        """
        # ── Step 1: Load news ─────────────────────────────────────────────
        news_by_symbol = load_news_from_elasticsearch(symbols, target_date)

        results: dict[str, dict] = {}
        all_texts     : list[str]  = []
        all_metadata  : list[tuple]= []  # (symbol, article_idx)

        # ── Step 2: Preprocess all texts ──────────────────────────────────
        for symbol in ["__MARKET__"] + symbols:
            articles = news_by_symbol.get(symbol, [])
            # Limit articles per symbol for GPU efficiency
            articles = articles[:NEWS_PER_STOCK_LIMIT]

            for art in articles:
                text = preprocess_text(
                    art.get("title", ""),
                    art.get("content", "")
                )
                if text:
                    all_texts.append(text)
                    all_metadata.append((symbol, len(all_texts) - 1))

        # ── Step 3: Batch FinBERT inference ──────────────────────────────
        if all_texts:
            logger.info(
                f"Running FinBERT on {len(all_texts)} articles "
                f"for {target_date} (batch_size={_get_batch_size()})..."
            )
            finbert_outputs = run_finbert_batch(all_texts)
        else:
            finbert_outputs = []
            logger.info(
                f"No news articles found for {target_date} — "
                f"using estimated neutral sentiment for all symbols"
            )

        # ── Step 4: Regroup outputs by symbol ────────────────────────────
        articles_by_symbol : dict[str, list[dict]] = {}
        outputs_by_symbol  : dict[str, list[dict]] = {}

        for (symbol, text_idx), fb_out in zip(all_metadata, finbert_outputs):
            articles_by_symbol.setdefault(symbol, [])
            outputs_by_symbol.setdefault(symbol, [])

            # Find the matching article
            sym_articles = news_by_symbol.get(symbol, [])[:NEWS_PER_STOCK_LIMIT]
            art_local_idx= len(articles_by_symbol[symbol])
            if art_local_idx < len(sym_articles):
                articles_by_symbol[symbol].append(sym_articles[art_local_idx])
                outputs_by_symbol[symbol].append(fb_out)

        # ── Step 5: Aggregate per symbol ──────────────────────────────────
        all_to_save: list[dict] = []

        for symbol in ["__MARKET__"] + symbols:
            arts  = articles_by_symbol.get(symbol, [])
            outs  = outputs_by_symbol.get(symbol, [])

            result = aggregate_article_scores(arts, outs, target_date)

            # Compute sentiment momentum from DB
            result["sentiment_momentum"] = compute_sentiment_momentum(
                symbol, target_date, self.conn
            )
            result["sentiment_strength"] = abs(result["sentiment_score"])

            # Market-level only: compute Fear & Greed index
            if symbol == "__MARKET__":
                result["market_fear_greed"] = compute_market_fear_greed(
                    result["sentiment_score"]
                )
            else:
                result["market_fear_greed"] = None

            # Last article age
            if arts:
                try:
                    latest = max(
                        pd.to_datetime(a.get("published_at", "")).date()
                        for a in arts
                        if a.get("published_at")
                    )
                    age_hours = int((
                        datetime.combine(target_date, datetime.max.time()) -
                        datetime.combine(latest, datetime.min.time())
                    ).total_seconds() / 3600)
                    result["last_article_age_hours"] = min(age_hours, 999)
                except Exception:
                    result["last_article_age_hours"] = 24
            else:
                result["last_article_age_hours"] = 999

            result["date"]   = target_date
            result["symbol"] = symbol
            results[symbol]  = result
            all_to_save.append(result)

        # ── Step 6: Save to DB ────────────────────────────────────────────
        if save_to_db and all_to_save:
            self._save_batch(all_to_save)

        return results

    def _save_batch(self, records: list[dict]):
        """Batch upserts sentiment results into features_sentiment."""

        insert_sql = """
            INSERT INTO features_sentiment (
                date, symbol,
                articles_processed, articles_positive,
                articles_negative, articles_neutral,
                raw_positive, raw_negative, raw_neutral,
                sentiment_score, sentiment_momentum, sentiment_strength,
                event_flag, event_type, event_weight_applied,
                market_fear_greed, is_estimated, last_article_age_hours
            ) VALUES (
                %(date)s, %(symbol)s,
                %(articles_processed)s, %(articles_positive)s,
                %(articles_negative)s, %(articles_neutral)s,
                %(raw_positive)s, %(raw_negative)s, %(raw_neutral)s,
                %(sentiment_score)s, %(sentiment_momentum)s,
                %(sentiment_strength)s,
                %(event_flag)s, %(event_type)s, %(event_weight_applied)s,
                %(market_fear_greed)s, %(is_estimated)s,
                %(last_article_age_hours)s
            )
            ON CONFLICT (date, symbol) DO UPDATE SET
                sentiment_score      = EXCLUDED.sentiment_score,
                sentiment_momentum   = EXCLUDED.sentiment_momentum,
                articles_processed   = EXCLUDED.articles_processed,
                event_flag           = EXCLUDED.event_flag,
                event_type           = EXCLUDED.event_type,
                market_fear_greed    = EXCLUDED.market_fear_greed,
                is_estimated         = EXCLUDED.is_estimated;
        """

        db_records = []
        for r in records:
            def _v(k, default=None):
                v = r.get(k, default)
                try:
                    if v is not None and not isinstance(v, (bool, str)) and pd.isna(v):
                        return default
                except (TypeError, ValueError):
                    pass
                if isinstance(v, (np.integer,)):   return int(v)
                if isinstance(v, (np.floating,)):  return float(v) if not np.isnan(v) else default
                if isinstance(v, (np.bool_,)):     return bool(v)
                return v

            db_records.append({
                "date"                  : r["date"],
                "symbol"                : r["symbol"],
                "articles_processed"    : _v("articles_processed", 0),
                "articles_positive"     : _v("articles_positive", 0),
                "articles_negative"     : _v("articles_negative", 0),
                "articles_neutral"      : _v("articles_neutral", 0),
                "raw_positive"          : _v("raw_positive", 0.333),
                "raw_negative"          : _v("raw_negative", 0.333),
                "raw_neutral"           : _v("raw_neutral", 0.333),
                "sentiment_score"       : _v("sentiment_score", 0.0),
                "sentiment_momentum"    : _v("sentiment_momentum", 0.0),
                "sentiment_strength"    : _v("sentiment_strength", 0.0),
                "event_flag"            : _v("event_flag", False),
                "event_type"            : _v("event_type", ""),
                "event_weight_applied"  : _v("event_weight_applied", 1.0),
                "market_fear_greed"     : _v("market_fear_greed"),
                "is_estimated"          : _v("is_estimated", True),
                "last_article_age_hours": _v("last_article_age_hours", 999),
            })

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur, insert_sql, db_records, page_size=200
                )
            self.conn.commit()
            logger.success(f"Sentiment: {len(db_records)} rows saved.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Sentiment DB save failed: {e}")
            raise

    def get_latest_score(self, symbol: str) -> Optional[dict]:
        """Returns latest sentiment score for a symbol (used by signal engine)."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT sentiment_score, sentiment_momentum,
                       event_flag, event_type, is_estimated,
                       last_article_age_hours
                FROM features_sentiment
                WHERE symbol = %s
                ORDER BY date DESC LIMIT 1;
            """, (symbol,))
            row = cur.fetchone()

        if not row:
            return None
        return {
            "sentiment_score"       : float(row[0]),
            "sentiment_momentum"    : float(row[1]),
            "event_flag"            : bool(row[2]),
            "event_type"            : str(row[3] or ""),
            "is_estimated"          : bool(row[4]),
            "last_article_age_hours": int(row[5]),
        }

    def get_market_sentiment(self, target_date: date) -> dict:
        """Returns market-level sentiment for a date (used by signal engine)."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT sentiment_score, market_fear_greed,
                       sentiment_momentum, is_estimated
                FROM features_sentiment
                WHERE symbol = '__MARKET__' AND date = %s;
            """, (target_date,))
            row = cur.fetchone()

        if not row:
            return {
                "sentiment_score" : 0.0,
                "market_fear_greed": 50.0,
                "sentiment_momentum": 0.0,
                "is_estimated"    : True,
            }
        return {
            "sentiment_score"   : float(row[0]),
            "market_fear_greed" : float(row[1]) if row[1] else 50.0,
            "sentiment_momentum": float(row[2]),
            "is_estimated"      : bool(row[3]),
        }

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


# ══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys, yaml

    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — Sentiment Extractor")
    parser.add_argument("--mode",   choices=["date", "score", "market"], default="market")
    parser.add_argument("--date",   type=str, default=str(date.today()))
    parser.add_argument("--symbol", type=str)
    args = parser.parse_args()

    target = date.fromisoformat(args.date)

    with SentimentExtractor() as extractor:
        if args.mode == "date":
            # Load universe and process one full date
            try:
                with open("config/universe.yaml") as f:
                    universe = yaml.safe_load(f).get("nifty500", [])
            except FileNotFoundError:
                universe = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]
            results = extractor.process_date(target, universe)
            mkt = results.get("__MARKET__", {})
            print(f"Market sentiment: {mkt.get('sentiment_score', 0):.3f} | "
                  f"Fear/Greed: {mkt.get('market_fear_greed', 50):.0f}")

        elif args.mode == "score":
            if not args.symbol:
                print("--symbol required"); sys.exit(1)
            score = extractor.get_latest_score(args.symbol)
            print(f"{args.symbol}: {score}")

        elif args.mode == "market":
            mkt = extractor.get_market_sentiment(target)
            print(f"Market sentiment for {target}: {mkt}")


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run: python -m pytest features/sentiment.py -v
#  Note: FinBERT model NOT loaded during tests (mocked for speed)
# ══════════════════════════════════════════════════════════════════════════

class TestSentimentFeatures:

    # ── Text preprocessing tests ──────────────────────────────────────────

    def test_preprocess_strips_html(self):
        text = preprocess_text("<b>Reliance profit</b> rises 20%", "<p>Strong results</p>")
        assert "<" not in text and ">" not in text

    def test_preprocess_strips_urls(self):
        text = preprocess_text("Read more at https://example.com/article", "")
        assert "http" not in text

    def test_preprocess_short_text_returns_empty(self):
        text = preprocess_text("Hi", "")
        assert text == "", f"Short text should return empty, got: '{text}'"

    def test_preprocess_combines_title_and_content(self):
        text = preprocess_text("Reliance Q4 results", "Revenue up 15%")
        assert "Reliance" in text
        assert "Revenue" in text

    def test_preprocess_content_truncated(self):
        long_content = "word " * 200   # 1000 chars
        text = preprocess_text("Title", long_content)
        assert len(text) < 500, "Content should be truncated"

    # ── Event detection tests ─────────────────────────────────────────────

    def test_event_detection_earnings(self):
        is_event, etype = detect_event("Reliance Q3 results beat estimates")
        assert is_event and etype == "earnings"

    def test_event_detection_regulatory(self):
        is_event, etype = detect_event("SEBI issues show cause notice to promoter")
        assert is_event and etype == "regulatory"

    def test_event_detection_management(self):
        is_event, etype = detect_event("CEO resigns from Wipro amid controversy")
        assert is_event and etype == "management"

    def test_event_detection_corporate_action(self):
        is_event, etype = detect_event("TCS announces buyback at 4200 per share")
        assert is_event and etype == "corporate_action"

    def test_event_detection_analyst(self):
        is_event, etype = detect_event("Goldman upgrade to buy with target price raised to 3000")
        assert is_event and etype == "analyst"

    def test_no_event_detected_for_generic_news(self):
        is_event, _ = detect_event("Market opens flat amid global cues")
        assert not is_event

    # ── Sentiment score conversion tests ─────────────────────────────────

    def test_score_positive(self):
        score = score_from_probabilities(positive=0.8, negative=0.1)
        assert score > 0 and score <= 1.0

    def test_score_negative(self):
        score = score_from_probabilities(positive=0.1, negative=0.8)
        assert score < 0 and score >= -1.0

    def test_score_neutral(self):
        score = score_from_probabilities(positive=0.333, negative=0.333)
        assert abs(score) < 0.01, f"Neutral probs should give ~0 score, got {score}"

    def test_score_clipped(self):
        score = score_from_probabilities(positive=1.0, negative=0.0)
        assert score == 1.0
        score = score_from_probabilities(positive=0.0, negative=1.0)
        assert score == -1.0

    # ── Aggregation tests ─────────────────────────────────────────────────

    def test_aggregate_no_articles_returns_neutral(self):
        result = aggregate_article_scores([], [], date(2024, 1, 1))
        assert result["sentiment_score"] == 0.0
        assert result["is_estimated"]    == True

    def test_aggregate_positive_articles(self):
        articles = [
            {"title": "Profit rises", "content": "", "published_at": "2024-01-01", "source": "ET"},
            {"title": "Revenue growth", "content": "", "published_at": "2024-01-01", "source": "ET"},
        ]
        finbert_out = [
            {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            {"positive": 0.7, "negative": 0.1, "neutral": 0.2},
        ]
        result = aggregate_article_scores(articles, finbert_out, date(2024, 1, 1))
        assert result["sentiment_score"] > 0

    def test_aggregate_event_flag_detected(self):
        articles = [
            {"title": "Q3 quarterly results beat estimates",
             "content": "Net profit up 25%",
             "published_at": "2024-01-01", "source": "ET"},
        ]
        finbert_out = [{"positive": 0.6, "negative": 0.2, "neutral": 0.2}]
        result = aggregate_article_scores(articles, finbert_out, date(2024, 1, 1))
        assert result["event_flag"]  == True
        assert result["event_type"]  == "earnings"
        assert result["event_weight_applied"] == EVENT_WEIGHT_MULT

    def test_aggregate_recency_decay(self):
        """Older articles should have less weight than recent ones."""
        old_article = [
            {"title": "Bad news", "content": "",
             "published_at": "2024-01-01", "source": "ET"}
        ]
        new_article = [
            {"title": "Good news", "content": "",
             "published_at": "2024-01-05", "source": "ET"}
        ]
        old_fb = [{"positive": 0.1, "negative": 0.8, "neutral": 0.1}]
        new_fb = [{"positive": 0.8, "negative": 0.1, "neutral": 0.1}]

        old_result = aggregate_article_scores(old_article, old_fb, date(2024, 1, 5))
        new_result = aggregate_article_scores(new_article, new_fb, date(2024, 1, 5))

        # Recent positive article should score higher than old negative one
        assert new_result["sentiment_score"] > old_result["sentiment_score"]

    def test_aggregate_score_range(self):
        articles = [
            {"title": f"News {i}", "content": "",
             "published_at": "2024-01-01", "source": "ET"}
            for i in range(10)
        ]
        finbert_out = [
            {"positive": np.random.random(),
             "negative": np.random.random(),
             "neutral" : np.random.random()}
            for _ in range(10)
        ]
        result = aggregate_article_scores(articles, finbert_out, date(2024, 1, 1))
        assert -1.0 <= result["sentiment_score"] <= 1.0

    # ── Fear & Greed tests ────────────────────────────────────────────────

    def test_fear_greed_range(self):
        for score in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            fg = compute_market_fear_greed(score)
            assert 0 <= fg <= 100, f"Fear/Greed out of range for score={score}"

    def test_fear_greed_neutral(self):
        assert compute_market_fear_greed(0.0) == 50.0

    def test_fear_greed_extreme_fear(self):
        assert compute_market_fear_greed(-1.0) == 0.0

    def test_fear_greed_extreme_greed(self):
        assert compute_market_fear_greed(1.0) == 100.0

    # ── FinBERT batch tests (no model required) ───────────────────────────

    def test_finbert_batch_empty_input(self):
        result = run_finbert_batch([])
        assert result == []

    def test_finbert_batch_all_empty_strings(self):
        result = run_finbert_batch(["", "  ", ""])
        assert len(result) == 3
        for r in result:
            assert "positive" in r and "negative" in r and "neutral" in r

    def test_neutral_result_structure(self):
        result = _neutral_sentiment_result(5)
        assert result["sentiment_score"]    == 0.0
        assert result["articles_processed"] == 5
        assert result["is_estimated"]       == True


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))