"""
╔══════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — FII/DII Flow Scraper                   ║
║         Project : MultiStockLSTMBot                             ║
║         File    : data/ingestion/fii_dii_scraper.py             ║
║         Phase   : 0 — Data Infrastructure                       ║
║         Purpose : Downloads daily FII/DII net flow from NSE     ║
║                   Feeds Pillar 3 (Market Direction Signal)       ║
╚══════════════════════════════════════════════════════════════════╝

What this file does:
--------------------
1. Scrapes provisional FII/DII data at ~4:00 PM (released by NSE)
2. Scrapes final FII/DII data at ~6:00 PM (confirmed figures)
3. Stores both in TimescaleDB with a provisional/final flag
4. Calculates 5-day cumulative FII flow (rolling trend)
5. Computes the Market Direction Signal (MDS) score [-3 to +3]
6. Caches MDS in Redis for use by the signal engine next morning

MDS Scoring Logic:
------------------
    +3 : FII net > +2000 Cr AND DII net > +500 Cr
    +2 : FII net > +1000 Cr OR DII net > +1000 Cr
    +1 : FII net > 0 (mild buying)
     0 : FII net between -500 and +500 Cr (neutral)
    -1 : FII net < -500 Cr (mild selling)
    -2 : FII net < -1500 Cr OR DII unable to absorb
    -3 : FII net < -3000 Cr (heavy panic selling)

Usage:
------
    # Run at 4 PM (provisional) — scheduled via Airflow
    python -m data.ingestion.fii_dii_scraper --type provisional

    # Run at 6 PM (final) — scheduled via Airflow
    python -m data.ingestion.fii_dii_scraper --type final

    # Backfill historical FII/DII data
    python -m data.ingestion.fii_dii_scraper --type backfill --start 2019-01-01

Dependencies:
-------------
    pip install requests beautifulsoup4 pandas psycopg2-binary redis loguru python-dotenv
"""

import os
import re
import json
import time
import argparse

import requests
import pandas as pd
import psycopg2
import psycopg2.extras
import redis

from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "fii_dii_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="30 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}",
)

# ── NSE FII/DII data endpoints ────────────────────────────────────────────
# Primary: NSE JSON API (most reliable)
NSE_FII_DII_API = "https://www.nseindia.com/api/fiidiiTradeReact"

# Backup: SEBI/NSE historical bulk data
NSE_HISTORICAL_URL = (
    "https://www.nseindia.com/api/historical/fiiDiiData"
    "?startDate={start}&endDate={end}&type=fii"
)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/",
}

# MDS scoring thresholds (in Crore INR)
MDS_THRESHOLDS = {
    "strong_bull_fii"  : 2000,
    "strong_bull_dii"  : 500,
    "bull_fii"         : 1000,
    "bull_dii"         : 1000,
    "mild_bull_fii"    : 0,
    "neutral_high"     : 500,
    "neutral_low"      : -500,
    "mild_bear_fii"    : -500,
    "bear_fii"         : -1500,
    "strong_bear_fii"  : -3000,
}


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE & REDIS
# ══════════════════════════════════════════════════════════════════════════

def get_db_connection():
    url = os.getenv("TIMESCALE_URL")
    if not url:
        raise EnvironmentError("TIMESCALE_URL not set in .env")
    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn


def get_redis_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://:godseye_redis_pass@localhost:6379")
    client = redis.from_url(url, decode_responses=True)
    client.ping()
    return client


def ensure_tables(conn):
    """Creates FII/DII storage tables"""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fii_dii_flow (
                date            DATE        NOT NULL,
                data_type       VARCHAR(15) NOT NULL,  -- 'provisional' or 'final'
                fii_buy         NUMERIC(15,2),
                fii_sell        NUMERIC(15,2),
                fii_net         NUMERIC(15,2),         -- buy - sell (positive = buying)
                dii_buy         NUMERIC(15,2),
                dii_sell        NUMERIC(15,2),
                dii_net         NUMERIC(15,2),
                mds_score       INTEGER,               -- Market Direction Signal [-3 to +3]
                fii_5d_cumul    NUMERIC(15,2),         -- 5-day rolling cumulative FII net
                loaded_at       TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (date, data_type)
            );
        """)
    conn.commit()
    logger.info("FII/DII tables verified/created")


# ══════════════════════════════════════════════════════════════════════════
#  NSE SESSION
# ══════════════════════════════════════════════════════════════════════════

def get_nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
    except Exception as e:
        logger.warning(f"NSE session init: {e}")
    return session


# ══════════════════════════════════════════════════════════════════════════
#  FII/DII FETCHER
# ══════════════════════════════════════════════════════════════════════════

def fetch_fii_dii_today(session: requests.Session) -> dict | None:
    """
    Fetches today's FII/DII data from NSE JSON API.
    Returns dict with fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net
    All values in Crore INR.
    """
    for attempt in range(1, 4):
        try:
            resp = session.get(NSE_FII_DII_API, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return _parse_nse_fii_dii_response(data)
            else:
                logger.warning(f"NSE FII/DII API returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"Attempt {attempt}/3 FII/DII fetch failed: {e}")
            time.sleep(3 * attempt)

    return None


def _parse_nse_fii_dii_response(data: dict) -> dict | None:
    """
    Parses NSE FII/DII JSON response.
    NSE returns data for multiple dates — we want today's figures.
    """
    try:
        # NSE returns a list; today's data is first entry
        records = data if isinstance(data, list) else data.get("data", [])

        if not records:
            return None

        today_record = records[0]

        # NSE field names vary — handle multiple possible formats
        fii_buy  = _safe_float(today_record.get("fiiBuy") or today_record.get("FII_BUY_VALUE"))
        fii_sell = _safe_float(today_record.get("fiiSell") or today_record.get("FII_SELL_VALUE"))
        dii_buy  = _safe_float(today_record.get("diiBuy") or today_record.get("DII_BUY_VALUE"))
        dii_sell = _safe_float(today_record.get("diiSell") or today_record.get("DII_SELL_VALUE"))

        if fii_buy is None:
            logger.warning("Could not parse FII/DII values from NSE response")
            return None

        return {
            "fii_buy"  : fii_buy,
            "fii_sell" : fii_sell,
            "fii_net"  : round(fii_buy - fii_sell, 2),
            "dii_buy"  : dii_buy,
            "dii_sell" : dii_sell,
            "dii_net"  : round(dii_buy - dii_sell, 2),
        }

    except Exception as e:
        logger.error(f"FII/DII parse error: {e}")
        return None


def fetch_fii_dii_historical(start_date: date, end_date: date,
                              session: requests.Session) -> list[dict]:
    """
    Fetches historical FII/DII data for a date range.
    Used during Phase 0 backfill.

    Returns list of daily records.
    """
    url = NSE_HISTORICAL_URL.format(
        start=start_date.strftime("%d-%m-%Y"),
        end=end_date.strftime("%d-%m-%Y")
    )

    try:
        resp = session.get(url, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"Historical FII/DII: HTTP {resp.status_code}")
            return []

        data    = resp.json()
        records = data if isinstance(data, list) else data.get("data", [])
        parsed  = []

        for rec in records:
            try:
                trade_date = _parse_date(rec.get("date") or rec.get("DATE"))
                if not trade_date:
                    continue

                fii_buy  = _safe_float(rec.get("fiiBuy")  or rec.get("FII_BUY_VALUE",  0))
                fii_sell = _safe_float(rec.get("fiiSell") or rec.get("FII_SELL_VALUE", 0))
                dii_buy  = _safe_float(rec.get("diiBuy")  or rec.get("DII_BUY_VALUE",  0))
                dii_sell = _safe_float(rec.get("diiSell") or rec.get("DII_SELL_VALUE", 0))

                parsed.append({
                    "date"     : trade_date,
                    "fii_buy"  : fii_buy,
                    "fii_sell" : fii_sell,
                    "fii_net"  : round(fii_buy - fii_sell, 2),
                    "dii_buy"  : dii_buy,
                    "dii_sell" : dii_sell,
                    "dii_net"  : round(dii_buy - dii_sell, 2),
                })
            except Exception:
                continue

        logger.info(f"Historical FII/DII: {len(parsed)} records from {start_date} to {end_date}")
        return parsed

    except Exception as e:
        logger.error(f"Historical FII/DII fetch error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════
#  MDS CALCULATOR
# ══════════════════════════════════════════════════════════════════════════

def calculate_mds(fii_net: float, dii_net: float,
                  fii_5d_cumul: float) -> int:
    """
    Calculates the Market Direction Signal (MDS) score.

    Args:
        fii_net      : Today's FII net flow in Crore INR
        dii_net      : Today's DII net flow in Crore INR
        fii_5d_cumul : 5-day cumulative FII net in Crore INR

    Returns:
        Integer score from -3 (strong bear) to +3 (strong bull)

    Logic:
        Primary driver: FII net (institutions drive direction)
        Secondary: DII net (counter or confirms FII)
        Modifier: 5-day cumulative (sustained trend adds weight)
    """
    t = MDS_THRESHOLDS

    # Base score from today's FII net
    if fii_net >= t["strong_bull_fii"]:
        base_score = 3
    elif fii_net >= t["bull_fii"]:
        base_score = 2
    elif fii_net >= t["mild_bull_fii"]:
        base_score = 1
    elif fii_net >= t["neutral_low"]:
        base_score = 0
    elif fii_net >= t["bear_fii"]:
        base_score = -1
    elif fii_net >= t["strong_bear_fii"]:
        base_score = -2
    else:
        base_score = -3

    # DII modifier
    if dii_net > t["bull_dii"] and base_score < 2:
        base_score = min(base_score + 1, 3)    # DII buying lifts sentiment
    elif dii_net < -t["bull_dii"] and base_score > -2:
        base_score = max(base_score - 1, -3)   # DII selling worsens sentiment

    # 5-day cumulative modifier (sustained trend)
    if fii_5d_cumul > 5000 and base_score > 0:
        base_score = min(base_score + 1, 3)    # Sustained buying
    elif fii_5d_cumul < -5000 and base_score < 0:
        base_score = max(base_score - 1, -3)   # Sustained selling

    return int(base_score)


def get_5day_cumulative_fii(conn, as_of_date: date) -> float:
    """Fetches 5-day cumulative FII net from DB"""
    sql = """
        SELECT COALESCE(SUM(fii_net), 0)
        FROM fii_dii_flow
        WHERE date >= %s AND date < %s
          AND data_type = 'final'
    """
    start = as_of_date - timedelta(days=7)  # 7 calendar days covers 5 trading days
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (start, as_of_date))
            result = cur.fetchone()
            return float(result[0]) if result else 0.0
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE WRITER
# ══════════════════════════════════════════════════════════════════════════

def write_to_db(flow_data: dict, trade_date: date,
                data_type: str, conn) -> bool:
    """
    Writes FII/DII flow record to TimescaleDB.

    Args:
        flow_data  : Dict with fii_buy, fii_sell, fii_net, dii_*
        trade_date : Trading date
        data_type  : 'provisional' or 'final'
        conn       : DB connection
    """
    fii_5d = get_5day_cumulative_fii(conn, trade_date)
    mds    = calculate_mds(
        flow_data["fii_net"], flow_data["dii_net"], fii_5d
    )

    sql = """
        INSERT INTO fii_dii_flow
            (date, data_type, fii_buy, fii_sell, fii_net,
             dii_buy, dii_sell, dii_net, mds_score, fii_5d_cumul)
        VALUES
            (%(date)s, %(data_type)s, %(fii_buy)s, %(fii_sell)s, %(fii_net)s,
             %(dii_buy)s, %(dii_sell)s, %(dii_net)s, %(mds_score)s, %(fii_5d_cumul)s)
        ON CONFLICT (date, data_type) DO UPDATE SET
            fii_buy      = EXCLUDED.fii_buy,
            fii_sell     = EXCLUDED.fii_sell,
            fii_net      = EXCLUDED.fii_net,
            dii_buy      = EXCLUDED.dii_buy,
            dii_sell     = EXCLUDED.dii_sell,
            dii_net      = EXCLUDED.dii_net,
            mds_score    = EXCLUDED.mds_score,
            fii_5d_cumul = EXCLUDED.fii_5d_cumul,
            loaded_at    = NOW();
    """

    record = {
        "date"        : trade_date,
        "data_type"   : data_type,
        "fii_buy"     : flow_data["fii_buy"],
        "fii_sell"    : flow_data["fii_sell"],
        "fii_net"     : flow_data["fii_net"],
        "dii_buy"     : flow_data["dii_buy"],
        "dii_sell"    : flow_data["dii_sell"],
        "dii_net"     : flow_data["dii_net"],
        "mds_score"   : mds,
        "fii_5d_cumul": fii_5d,
    }

    try:
        with conn.cursor() as cur:
            cur.execute(sql, record)
        conn.commit()
        logger.success(
            f"FII/DII {data_type} saved for {trade_date} | "
            f"FII Net: ₹{flow_data['fii_net']:,.0f} Cr | "
            f"DII Net: ₹{flow_data['dii_net']:,.0f} Cr | "
            f"MDS: {mds}"
        )
        return True
    except Exception as e:
        logger.error(f"FII/DII DB write failed: {e}")
        conn.rollback()
        return False


def cache_mds_in_redis(mds_score: int, fii_net: float,
                       dii_net: float, redis_client: redis.Redis):
    """
    Caches the MDS score in Redis.
    The signal engine reads this every morning before generating signals.
    TTL: 26 hours (covers overnight + next trading day)
    """
    mds_data = {
        "mds_score" : mds_score,
        "fii_net"   : fii_net,
        "dii_net"   : dii_net,
        "updated_at": datetime.now().isoformat(),
        "interpretation": {
            3:  "Strong Bullish — FII + DII both buying",
            2:  "Bullish — Institutional support",
            1:  "Mild Bullish — Slight FII positive",
            0:  "Neutral — Balanced flows",
            -1: "Mild Bearish — Slight FII selling",
            -2: "Bearish — FII selling pressure",
            -3: "Strong Bearish — Heavy institutional selling",
        }.get(mds_score, "Unknown")
    }

    redis_client.setex("mds:current", 93600, json.dumps(mds_data))  # 26 hours
    logger.info(f"MDS cached in Redis: {mds_score} ({mds_data['interpretation']})")


# ══════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def _safe_float(value) -> float:
    """Safely converts value to float, handling commas and None"""
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _parse_date(date_str: str) -> date | None:
    """Parses various date string formats from NSE"""
    if not date_str:
        return None
    formats = ["%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt).date()
        except ValueError:
            continue
    return None


# ══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════

def run_provisional():
    """Fetches provisional FII/DII data (~4 PM). Run via Airflow."""
    logger.info("Fetching provisional FII/DII data...")
    conn         = get_db_connection()
    redis_client = get_redis_client()
    session      = get_nse_session()

    ensure_tables(conn)

    flow = fetch_fii_dii_today(session)
    if flow:
        today = date.today()
        write_to_db(flow, today, "provisional", conn)
        fii_5d = get_5day_cumulative_fii(conn, today)
        mds    = calculate_mds(flow["fii_net"], flow["dii_net"], fii_5d)
        cache_mds_in_redis(mds, flow["fii_net"], flow["dii_net"], redis_client)
    else:
        logger.error("Failed to fetch provisional FII/DII data")

    conn.close()


def run_final():
    """Fetches final FII/DII data (~6 PM). Run via Airflow."""
    logger.info("Fetching final FII/DII data...")
    conn         = get_db_connection()
    redis_client = get_redis_client()
    session      = get_nse_session()

    ensure_tables(conn)

    flow = fetch_fii_dii_today(session)
    if flow:
        today = date.today()
        write_to_db(flow, today, "final", conn)
        fii_5d = get_5day_cumulative_fii(conn, today)
        mds    = calculate_mds(flow["fii_net"], flow["dii_net"], fii_5d)
        cache_mds_in_redis(mds, flow["fii_net"], flow["dii_net"], redis_client)
    else:
        logger.error("Failed to fetch final FII/DII data")

    conn.close()


def run_backfill(start_date: date):
    """Backfills historical FII/DII data. Run once during Phase 0 setup."""
    logger.info(f"Backfilling FII/DII from {start_date} to today...")
    conn    = get_db_connection()
    session = get_nse_session()

    ensure_tables(conn)

    end_date = date.today() - timedelta(days=1)
    # Fetch in monthly chunks to avoid NSE timeout
    current = start_date
    total   = 0

    while current <= end_date:
        chunk_end = min(current + timedelta(days=30), end_date)
        records   = fetch_fii_dii_historical(current, chunk_end, session)

        for rec in records:
            write_to_db(rec, rec["date"], "final", conn)
            total += 1

        current = chunk_end + timedelta(days=1)
        time.sleep(1)  # polite delay

    logger.success(f"FII/DII backfill complete: {total} records loaded")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — FII/DII Scraper")
    parser.add_argument(
        "--type",
        choices=["provisional", "final", "backfill"],
        required=True,
        help=(
            "provisional : Fetch 4 PM provisional data\n"
            "final       : Fetch 6 PM final data\n"
            "backfill    : Load full history from --start"
        )
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date(2019, 1, 1),
        help="Start date for backfill (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    if args.type == "provisional":
        run_provisional()
    elif args.type == "final":
        run_final()
    elif args.type == "backfill":
        run_backfill(args.start)