"""
╔══════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — NSE Options Chain Scraper              ║
║         Project : MultiStockLSTMBot                             ║
║         File    : data/ingestion/options_chain.py               ║
║         Phase   : 0 — Data Infrastructure                       ║
║         Purpose : Fetches NSE options OI every 3 minutes        ║
║                   Feeds Pillar 2 (MSI - PCR) and                ║
║                   Pillar 3 (FII/DII net gamma)                  ║
╚══════════════════════════════════════════════════════════════════╝

What this file does:
--------------------
1. Fetches the full options chain for Nifty 50 index every 3 minutes
2. Fetches stock-level options chain for all optionable Nifty 500 stocks
3. Calculates Put-Call Ratio (PCR) of OI for each symbol
4. Stores raw options data in TimescaleDB
5. Caches current PCR values in Redis for real-time feature access

Key Metrics Extracted:
----------------------
    PCR (Put-Call Ratio)  : Total Put OI / Total Call OI
                            PCR < 0.7  = overbought (too bullish)
                            PCR > 1.3  = oversold (too bearish)
    ATM IV                : Implied Volatility of At-The-Money strike
    Max Pain              : Strike where maximum options expire worthless
    OI Change             : Delta in OI since last snapshot

Usage:
------
    # Run continuously during market hours (every 3 mins)
    python -m data.ingestion.options_chain --mode live

    # Single fetch (for testing)
    python -m data.ingestion.options_chain --mode single --symbol NIFTY

Dependencies:
-------------
    pip install requests pandas psycopg2-binary redis loguru python-dotenv
"""

import os
import json
import time
import argparse

import requests
import pandas as pd
import psycopg2
import psycopg2.extras
import redis

from datetime import datetime
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "options_chain_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="7 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}",
)

# ── NSE API endpoints ─────────────────────────────────────────────────────
NSE_OPTIONS_URL = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
NSE_STOCK_OPTIONS_URL = "https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Fetch interval during market hours (seconds)
FETCH_INTERVAL = 180  # 3 minutes

# Index symbols to always fetch
INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY"]

# Top optionable stocks from Nifty 500 (F&O eligible)
# Full list loaded from config/universe.yaml filtered by F&O eligibility
FNO_STOCKS_FALLBACK = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "KOTAKBANK", "SBIN", "AXISBANK", "BAJFINANCE", "MARUTI",
    "TATAMOTORS", "TATASTEEL", "WIPRO", "HCLTECH", "SUNPHARMA",
]


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE & REDIS SETUP
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
    """Creates options chain storage tables"""
    with conn.cursor() as cur:

        # Raw options snapshot table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS options_snapshot (
                ts              TIMESTAMPTZ NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                expiry          DATE        NOT NULL,
                strike          NUMERIC(10,2) NOT NULL,
                option_type     CHAR(2)     NOT NULL,  -- CE or PE
                oi              BIGINT,
                oi_change       BIGINT,
                volume          BIGINT,
                iv              NUMERIC(8,4),
                ltp             NUMERIC(10,2),
                bid             NUMERIC(10,2),
                ask             NUMERIC(10,2),
                PRIMARY KEY (ts, symbol, expiry, strike, option_type)
            );
        """)
        cur.execute("""
            SELECT create_hypertable(
                'options_snapshot', 'ts',
                if_not_exists => TRUE, migrate_data => TRUE
            );
        """)

        # PCR summary table (one row per symbol per snapshot)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS options_pcr (
                ts          TIMESTAMPTZ NOT NULL,
                symbol      VARCHAR(20) NOT NULL,
                pcr_oi      NUMERIC(8,4),   -- put OI / call OI
                pcr_vol     NUMERIC(8,4),   -- put vol / call vol
                atm_iv_ce   NUMERIC(8,4),   -- ATM call IV
                atm_iv_pe   NUMERIC(8,4),   -- ATM put IV
                max_pain    NUMERIC(10,2),  -- max pain strike
                total_call_oi BIGINT,
                total_put_oi  BIGINT,
                PRIMARY KEY (ts, symbol)
            );
        """)
        cur.execute("""
            SELECT create_hypertable(
                'options_pcr', 'ts',
                if_not_exists => TRUE, migrate_data => TRUE
            );
        """)

    conn.commit()
    logger.info("Options chain tables verified/created")


# ══════════════════════════════════════════════════════════════════════════
#  NSE SESSION
# ══════════════════════════════════════════════════════════════════════════

def get_nse_session() -> requests.Session:
    """NSE requires homepage visit for session cookies"""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
    except Exception as e:
        logger.warning(f"NSE session init warning: {e}")
    return session


# ══════════════════════════════════════════════════════════════════════════
#  OPTIONS CHAIN FETCHER
# ══════════════════════════════════════════════════════════════════════════

def fetch_options_chain(symbol: str, session: requests.Session,
                        is_index: bool = False) -> dict | None:
    """
    Fetches raw options chain JSON from NSE for a given symbol.

    Args:
        symbol   : NSE symbol (e.g. 'NIFTY', 'RELIANCE')
        session  : NSE session with cookies
        is_index : True for index options, False for stock options

    Returns:
        Raw NSE options chain dict, or None on failure
    """
    if is_index:
        url = NSE_OPTIONS_URL.format(symbol=symbol)
    else:
        url = NSE_STOCK_OPTIONS_URL.format(symbol=symbol)

    for attempt in range(1, 4):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                # Session expired — refresh cookies
                logger.warning(f"NSE session expired for {symbol} — refreshing")
                session.get("https://www.nseindia.com", timeout=15)
                time.sleep(2)
            else:
                logger.warning(f"HTTP {resp.status_code} for {symbol} options")
        except Exception as e:
            logger.warning(f"Attempt {attempt}/3 failed for {symbol}: {e}")
            time.sleep(2 * attempt)

    return None


# ══════════════════════════════════════════════════════════════════════════
#  PARSER & METRICS CALCULATOR
# ══════════════════════════════════════════════════════════════════════════

def parse_options_chain(raw_data: dict, symbol: str,
                        snapshot_ts: datetime) -> tuple[pd.DataFrame, dict]:
    """
    Parses NSE options chain JSON into structured DataFrames.

    Returns:
        Tuple of:
            - options_df : Row per strike/expiry/type
            - pcr_metrics: Dict with PCR, ATM IV, max pain
    """
    records = []

    try:
        data       = raw_data.get("records", {})
        chain_data = data.get("data", [])
        spot_price = raw_data.get("records", {}).get("underlyingValue", 0)

        # Get nearest expiry only (current month)
        expiry_dates = sorted(set(
            item["expiryDate"] for item in chain_data
            if "expiryDate" in item
        ))
        nearest_expiry = expiry_dates[0] if expiry_dates else None

        total_call_oi = 0
        total_put_oi  = 0
        total_call_vol = 0
        total_put_vol  = 0
        max_pain_data  = {}   # strike → total OI loss at that strike

        for item in chain_data:
            if item.get("expiryDate") != nearest_expiry:
                continue

            strike = item.get("strikePrice", 0)

            # ── Call side ─────────────────────────────────────────────────
            ce = item.get("CE", {})
            if ce:
                ce_oi     = ce.get("openInterest", 0) or 0
                ce_change = ce.get("changeinOpenInterest", 0) or 0
                ce_vol    = ce.get("totalTradedVolume", 0) or 0
                ce_iv     = ce.get("impliedVolatility", 0) or 0
                ce_ltp    = ce.get("lastPrice", 0) or 0

                total_call_oi  += ce_oi
                total_call_vol += ce_vol

                records.append({
                    "ts": snapshot_ts, "symbol": symbol,
                    "expiry": nearest_expiry, "strike": strike,
                    "option_type": "CE", "oi": ce_oi,
                    "oi_change": ce_change, "volume": ce_vol,
                    "iv": ce_iv, "ltp": ce_ltp,
                    "bid": ce.get("bidprice", 0),
                    "ask": ce.get("askPrice", 0),
                })

            # ── Put side ──────────────────────────────────────────────────
            pe = item.get("PE", {})
            if pe:
                pe_oi     = pe.get("openInterest", 0) or 0
                pe_change = pe.get("changeinOpenInterest", 0) or 0
                pe_vol    = pe.get("totalTradedVolume", 0) or 0
                pe_iv     = pe.get("impliedVolatility", 0) or 0
                pe_ltp    = pe.get("lastPrice", 0) or 0

                total_put_oi  += pe_oi
                total_put_vol += pe_vol

                records.append({
                    "ts": snapshot_ts, "symbol": symbol,
                    "expiry": nearest_expiry, "strike": strike,
                    "option_type": "PE", "oi": pe_oi,
                    "oi_change": pe_change, "volume": pe_vol,
                    "iv": pe_iv, "ltp": pe_ltp,
                    "bid": pe.get("bidprice", 0),
                    "ask": pe.get("askPrice", 0),
                })

            # Max pain contribution
            max_pain_data[strike] = max_pain_data.get(strike, 0) + (ce_oi + (pe_oi if pe else 0))

        # ── Calculate PCR ─────────────────────────────────────────────────
        pcr_oi  = round(total_put_oi  / total_call_oi,  4) if total_call_oi  > 0 else 1.0
        pcr_vol = round(total_put_vol / total_call_vol, 4) if total_call_vol > 0 else 1.0

        # ── ATM IV (nearest strike to spot) ──────────────────────────────
        atm_strike  = min(max_pain_data.keys(), key=lambda s: abs(s - spot_price), default=0)
        atm_records = [r for r in records if r["strike"] == atm_strike and r["expiry"] == nearest_expiry]
        atm_iv_ce   = next((r["iv"] for r in atm_records if r["option_type"] == "CE"), 0)
        atm_iv_pe   = next((r["iv"] for r in atm_records if r["option_type"] == "PE"), 0)

        # ── Max Pain (strike with minimum total OI pain) ──────────────────
        max_pain_strike = min(max_pain_data, key=max_pain_data.get) if max_pain_data else 0

        pcr_metrics = {
            "ts"            : snapshot_ts,
            "symbol"        : symbol,
            "pcr_oi"        : pcr_oi,
            "pcr_vol"       : pcr_vol,
            "atm_iv_ce"     : atm_iv_ce,
            "atm_iv_pe"     : atm_iv_pe,
            "max_pain"      : max_pain_strike,
            "total_call_oi" : total_call_oi,
            "total_put_oi"  : total_put_oi,
        }

        df = pd.DataFrame(records) if records else pd.DataFrame()
        return df, pcr_metrics

    except Exception as e:
        logger.error(f"Failed to parse options chain for {symbol}: {e}")
        return pd.DataFrame(), {}


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE & REDIS WRITERS
# ══════════════════════════════════════════════════════════════════════════

def write_to_db(df: pd.DataFrame, pcr_metrics: dict, conn):
    """Writes options snapshot and PCR metrics to TimescaleDB"""
    if df.empty and not pcr_metrics:
        return

    try:
        with conn.cursor() as cur:
            # Insert options snapshot rows
            if not df.empty:
                records = df.to_dict("records")
                sql = """
                    INSERT INTO options_snapshot
                        (ts, symbol, expiry, strike, option_type,
                         oi, oi_change, volume, iv, ltp, bid, ask)
                    VALUES
                        (%(ts)s, %(symbol)s, %(expiry)s, %(strike)s, %(option_type)s,
                         %(oi)s, %(oi_change)s, %(volume)s, %(iv)s, %(ltp)s, %(bid)s, %(ask)s)
                    ON CONFLICT (ts, symbol, expiry, strike, option_type)
                    DO UPDATE SET
                        oi = EXCLUDED.oi, oi_change = EXCLUDED.oi_change,
                        volume = EXCLUDED.volume, iv = EXCLUDED.iv,
                        ltp = EXCLUDED.ltp;
                """
                psycopg2.extras.execute_batch(cur, sql, records, page_size=500)

            # Insert PCR summary
            if pcr_metrics:
                cur.execute("""
                    INSERT INTO options_pcr
                        (ts, symbol, pcr_oi, pcr_vol, atm_iv_ce, atm_iv_pe,
                         max_pain, total_call_oi, total_put_oi)
                    VALUES
                        (%(ts)s, %(symbol)s, %(pcr_oi)s, %(pcr_vol)s,
                         %(atm_iv_ce)s, %(atm_iv_pe)s, %(max_pain)s,
                         %(total_call_oi)s, %(total_put_oi)s)
                    ON CONFLICT (ts, symbol) DO UPDATE SET
                        pcr_oi = EXCLUDED.pcr_oi,
                        pcr_vol = EXCLUDED.pcr_vol,
                        atm_iv_ce = EXCLUDED.atm_iv_ce,
                        atm_iv_pe = EXCLUDED.atm_iv_pe;
                """, pcr_metrics)

        conn.commit()

    except Exception as e:
        logger.error(f"DB write failed: {e}")
        conn.rollback()


def cache_pcr_in_redis(pcr_metrics: dict, redis_client: redis.Redis):
    """Caches current PCR value in Redis for real-time feature access"""
    if not pcr_metrics:
        return
    symbol = pcr_metrics.get("symbol")
    redis_client.setex(
        f"pcr:{symbol}",
        300,  # 5 minute TTL
        json.dumps({
            "pcr_oi"   : pcr_metrics.get("pcr_oi"),
            "atm_iv_ce": pcr_metrics.get("atm_iv_ce"),
            "atm_iv_pe": pcr_metrics.get("atm_iv_pe"),
            "max_pain" : pcr_metrics.get("max_pain"),
        })
    )


# ══════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

def run_once(symbols: list[str], index_symbols: list[str],
             session: requests.Session, conn, redis_client: redis.Redis):
    """Fetches options chain for all symbols once"""
    snapshot_ts = datetime.now()
    logger.info(f"Options chain snapshot at {snapshot_ts.strftime('%H:%M:%S')}")

    # Fetch indices
    for symbol in index_symbols:
        raw = fetch_options_chain(symbol, session, is_index=True)
        if raw:
            df, pcr = parse_options_chain(raw, symbol, snapshot_ts)
            write_to_db(df, pcr, conn)
            cache_pcr_in_redis(pcr, redis_client)
            logger.info(f"{symbol}: PCR={pcr.get('pcr_oi', 'N/A'):.2f} | "
                        f"ATM IV CE={pcr.get('atm_iv_ce', 0):.1f}%")
        time.sleep(0.5)

    # Fetch F&O stocks
    for symbol in symbols:
        raw = fetch_options_chain(symbol, session, is_index=False)
        if raw:
            df, pcr = parse_options_chain(raw, symbol, snapshot_ts)
            write_to_db(df, pcr, conn)
            cache_pcr_in_redis(pcr, redis_client)
        time.sleep(0.3)  # polite delay between requests

    logger.info(f"Snapshot complete: {len(index_symbols) + len(symbols)} symbols processed")


def run_live():
    """Runs options chain fetching every 3 minutes during market hours"""
    logger.info("Options chain scraper starting...")

    conn         = get_db_connection()
    redis_client = get_redis_client()
    session      = get_nse_session()

    ensure_tables(conn)

    # Load F&O stock list
    try:
        import yaml
        with open("config/universe.yaml") as f:
            data = yaml.safe_load(f)
        fno_stocks = data.get("fno_stocks", FNO_STOCKS_FALLBACK)
    except Exception:
        fno_stocks = FNO_STOCKS_FALLBACK
        logger.warning("Using fallback F&O stock list")

    try:
        while True:
            now = datetime.now()
            hour_min = now.strftime("%H:%M")

            # Only fetch during market hours
            if "09:15" <= hour_min <= "15:30":
                run_once(fno_stocks, INDEX_SYMBOLS, session, conn, redis_client)

                # Refresh NSE session every 30 mins
                if now.minute % 30 == 0:
                    session = get_nse_session()

                time.sleep(FETCH_INTERVAL)
            else:
                logger.info(f"Market closed ({hour_min}) — sleeping 5 mins")
                time.sleep(300)

    except KeyboardInterrupt:
        logger.info("Options chain scraper stopped")
    finally:
        conn.close()


def run_single(symbol: str):
    """Fetches options chain for a single symbol — for testing"""
    session      = get_nse_session()
    conn         = get_db_connection()
    redis_client = get_redis_client()

    ensure_tables(conn)

    is_index = symbol in INDEX_SYMBOLS
    raw      = fetch_options_chain(symbol, session, is_index=is_index)

    if raw:
        df, pcr = parse_options_chain(raw, symbol, datetime.now())
        write_to_db(df, pcr, conn)
        cache_pcr_in_redis(pcr, redis_client)
        print(f"\nPCR Summary for {symbol}:")
        for k, v in pcr.items():
            print(f"  {k}: {v}")
        print(f"\nOptions rows captured: {len(df)}")
    else:
        print(f"Failed to fetch options chain for {symbol}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — Options Chain Scraper")
    parser.add_argument(
        "--mode", choices=["live", "single"], required=True,
        help="live: continuous 3-min fetch | single: one fetch for testing"
    )
    parser.add_argument("--symbol", default="NIFTY", help="Symbol for single mode")
    args = parser.parse_args()

    if args.mode == "live":
        run_live()
    elif args.mode == "single":
        run_single(args.symbol.upper())