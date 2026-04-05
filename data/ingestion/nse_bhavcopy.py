"""
╔══════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Daily Data Ingestion v4                ║
║         Project : MultiStockLSTMBot                             ║
║         File    : data/ingestion/nse_bhavcopy.py                ║
║         Phase   : 0 — Data Infrastructure                       ║
║                                                                  ║
║  Data Sources:                                                   ║
║    OHLCV      → Zerodha Kite Connect (chunked, 1800 days/call)  ║
║    Delivery % → NSE Bhavcopy / MTO file                        ║
║                                                                  ║
║  Fix v4: Kite has 2000 day limit per API call.                  ║
║          We now fetch in 1800-day chunks automatically.          ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
------
    # One-time 5-year backfill
    python -m data.ingestion.nse_bhavcopy --mode backfill --start 2019-01-01

    # Daily update after market close
    python -m data.ingestion.nse_bhavcopy --mode daily

    # Single date for testing
    python -m data.ingestion.nse_bhavcopy --mode single --date 2024-03-15
"""

import os
import io
import json
import time
import argparse
import requests
import pandas as pd
import psycopg2
import psycopg2.extras
import yaml

from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "daily_ingestion_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# ── Constants ─────────────────────────────────────────────────────────────
ACCESS_TOKEN_FILE  = Path("config/.kite_access_token")
INSTRUMENT_FILE    = Path("config/.kite_instruments.json")
KITE_CHUNK_DAYS    = 1800      # Safe below Kite's 2000-day hard limit
KITE_REQUEST_DELAY = 0.35      # Seconds between Kite API calls (rate limit ~3/sec)
NSE_REQUEST_DELAY  = 1.5       # Seconds between NSE requests
MAX_RETRIES        = 3

# NSE URLs — used only for delivery % (not OHLCV)
DELIVERY_URL  = "https://www.nseindia.com/archives/equities/mto/MT{date}.DAT"
BHAVCOPY_URL  = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept"  : "*/*",
    "Referer" : "https://www.nseindia.com/",
}


# ══════════════════════════════════════════════════════════════════════════
#  UNIVERSE LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_universe() -> set:
    """Loads Nifty 500 symbols from config/universe.yaml"""
    path = Path("config/universe.yaml")
    if not path.exists():
        raise FileNotFoundError("config/universe.yaml not found.")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        raise ValueError("config/universe.yaml is empty or invalid.")
    symbols = set(data.get("nifty500", []))
    if not symbols:
        raise ValueError("nifty500 key is empty in universe.yaml.")
    logger.info(f"Universe loaded: {len(symbols)} symbols")
    return symbols


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════

def get_db_connection():
    url = os.getenv("TIMESCALE_URL")
    if not url:
        raise EnvironmentError("TIMESCALE_URL not set in .env")
    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn


def ensure_tables_exist(conn):
    """Creates all required TimescaleDB tables"""
    with conn.cursor() as cur:

        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_ohlcv (
                date            DATE        NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                series          VARCHAR(5)  NOT NULL DEFAULT 'EQ',
                open            NUMERIC(12,2),
                high            NUMERIC(12,2),
                low             NUMERIC(12,2),
                close           NUMERIC(12,2),
                prev_close      NUMERIC(12,2),
                volume          NUMERIC(22,0),
                turnover        NUMERIC(22,2),
                trades          NUMERIC(12,0),
                deliverable_qty NUMERIC(22,0),
                delivery_pct    NUMERIC(6,2),
                PRIMARY KEY (date, symbol)
            );
        """)

        cur.execute("""
            SELECT create_hypertable(
                'daily_ohlcv', 'date',
                if_not_exists => TRUE,
                migrate_data  => TRUE
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_ohlcv_symbol
            ON daily_ohlcv (symbol, date DESC);
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS data_load_log (
                date          DATE        PRIMARY KEY,
                ohlcv_rows    INTEGER,
                delivery_rows INTEGER,
                merged_rows   INTEGER,
                source        VARCHAR(20) DEFAULT 'kite',
                loaded_at     TIMESTAMP   DEFAULT NOW(),
                status        VARCHAR(20) DEFAULT 'success'
            );
        """)

    conn.commit()
    logger.info("Database tables verified/created")


def is_already_loaded(trade_date: date, conn) -> bool:
    """Returns True if date already successfully loaded"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM data_load_log WHERE date = %s AND status = 'success'",
            (trade_date,)
        )
        return cur.fetchone() is not None


# ══════════════════════════════════════════════════════════════════════════
#  KITE CLIENT & INSTRUMENTS
# ══════════════════════════════════════════════════════════════════════════

def get_kite_client() -> KiteConnect:
    """
    Returns authenticated KiteConnect client using cached access token.
    Run kite_feed.py --mode auth to refresh token if expired.
    """
    api_key = os.getenv("KITE_API_KEY")
    if not api_key:
        raise EnvironmentError("KITE_API_KEY not set in .env")

    kite = KiteConnect(api_key=api_key)

    if not ACCESS_TOKEN_FILE.exists():
        raise FileNotFoundError(
            "Kite access token not found.\n"
            "Run: python -m data.ingestion.kite_feed --mode auth"
        )

    token_data = json.loads(ACCESS_TOKEN_FILE.read_text())
    token_date = token_data.get("date", "")

    if token_date != str(date.today()):
        raise ValueError(
            f"Kite access token expired (date: {token_date}).\n"
            "Run: python -m data.ingestion.kite_feed --mode auth"
        )

    kite.set_access_token(token_data["access_token"])
    logger.info("Kite client authenticated successfully")
    return kite


def load_instrument_tokens(kite: KiteConnect, universe: set) -> dict:
    """
    Returns {symbol: instrument_token} for all universe symbols.
    Instrument token is the numeric ID required by Kite historical API.
    Result is cached daily to avoid repeated API calls.
    """
    # Use cache if available and fresh
    if INSTRUMENT_FILE.exists():
        cache = json.loads(INSTRUMENT_FILE.read_text())
        if cache.get("date") == str(date.today()):
            token_map = {v: int(k) for k, v in cache["tokens"].items()}
            logger.info(f"Loaded {len(token_map)} instrument tokens from cache")
            return token_map

    logger.info("Fetching instrument list from Kite...")
    instruments = kite.instruments("NSE")

    token_map = {}
    for inst in instruments:
        symbol = inst["tradingsymbol"]
        if (
            symbol in universe
            and inst["segment"] == "NSE"
            and inst["instrument_type"] == "EQ"
        ):
            token_map[symbol] = inst["instrument_token"]

    # Save cache
    INSTRUMENT_FILE.parent.mkdir(exist_ok=True)
    INSTRUMENT_FILE.write_text(json.dumps({
        "date"  : str(date.today()),
        "tokens": {str(v): k for k, v in token_map.items()}
    }, indent=2))

    logger.info(f"Instrument tokens loaded: {len(token_map)} symbols")
    return token_map


# ══════════════════════════════════════════════════════════════════════════
#  KITE OHLCV FETCHER — WITH CHUNKING
# ══════════════════════════════════════════════════════════════════════════

def fetch_ohlcv_from_kite(
    kite: KiteConnect,
    token_map: dict,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetches daily OHLCV for all universe symbols from Kite.

    IMPORTANT: Kite has a hard limit of 2000 days per API call.
    This function automatically splits requests into KITE_CHUNK_DAYS
    (1800 day) chunks to stay safely below the limit.

    For a 5-year backfill (2019-2026 = ~1850 trading days):
        Chunk 1: 2019-01-01 → 2023-12-14 (1800 days)
        Chunk 2: 2023-12-15 → today      (~490 days)

    Args:
        kite       : Authenticated KiteConnect client
        token_map  : {symbol: instrument_token}
        start_date : Start of full date range
        end_date   : End of full date range

    Returns:
        DataFrame with columns: date, symbol, series, open, high, low, close, volume
    """
    all_records = []
    total       = len(token_map)

    logger.info(
        f"Fetching OHLCV from Kite: {total} symbols | "
        f"{start_date} to {end_date} | "
        f"chunk size: {KITE_CHUNK_DAYS} days"
    )

    for i, (symbol, token) in enumerate(token_map.items(), 1):
        symbol_candles = []

        # ── Split into chunks to respect 2000-day limit ───────────────────
        chunk_start = start_date
        while chunk_start <= end_date:
            chunk_end = min(
                chunk_start + timedelta(days=KITE_CHUNK_DAYS),
                end_date
            )

            try:
                candles = kite.historical_data(
                    instrument_token = token,
                    from_date        = chunk_start,
                    to_date          = chunk_end,
                    interval         = "day",
                    continuous       = False,
                    oi               = False,
                )
                symbol_candles.extend(candles or [])

            except Exception as chunk_err:
                logger.warning(
                    f"{symbol} chunk {chunk_start}→{chunk_end} failed: {chunk_err}"
                )

            # Move to next chunk
            chunk_start = chunk_end + timedelta(days=1)

            # Rate limit between chunk calls
            time.sleep(KITE_REQUEST_DELAY)

        # ── Process all candles for this symbol ───────────────────────────
        if not symbol_candles:
            logger.debug(f"{symbol}: No candles returned")
            continue

        for candle in symbol_candles:
            candle_date = candle["date"]
            if hasattr(candle_date, "date"):
                candle_date = candle_date.date()

            all_records.append({
                "date"  : candle_date,
                "symbol": symbol,
                "series": "EQ",
                "open"  : candle["open"],
                "high"  : candle["high"],
                "low"   : candle["low"],
                "close" : candle["close"],
                "volume": candle["volume"],
            })

        # Progress log every 50 symbols
        if i % 50 == 0:
            logger.info(
                f"Kite OHLCV progress: {i}/{total} symbols | "
                f"{len(all_records):,} records so far"
            )

    # ── Build DataFrame ───────────────────────────────────────────────────
    if not all_records:
        logger.error("No OHLCV data returned from Kite")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    logger.success(
        f"Kite OHLCV fetch complete: "
        f"{len(df):,} records | "
        f"{df['symbol'].nunique()} symbols | "
        f"{df['date'].nunique()} dates"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════
#  NSE DELIVERY FETCHER
# ══════════════════════════════════════════════════════════════════════════

def get_nse_session() -> requests.Session:
    """Creates NSE session with required cookies"""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1.5)
    except Exception as e:
        logger.warning(f"NSE session init: {e}")
    return session


def fetch_delivery_from_nse(
    trade_date: date,
    session: requests.Session,
) -> pd.DataFrame:
    """
    Fetches delivery % from NSE for a single trading date.
    This is the ONLY thing we need from NSE — Kite handles all OHLCV.

    Tries MTO DAT file first, falls back to Bhavcopy CSV.
    Returns empty DataFrame if neither works — delivery will be NULL in DB.
    NULL delivery is handled gracefully by the MSI feature calculator.
    """
    date_str = trade_date.strftime("%d%m%Y")

    # ── Primary: MTO delivery file ────────────────────────────────────────
    mto_url = DELIVERY_URL.format(date=date_str)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(mto_url, timeout=15)
            if resp.status_code == 200:
                df = _parse_mto_file(resp.content)
                if not df.empty:
                    return df
            elif resp.status_code == 404:
                break
            time.sleep(NSE_REQUEST_DELAY * attempt)
        except Exception as e:
            logger.debug(f"MTO attempt {attempt} for {trade_date}: {e}")
            time.sleep(NSE_REQUEST_DELAY * attempt)

    # ── Fallback: Bhavcopy CSV for delivery ───────────────────────────────
    bhav_url = BHAVCOPY_URL.format(date=date_str)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(bhav_url, timeout=15)
            if resp.status_code == 200:
                df = _parse_bhavcopy_for_delivery(resp.content)
                if not df.empty:
                    return df
            elif resp.status_code == 404:
                break
            time.sleep(NSE_REQUEST_DELAY * attempt)
        except Exception as e:
            logger.debug(f"Bhavcopy fallback attempt {attempt} for {trade_date}: {e}")
            time.sleep(NSE_REQUEST_DELAY * attempt)

    logger.warning(f"Delivery data unavailable for {trade_date} — will store NULL")
    return pd.DataFrame(columns=["symbol", "deliverable_qty", "delivery_pct", "trades"])


def _parse_mto_file(raw_bytes: bytes) -> pd.DataFrame:
    """Parses NSE MTO DAT file for delivery data"""
    try:
        content = raw_bytes.decode("utf-8", errors="ignore")
        lines   = content.strip().split("\n")
        records = []

        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7 and parts[0] == "90":
                try:
                    records.append({
                        "symbol"         : parts[2].strip().upper(),
                        "series"         : parts[3].strip(),
                        "deliverable_qty": int(float(parts[5].replace(",", ""))),
                        "delivery_pct"   : float(parts[6].replace(",", "")),
                        "trades"         : None,
                    })
                except (ValueError, IndexError):
                    continue

        if not records:
            return pd.DataFrame(columns=["symbol", "deliverable_qty", "delivery_pct", "trades"])

        df = pd.DataFrame(records)
        df = df[df["series"] == "EQ"][
            ["symbol", "deliverable_qty", "delivery_pct", "trades"]
        ].copy()
        return df

    except Exception as e:
        logger.error(f"MTO parse error: {e}")
        return pd.DataFrame(columns=["symbol", "deliverable_qty", "delivery_pct", "trades"])


def _parse_bhavcopy_for_delivery(raw_bytes: bytes) -> pd.DataFrame:
    """
    Parses Bhavcopy CSV extracting ONLY delivery % and trades.
    OHLCV columns from here are ignored — Kite is authoritative for those.
    """
    try:
        content = raw_bytes.decode("utf-8", errors="ignore")
        df      = pd.read_csv(io.StringIO(content))
        df.columns = df.columns.str.strip().str.upper()

        column_map = {
            "SYMBOL"      : "symbol",
            "SERIES"      : "series",
            "TOTALTRADES" : "trades",
            "NOOFTRADES"  : "trades",
            "NO OF TRADES": "trades",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        if "series" in df.columns:
            df = df[df["series"].str.strip() == "EQ"]
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].str.strip().str.upper()

        if "trades" in df.columns:
            df["trades"] = pd.to_numeric(df["trades"], errors="coerce")
        else:
            df["trades"] = None

        df["deliverable_qty"] = None
        df["delivery_pct"]    = None

        return df[["symbol", "deliverable_qty", "delivery_pct", "trades"]].copy()

    except Exception as e:
        logger.error(f"Bhavcopy delivery parse error: {e}")
        return pd.DataFrame(columns=["symbol", "deliverable_qty", "delivery_pct", "trades"])


# ══════════════════════════════════════════════════════════════════════════
#  MERGE & VALIDATE
# ══════════════════════════════════════════════════════════════════════════

def merge_and_validate(
    ohlcv_df: pd.DataFrame,
    delivery_df: pd.DataFrame,
    trade_date: date,
) -> pd.DataFrame:
    """
    Merges Kite OHLCV with NSE delivery data for a single trading date.
    Kite OHLCV is authoritative. NSE delivery augments it.
    """
    # Filter ohlcv_df to this specific date
    df = ohlcv_df[ohlcv_df["date"] == trade_date].copy()

    if df.empty:
        return df

    # Merge delivery
    if not delivery_df.empty:
        df = df.merge(delivery_df, on="symbol", how="left")
    else:
        df["deliverable_qty"] = None
        df["delivery_pct"]    = None
        df["trades"]          = None

    # Add missing columns
    if "prev_close" not in df.columns:
        df["prev_close"] = None

    # Estimate turnover (Kite doesn't provide it)
    if "turnover" not in df.columns:
        df["turnover"] = (
            df["volume"] * ((df["open"] + df["close"]) / 2)
        ).round(2)

    # Quality checks
    df = df[df["close"]  > 0]
    df = df[df["volume"] > 0]

    # OHLC consistency
    if len(df) > 0:
        valid = (
            (df["high"] >= df["low"])   &
            (df["high"] >= df["open"])  &
            (df["high"] >= df["close"]) &
            (df["low"]  <= df["open"])  &
            (df["low"]  <= df["close"])
        )
        removed = (~valid).sum()
        if removed > 0:
            logger.warning(f"{removed} OHLC inconsistencies removed for {trade_date}")
        df = df[valid]

    # Final column selection
    final_cols = [
        "date", "symbol", "series", "open", "high", "low", "close",
        "prev_close", "volume", "turnover", "trades",
        "deliverable_qty", "delivery_pct"
    ]
    for col in final_cols:
        if col not in df.columns:
            df[col] = None

    df = df[final_cols].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    logger.info(f"{trade_date}: {len(df)} clean records")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE INSERTION
# ══════════════════════════════════════════════════════════════════════════

def insert_to_db(df: pd.DataFrame, conn, trade_date: date) -> int:
    """Batch upserts clean DataFrame to TimescaleDB"""
    if df.empty:
        return 0

    records = df.to_dict("records")

    insert_sql = """
        INSERT INTO daily_ohlcv (
            date, symbol, series, open, high, low, close,
            prev_close, volume, turnover, trades,
            deliverable_qty, delivery_pct
        ) VALUES (
            %(date)s, %(symbol)s, %(series)s, %(open)s, %(high)s,
            %(low)s, %(close)s, %(prev_close)s, %(volume)s,
            %(turnover)s, %(trades)s, %(deliverable_qty)s, %(delivery_pct)s
        )
        ON CONFLICT (date, symbol) DO UPDATE SET
            open            = EXCLUDED.open,
            high            = EXCLUDED.high,
            low             = EXCLUDED.low,
            close           = EXCLUDED.close,
            volume          = EXCLUDED.volume,
            turnover        = EXCLUDED.turnover,
            trades          = EXCLUDED.trades,
            deliverable_qty = EXCLUDED.deliverable_qty,
            delivery_pct    = EXCLUDED.delivery_pct;
    """

    log_sql = """
        INSERT INTO data_load_log
            (date, ohlcv_rows, delivery_rows, merged_rows, source, status)
        VALUES (%s, %s, %s, %s, 'kite', 'success')
        ON CONFLICT (date) DO UPDATE SET
            merged_rows = EXCLUDED.merged_rows,
            loaded_at   = NOW(),
            status      = 'success';
    """

    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=500)
            delivery_rows = int(df["deliverable_qty"].notna().sum())
            cur.execute(log_sql, (trade_date, len(df), delivery_rows, len(df)))
        conn.commit()
        logger.success(f"✓ {trade_date} — {len(df)} rows inserted")
        return len(df)
    except Exception as e:
        conn.rollback()
        logger.error(f"DB insert failed for {trade_date}: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════
#  PREV_CLOSE UPDATER
# ══════════════════════════════════════════════════════════════════════════

def update_prev_close(conn):
    """
    Updates prev_close for all rows using previous trading day's close.
    Run once after backfill completes — much faster than computing per-row.
    """
    logger.info("Updating prev_close values (one-time SQL pass)...")
    sql = """
        UPDATE daily_ohlcv d
        SET prev_close = prev.prev_close
        FROM (
            SELECT
                date,
                symbol,
                LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close
            FROM daily_ohlcv
        ) prev
        WHERE d.date   = prev.date
          AND d.symbol = prev.symbol
          AND prev.prev_close IS NOT NULL;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.success("prev_close updated for all rows")


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_trading_dates(start: date, end: date) -> list[date]:
    """Returns all weekdays between start and end inclusive"""
    dates   = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


# ══════════════════════════════════════════════════════════════════════════
#  PIPELINE MODES
# ══════════════════════════════════════════════════════════════════════════

def run_backfill(start_date: date):
    """
    Full historical backfill.

    Step 1: Fetch ALL OHLCV from Kite in 1800-day chunks per symbol
    Step 2: For each trading date, fetch delivery from NSE, merge, insert
    Step 3: Update prev_close in a single SQL pass
    """
    end_date     = date.today() - timedelta(days=1)
    trading_days = get_trading_dates(start_date, end_date)

    logger.info(
        f"Backfill: {len(trading_days)} trading days | "
        f"{start_date} to {end_date}"
    )

    universe = load_universe()
    kite     = get_kite_client()
    conn     = get_db_connection()
    nse_sess = get_nse_session()

    ensure_tables_exist(conn)

    # ── Step 1: Fetch all OHLCV from Kite upfront ─────────────────────────
    logger.info("Step 1/3: Fetching OHLCV from Kite (chunked)...")
    token_map = load_instrument_tokens(kite, universe)
    ohlcv_df  = fetch_ohlcv_from_kite(kite, token_map, start_date, end_date)

    if ohlcv_df.empty:
        logger.error("No OHLCV data from Kite — check access token and retry")
        conn.close()
        return

    # ── Step 2: Process each date ─────────────────────────────────────────
    logger.info("Step 2/3: Merging with NSE delivery data and inserting...")
    loaded  = 0
    skipped = 0

    for i, trade_date in enumerate(trading_days, 1):
        logger.info(f"Progress: {i}/{len(trading_days)} — {trade_date}")

        if is_already_loaded(trade_date, conn):
            logger.info(f"Skipping {trade_date} — already loaded")
            skipped += 1
            continue

        # Fetch delivery from NSE (lightweight — only delivery %)
        delivery_df = fetch_delivery_from_nse(trade_date, nse_sess)

        # Merge and validate
        clean_df = merge_and_validate(ohlcv_df, delivery_df, trade_date)

        if clean_df.empty:
            logger.info(f"Skipping {trade_date} — no data (likely holiday)")
            skipped += 1
            continue

        # Insert to TimescaleDB
        try:
            insert_to_db(clean_df, conn, trade_date)
            loaded += 1
        except Exception as e:
            logger.error(f"Insert failed for {trade_date}: {e}")

        # Refresh NSE session every 100 dates
        if i % 100 == 0:
            nse_sess = get_nse_session()

        time.sleep(0.1)

    # ── Step 3: Update prev_close ─────────────────────────────────────────
    logger.info("Step 3/3: Updating prev_close...")
    update_prev_close(conn)
    conn.close()

    logger.success(
        f"Backfill complete — "
        f"Loaded: {loaded} dates | Skipped: {skipped} dates | "
        f"Approx rows: ~{loaded * 450:,}"
    )


def run_daily():
    """Daily update — fetches yesterday's data via Kite + NSE delivery"""
    now = datetime.now()
    target_date = (now - timedelta(days=1)).date() if now.hour < 18 else now.date()
    while target_date.weekday() >= 5:
        target_date -= timedelta(days=1)

    logger.info(f"Daily mode: {target_date}")

    universe = load_universe()
    kite     = get_kite_client()
    conn     = get_db_connection()
    nse_sess = get_nse_session()

    ensure_tables_exist(conn)

    if is_already_loaded(target_date, conn):
        logger.info(f"{target_date} already loaded")
        conn.close()
        return

    token_map   = load_instrument_tokens(kite, universe)
    ohlcv_df    = fetch_ohlcv_from_kite(kite, token_map, target_date, target_date)
    delivery_df = fetch_delivery_from_nse(target_date, nse_sess)
    clean_df    = merge_and_validate(ohlcv_df, delivery_df, target_date)

    insert_to_db(clean_df, conn, target_date)
    update_prev_close(conn)
    conn.close()

    logger.success(f"Daily update complete: {target_date}")


def run_single(target_date: date):
    """Single date — for testing or gap fill"""
    logger.info(f"Single mode: {target_date}")

    universe = load_universe()
    kite     = get_kite_client()
    conn     = get_db_connection()
    nse_sess = get_nse_session()

    ensure_tables_exist(conn)

    token_map   = load_instrument_tokens(kite, universe)
    ohlcv_df    = fetch_ohlcv_from_kite(kite, token_map, target_date, target_date)
    delivery_df = fetch_delivery_from_nse(target_date, nse_sess)
    clean_df    = merge_and_validate(ohlcv_df, delivery_df, target_date)

    insert_to_db(clean_df, conn, target_date)
    conn.close()

    logger.success(f"Single date complete: {target_date} — {len(clean_df)} rows")


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G.O.D.S E.Y.E — Daily Data Ingestion (Kite + NSE)"
    )
    parser.add_argument(
        "--mode",
        choices=["daily", "backfill", "single"],
        required=True,
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date(2019, 1, 1),
        help="Start date for backfill (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Specific date for single mode (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    if args.mode == "daily":
        run_daily()
    elif args.mode == "backfill":
        run_backfill(args.start)
    elif args.mode == "single":
        if not args.date:
            parser.error("--date is required for single mode")
        run_single(args.date)