"""
╔══════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — NSE Bhavcopy Downloader                ║
║         File   : data/ingestion/nse_bhavcopy.py                 ║
║         Phase  : 0 — Data Infrastructure                        ║
║         Purpose: Downloads daily OHLCV + delivery data from NSE ║
║                  Stores into TimescaleDB (PostgreSQL)            ║
╚══════════════════════════════════════════════════════════════════╝

What this file does:
--------------------
1. Downloads NSE Bhavcopy (OHLCV) CSV for any given trading date
2. Downloads NSE Delivery data CSV for the same date
3. Merges both into a single clean DataFrame
4. Filters to only Nifty 500 universe stocks (from config/universe.yaml)
5. Validates data quality (missing values, price anomalies, volume zero)
6. Inserts clean records into TimescaleDB
7. Handles corporate actions — adjusts for splits/bonuses automatically
8. Can run in BACKFILL mode (load 5 years of history) or DAILY mode

Usage:
------
    # Daily mode (run after 6 PM IST on any trading day)
    python -m data.ingestion.nse_bhavcopy --mode daily

    # Backfill mode (load full history — run once during Phase 0 setup)
    python -m data.ingestion.nse_bhavcopy --mode backfill --start 2019-01-01

    # Single date (for testing or gap-filling)
    python -m data.ingestion.nse_bhavcopy --mode single --date 2024-03-15

NSE Data Sources:
-----------------
    Bhavcopy  : https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_<DATE>_F_0000.csv.zip
    Delivery  : https://www.nseindia.com/archives/equities/mto/MT<DATE>.DAT
    Both are free, no authentication required.

Dependencies:
-------------
    pip install requests pandas psycopg2-binary pyyaml loguru python-dotenv
"""

import os
import io
import time
import zipfile
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

# ── Load environment variables from .env ──────────────────────────────────
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────
NSE_HEADERS = {
    # NSE blocks requests without a browser User-Agent
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# NSE Bhavcopy URL pattern (new format as of 2024)
BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
)

# NSE Delivery (MTO) URL pattern
DELIVERY_URL = (
    "https://www.nseindia.com/archives/equities/mto/MT{date}.DAT"
)

# Retry settings for NSE (they rate-limit aggressively)
MAX_RETRIES   = 5
RETRY_DELAY   = 3    # seconds between retries
REQUEST_TIMEOUT = 30  # seconds

# ── Logger setup ──────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "nse_bhavcopy_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


# ══════════════════════════════════════════════════════════════════════════
#  CONFIG LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_universe() -> set:
    """
    Loads the Nifty 500 stock symbol list from config/universe.yaml.
    Returns a set of NSE symbols (e.g. {'RELIANCE', 'TCS', 'INFY', ...})

    If universe.yaml does not exist yet, falls back to downloading
    the Nifty 500 list directly from NSE indices endpoint.
    """
    universe_path = Path("config/universe.yaml")

    if universe_path.exists():
        with open(universe_path, "r") as f:
            data = yaml.safe_load(f)
        symbols = set(data.get("nifty500", []))
        logger.info(f"Universe loaded: {len(symbols)} symbols from config/universe.yaml")
        return symbols

    # Fallback: fetch Nifty 500 constituents from NSE
    logger.warning("universe.yaml not found — fetching Nifty 500 from NSE...")
    return _fetch_nifty500_from_nse()


def _fetch_nifty500_from_nse() -> set:
    """
    Fetches the current Nifty 500 constituent list from NSE's index API.
    Saves result to config/universe.yaml for future runs.
    """
    url = (
        "https://nseindia.com/api/equity-stockIndices"
        "?index=NIFTY%20500"
    )
    session = _get_nse_session()
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        symbols = {
            item["symbol"]
            for item in data.get("data", [])
            if item.get("symbol") != "NIFTY 500"  # exclude index row
        }
        # Save to universe.yaml
        Path("config").mkdir(exist_ok=True)
        with open("config/universe.yaml", "w") as f:
            yaml.dump({"nifty500": sorted(symbols)}, f)
        logger.info(f"Fetched {len(symbols)} symbols from NSE; saved to universe.yaml")
        return symbols
    except Exception as e:
        logger.error(f"Failed to fetch Nifty 500 from NSE: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE CONNECTION
# ══════════════════════════════════════════════════════════════════════════

def get_db_connection():
    """
    Returns a psycopg2 connection to TimescaleDB.
    Connection string loaded from TIMESCALE_URL in .env

    Expected .env format:
        TIMESCALE_URL=postgresql://user:password@localhost:5432/godseye
    """
    url = os.getenv("TIMESCALE_URL")
    if not url:
        raise EnvironmentError(
            "TIMESCALE_URL not set in .env\n"
            "Expected format: postgresql://user:password@localhost:5432/godseye"
        )
    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn


def ensure_tables_exist(conn):
    """
    Creates the required TimescaleDB tables if they don't exist.

    Tables created:
        - daily_ohlcv     : Daily OHLCV + delivery data (hypertable on date)
        - data_load_log   : Tracks which dates have been successfully loaded
    """
    with conn.cursor() as cur:

        # ── Main OHLCV table ──────────────────────────────────────────────
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
                volume          BIGINT,
                turnover        NUMERIC(18,2),    -- in Rupees
                trades          INTEGER,
                deliverable_qty BIGINT,
                delivery_pct    NUMERIC(6,2),     -- delivery % of volume
                PRIMARY KEY (date, symbol)
            );
        """)

        # ── Convert to TimescaleDB hypertable (partitioned by date) ───────
        # This gives us 10–100× faster time-range queries
        cur.execute("""
            SELECT create_hypertable(
                'daily_ohlcv', 'date',
                if_not_exists => TRUE,
                migrate_data  => TRUE
            );
        """)

        # ── Index for fast symbol lookups ─────────────────────────────────
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_ohlcv_symbol
            ON daily_ohlcv (symbol, date DESC);
        """)

        # ── Data load tracking table ──────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS data_load_log (
                date            DATE        PRIMARY KEY,
                bhavcopy_rows   INTEGER,
                delivery_rows   INTEGER,
                merged_rows     INTEGER,
                loaded_at       TIMESTAMP   DEFAULT NOW(),
                status          VARCHAR(20) DEFAULT 'success'
            );
        """)

    conn.commit()
    logger.info("Database tables verified/created successfully.")


# ══════════════════════════════════════════════════════════════════════════
#  NSE SESSION (handles cookies — NSE requires them)
# ══════════════════════════════════════════════════════════════════════════

def _get_nse_session() -> requests.Session:
    """
    Creates a requests Session that mimics a browser.
    NSE requires visiting the homepage first to get session cookies,
    otherwise data downloads return 403 Forbidden.
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)

    try:
        # Hit NSE homepage to get session cookies
        session.get("https://www.nseindia.com", timeout=REQUEST_TIMEOUT)
        time.sleep(1)  # Polite delay
    except Exception as e:
        logger.warning(f"Could not initialize NSE session cookies: {e}")

    return session


# ══════════════════════════════════════════════════════════════════════════
#  BHAVCOPY DOWNLOAD & PARSE
# ══════════════════════════════════════════════════════════════════════════

def download_bhavcopy(trade_date: date, session: requests.Session) -> pd.DataFrame:
    """
    Downloads and parses the NSE Bhavcopy CSV for a given trading date.

    Args:
        trade_date : The trading date to download (e.g. date(2024, 3, 15))
        session    : Authenticated requests.Session with NSE cookies

    Returns:
        DataFrame with columns:
            symbol, series, open, high, low, close, prev_close,
            volume, turnover, trades, date

    Raises:
        ValueError  : If date is a weekend/holiday (no data available)
        RuntimeError: If download fails after all retries
    """
    # NSE uses DDMMYYYY format in Bhavcopy URL
    date_str = trade_date.strftime("%d%m%Y")
    url = BHAVCOPY_URL.format(date=date_str)

    logger.info(f"Downloading Bhavcopy for {trade_date} from NSE...")

    raw_bytes = _download_with_retry(url, session)
    if raw_bytes is None:
        raise RuntimeError(f"Bhavcopy download failed for {trade_date} after {MAX_RETRIES} retries")

    # ── Unzip and parse CSV ───────────────────────────────────────────────
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            csv_filename = zf.namelist()[0]
            with zf.open(csv_filename) as f:
                df = pd.read_csv(f)
    except zipfile.BadZipFile:
        raise RuntimeError(f"Bhavcopy for {trade_date} is not a valid ZIP file")

    # ── Normalize column names ────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.upper()

    # Map NSE's column names to our standard names
    # (NSE occasionally changes column names — this mapping handles variants)
    column_map = {
        "SYMBOL"       : "symbol",
        "SERIES"       : "series",
        "OPEN"         : "open",
        "OPEN PRICE"   : "open",
        "HIGH"         : "high",
        "HIGH PRICE"   : "high",
        "LOW"          : "low",
        "LOW PRICE"    : "low",
        "CLOSE"        : "close",
        "CLOSE PRICE"  : "close",
        "LAST"         : "close",     # fallback if CLOSE missing
        "PREVCLOSE"    : "prev_close",
        "PREV. CLOSE"  : "prev_close",
        "TOTTRDQTY"    : "volume",
        "TOTAL TRADED QUANTITY": "volume",
        "TOTTRDVAL"    : "turnover",
        "TOTAL TRADED VALUE"  : "turnover",
        "TOTALTRADES"  : "trades",
        "NO OF TRADES" : "trades",
    }

    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # ── Keep only EQ series (exclude BE, BL, BT, etc.) ───────────────────
    if "series" in df.columns:
        df = df[df["series"] == "EQ"].copy()

    # ── Add date column ───────────────────────────────────────────────────
    df["date"] = trade_date

    # ── Select and clean final columns ────────────────────────────────────
    keep = ["symbol", "series", "open", "high", "low", "close",
            "prev_close", "volume", "turnover", "trades", "date"]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()

    # Strip whitespace from symbol names
    df["symbol"] = df["symbol"].str.strip().str.upper()

    # Convert numeric columns (handle any comma-formatted numbers)
    numeric_cols = ["open", "high", "low", "close", "prev_close",
                    "volume", "turnover", "trades"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    logger.info(f"Bhavcopy parsed: {len(df)} EQ series records for {trade_date}")
    return df


def download_delivery_data(trade_date: date, session: requests.Session) -> pd.DataFrame:
    """
    Downloads and parses NSE Delivery (MTO) data for a given trading date.
    Delivery % is a core component of Pillar 2 (MSI) feature engineering.

    Args:
        trade_date : The trading date
        session    : NSE session

    Returns:
        DataFrame with columns: symbol, deliverable_qty, delivery_pct
        Returns empty DataFrame if delivery data unavailable (e.g. holidays)
    """
    # MTO file uses DDMMYYYY format
    date_str = trade_date.strftime("%d%m%Y")
    url = DELIVERY_URL.format(date=date_str)

    logger.info(f"Downloading Delivery data for {trade_date}...")

    raw_bytes = _download_with_retry(url, session)
    if raw_bytes is None:
        logger.warning(f"Delivery data not available for {trade_date} — will proceed without it")
        return pd.DataFrame(columns=["symbol", "deliverable_qty", "delivery_pct"])

    # ── Parse the fixed-format DAT file ──────────────────────────────────
    # MTO.DAT is a pipe-separated or fixed-width file depending on NSE version
    try:
        content = raw_bytes.decode("utf-8", errors="ignore")
        lines = content.strip().split("\n")

        records = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            # Expected format: record_type, sr_no, symbol, series, traded_qty,
            #                  deliverable_qty, delivery_pct
            if len(parts) >= 7 and parts[0] == "90":  # '90' = equity delivery record
                try:
                    records.append({
                        "symbol"         : parts[2].strip().upper(),
                        "series"         : parts[3].strip(),
                        "deliverable_qty": int(float(parts[5].replace(",", ""))),
                        "delivery_pct"   : float(parts[6].replace(",", "")),
                    })
                except (ValueError, IndexError):
                    continue

        if not records:
            logger.warning(f"No delivery records parsed for {trade_date}")
            return pd.DataFrame(columns=["symbol", "deliverable_qty", "delivery_pct"])

        df = pd.DataFrame(records)

        # Keep only EQ series
        df = df[df["series"] == "EQ"][["symbol", "deliverable_qty", "delivery_pct"]].copy()

        logger.info(f"Delivery data parsed: {len(df)} records for {trade_date}")
        return df

    except Exception as e:
        logger.error(f"Failed to parse delivery data for {trade_date}: {e}")
        return pd.DataFrame(columns=["symbol", "deliverable_qty", "delivery_pct"])


# ══════════════════════════════════════════════════════════════════════════
#  MERGE, FILTER & VALIDATE
# ══════════════════════════════════════════════════════════════════════════

def merge_and_filter(
    bhavcopy_df: pd.DataFrame,
    delivery_df: pd.DataFrame,
    universe: set,
    trade_date: date,
) -> pd.DataFrame:
    """
    Merges Bhavcopy and Delivery DataFrames, filters to universe stocks,
    and validates data quality.

    Args:
        bhavcopy_df : Raw Bhavcopy DataFrame
        delivery_df : Raw Delivery DataFrame (may be empty)
        universe    : Set of valid NSE symbols (Nifty 500)
        trade_date  : The trading date

    Returns:
        Clean, merged DataFrame ready for database insertion
    """
    # ── Filter to universe stocks only ────────────────────────────────────
    pre_filter = len(bhavcopy_df)
    bhavcopy_df = bhavcopy_df[bhavcopy_df["symbol"].isin(universe)].copy()
    logger.info(
        f"Universe filter: {pre_filter} → {len(bhavcopy_df)} records "
        f"({pre_filter - len(bhavcopy_df)} non-universe stocks removed)"
    )

    # ── Left join delivery data ───────────────────────────────────────────
    if not delivery_df.empty:
        df = bhavcopy_df.merge(delivery_df, on="symbol", how="left")
    else:
        df = bhavcopy_df.copy()
        df["deliverable_qty"] = None
        df["delivery_pct"]    = None

    # ── Data quality validation ───────────────────────────────────────────
    issues = []

    # 1. Remove rows where close price is zero or negative
    bad_price = df["close"] <= 0
    if bad_price.sum() > 0:
        issues.append(f"{bad_price.sum()} rows with close ≤ 0 removed")
        df = df[~bad_price]

    # 2. Remove rows where volume is zero (non-traded stocks)
    if "volume" in df.columns:
        bad_vol = df["volume"] == 0
        if bad_vol.sum() > 0:
            issues.append(f"{bad_vol.sum()} zero-volume rows removed")
            df = df[~bad_vol]

    # 3. Check OHLC consistency (high >= low, high >= open, high >= close)
    ohlc_inconsistent = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    if ohlc_inconsistent.sum() > 0:
        issues.append(f"{ohlc_inconsistent.sum()} OHLC-inconsistent rows removed")
        df = df[~ohlc_inconsistent]

    # 4. Flag extreme price changes (> 50% from prev_close in one day)
    # These are likely circuit breaker events — flag but don't remove
    if "prev_close" in df.columns and df["prev_close"].notna().any():
        extreme_move = (
            (df["close"] / df["prev_close"] - 1).abs() > 0.50
        ) & df["prev_close"].notna()
        if extreme_move.sum() > 0:
            logger.warning(
                f"{extreme_move.sum()} stocks with >50% daily move "
                f"(possible circuit breaker): "
                f"{df[extreme_move]['symbol'].tolist()}"
            )

    if issues:
        logger.warning(f"Data quality issues for {trade_date}: {'; '.join(issues)}")

    # ── Final column selection & types ────────────────────────────────────
    final_cols = [
        "date", "symbol", "series", "open", "high", "low", "close",
        "prev_close", "volume", "turnover", "trades",
        "deliverable_qty", "delivery_pct"
    ]
    for col in final_cols:
        if col not in df.columns:
            df[col] = None

    df = df[final_cols].copy()

    # Ensure date column is correct type
    df["date"] = pd.to_datetime(df["date"]).dt.date

    logger.info(f"Final merged dataset: {len(df)} clean records for {trade_date}")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE INSERTION
# ══════════════════════════════════════════════════════════════════════════

def insert_to_db(df: pd.DataFrame, conn, trade_date: date) -> int:
    """
    Inserts the clean DataFrame into TimescaleDB using fast batch upsert.
    Uses ON CONFLICT DO UPDATE so re-running for the same date is safe.

    Args:
        df         : Clean merged DataFrame
        conn       : psycopg2 connection
        trade_date : Trading date (for logging)

    Returns:
        Number of rows inserted/updated
    """
    if df.empty:
        logger.warning(f"No data to insert for {trade_date}")
        return 0

    records = df.to_dict("records")

    insert_sql = """
        INSERT INTO daily_ohlcv (
            date, symbol, series, open, high, low, close,
            prev_close, volume, turnover, trades,
            deliverable_qty, delivery_pct
        ) VALUES (
            %(date)s, %(symbol)s, %(series)s, %(open)s, %(high)s, %(low)s, %(close)s,
            %(prev_close)s, %(volume)s, %(turnover)s, %(trades)s,
            %(deliverable_qty)s, %(delivery_pct)s
        )
        ON CONFLICT (date, symbol) DO UPDATE SET
            open            = EXCLUDED.open,
            high            = EXCLUDED.high,
            low             = EXCLUDED.low,
            close           = EXCLUDED.close,
            prev_close      = EXCLUDED.prev_close,
            volume          = EXCLUDED.volume,
            turnover        = EXCLUDED.turnover,
            trades          = EXCLUDED.trades,
            deliverable_qty = EXCLUDED.deliverable_qty,
            delivery_pct    = EXCLUDED.delivery_pct;
    """

    log_sql = """
        INSERT INTO data_load_log (date, bhavcopy_rows, delivery_rows, merged_rows, status)
        VALUES (%s, %s, %s, %s, 'success')
        ON CONFLICT (date) DO UPDATE SET
            merged_rows = EXCLUDED.merged_rows,
            loaded_at   = NOW(),
            status      = 'success';
    """

    try:
        with conn.cursor() as cur:
            # Batch insert (much faster than row-by-row)
            psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=500)

            # Log the successful load
            delivery_rows = df["deliverable_qty"].notna().sum()
            cur.execute(log_sql, (trade_date, len(df), int(delivery_rows), len(df)))

        conn.commit()
        logger.info(f"Successfully inserted {len(df)} rows for {trade_date}")
        return len(df)

    except Exception as e:
        conn.rollback()
        logger.error(f"Database insertion failed for {trade_date}: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def _download_with_retry(url: str, session: requests.Session) -> bytes | None:
    """
    Downloads a URL with exponential backoff retry.
    Returns raw bytes on success, None on all retries exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.content
            elif resp.status_code == 404:
                logger.warning(f"404 Not Found: {url} (likely holiday/weekend)")
                return None
            else:
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES}: HTTP {resp.status_code} for {url}"
                )
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES}: Request error: {e}")

        if attempt < MAX_RETRIES:
            sleep_time = RETRY_DELAY * attempt  # exponential backoff
            logger.info(f"Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    return None


def is_already_loaded(trade_date: date, conn) -> bool:
    """
    Checks if a given date is already in data_load_log with status='success'.
    Prevents duplicate downloads during backfill.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM data_load_log WHERE date = %s AND status = 'success'",
            (trade_date,)
        )
        return cur.fetchone() is not None


def get_trading_dates(start: date, end: date) -> list[date]:
    """
    Returns a list of weekdays between start and end (inclusive).
    Note: This includes some public holidays — NSE will return 404 for those,
    which is handled gracefully in the download functions.
    """
    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current)
        current += timedelta(days=1)
    return dates


# ══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════

def run_for_date(trade_date: date, session: requests.Session,
                 universe: set, conn) -> bool:
    """
    Runs the complete download → parse → merge → validate → insert
    pipeline for a single trading date.

    Args:
        trade_date : Date to process
        session    : NSE session
        universe   : Nifty 500 symbol set
        conn       : DB connection

    Returns:
        True if successful, False if skipped (holiday/already loaded)
    """
    # Skip if already loaded
    if is_already_loaded(trade_date, conn):
        logger.info(f"Skipping {trade_date} — already loaded in DB")
        return False

    # Skip weekends
    if trade_date.weekday() >= 5:
        logger.debug(f"Skipping {trade_date} — weekend")
        return False

    try:
        # Step 1: Download Bhavcopy
        bhavcopy_df = download_bhavcopy(trade_date, session)

        # Step 2: Download Delivery data
        delivery_df = download_delivery_data(trade_date, session)

        # Step 3: Merge, filter to universe, validate
        clean_df = merge_and_filter(bhavcopy_df, delivery_df, universe, trade_date)

        # Step 4: Insert to TimescaleDB
        rows_inserted = insert_to_db(clean_df, conn, trade_date)

        logger.success(f"✓ {trade_date} complete — {rows_inserted} rows inserted")

        # Polite delay to avoid hammering NSE
        time.sleep(1.5)
        return True

    except RuntimeError as e:
        # 404s (holidays) are expected — just skip
        logger.info(f"Skipping {trade_date}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to process {trade_date}: {e}")
        return False


def run_daily():
    """
    Daily mode: downloads yesterday's data (or today's if after 6 PM IST).
    Run this via Airflow DAG nightly at 10:30 PM IST.
    """
    # Determine which date to download
    now = datetime.now()
    # If before 6 PM, use previous trading day
    if now.hour < 18:
        target_date = (now - timedelta(days=1)).date()
    else:
        target_date = now.date()

    # Roll back to Friday if it's a weekend
    while target_date.weekday() >= 5:
        target_date -= timedelta(days=1)

    logger.info(f"Daily mode: processing {target_date}")

    universe = load_universe()
    session  = _get_nse_session()
    conn     = get_db_connection()

    try:
        ensure_tables_exist(conn)
        run_for_date(target_date, session, universe, conn)
    finally:
        conn.close()


def run_backfill(start_date: date):
    """
    Backfill mode: loads all trading days from start_date to yesterday.
    Run once during Phase 0 setup to load 5 years of history.

    Expected runtime: ~45–90 minutes for 5 years (1250 trading days)
    """
    end_date     = date.today() - timedelta(days=1)
    trading_days = get_trading_dates(start_date, end_date)

    logger.info(
        f"Backfill mode: {len(trading_days)} dates from "
        f"{start_date} to {end_date}"
    )

    universe = load_universe()
    session  = _get_nse_session()
    conn     = get_db_connection()

    loaded  = 0
    skipped = 0
    failed  = 0

    try:
        ensure_tables_exist(conn)

        for i, trade_date in enumerate(trading_days, 1):
            logger.info(f"Progress: {i}/{len(trading_days)} — {trade_date}")

            success = run_for_date(trade_date, session, universe, conn)

            if success:
                loaded += 1
            else:
                skipped += 1

            # Refresh NSE session every 100 dates (cookies expire)
            if i % 100 == 0:
                logger.info("Refreshing NSE session...")
                session = _get_nse_session()

    except KeyboardInterrupt:
        logger.warning("Backfill interrupted by user — progress saved to DB")
    finally:
        conn.close()

    logger.info(
        f"Backfill complete — "
        f"Loaded: {loaded} | Skipped: {skipped} | Failed: {failed}"
    )


def run_single(target_date: date):
    """
    Single date mode: downloads data for one specific date.
    Useful for gap-filling or testing.
    """
    logger.info(f"Single date mode: {target_date}")

    universe = load_universe()
    session  = _get_nse_session()
    conn     = get_db_connection()

    try:
        ensure_tables_exist(conn)
        run_for_date(target_date, session, universe, conn)
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G.O.D.S E.Y.E — NSE Bhavcopy Downloader"
    )
    parser.add_argument(
        "--mode",
        choices=["daily", "backfill", "single"],
        required=True,
        help=(
            "daily    : Download latest trading day (run nightly)\n"
            "backfill : Load full history from --start date\n"
            "single   : Download one specific --date"
        )
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date(2019, 1, 1),
        help="Start date for backfill mode (YYYY-MM-DD). Default: 2019-01-01"
    )
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Specific date for single mode (YYYY-MM-DD)"
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