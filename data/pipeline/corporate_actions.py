"""
╔══════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Corporate Action Adjuster              ║
║         Project : MultiStockLSTMBot                             ║
║         File    : data/pipeline/corporate_actions.py            ║
║         Phase   : 0 — Data Infrastructure                       ║
║         Purpose : Adjusts historical OHLCV for splits,          ║
║                   bonuses, and dividends                         ║
╚══════════════════════════════════════════════════════════════════╝

Why this is critical:
----------------------
    If RELIANCE did a 1:1 bonus issue in September 2017, its price
    halved overnight from ~₹900 to ~₹450. Without adjustment, the
    model sees this as a catastrophic 50% crash — which poisons
    every training example from 2017 onward.

    Similarly, if HDFC Bank split 1:2 in 2019, all historical prices
    need to be halved and all historical volumes doubled to maintain
    a consistent price series.

    This module fetches corporate action history from NSE, stores it
    in the database, and applies adjustments to daily_ohlcv table.

Corporate Action Types:
-----------------------
    SPLIT   : Company splits shares. e.g. 2:1 → each share becomes 2
              Adjustment: divide historical prices by 2, multiply volume by 2
              Ratio stored: 2.0 (new shares per old share)

    BONUS   : Company issues free bonus shares. e.g. 1:1 → 1 bonus per 1 held
              Identical to split for price adjustment purposes.
              Ratio stored: 2.0 (for 1:1 bonus = price halves)

    DIVIDEND: Company pays cash dividend per share.
              Minor price impact — subtract dividend from historical prices.
              (Less critical than splits; price series remains mostly intact)

    RIGHTS  : Company offers new shares at discount to existing holders.
              Complex calculation — logged but not auto-adjusted.
              Requires manual review.

Usage:
------
    from data.pipeline.corporate_actions import CorporateActionAdjuster

    adjuster = CorporateActionAdjuster()

    # Fetch and store corporate actions for a symbol
    actions = adjuster.fetch_and_store_actions("RELIANCE")

    # Apply all pending adjustments for a symbol
    adjuster.apply_adjustments("RELIANCE", conn)

    # Adjust a DataFrame directly (without DB)
    adjusted_df = adjuster.adjust_dataframe(df, actions)

    # Backfill corporate actions for all 500 stocks (Phase 0 setup)
    adjuster.backfill_all_symbols(universe, conn)
"""

import os
import re
import time
import requests
import pandas as pd
import psycopg2
import psycopg2.extras

from datetime import datetime, date
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "corporate_actions_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="30 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}",
)

# ── NSE Corporate Actions API ─────────────────────────────────────────────
NSE_CORP_ACTIONS_URL = (
    "https://www.nseindia.com/api/corporates-corporateActions"
    "?index=equities&symbol={symbol}"
)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept"        : "application/json",
    "Referer"       : "https://www.nseindia.com/",
    "Accept-Language": "en-US,en;q=0.9",
}


# ══════════════════════════════════════════════════════════════════════════
#  RATIO PARSER
# ══════════════════════════════════════════════════════════════════════════

def parse_split_ratio(subject: str) -> float | None:
    """
    Parses a split/bonus ratio from NSE corporate action subject text.

    Examples of NSE subject strings:
        "Face Value Split From Rs.10/- to Rs.5/-"  → ratio = 2.0
        "Bonus 1:1"                                 → ratio = 2.0
        "Bonus 1:2"                                 → ratio = 1.5
        "Stock Split 5:1"                           → ratio = 5.0
        "Sub Division of Share from Rs 10 to Rs 2" → ratio = 5.0

    Args:
        subject : Raw NSE subject string

    Returns:
        Float ratio (new shares per old share), or None if unparseable
    """
    subject_lower = subject.lower().strip()

    # ── Pattern 1: Bonus X:Y format ──────────────────────────────────────
    # "Bonus 1:1" → 1 bonus per 1 held → ratio = 2.0 (shares double)
    bonus_match = re.search(r"bonus\s+(\d+)\s*:\s*(\d+)", subject_lower)
    if bonus_match:
        new_shares = int(bonus_match.group(1))  # bonus shares received
        old_shares = int(bonus_match.group(2))  # for each held
        ratio = (old_shares + new_shares) / old_shares
        return round(ratio, 4)

    # ── Pattern 2: Stock Split X:Y format ────────────────────────────────
    # "Stock Split 2:1" → 2 new for 1 old → ratio = 2.0
    split_match = re.search(r"split\s+(\d+)\s*:\s*(\d+)", subject_lower)
    if split_match:
        new_shares = int(split_match.group(1))
        old_shares = int(split_match.group(2))
        ratio = new_shares / old_shares
        return round(ratio, 4)

    # ── Pattern 3: Face value reduction ──────────────────────────────────
    # "Face Value Split From Rs.10/- to Rs.5/-" → ratio = 10/5 = 2.0
    fv_match = re.search(
        r"(?:face value|fv|nominal).*?(?:from|rs\.?)\s*([\d.]+).*?(?:to|rs\.?)\s*([\d.]+)",
        subject_lower
    )
    if fv_match:
        old_fv = float(fv_match.group(1))
        new_fv = float(fv_match.group(2))
        if new_fv > 0 and old_fv > new_fv:
            return round(old_fv / new_fv, 4)

    # ── Pattern 4: Sub-division ───────────────────────────────────────────
    # "Sub Division from Rs 10 to Rs 2" → ratio = 10/2 = 5.0
    sub_match = re.search(
        r"sub.?div.*?(?:from)?\s*rs\.?\s*([\d.]+).*?(?:to)?\s*rs\.?\s*([\d.]+)",
        subject_lower
    )
    if sub_match:
        old_fv = float(sub_match.group(1))
        new_fv = float(sub_match.group(2))
        if new_fv > 0 and old_fv > new_fv:
            return round(old_fv / new_fv, 4)

    return None


def parse_dividend_amount(subject: str) -> float | None:
    """
    Parses dividend amount per share from NSE subject text.

    Examples:
        "Dividend - Rs.5/- Per Share"  → 5.0
        "Interim Dividend Rs 2.50"     → 2.50
        "Final Dividend @ 150%"        → None (percentage-based — skip)

    Args:
        subject : Raw NSE subject string

    Returns:
        Dividend amount per share in Rupees, or None if unparseable
    """
    subject_lower = subject.lower().strip()

    # Skip percentage-based dividends (complex to compute)
    if "%" in subject:
        return None

    # Match "Rs. X" or "Rs X" patterns
    amount_match = re.search(
        r"(?:rs\.?|rupees?)\s*([\d.]+)\s*(?:/-|per share)?",
        subject_lower
    )
    if amount_match:
        try:
            return float(amount_match.group(1))
        except ValueError:
            return None

    return None


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_db_connection():
    url = os.getenv("TIMESCALE_URL")
    if not url:
        raise EnvironmentError("TIMESCALE_URL not set in .env")
    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn


def ensure_table(conn):
    """Creates corporate_actions table if not exists"""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                symbol          VARCHAR(20)  NOT NULL,
                ex_date         DATE         NOT NULL,
                action_type     VARCHAR(20)  NOT NULL,
                ratio           NUMERIC(10,4),
                amount          NUMERIC(10,4),
                raw_subject     TEXT,
                adjusted        BOOLEAN DEFAULT FALSE,
                fetched_at      TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (symbol, ex_date, action_type)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol
            ON corporate_actions (symbol, ex_date DESC);
        """)
    conn.commit()
    logger.info("corporate_actions table verified/created")


# ══════════════════════════════════════════════════════════════════════════
#  NSE FETCHER
# ══════════════════════════════════════════════════════════════════════════

def get_nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(0.5)
    except Exception:
        pass
    return session


def fetch_corporate_actions_nse(
    symbol: str,
    session: requests.Session,
) -> list[dict]:
    """
    Fetches corporate actions for a symbol from NSE API.

    Args:
        symbol  : NSE trading symbol (e.g. 'RELIANCE')
        session : NSE session with cookies

    Returns:
        List of parsed action dicts with keys:
            symbol, ex_date, action_type, ratio, amount, raw_subject
    """
    url = NSE_CORP_ACTIONS_URL.format(symbol=symbol)
    actions = []

    for attempt in range(1, 4):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                time.sleep(2 * attempt)
                continue

            data = resp.json()
            items = data if isinstance(data, list) else []

            for item in items:
                subject  = item.get("subject", "") or ""
                ex_date  = _parse_date(item.get("exDate", ""))

                if not ex_date or not subject:
                    continue

                subject_lower = subject.lower()

                # Classify action
                if any(k in subject_lower for k in ["split", "sub division", "sub-division", "face value"]):
                    action_type = "split"
                    ratio       = parse_split_ratio(subject)
                    amount      = None

                elif "bonus" in subject_lower:
                    action_type = "bonus"
                    ratio       = parse_split_ratio(subject)
                    amount      = None

                elif any(k in subject_lower for k in ["dividend", "div"]):
                    action_type = "dividend"
                    ratio       = None
                    amount      = parse_dividend_amount(subject)

                elif "rights" in subject_lower:
                    action_type = "rights"
                    ratio       = None
                    amount      = None
                    logger.info(f"{symbol}: Rights issue on {ex_date} — manual review needed")

                else:
                    continue  # Skip unknown action types

                actions.append({
                    "symbol"     : symbol,
                    "ex_date"    : ex_date,
                    "action_type": action_type,
                    "ratio"      : ratio,
                    "amount"     : amount,
                    "raw_subject": subject[:500],
                })

            return actions

        except Exception as e:
            logger.warning(f"NSE corp action fetch attempt {attempt}/3 for {symbol}: {e}")
            time.sleep(2 * attempt)

    return actions


# ══════════════════════════════════════════════════════════════════════════
#  PRICE ADJUSTMENT ENGINE
# ══════════════════════════════════════════════════════════════════════════

class CorporateActionAdjuster:
    """
    Fetches corporate actions from NSE and applies price adjustments
    to historical OHLCV data in TimescaleDB.
    """

    def fetch_and_store_actions(
        self,
        symbol: str,
        conn,
        session: requests.Session = None,
    ) -> list[dict]:
        """
        Fetches corporate actions for a symbol from NSE and stores in DB.

        Args:
            symbol  : NSE symbol
            conn    : DB connection
            session : NSE session (created if not provided)

        Returns:
            List of action dicts stored
        """
        if session is None:
            session = get_nse_session()

        actions = fetch_corporate_actions_nse(symbol, session)

        if not actions:
            logger.debug(f"{symbol}: No corporate actions found")
            return []

        sql = """
            INSERT INTO corporate_actions
                (symbol, ex_date, action_type, ratio, amount, raw_subject)
            VALUES
                (%(symbol)s, %(ex_date)s, %(action_type)s,
                 %(ratio)s, %(amount)s, %(raw_subject)s)
            ON CONFLICT (symbol, ex_date, action_type) DO NOTHING;
        """

        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, actions)
            conn.commit()
            logger.info(f"{symbol}: Stored {len(actions)} corporate actions")
        except Exception as e:
            logger.error(f"{symbol}: Failed to store corporate actions: {e}")
            conn.rollback()

        return actions

    def apply_adjustments(self, symbol: str, conn) -> int:
        """
        Applies all unadjusted corporate actions for a symbol
        to the daily_ohlcv table.

        Steps:
            1. Fetch unadjusted split/bonus actions from DB
            2. For each action (sorted desc by ex_date):
               Adjust all daily_ohlcv rows with date < ex_date
            3. Mark action as adjusted = TRUE

        Args:
            symbol : NSE symbol
            conn   : DB connection

        Returns:
            Number of rows adjusted in daily_ohlcv
        """
        # Get unadjusted split/bonus actions (oldest first)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ex_date, action_type, ratio, amount
                FROM corporate_actions
                WHERE symbol = %s
                  AND adjusted = FALSE
                  AND action_type IN ('split', 'bonus', 'dividend')
                  AND ratio IS NOT NULL OR amount IS NOT NULL
                ORDER BY ex_date ASC;
            """, (symbol,))
            pending_actions = cur.fetchall()

        if not pending_actions:
            return 0

        total_adjusted = 0

        for ex_date, action_type, ratio, amount in pending_actions:
            try:
                if action_type in ("split", "bonus") and ratio and ratio > 1.0:
                    adjusted = self._apply_split_to_db(symbol, ex_date, ratio, conn)
                    total_adjusted += adjusted

                elif action_type == "dividend" and amount and amount > 0:
                    adjusted = self._apply_dividend_to_db(symbol, ex_date, amount, conn)
                    total_adjusted += adjusted

                # Mark as adjusted
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE corporate_actions
                        SET adjusted = TRUE
                        WHERE symbol = %s AND ex_date = %s AND action_type = %s;
                    """, (symbol, ex_date, action_type))
                conn.commit()

            except Exception as e:
                logger.error(f"{symbol}: Failed to apply {action_type} adjustment on {ex_date}: {e}")
                conn.rollback()

        return total_adjusted

    def adjust_dataframe(
        self,
        df: pd.DataFrame,
        actions: list[dict],
    ) -> pd.DataFrame:
        """
        Applies corporate action adjustments directly to a DataFrame.
        Used during feature engineering when loading per-stock data.

        Args:
            df      : Single-stock OHLCV DataFrame, sorted ascending by date
            actions : List of action dicts from DB or NSE

        Returns:
            Adjusted DataFrame with continuous price series
        """
        df = df.copy().sort_values("date").reset_index(drop=True)

        # Sort actions by ex_date descending
        # (apply most recent first to avoid double-adjusting)
        sorted_actions = sorted(
            actions,
            key=lambda x: x["ex_date"],
            reverse=True,
        )

        for action in sorted_actions:
            ex_date     = action["ex_date"]
            action_type = action["action_type"]
            ratio       = action.get("ratio")
            amount      = action.get("amount")

            if isinstance(ex_date, str):
                ex_date = datetime.strptime(ex_date, "%Y-%m-%d").date()

            if action_type in ("split", "bonus") and ratio and ratio > 1.0:
                pre_action = df["date"].apply(
                    lambda d: (d.date() if hasattr(d, "date") else d) < ex_date
                )

                price_cols = ["open", "high", "low", "close", "prev_close"]
                for col in price_cols:
                    if col in df.columns:
                        df.loc[pre_action, col] = df.loc[pre_action, col] / ratio

                if "volume" in df.columns:
                    df.loc[pre_action, "volume"] = (
                        df.loc[pre_action, "volume"] * ratio
                    ).astype("Int64")

                logger.debug(
                    f"Applied {action_type} ratio={ratio} before {ex_date} "
                    f"({pre_action.sum()} rows adjusted)"
                )

            elif action_type == "dividend" and amount and amount > 0:
                pre_action = df["date"].apply(
                    lambda d: (d.date() if hasattr(d, "date") else d) < ex_date
                )
                price_cols = ["open", "high", "low", "close"]
                for col in price_cols:
                    if col in df.columns:
                        df.loc[pre_action, col] = (
                            df.loc[pre_action, col] - amount
                        ).clip(lower=0.01)

        return df

    def backfill_all_symbols(
        self,
        universe: set,
        conn,
        delay_seconds: float = 0.5,
    ) -> dict:
        """
        Fetches and stores corporate actions for all symbols in universe.
        Run once during Phase 0 setup.

        Args:
            universe       : Set of NSE symbols
            conn           : DB connection
            delay_seconds  : Delay between NSE requests (polite scraping)

        Returns:
            Summary dict with counts
        """
        session      = get_nse_session()
        success      = 0
        failed       = 0
        total_actions = 0

        logger.info(f"Starting corporate action backfill for {len(universe)} symbols...")

        for i, symbol in enumerate(sorted(universe), 1):
            try:
                actions = self.fetch_and_store_actions(symbol, conn, session)
                total_actions += len(actions)
                success += 1

                if i % 50 == 0:
                    logger.info(f"Progress: {i}/{len(universe)} symbols processed")
                    # Refresh NSE session periodically
                    session = get_nse_session()

                time.sleep(delay_seconds)

            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                failed += 1

        summary = {
            "total_symbols"  : len(universe),
            "success"        : success,
            "failed"         : failed,
            "total_actions"  : total_actions,
        }

        logger.success(
            f"Corporate action backfill complete: "
            f"{success} symbols, {total_actions} actions found, {failed} failed"
        )
        return summary

    # ── Private DB Methods ────────────────────────────────────────────────

    def _apply_split_to_db(
        self,
        symbol: str,
        ex_date: date,
        ratio: float,
        conn,
    ) -> int:
        """Applies split/bonus adjustment to daily_ohlcv rows before ex_date"""
        sql = """
            UPDATE daily_ohlcv SET
                open       = ROUND(open       / %s, 2),
                high       = ROUND(high       / %s, 2),
                low        = ROUND(low        / %s, 2),
                close      = ROUND(close      / %s, 2),
                prev_close = ROUND(prev_close / %s, 2),
                volume     = ROUND(volume     * %s)
            WHERE symbol = %s AND date < %s;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (ratio, ratio, ratio, ratio, ratio, ratio, symbol, ex_date))
            rows_updated = cur.rowcount
        conn.commit()

        logger.info(
            f"{symbol}: Applied split/bonus ratio={ratio} before {ex_date} "
            f"— {rows_updated} rows adjusted in DB"
        )
        return rows_updated

    def _apply_dividend_to_db(
        self,
        symbol: str,
        ex_date: date,
        amount: float,
        conn,
    ) -> int:
        """Applies dividend adjustment to daily_ohlcv rows before ex_date"""
        sql = """
            UPDATE daily_ohlcv SET
                open       = GREATEST(ROUND(open       - %s, 2), 0.01),
                high       = GREATEST(ROUND(high       - %s, 2), 0.01),
                low        = GREATEST(ROUND(low        - %s, 2), 0.01),
                close      = GREATEST(ROUND(close      - %s, 2), 0.01),
                prev_close = GREATEST(ROUND(prev_close - %s, 2), 0.01)
            WHERE symbol = %s AND date < %s;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (amount, amount, amount, amount, amount, symbol, ex_date))
            rows_updated = cur.rowcount
        conn.commit()

        logger.info(
            f"{symbol}: Applied dividend ₹{amount}/share before {ex_date} "
            f"— {rows_updated} rows adjusted"
        )
        return rows_updated


# ══════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════

def _parse_date(date_str: str) -> date | None:
    """Parses NSE date strings to Python date"""
    if not date_str:
        return None
    formats = ["%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%b %d, %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


# ══════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing CorporateActionAdjuster...")

    # Test ratio parser
    test_cases = [
        ("Bonus 1:1",                                    2.0),
        ("Bonus 1:2",                                    1.5),
        ("Stock Split 2:1",                              2.0),
        ("Face Value Split From Rs.10/- to Rs.5/-",      2.0),
        ("Sub Division of Share from Rs 10 to Rs 2",     5.0),
    ]

    all_passed = True
    for subject, expected in test_cases:
        result = parse_split_ratio(subject)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} '{subject}' → {result} (expected {expected})")

    print()

    # Test adjust_dataframe
    import numpy as np

    adjuster = CorporateActionAdjuster()

    # Simulate RELIANCE with a 1:1 bonus on 2022-05-01
    dates  = pd.date_range("2022-04-01", "2022-06-01", freq="B")
    prices = [2500.0] * 22 + [1250.0] * 23   # Price halved on bonus date

    df = pd.DataFrame({
        "date"  : dates,
        "open"  : prices,
        "high"  : [p * 1.01 for p in prices],
        "low"   : [p * 0.99 for p in prices],
        "close" : prices,
        "volume": [1_000_000] * len(dates),
    })

    actions = [{
        "symbol"     : "RELIANCE",
        "ex_date"    : date(2022, 5, 2),
        "action_type": "bonus",
        "ratio"      : 2.0,
        "amount"     : None,
    }]

    adjusted = adjuster.adjust_dataframe(df, actions)

    # All prices should now be ~1250 (pre-bonus prices halved)
    pre_bonus_close = adjusted[adjusted["date"] < pd.Timestamp("2022-05-02")]["close"]
    assert (pre_bonus_close - 1250.0).abs().max() < 1.0, "Price adjustment failed"
    print(f"  adjust_dataframe ✓ (pre-bonus prices correctly halved to ~₹1250)")

    if all_passed:
        print("\nAll CorporateActionAdjuster tests passed.")
    else:
        print("\nSome tests failed — check ratio parser logic.")