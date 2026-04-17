"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Pillar 2: MSI (Market Sentiment Index)         ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : features/msi.py                                        ║
║         Phase   : 1 — Feature Engineering                               ║
║                                                                          ║
║  What this pillar learns:                                                ║
║    Whether a stock is in an overbought or oversold condition —           ║
║    weighted by the actual volume of money flowing in and out.            ║
║    Unlike standard RSI (price-only), MSI incorporates:                  ║
║      - Volume-Weighted RSI  (how much money drives each move)           ║
║      - Money Flow Index     (typical price × volume pressure)            ║
║      - Delivery % Momentum  (NSE delivery data — smart money proxy)     ║
║      - Put-Call Ratio       (options market sentiment)                   ║
║                                                                          ║
║  MSI Formula:                                                            ║
║    MSI = 0.35×VRSI + 0.25×MFI + 0.25×Delivery_MOM + 0.15×PCR_score    ║
║    Range: 0 (extremely oversold) → 100 (extremely overbought)           ║
║                                                                          ║
║  Signal interpretation:                                                  ║
║    MSI 0  – 25  : Extremely oversold  → high-probability bounce zone    ║
║    MSI 25 – 40  : Oversold            → buy zone (with trend confirm)   ║
║    MSI 40 – 60  : Neutral             → no OB/OS edge                   ║
║    MSI 60 – 75  : Overbought          → caution on new longs            ║
║    MSI 75 – 100 : Extremely overbought→ sell/short zone                  ║
║                                                                          ║
║  Key differentiator vs standard RSI:                                     ║
║    MSI divergence from price = highest-conviction signal in this pillar  ║
║    (price makes new high but MSI makes lower high = distribution)        ║
║                                                                          ║
║  Output:                                                                 ║
║    msi_score    : float in [0, 100]  (composite OB/OS level)            ║
║    msi_signal   : float in [-1, +1]  (normalized for LSTM input)        ║
║    divergence   : int  (-1, 0, +1)   (bearish/none/bullish divergence)  ║
║                                                                          ║
║  Database:                                                               ║
║    Reads from : daily_ohlcv (OHLCV + delivery_pct)                      ║
║    Reads from : options_chain (PCR — populated in Phase 0 options file) ║
║    Writes to  : features_msi                                             ║
║                                                                          ║
║  Note on delivery_pct:                                                   ║
║    NULL for most historical dates (NSE doesn't serve old MTO files).    ║
║    Handled gracefully — delivery component weight redistributed when     ║
║    data is absent. System still produces valid MSI without it.           ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install pandas numpy psycopg2-binary pandas-ta loguru             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import psycopg2
import psycopg2.extras

from datetime import date, timedelta
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

load_dotenv()

# ── Database ───────────────────────────────────────────────────────────────
DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)

# ── MSI component parameters ──────────────────────────────────────────────
VRSI_PERIOD        = 14     # Volume-Weighted RSI period
MFI_PERIOD         = 14     # Money Flow Index period
DELIVERY_MOM_DAYS  = 5      # Delivery % rate-of-change window
PCR_SMOOTH         = 5      # PCR smoothing period (reduces noise)
DIVERGENCE_LOOKBACK= 14     # Bars to look back for divergence detection

# MSI component weights — must sum to 1.0
# When delivery_pct is NULL, its weight is redistributed to VRSI and MFI
W_VRSI     = 0.35
W_MFI      = 0.25
W_DELIVERY = 0.25
W_PCR      = 0.15

# Thresholds
MSI_OVERSOLD    = 30.0   # Below this = oversold
MSI_OVERBOUGHT  = 70.0   # Above this = overbought
MIN_BARS        = 60     # Minimum bars needed for reliable MSI


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_conn():
    return psycopg2.connect(DB_URL)


def _ensure_features_table(conn):
    """
    Creates features_msi table if it doesn't exist.
    Stores all MSI sub-components and the final MSI score.
    """
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS features_msi (
                date                DATE        NOT NULL,
                symbol              VARCHAR(20) NOT NULL,

                -- Sub-component values [0, 100]
                vrsi                NUMERIC(6,2),   -- Volume-Weighted RSI
                mfi                 NUMERIC(6,2),   -- Money Flow Index
                delivery_mom_score  NUMERIC(6,2),   -- Delivery momentum [0,100]
                pcr_score           NUMERIC(6,2),   -- Put-Call Ratio score [0,100]

                -- Intermediate values (for debugging / LSTM raw features)
                delivery_pct        NUMERIC(6,2),   -- Raw delivery % (may be NULL)
                delivery_5d_mom     NUMERIC(8,4),   -- 5-day rate of change of delivery %
                pcr_raw             NUMERIC(6,3),   -- Raw put-call ratio
                has_delivery        BOOLEAN,        -- whether delivery data was available
                has_pcr             BOOLEAN,        -- whether PCR data was available

                -- Effective weights used (change when data missing)
                w_vrsi_used         NUMERIC(4,2),
                w_mfi_used          NUMERIC(4,2),
                w_delivery_used     NUMERIC(4,2),
                w_pcr_used          NUMERIC(4,2),

                -- MSI divergence
                price_high_14d      NUMERIC(12,4),  -- 14-day price high
                msi_high_14d        NUMERIC(6,2),   -- 14-day MSI high
                divergence          SMALLINT,       -- -1 bearish, 0 none, +1 bullish

                -- Final outputs
                msi_score           NUMERIC(6,2),   -- [0, 100]
                msi_signal          NUMERIC(5,4),   -- [-1.0, +1.0] for LSTM

                PRIMARY KEY (date, symbol)
            );
        """)

        # Convert to TimescaleDB hypertable
        try:
            cur.execute("""
                        SELECT create_hypertable(
                            'features_msi', 'date',
                            if_not_exists => TRUE,
                            migrate_data  => TRUE
                        )
                    """)
        except Exception:
            pass  # already a hypertable or not supported — continue

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_msi_symbol
            ON features_msi (symbol, date DESC);
        """)

    conn.commit()
    logger.info("features_msi table ready.")


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════

def load_ohlcv_with_delivery(
    symbol: str,
    start_date: date,
    end_date: date,
    conn,
) -> pd.DataFrame:
    """
    Loads OHLCV + delivery_pct for a symbol from TimescaleDB.
    delivery_pct will be NULL for most historical dates — handled gracefully.

    Returns:
        DataFrame indexed by date with columns:
            open, high, low, close, volume, turnover, delivery_pct
    """
    # Fetch extra history for warm-up (MFI needs 14 bars, delivery mom needs 5)
    history_start = start_date - timedelta(days=MIN_BARS + 10)

    sql = """
        SELECT
            date, open, high, low, close,
            volume, turnover, delivery_pct
        FROM daily_ohlcv
        WHERE symbol = %s
          AND date BETWEEN %s AND %s
        ORDER BY date ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, history_start, end_date))
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close",
                 "volume", "turnover", "delivery_pct"]
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    numeric = ["open", "high", "low", "close", "volume", "turnover", "delivery_pct"]
    df[numeric] = df[numeric].astype(float)
    df = df.dropna(subset=["close"])

    return df


def load_pcr(
    symbol: str,
    start_date: date,
    end_date: date,
    conn,
) -> pd.Series:
    """
    Loads Put-Call Ratio (PCR) from options_chain table if available.
    PCR = total put OI / total call OI for the stock's nearest expiry.

    Returns:
        pd.Series indexed by date, or empty Series if table doesn't exist yet.
        Empty Series is handled gracefully — PCR weight redistributed.
    """
    history_start = start_date - timedelta(days=PCR_SMOOTH + 5)

    try:
        sql = """
            SELECT
                date,
                SUM(put_oi) / NULLIF(SUM(call_oi), 0) AS pcr
            FROM options_chain
            WHERE symbol = %s
              AND date BETWEEN %s AND %s
            GROUP BY date
            ORDER BY date ASC;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, history_start, end_date))
            rows = cur.fetchall()

        if not rows:
            return pd.Series(dtype=float)

        pcr = pd.Series(
            {row[0]: float(row[1]) for row in rows if row[1] is not None}
        )
        pcr.index = pd.to_datetime(pcr.index)
        return pcr

    except psycopg2.errors.UndefinedTable:
        # options_chain table doesn't exist yet — perfectly fine for Phase 1
        return pd.Series(dtype=float)
    except Exception as e:
        logger.warning(f"PCR load failed for {symbol}: {e} — proceeding without PCR")
        return pd.Series(dtype=float)


def load_all_symbols(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM daily_ohlcv ORDER BY symbol;")
        return [row[0] for row in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════
#  MSI COMPONENT COMPUTATIONS
# ══════════════════════════════════════════════════════════════════════════

def compute_vrsi(df: pd.DataFrame, period: int = VRSI_PERIOD) -> pd.Series:
    """
    Computes Volume-Weighted RSI (VRSI).

    Standard RSI treats every bar equally regardless of volume.
    VRSI weights each bar's gain/loss by its relative volume,
    so a 2% gain on 10× average volume counts far more than
    a 2% gain on 0.1× average volume.

    Method:
        1. Compute daily price change
        2. Separate into gains and losses
        3. Weight each by volume relative to rolling average volume
        4. Apply Wilder's smoothing (same as standard RSI)
        5. Compute RS = avg_weighted_gain / avg_weighted_loss → RSI formula

    Args:
        df     : OHLCV DataFrame with 'close' and 'volume'
        period : Lookback period (default 14)

    Returns:
        pd.Series of VRSI values in [0, 100]
    """
    close  = df["close"]
    volume = df["volume"].replace(0, np.nan)

    # Price change
    delta = close.diff()

    # Relative volume (volume / rolling mean volume)
    avg_vol = volume.rolling(window=period * 2, min_periods=period).mean()
    rel_vol = (volume / avg_vol).fillna(1.0).clip(0.1, 10.0)

    # Volume-weighted gains and losses
    gain = (delta.clip(lower=0) * rel_vol)
    loss = (-delta.clip(upper=0) * rel_vol)

    # Wilder's smoothing (same exponential method as standard RSI)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs   = avg_gain / avg_loss.replace(0, np.nan)
    vrsi = 100 - (100 / (1 + rs))

    return vrsi.clip(0, 100)


def compute_mfi(df: pd.DataFrame, period: int = MFI_PERIOD) -> pd.Series:
    """
    Computes Money Flow Index (MFI).

    MFI is often called "RSI with volume" but uses a different formula.
    It measures the ratio of positive to negative money flow
    where money flow = typical price × volume.

    Formula:
        Typical Price (TP) = (High + Low + Close) / 3
        Raw Money Flow     = TP × Volume
        Positive MF        = sum of RMF where TP > prev TP (14 bars)
        Negative MF        = sum of RMF where TP < prev TP (14 bars)
        MFI = 100 - (100 / (1 + Positive MF / Negative MF))

    Args:
        df     : OHLCV DataFrame
        period : Lookback period (default 14)

    Returns:
        pd.Series of MFI values in [0, 100]
    """
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan)
    rmf = tp * vol

    # Direction: positive flow when TP rises, negative when TP falls
    tp_change = tp.diff()
    pos_flow  = rmf.where(tp_change > 0, 0.0)
    neg_flow  = rmf.where(tp_change < 0, 0.0)

    pos_sum = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=period).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))

    return mfi.clip(0, 100)


def compute_delivery_momentum_score(
    delivery_pct: pd.Series,
    window: int = DELIVERY_MOM_DAYS,
) -> pd.Series:
    """
    Converts raw delivery percentage into an OB/OS score [0, 100].

    Delivery % = what fraction of traded volume resulted in actual
    delivery (demat transfer). High and rising delivery = smart money
    accumulating (bullish). Low and falling delivery = speculation
    and distribution (bearish).

    Logic:
        1. Compute 5-day rate of change of delivery %
        2. Normalize using rolling z-score to [0, 100]
        3. High score (>70) = delivery rising fast = bullish pressure
        4. Low score (<30) = delivery falling fast = bearish pressure

    Args:
        delivery_pct : Raw NSE delivery % series (may contain NaN)
        window       : Rate-of-change window (default 5 days)

    Returns:
        pd.Series of delivery momentum score in [0, 100]
        Returns all-50 (neutral) when delivery data unavailable
    """
    # Check if we have meaningful delivery data
    valid_count = delivery_pct.notna().sum()

    if valid_count < window + 2:
        # Not enough delivery data — return neutral score
        return pd.Series(50.0, index=delivery_pct.index)

    # 5-day rate of change
    mom = delivery_pct.pct_change(periods=window)

    # Rolling z-score normalization
    roll_mean = mom.rolling(window=60, min_periods=10).mean()
    roll_std  = mom.rolling(window=60, min_periods=10).std().replace(0, np.nan)
    z_score   = (mom - roll_mean) / roll_std

    # Map z-score to [0, 100] using sigmoid-like transform
    # z = +2 → score ≈ 95 (strong accumulation)
    # z =  0 → score = 50 (neutral)
    # z = -2 → score ≈  5 (strong distribution)
    score = 50 + (z_score.clip(-3, 3) / 3) * 47

    # Fill NaN with neutral 50 (not enough history or missing data)
    score = score.fillna(50.0).clip(0, 100)

    return score


def compute_pcr_score(pcr_series: pd.Series, smooth: int = PCR_SMOOTH) -> pd.Series:
    """
    Converts raw Put-Call Ratio into a contrarian OB/OS score [0, 100].

    PCR interpretation (contrarian):
        PCR < 0.7 : Too many calls = retail is very bullish = overbought risk
                    → Low score (bearish signal from this component)
        PCR > 1.3 : Too many puts = retail is very bearish = oversold opportunity
                    → High score (bullish signal from this component)
        PCR ~ 1.0 : Balanced = neutral

    Note: This is CONTRARIAN — high PCR (fear) → high score (buy signal).
    This is consistent with how MSI overall works (high score = oversold).

    Wait — correction. MSI high score = overbought. So:
        PCR < 0.7 (complacency/overbought) → high score (contributes to OB)
        PCR > 1.3 (fear/oversold)          → low score (contributes to OS)

    Args:
        pcr_series : Raw PCR values (put OI / call OI)
        smooth     : Smoothing period to reduce noise (default 5)

    Returns:
        pd.Series of PCR score in [0, 100]
        Returns empty Series if pcr_series is empty
    """
    if pcr_series.empty:
        return pd.Series(dtype=float)

    # Smooth to reduce single-day noise
    pcr_smooth = pcr_series.rolling(window=smooth, min_periods=1).mean()

    # Map to [0, 100]:
    # PCR = 0.5 (very bullish market) → score = 90 (overbought signal)
    # PCR = 1.0 (neutral)             → score = 50
    # PCR = 2.0 (very bearish market) → score = 10 (oversold signal)
    # Using inverse sigmoid: score = 100 - 50 × tanh((PCR - 1) × 1.5) + 50
    pcr_clipped = pcr_smooth.clip(0.3, 3.0)
    score = 50 - 45 * np.tanh((pcr_clipped - 1.0) * 1.5)

    return score.clip(0, 100)


# ══════════════════════════════════════════════════════════════════════════
#  MSI DIVERGENCE DETECTION
# ══════════════════════════════════════════════════════════════════════════

def compute_divergence(
    close: pd.Series,
    msi: pd.Series,
    lookback: int = DIVERGENCE_LOOKBACK,
) -> pd.Series:
    """
    Detects bullish and bearish divergences between price and MSI.

    Bearish divergence (-1):
        Price makes a new 14-day HIGH but MSI makes a LOWER high
        → distribution — smart money selling into strength

    Bullish divergence (+1):
        Price makes a new 14-day LOW but MSI makes a HIGHER low
        → accumulation — smart money buying into weakness

    No divergence (0):
        Price and MSI are in agreement

    This is the highest-conviction signal in Pillar 2.
    When trend (Pillar 1) confirms divergence, signal quality
    increases substantially.

    Args:
        close    : Close price series
        msi      : MSI score series [0, 100]
        lookback : Window for detecting new highs/lows (default 14)

    Returns:
        pd.Series of divergence values: -1, 0, or +1
    """
    n          = len(close)
    divergence = np.zeros(n, dtype=int)

    close_arr = close.values
    msi_arr   = msi.values

    for i in range(lookback, n):
        window_c = close_arr[i - lookback: i + 1]
        window_m = msi_arr[i - lookback: i + 1]

        if np.any(np.isnan(window_m)):
            continue

        price_is_new_high = close_arr[i] >= np.max(window_c[:-1])
        price_is_new_low  = close_arr[i] <= np.min(window_c[:-1])
        msi_lower_high    = msi_arr[i]   <  np.max(window_m[:-1])
        msi_higher_low    = msi_arr[i]   >  np.min(window_m[:-1])

        if price_is_new_high and msi_lower_high:
            divergence[i] = -1   # Bearish divergence
        elif price_is_new_low and msi_higher_low:
            divergence[i] = 1    # Bullish divergence

    return pd.Series(divergence, index=close.index)


# ══════════════════════════════════════════════════════════════════════════
#  MSI AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════

def compute_msi(
    df: pd.DataFrame,
    pcr_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Computes the complete MSI composite score for a stock.

    Aggregates all four sub-components with dynamic weight adjustment
    when data is missing (delivery_pct NULL or no options chain).

    MSI Score → MSI Signal mapping:
        MSI [0, 100] → Signal [-1.0, +1.0]
        MSI =   0 → Signal = +1.0  (max oversold = strong buy signal)
        MSI =  50 → Signal =  0.0  (neutral)
        MSI = 100 → Signal = -1.0  (max overbought = strong sell signal)

    Args:
        df         : OHLCV DataFrame with delivery_pct column
        pcr_series : Optional PCR pd.Series indexed by date

    Returns:
        DataFrame with all MSI components and final msi_score, msi_signal
    """
    result = df.copy()

    # ── Component 1: VRSI ─────────────────────────────────────────────────
    result["vrsi"] = compute_vrsi(df)

    # ── Component 2: MFI ─────────────────────────────────────────────────
    result["mfi"] = compute_mfi(df)

    # ── Component 3: Delivery Momentum ───────────────────────────────────
    delivery_available = "delivery_pct" in df.columns and \
                         df["delivery_pct"].notna().sum() > DELIVERY_MOM_DAYS + 2

    if delivery_available:
        result["delivery_mom_score"] = compute_delivery_momentum_score(
            df["delivery_pct"]
        )
        result["delivery_pct_raw"]   = df["delivery_pct"]
        result["has_delivery"]       = True
    else:
        result["delivery_mom_score"] = 50.0   # neutral — no data
        result["delivery_pct_raw"]   = np.nan
        result["has_delivery"]       = False

    # ── Component 4: PCR ──────────────────────────────────────────────────
    pcr_score_series = pd.Series(dtype=float)
    pcr_raw_series   = pd.Series(dtype=float)

    if pcr_series is not None and not pcr_series.empty:
        pcr_score_series = compute_pcr_score(pcr_series)
        pcr_raw_series   = pcr_series
        result["has_pcr"] = True
    else:
        result["has_pcr"] = False

    # Align PCR to df index
    if not pcr_score_series.empty:
        result["pcr_score"] = pcr_score_series.reindex(result.index)
        result["pcr_raw"]   = pcr_raw_series.reindex(result.index)
    else:
        result["pcr_score"] = np.nan
        result["pcr_raw"]   = np.nan

    # ── Dynamic weight adjustment ─────────────────────────────────────────
    # When data is missing, redistribute its weight proportionally
    # to the remaining components
    w_vrsi     = W_VRSI
    w_mfi      = W_MFI
    w_delivery = W_DELIVERY if delivery_available else 0.0
    w_pcr      = W_PCR      if not pcr_score_series.empty else 0.0

    total_w = w_vrsi + w_mfi + w_delivery + w_pcr
    if total_w == 0:
        total_w = 1.0  # safety guard

    # Normalize weights to sum to 1.0
    w_vrsi     /= total_w
    w_mfi      /= total_w
    w_delivery /= total_w
    w_pcr      /= total_w

    result["w_vrsi_used"]     = round(w_vrsi, 3)
    result["w_mfi_used"]      = round(w_mfi, 3)
    result["w_delivery_used"] = round(w_delivery, 3)
    result["w_pcr_used"]      = round(w_pcr, 3)

    # ── Composite MSI Score ───────────────────────────────────────────────
    msi = (
        result["vrsi"]               * w_vrsi +
        result["mfi"]                * w_mfi  +
        result["delivery_mom_score"] * w_delivery
    )

    if not pcr_score_series.empty:
        pcr_aligned = result["pcr_score"].fillna(50.0)
        msi += pcr_aligned * w_pcr

    result["msi_score"] = msi.clip(0, 100)

    # ── MSI Signal: map [0,100] → [-1, +1] (inverted) ─────────────────
    # MSI = 0  (oversold)   → signal = +1.0 (bullish)
    # MSI = 50 (neutral)    → signal =  0.0
    # MSI = 100 (overbought)→ signal = -1.0 (bearish)
    result["msi_signal"] = ((50.0 - result["msi_score"]) / 50.0).clip(-1.0, 1.0)

    # ── Divergence ────────────────────────────────────────────────────────
    result["divergence"] = compute_divergence(df["close"], result["msi_score"])

    # ── 14-day highs for divergence context ──────────────────────────────
    result["price_high_14d"] = df["close"].rolling(DIVERGENCE_LOOKBACK).max()
    result["msi_high_14d"]   = result["msi_score"].rolling(DIVERGENCE_LOOKBACK).max()

    return result


# ══════════════════════════════════════════════════════════════════════════
#  MAIN FEATURE EXTRACTOR CLASS
# ══════════════════════════════════════════════════════════════════════════

class MSIExtractor:
    """
    Main interface for Pillar 2 — Market Sentiment Index.

    Usage:
        extractor = MSIExtractor()

        # Single stock
        result = extractor.compute("RELIANCE", date(2024, 1, 1), date(2024, 12, 31))

        # All stocks (Phase 1 full run)
        extractor.run_all(end_date=date(2024, 12, 31))

        # Latest score for signal engine
        score = extractor.get_latest_score("TCS")
    """

    def __init__(self):
        self.conn = _get_conn()
        _ensure_features_table(self.conn)

    def compute(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        save_to_db: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Computes MSI features for a single symbol over a date range.

        Gracefully handles:
            - NULL delivery_pct (redistributes weight to VRSI + MFI)
            - Missing options_chain table (proceeds without PCR)
            - Insufficient history (returns None with warning)
        """
        df = load_ohlcv_with_delivery(symbol, start_date, end_date, self.conn)

        if df.empty:
            logger.warning(f"{symbol}: No OHLCV data, skipping.")
            return None

        if len(df) < MIN_BARS:
            logger.warning(f"{symbol}: Only {len(df)} bars, need {MIN_BARS}. Skipping.")
            return None

        # Load PCR (may return empty Series — handled gracefully)
        pcr = load_pcr(symbol, start_date, end_date, self.conn)

        # Compute MSI
        result = compute_msi(df, pcr_series=pcr if not pcr.empty else None)

        # Trim to requested date range (remove warm-up period)
        result = result[result.index >= pd.Timestamp(start_date)].copy()

        if result.empty:
            return None

        if save_to_db:
            self._save(symbol, result)

        return result

    def _save(self, symbol: str, df: pd.DataFrame):
        """Upserts MSI features into features_msi. Safe to re-run."""
        self.conn.rollback()
        records = []
        for ts, row in df.iterrows():
            def _val(col, default=None):
                v = row.get(col, default)
                if v is None:
                    return None
                try:
                    if pd.isna(v):
                        return default
                except (TypeError, ValueError):
                    pass
                if isinstance(v, (np.integer,)):
                    return int(v)
                if isinstance(v, (np.floating,)):
                    return None if np.isnan(v) else float(v)
                if isinstance(v, (np.bool_,)):
                    return bool(v)
                return v

            records.append({
                "date"               : ts.date(),
                "symbol"             : symbol,
                "vrsi"               : _val("vrsi"),
                "mfi"                : _val("mfi"),
                "delivery_mom_score" : _val("delivery_mom_score"),
                "pcr_score"          : _val("pcr_score"),
                "delivery_pct"       : _val("delivery_pct_raw"),
                "delivery_5d_mom"    : _val("delivery_5d_mom"),
                "pcr_raw"            : _val("pcr_raw"),
                "has_delivery"       : _val("has_delivery", False),
                "has_pcr"            : _val("has_pcr", False),
                "w_vrsi_used"        : _val("w_vrsi_used"),
                "w_mfi_used"         : _val("w_mfi_used"),
                "w_delivery_used"    : _val("w_delivery_used"),
                "w_pcr_used"         : _val("w_pcr_used"),
                "price_high_14d"     : _val("price_high_14d"),
                "msi_high_14d"       : _val("msi_high_14d"),
                "divergence"         : _val("divergence", 0),
                "msi_score"          : _val("msi_score"),
                "msi_signal"         : _val("msi_signal"),
            })

        if not records:
            return

        insert_sql = """
            INSERT INTO features_msi (
                date, symbol,
                vrsi, mfi, delivery_mom_score, pcr_score,
                delivery_pct, delivery_5d_mom, pcr_raw,
                has_delivery, has_pcr,
                w_vrsi_used, w_mfi_used, w_delivery_used, w_pcr_used,
                price_high_14d, msi_high_14d, divergence,
                msi_score, msi_signal
            ) VALUES (
                %(date)s, %(symbol)s,
                %(vrsi)s, %(mfi)s, %(delivery_mom_score)s, %(pcr_score)s,
                %(delivery_pct)s, %(delivery_5d_mom)s, %(pcr_raw)s,
                %(has_delivery)s, %(has_pcr)s,
                %(w_vrsi_used)s, %(w_mfi_used)s, %(w_delivery_used)s, %(w_pcr_used)s,
                %(price_high_14d)s, %(msi_high_14d)s, %(divergence)s,
                %(msi_score)s, %(msi_signal)s
            )
            ON CONFLICT (date, symbol) DO UPDATE SET
                msi_score           = EXCLUDED.msi_score,
                msi_signal          = EXCLUDED.msi_signal,
                vrsi                = EXCLUDED.vrsi,
                mfi                 = EXCLUDED.mfi,
                delivery_mom_score  = EXCLUDED.delivery_mom_score,
                divergence          = EXCLUDED.divergence,
                has_delivery        = EXCLUDED.has_delivery,
                has_pcr             = EXCLUDED.has_pcr;
        """

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=500)
            self.conn.commit()
            logger.success(f"{symbol}: {len(records)} MSI rows saved.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"{symbol}: MSI DB save failed — {e}")
            raise

    def run_all(
        self,
        end_date: Optional[date] = None,
        start_date: Optional[date] = None,
        symbols: Optional[list[str]] = None,
    ):
        """
        Computes MSI for all symbols. Same interface as TrendExtractor.run_all().
        """
        if end_date   is None: end_date   = date.today()
        if start_date is None: start_date = date(2019, 1, 1)
        if symbols    is None: symbols    = load_all_symbols(self.conn)

        logger.info(
            f"MSIExtractor.run_all: {len(symbols)} symbols | "
            f"{start_date} → {end_date}"
        )

        success, failed, skipped = 0, 0, 0

        for i, symbol in enumerate(symbols, 1):
            try:
                result = self.compute(symbol, start_date, end_date, save_to_db=True)
                if result is not None: success += 1
                else:                  skipped += 1
            except Exception as e:
                logger.error(f"[{i}/{len(symbols)}] {symbol} failed: {e}")
                failed += 1

            if i % 50 == 0:
                logger.info(
                    f"Progress: {i}/{len(symbols)} | "
                    f"Success: {success} | Skipped: {skipped} | Failed: {failed}"
                )

        logger.info(
            f"MSI run_all complete — "
            f"Success: {success} | Skipped: {skipped} | Failed: {failed}"
        )

    def get_latest_score(self, symbol: str) -> Optional[dict]:
        """
        Returns latest MSI score and signal for a symbol.
        Used by signal engine during live trading.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT msi_score, msi_signal, divergence, has_delivery, has_pcr
                FROM features_msi
                WHERE symbol = %s
                ORDER BY date DESC LIMIT 1;
            """, (symbol,))
            row = cur.fetchone()

        if not row:
            return None
        return {
            "msi_score"   : float(row[0]),
            "msi_signal"  : float(row[1]),
            "divergence"  : int(row[2]),
            "has_delivery": bool(row[3]),
            "has_pcr"     : bool(row[4]),
        }

    def get_scores_for_date(self, target_date: date) -> pd.DataFrame:
        """
        Returns MSI scores for ALL symbols on a specific date.
        Sorted by msi_signal descending (most oversold = best buy candidates first).
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, msi_score, msi_signal, divergence
                FROM features_msi
                WHERE date = %s
                ORDER BY msi_signal DESC;
            """, (target_date,))
            rows = cur.fetchall()

        return pd.DataFrame(
            rows, columns=["symbol", "msi_score", "msi_signal", "divergence"]
        )

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


# ══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — MSI Feature Extractor")
    parser.add_argument("--mode",   choices=["all", "single", "score"], default="all")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--start",  type=str, default="2019-01-01")
    parser.add_argument("--end",    type=str, default=str(date.today()))
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    with MSIExtractor() as extractor:
        if args.mode == "all":
            extractor.run_all(start_date=start, end_date=end)
        elif args.mode == "single":
            if not args.symbol:
                print("--symbol required"); sys.exit(1)
            df = extractor.compute(args.symbol, start, end)
            if df is not None:
                print(df[["msi_score", "msi_signal", "vrsi",
                           "mfi", "divergence"]].tail(10))
        elif args.mode == "score":
            if not args.symbol:
                print("--symbol required"); sys.exit(1)
            score = extractor.get_latest_score(args.symbol)
            print(f"{args.symbol} latest MSI: {score}")


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run: python -m pytest features/msi.py -v
# ══════════════════════════════════════════════════════════════════════════

def _make_sample_ohlcv(n: int = 200, trend: str = "up") -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    if trend == "up":
        close = 1000 + np.cumsum(np.random.randn(n) * 5 + 0.4)
    elif trend == "down":
        close = 1000 + np.cumsum(np.random.randn(n) * 5 - 0.4)
    else:
        close = 1000 + np.cumsum(np.random.randn(n) * 3)

    close  = np.maximum(close, 10)
    high   = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low    = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_  = close * (1 + np.random.randn(n) * 0.005)
    vol    = np.random.randint(500_000, 10_000_000, n).astype(float)

    # Simulate delivery_pct: available for last 30 bars only (like real NSE data)
    delivery = np.full(n, np.nan)
    delivery[-30:] = np.random.uniform(20, 80, 30)

    return pd.DataFrame({
        "open"        : open_,
        "high"        : high,
        "low"         : low,
        "close"       : close,
        "volume"      : vol,
        "turnover"    : close * vol,
        "delivery_pct": delivery,
    }, index=dates)


class TestMSIFeatures:

    def setup_method(self):
        self.df_up   = _make_sample_ohlcv(200, "up")
        self.df_down = _make_sample_ohlcv(200, "down")
        self.df_side = _make_sample_ohlcv(200, "sideways")

    # ── VRSI Tests ────────────────────────────────────────────────────────

    def test_vrsi_range(self):
        vrsi = compute_vrsi(self.df_up)
        valid = vrsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), \
            "VRSI must be in [0, 100]"

    def test_vrsi_uptrend_high(self):
        """Uptrend VRSI should be mostly above 50."""
        vrsi = compute_vrsi(self.df_up).dropna()
        pct_above = (vrsi > 50).mean()
        assert pct_above > 0.55, \
            f"Uptrend VRSI should be >50 most of the time, got {pct_above:.1%}"

    def test_vrsi_downtrend_low(self):
        """Downtrend VRSI should be mostly below 50."""
        vrsi = compute_vrsi(self.df_down).dropna()
        pct_below = (vrsi < 50).mean()
        assert pct_below > 0.55, \
            f"Downtrend VRSI should be <50 most of the time, got {pct_below:.1%}"

    def test_vrsi_no_volume_zero_crash(self):
        """VRSI should not crash when volume contains zeros."""
        df = self.df_up.copy()
        df.loc[df.index[:10], "volume"] = 0
        vrsi = compute_vrsi(df)
        assert vrsi.notna().sum() > 100, "VRSI should still produce values with zero volumes"

    # ── MFI Tests ─────────────────────────────────────────────────────────

    def test_mfi_range(self):
        mfi = compute_mfi(self.df_up)
        valid = mfi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), \
            "MFI must be in [0, 100]"

    def test_mfi_uptrend_bias(self):
        """MFI in uptrend should trend above 50."""
        mfi = compute_mfi(self.df_up).dropna()
        assert (mfi > 50).mean() > 0.50

    def test_mfi_columns_not_nan_after_warmup(self):
        """After 14-bar warm-up, MFI should have no NaN."""
        mfi = compute_mfi(self.df_up)
        post_warmup = mfi.iloc[MFI_PERIOD + 2:]
        nan_count = post_warmup.isna().sum()
        assert nan_count == 0, f"MFI has {nan_count} NaN after warm-up"

    # ── Delivery Momentum Tests ───────────────────────────────────────────

    def test_delivery_mom_neutral_when_no_data(self):
        """When delivery_pct is all NaN, score should be 50 (neutral)."""
        delivery = pd.Series(np.nan, index=self.df_up.index)
        score = compute_delivery_momentum_score(delivery)
        assert (score == 50.0).all(), "Should return 50 when no delivery data"

    def test_delivery_mom_range(self):
        """Delivery momentum score must be in [0, 100]."""
        delivery = self.df_up["delivery_pct"]
        score = compute_delivery_momentum_score(delivery)
        assert (score >= 0).all() and (score <= 100).all()

    def test_delivery_mom_length_matches_input(self):
        delivery = self.df_up["delivery_pct"]
        score = compute_delivery_momentum_score(delivery)
        assert len(score) == len(delivery)

    # ── PCR Score Tests ───────────────────────────────────────────────────

    def test_pcr_empty_returns_empty(self):
        result = compute_pcr_score(pd.Series(dtype=float))
        assert result.empty

    def test_pcr_range(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        pcr = pd.Series(np.random.uniform(0.5, 2.0, 100), index=dates)
        score = compute_pcr_score(pcr)
        assert (score >= 0).all() and (score <= 100).all()

    def test_pcr_high_pcr_gives_low_score(self):
        """High PCR (fear/oversold) → low PCR score (NOT overbought)."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        pcr_high = pd.Series(2.0, index=dates)   # extreme puts
        score = compute_pcr_score(pcr_high)
        assert score.mean() < 30, f"High PCR should give low score, got {score.mean():.1f}"

    def test_pcr_low_pcr_gives_high_score(self):
        """Low PCR (complacency/overbought) → high PCR score."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        pcr_low = pd.Series(0.4, index=dates)    # extreme calls
        score = compute_pcr_score(pcr_low)
        assert score.mean() > 70, f"Low PCR should give high score, got {score.mean():.1f}"

    # ── MSI Composite Tests ───────────────────────────────────────────────

    def test_msi_score_range(self):
        """MSI score must always be in [0, 100]."""
        result = compute_msi(self.df_up)
        valid = result["msi_score"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_msi_signal_range(self):
        """MSI signal must always be in [-1.0, +1.0]."""
        result = compute_msi(self.df_up)
        valid = result["msi_signal"].dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_msi_signal_inverted(self):
        """MSI signal should be negative when msi_score > 50 and vice versa."""
        result = compute_msi(self.df_up).dropna(subset=["msi_score"])
        ob_rows = result[result["msi_score"] > 60]
        if len(ob_rows) > 5:
            assert (ob_rows["msi_signal"] < 0).all(), \
                "Overbought MSI score should give negative msi_signal"

    def test_msi_no_delivery_graceful(self):
        """MSI should work fine with all-NaN delivery_pct."""
        df = self.df_up.copy()
        df["delivery_pct"] = np.nan
        result = compute_msi(df)
        assert result["msi_score"].notna().sum() > 100
        assert (result["has_delivery"] == False).all()

    def test_msi_with_pcr(self):
        """MSI should incorporate PCR when provided."""
        pcr = pd.Series(
            np.random.uniform(0.7, 1.5, len(self.df_up)),
            index=self.df_up.index
        )
        result_no_pcr   = compute_msi(self.df_up.copy())
        result_with_pcr = compute_msi(self.df_up.copy(), pcr_series=pcr)

        # Scores should differ when PCR is included
        diff = (result_with_pcr["msi_score"] - result_no_pcr["msi_score"]).abs()
        assert diff.mean() > 0.01, "PCR should change MSI scores"
        assert (result_with_pcr["has_pcr"] == True).all()

    def test_msi_weights_sum_to_one(self):
        """Effective weights must always sum to 1.0."""
        result = compute_msi(self.df_up)
        w_sum = (
            result["w_vrsi_used"] +
            result["w_mfi_used"] +
            result["w_delivery_used"] +
            result["w_pcr_used"]
        )
        assert (w_sum - 1.0).abs().max() < 0.01, \
            f"Weights don't sum to 1.0: {w_sum.unique()}"

    # ── Divergence Tests ──────────────────────────────────────────────────

    def test_divergence_valid_values(self):
        """Divergence must only be -1, 0, or +1."""
        result = compute_msi(self.df_up)
        valid = {-1, 0, 1}
        assert set(result["divergence"].unique()).issubset(valid)

    def test_divergence_not_all_zero(self):
        """Should detect at least some divergences over 200 bars."""
        result = compute_msi(self.df_side)
        non_zero = (result["divergence"] != 0).sum()
        assert non_zero > 0, "Expected at least 1 divergence signal over 200 bars"

    def test_divergence_length_matches(self):
        result = compute_msi(self.df_up)
        assert len(result["divergence"]) == len(self.df_up)


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))