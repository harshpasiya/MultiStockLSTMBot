"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Pillar 1: Trend Analysis                       ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : features/trend.py                                      ║
║         Phase   : 1 — Feature Engineering                               ║
║                                                                          ║
║  What this pillar learns:                                                ║
║    The direction, strength, and maturity of price trends across          ║
║    multiple timeframes simultaneously. It answers three questions:       ║
║      1. Which direction is the stock trending? (up / down / sideways)   ║
║      2. How strong is that trend? (weak / moderate / strong)             ║
║      3. Is the trend mature (about to end) or young (room to run)?      ║
║                                                                          ║
║  Features computed:                                                      ║
║    EMA Ribbon       → 9, 21, 50, 100, 200-period EMAs + alignments      ║
║    ADX(14)          → trend strength + directional indicators (+DI/-DI)  ║
║    Supertrend(7,3)  → ATR-based dynamic trend direction                  ║
║    VWAP Deviation   → price distance from daily VWAP (%)                ║
║    Higher Highs/Lows→ structural swing-point based trend detection       ║
║    Ichimoku Cloud   → Tenkan, Kijun, Senkou A/B, Chikou                 ║
║                                                                          ║
║  Output:                                                                 ║
║    trend_score : float in [-1.0, +1.0]                                  ║
║                  +1.0 = strongest possible uptrend                       ║
║                  -1.0 = strongest possible downtrend                     ║
║                   0.0 = no trend / sideways                              ║
║    raw_features : dict of all intermediate values (for LSTM input)       ║
║                                                                          ║
║  Database:                                                               ║
║    Reads from : daily_ohlcv (TimescaleDB, port 5433)                    ║
║    Writes to  : features_trend (created by this module)                  ║
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

# Suppress pandas_ta warnings about deprecated numpy types
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

load_dotenv()

# ── Database connection string ────────────────────────────────────────────
# TimescaleDB is on port 5433 (5432 is taken by local PostgreSQL)
DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)

# ── Feature engineering constants ─────────────────────────────────────────
EMA_PERIODS       = [9, 21, 50, 100, 200]   # EMA ribbon periods
ADX_PERIOD        = 14                        # ADX lookback
SUPERTREND_PERIOD = 7                         # Supertrend ATR period
SUPERTREND_MULT   = 3.0                       # Supertrend ATR multiplier
ICHIMOKU_FAST     = 9                         # Tenkan-sen period
ICHIMOKU_MED      = 26                        # Kijun-sen period
ICHIMOKU_SLOW     = 52                        # Senkou Span B period
SWING_LOOKBACK    = 5                         # bars each side for swing detection
MIN_BARS_REQUIRED = 210                       # need 200 EMA + buffer


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_conn():
    """Returns a psycopg2 connection to TimescaleDB."""
    return psycopg2.connect(DB_URL)


def _ensure_features_table(conn):
    """
    Creates the features_trend table in TimescaleDB if it doesn't exist.
    This table stores all computed trend features + the final trend_score.
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS features_trend (
                date              DATE        NOT NULL,
                symbol            VARCHAR(20) NOT NULL,

                -- EMA values (raw prices)
                ema_9             NUMERIC(12,4),
                ema_21            NUMERIC(12,4),
                ema_50            NUMERIC(12,4),
                ema_100           NUMERIC(12,4),
                ema_200           NUMERIC(12,4),

                -- EMA ribbon signals (normalized distances)
                ema_9_21_gap      NUMERIC(8,6),   -- (EMA9 - EMA21) / close
                ema_21_50_gap     NUMERIC(8,6),
                ema_50_200_gap    NUMERIC(8,6),
                ribbon_aligned_up   BOOLEAN,      -- all EMAs in bullish order
                ribbon_aligned_dn   BOOLEAN,      -- all EMAs in bearish order
                price_vs_ema200   NUMERIC(8,6),   -- (close - EMA200) / EMA200

                -- ADX features
                adx               NUMERIC(6,2),
                plus_di           NUMERIC(6,2),
                minus_di          NUMERIC(6,2),
                di_crossover      SMALLINT,       -- +1 bull cross, -1 bear cross, 0 none
                adx_trending      BOOLEAN,        -- ADX > 25

                -- Supertrend
                supertrend_dir    SMALLINT,       -- +1 uptrend, -1 downtrend
                supertrend_val    NUMERIC(12,4),  -- supertrend line value
                price_vs_st       NUMERIC(8,6),   -- (close - supertrend) / close

                -- VWAP deviation (intraday proxy using daily OHLCV)
                typical_price     NUMERIC(12,4),  -- (H+L+C)/3
                vwap_20d          NUMERIC(12,4),  -- 20-day VWAP
                vwap_deviation    NUMERIC(8,6),   -- (close - vwap) / vwap

                -- Higher Highs / Higher Lows structure
                swing_structure   SMALLINT,       -- +1 HH+HL, -1 LH+LL, 0 mixed
                last_swing_high   NUMERIC(12,4),
                last_swing_low    NUMERIC(12,4),

                -- Ichimoku Cloud
                tenkan            NUMERIC(12,4),
                kijun             NUMERIC(12,4),
                senkou_a          NUMERIC(12,4),
                senkou_b          NUMERIC(12,4),
                chikou_above      BOOLEAN,        -- chikou above price 26 bars ago
                price_above_cloud BOOLEAN,        -- close above both senkou lines
                tk_cross          SMALLINT,       -- +1 bull TK cross, -1 bear, 0 none

                -- Final output
                trend_score       NUMERIC(5,4),   -- [-1.0000, +1.0000]

                PRIMARY KEY (date, symbol)
            );
        """)

        # Hypertable for fast time-range queries
        cur.execute("""
            SELECT create_hypertable(
                'features_trend', 'date',
                if_not_exists => TRUE,
                migrate_data  => TRUE
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_trend_symbol
            ON features_trend (symbol, date DESC);
        """)

    conn.commit()
    logger.info("features_trend table ready.")


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_ohlcv(
    symbol: str,
    start_date: date,
    end_date: date,
    conn,
) -> pd.DataFrame:
    """
    Loads daily OHLCV data for a single symbol from TimescaleDB.

    Args:
        symbol     : NSE symbol e.g. 'RELIANCE'
        start_date : Fetch from this date (include enough history for EMA200)
        end_date   : Fetch up to this date
        conn       : psycopg2 connection

    Returns:
        DataFrame indexed by date with columns:
            open, high, low, close, prev_close, volume, turnover
        Empty DataFrame if symbol not found or insufficient data.
    """
    sql = """
        SELECT
            date, open, high, low, close,
            prev_close, volume, turnover
        FROM daily_ohlcv
        WHERE symbol    = %s
          AND date BETWEEN %s AND %s
        ORDER BY date ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, start_date, end_date))
        rows = cur.fetchall()

    if not rows:
        logger.warning(f"No OHLCV data found for {symbol} between {start_date} and {end_date}")
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close",
                 "prev_close", "volume", "turnover"]
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Convert to float (stored as NUMERIC in DB)
    numeric_cols = ["open", "high", "low", "close", "prev_close", "volume", "turnover"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Drop rows with null close (data quality guard)
    df = df.dropna(subset=["close"])

    return df


def load_all_symbols(conn) -> list[str]:
    """Returns list of all distinct symbols in daily_ohlcv."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM daily_ohlcv ORDER BY symbol;")
        return [row[0] for row in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE COMPUTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def compute_ema_ribbon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes EMA ribbon: 9, 21, 50, 100, 200 period EMAs.

    Also computes:
        - Gap between adjacent EMAs (normalized by close price)
        - Ribbon alignment flags (all bullish / all bearish order)
        - Price distance above/below EMA200

    Args:
        df : OHLCV DataFrame with 'close' column

    Returns:
        DataFrame with EMA columns added in-place
    """
    close = df["close"]

    for period in EMA_PERIODS:
        df[f"ema_{period}"] = ta.ema(close, length=period)

    # Normalized gaps between adjacent EMAs
    # Positive gap = faster EMA above slower EMA = bullish alignment
    df["ema_9_21_gap"]  = (df["ema_9"]  - df["ema_21"])  / close
    df["ema_21_50_gap"] = (df["ema_21"] - df["ema_50"])  / close
    df["ema_50_200_gap"]= (df["ema_50"] - df["ema_200"]) / close

    # Ribbon fully aligned bullish: EMA9 > EMA21 > EMA50 > EMA100 > EMA200
    df["ribbon_aligned_up"] = (
        (df["ema_9"]   > df["ema_21"]) &
        (df["ema_21"]  > df["ema_50"]) &
        (df["ema_50"]  > df["ema_100"]) &
        (df["ema_100"] > df["ema_200"])
    )

    # Ribbon fully aligned bearish: EMA9 < EMA21 < EMA50 < EMA100 < EMA200
    df["ribbon_aligned_dn"] = (
        (df["ema_9"]   < df["ema_21"]) &
        (df["ema_21"]  < df["ema_50"]) &
        (df["ema_50"]  < df["ema_100"]) &
        (df["ema_100"] < df["ema_200"])
    )

    # Price position relative to EMA200
    # Positive = price above 200 EMA = long-term uptrend
    df["price_vs_ema200"] = (close - df["ema_200"]) / df["ema_200"]

    return df


def compute_adx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes ADX(14) with +DI and -DI directional indicators.

    ADX interpretation:
        ADX < 20  : No trend (ranging market — signals less reliable)
        ADX 20–25 : Weak trend developing
        ADX 25–40 : Moderate trend (sweet spot for trend-following)
        ADX > 40  : Strong trend (often near exhaustion)

    Also detects +DI/-DI crossovers which signal trend direction changes.

    Args:
        df : OHLCV DataFrame

    Returns:
        DataFrame with adx, plus_di, minus_di, di_crossover, adx_trending columns
    """
    adx_result = ta.adx(df["high"], df["low"], df["close"], length=ADX_PERIOD)

    if adx_result is None or adx_result.empty:
        df["adx"]      = np.nan
        df["plus_di"]  = np.nan
        df["minus_di"] = np.nan
    else:
        # pandas_ta returns columns like ADX_14, DMP_14, DMN_14
        adx_col  = [c for c in adx_result.columns if c.startswith("ADX_")]
        dmp_col  = [c for c in adx_result.columns if c.startswith("DMP_")]
        dmn_col  = [c for c in adx_result.columns if c.startswith("DMN_")]

        df["adx"]      = adx_result[adx_col[0]].values  if adx_col  else np.nan
        df["plus_di"]  = adx_result[dmp_col[0]].values  if dmp_col  else np.nan
        df["minus_di"] = adx_result[dmn_col[0]].values  if dmn_col  else np.nan

    # ADX trending flag: True when ADX > 25
    df["adx_trending"] = df["adx"] > 25

    # Detect +DI / -DI crossovers
    # +1 when +DI crosses above -DI (bullish), -1 when crosses below (bearish)
    prev_plus  = df["plus_di"].shift(1)
    prev_minus = df["minus_di"].shift(1)

    bull_cross = (df["plus_di"] > df["minus_di"]) & (prev_plus <= prev_minus)
    bear_cross = (df["plus_di"] < df["minus_di"]) & (prev_plus >= prev_minus)

    df["di_crossover"] = 0
    df.loc[bull_cross, "di_crossover"] = 1
    df.loc[bear_cross, "di_crossover"] = -1

    return df


def compute_supertrend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Supertrend indicator (period=7, multiplier=3.0).

    Supertrend is an ATR-based dynamic support/resistance line that
    flips direction when price breaks through it. It gives a clean
    binary trend direction signal that is harder to whipsaw than EMAs.

    Supertrend direction:
        +1 = uptrend  (price above supertrend line)
        -1 = downtrend (price below supertrend line)

    Args:
        df : OHLCV DataFrame

    Returns:
        DataFrame with supertrend_dir, supertrend_val, price_vs_st columns
    """
    st_result = ta.supertrend(
        df["high"], df["low"], df["close"],
        length=SUPERTREND_PERIOD,
        multiplier=SUPERTREND_MULT
    )

    if st_result is None or st_result.empty:
        df["supertrend_dir"] = 0
        df["supertrend_val"] = np.nan
        df["price_vs_st"]    = np.nan
        return df

    # pandas_ta supertrend returns:
    #   SUPERT_7_3.0       = the supertrend line value
    #   SUPERTd_7_3.0      = direction (1 = up, -1 = down)
    dir_col = [c for c in st_result.columns if "SUPERTd" in c]
    val_col = [c for c in st_result.columns if c.startswith("SUPERT_")]

    if dir_col:
        raw = st_result[dir_col[0]]
        # pandas_ta fills warm-up bars with NaN — convert via float
        # to avoid the int64 sentinel (-9223372036854775808) bug
        df["supertrend_dir"] = (
            raw.astype(float)
            .apply(lambda x: int(x) if pd.notna(x) else 0)
        )
    else:
        df["supertrend_dir"] = 0

    if val_col:
        df["supertrend_val"] = st_result[val_col[0]].values
        # Normalized distance from price to supertrend line
        df["price_vs_st"] = (df["close"] - df["supertrend_val"]) / df["close"]
    else:
        df["supertrend_val"] = np.nan
        df["price_vs_st"]    = np.nan

    return df


def compute_vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Computes a 20-day rolling VWAP and the price's deviation from it.

    Note: True intraday VWAP requires tick data. This is a daily-bar
    approximation using typical price × volume, which is a valid and
    widely-used proxy for identifying mean-reversion setups on daily charts.

    Typical Price = (High + Low + Close) / 3
    VWAP_20d = rolling_sum(TP × Volume, 20) / rolling_sum(Volume, 20)
    VWAP_Deviation = (Close - VWAP) / VWAP

    Interpretation:
        Deviation > +0.03  : Price significantly above VWAP (overbought on daily)
        Deviation < -0.03  : Price significantly below VWAP (oversold on daily)

    Args:
        df     : OHLCV DataFrame
        window : Rolling window for VWAP (default 20 trading days)

    Returns:
        DataFrame with typical_price, vwap_20d, vwap_deviation columns
    """
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

    # Handle zero/NaN volume gracefully
    vol = df["volume"].replace(0, np.nan)

    tp_vol = df["typical_price"] * vol
    df["vwap_20d"] = (
        tp_vol.rolling(window=window, min_periods=max(1, window // 2)).sum() /
        vol.rolling(window=window, min_periods=max(1, window // 2)).sum()
    )

    df["vwap_deviation"] = (df["close"] - df["vwap_20d"]) / df["vwap_20d"]

    return df


def compute_swing_structure(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    Detects Higher Highs / Higher Lows (uptrend) and
    Lower Highs / Lower Lows (downtrend) using swing point analysis.

    A swing high is a candle whose high is higher than the `lookback`
    candles on both sides. A swing low is the opposite.

    Structure classification:
        +1 : Last two swing highs are HH AND last two swing lows are HL
             → Confirmed uptrend structure
        -1 : Last two swing highs are LH AND last two swing lows are LL
             → Confirmed downtrend structure
         0 : Mixed structure (consolidation / transition)

    Args:
        df       : OHLCV DataFrame
        lookback : Number of bars each side to check for swing point

    Returns:
        DataFrame with swing_structure, last_swing_high, last_swing_low columns
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    swing_highs = []   # (index, price)
    swing_lows  = []   # (index, price)

    for i in range(lookback, n - lookback):
        # Swing high: this bar's high is highest in the window
        if highs[i] == max(highs[i - lookback: i + lookback + 1]):
            swing_highs.append((i, highs[i]))

        # Swing low: this bar's low is lowest in the window
        if lows[i] == min(lows[i - lookback: i + lookback + 1]):
            swing_lows.append((i, lows[i]))

    # Determine structure for each bar based on last 2 swing points seen
    structure  = np.zeros(n, dtype=int)
    last_sh    = np.full(n, np.nan)
    last_sl    = np.full(n, np.nan)

    sh_ptr = 0   # pointer into swing_highs
    sl_ptr = 0   # pointer into swing_lows

    for i in range(n):
        # Advance swing high pointer to include all swing highs up to bar i
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= i:
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= i:
            sl_ptr += 1

        visible_sh = swing_highs[:sh_ptr]
        visible_sl = swing_lows[:sl_ptr]

        if visible_sh:
            last_sh[i] = visible_sh[-1][1]
        if visible_sl:
            last_sl[i] = visible_sl[-1][1]

        # Need at least 2 swing highs AND 2 swing lows to classify structure
        if len(visible_sh) >= 2 and len(visible_sl) >= 2:
            hh = visible_sh[-1][1] > visible_sh[-2][1]  # Higher High
            hl = visible_sl[-1][1] > visible_sl[-2][1]  # Higher Low
            lh = visible_sh[-1][1] < visible_sh[-2][1]  # Lower High
            ll = visible_sl[-1][1] < visible_sl[-2][1]  # Lower Low

            if hh and hl:
                structure[i] = 1    # Uptrend
            elif lh and ll:
                structure[i] = -1   # Downtrend
            else:
                structure[i] = 0    # Mixed / consolidation

    df["swing_structure"] = structure
    df["last_swing_high"] = last_sh
    df["last_swing_low"]  = last_sl

    return df


def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Ichimoku Kinko Hyo (Cloud) indicator.

    Components:
        Tenkan-sen  (9)  : (Max_high + Min_low) / 2 over 9 periods
        Kijun-sen   (26) : (Max_high + Min_low) / 2 over 26 periods
        Senkou A         : (Tenkan + Kijun) / 2, plotted 26 bars forward
        Senkou B    (52) : (Max_high + Min_low) / 2 over 52 periods, plotted 26 forward
        Chikou          : Close plotted 26 bars backward

    Signal derivations:
        price_above_cloud : Close > max(Senkou A, Senkou B) = bullish
        chikou_above      : Chikou (current close) > price 26 bars ago = bullish
        tk_cross          : Tenkan crosses above/below Kijun

    Args:
        df : OHLCV DataFrame

    Returns:
        DataFrame with all Ichimoku columns
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    def mid(series_high, series_low, period):
        return (
            series_high.rolling(period).max() +
            series_low.rolling(period).min()
        ) / 2

    # Core lines
    tenkan = mid(high, low, ICHIMOKU_FAST)
    kijun  = mid(high, low, ICHIMOKU_MED)

    # Senkou lines shifted 26 bars into the future
    # For backtesting we use current-bar values (no look-ahead)
    senkou_a = ((tenkan + kijun) / 2).shift(ICHIMOKU_MED)
    senkou_b = mid(high, low, ICHIMOKU_SLOW).shift(ICHIMOKU_MED)

    # Chikou: current close shifted 26 bars back
    chikou = close.shift(-ICHIMOKU_MED)

    df["tenkan"]   = tenkan
    df["kijun"]    = kijun
    df["senkou_a"] = senkou_a
    df["senkou_b"] = senkou_b

    # Price above cloud: bullish when close is above BOTH senkou lines
    df["price_above_cloud"] = (
        (close > senkou_a.fillna(0)) &
        (close > senkou_b.fillna(0))
    )

    # Chikou above price 26 bars ago: bullish momentum confirmation
    price_26_ago = close.shift(ICHIMOKU_MED)
    df["chikou_above"] = close > price_26_ago

    # Tenkan/Kijun crossover: bullish when Tenkan crosses above Kijun
    prev_tenkan = tenkan.shift(1)
    prev_kijun  = kijun.shift(1)

    bull_tk = (tenkan > kijun) & (prev_tenkan <= prev_kijun)
    bear_tk = (tenkan < kijun) & (prev_tenkan >= prev_kijun)

    df["tk_cross"] = 0
    df.loc[bull_tk, "tk_cross"] = 1
    df.loc[bear_tk, "tk_cross"] = -1

    return df


# ══════════════════════════════════════════════════════════════════════════
#  TREND SCORE AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════

def compute_trend_score(df: pd.DataFrame) -> pd.Series:
    """
    Aggregates all trend sub-signals into a single trend_score in [-1.0, +1.0].

    Scoring methodology:
        Each sub-signal contributes a directional vote in [-1, +1].
        Votes are weighted by reliability and then averaged.
        The final score is clipped to [-1.0, +1.0].

    Weights (sum to 1.0):
        EMA Ribbon alignment   : 0.25  (primary trend direction)
        ADX + DI direction     : 0.20  (trend strength confirmation)
        Supertrend direction   : 0.20  (clean binary trend filter)
        Swing structure        : 0.15  (structural price action)
        Ichimoku cloud         : 0.15  (multi-dimensional confirmation)
        VWAP deviation         : 0.05  (minor mean-reversion component)

    Args:
        df : DataFrame with all computed features

    Returns:
        pd.Series of trend_score values, same index as df
    """
    scores = pd.DataFrame(index=df.index)

    # ── 1. EMA Ribbon (weight 0.25) ───────────────────────────────────────
    # Full bull alignment = +1, full bear = -1, mixed = proportional
    ribbon_score = pd.Series(0.0, index=df.index)
    ribbon_score += np.where(df["ema_9_21_gap"]   > 0, 0.33, -0.33)
    ribbon_score += np.where(df["ema_21_50_gap"]  > 0, 0.33, -0.33)
    ribbon_score += np.where(df["ema_50_200_gap"] > 0, 0.34, -0.34)
    # Bonus for full alignment
    ribbon_score = np.where(df["ribbon_aligned_up"], 1.0,
                   np.where(df["ribbon_aligned_dn"], -1.0, ribbon_score))
    scores["ema_ribbon"] = pd.Series(ribbon_score, index=df.index) * 0.25

    # ── 2. ADX + DI Direction (weight 0.20) ──────────────────────────────
    # Direction from +DI vs -DI; scaled by ADX strength
    di_direction = np.where(
        df["plus_di"].notna() & df["minus_di"].notna(),
        np.sign(df["plus_di"] - df["minus_di"]),
        0.0
    )
    # Scale by normalized ADX (0 when ADX=0, 1 when ADX>=40)
    adx_scale = np.clip(df["adx"].fillna(0) / 40.0, 0, 1)
    scores["adx_signal"] = pd.Series(di_direction * adx_scale, index=df.index) * 0.20

    # ── 3. Supertrend Direction (weight 0.20) ─────────────────────────────
    # Clean binary: +1 uptrend, -1 downtrend
    st_signal = df["supertrend_dir"].fillna(0).astype(float)
    scores["supertrend"] = st_signal * 0.20

    # ── 4. Swing Structure (weight 0.15) ──────────────────────────────────
    swing_signal = df["swing_structure"].fillna(0).astype(float)
    scores["swing"] = swing_signal * 0.15

    # ── 5. Ichimoku Cloud (weight 0.15) ───────────────────────────────────
    ichi_score = pd.Series(0.0, index=df.index)

    # Price above cloud = +0.5 component
    ichi_score += np.where(df["price_above_cloud"].fillna(False), 0.5, -0.5)

    # Chikou confirmation = +0.3 component
    ichi_score += np.where(df["chikou_above"].fillna(False), 0.3, -0.3)

    # TK cross = +0.2 component (only on crossover bar, fades to 0)
    ichi_score += df["tk_cross"].fillna(0) * 0.2

    # Clip to [-1, +1]
    ichi_score = ichi_score.clip(-1.0, 1.0)
    scores["ichimoku"] = ichi_score * 0.15

    # ── 6. VWAP Deviation (weight 0.05) ───────────────────────────────────
    # Normalized deviation: >+5% = overbought (-1), <-5% = oversold (+1)
    # Note: mean-reverting signal — contrary to trend direction
    vwap_dev = df["vwap_deviation"].fillna(0)
    vwap_signal = -np.clip(vwap_dev / 0.05, -1.0, 1.0)  # inverted (contrarian)
    scores["vwap"] = pd.Series(vwap_signal, index=df.index) * 0.05

    # ── Final aggregation ─────────────────────────────────────────────────
    trend_score = scores.sum(axis=1).clip(-1.0, 1.0)

    return trend_score


# ══════════════════════════════════════════════════════════════════════════
#  MAIN FEATURE EXTRACTOR CLASS
# ══════════════════════════════════════════════════════════════════════════

class TrendExtractor:
    """
    Main interface for Pillar 1 — Trend Analysis.

    Usage:
        extractor = TrendExtractor()

        # Compute features for one stock
        result = extractor.compute("RELIANCE", date(2024, 1, 1), date(2024, 12, 31))

        # Compute and save for all Nifty 500 stocks
        extractor.run_all(end_date=date(2024, 12, 31))

        # Get latest trend score for one stock
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
        Computes all trend features for a single symbol over a date range.

        Args:
            symbol     : NSE symbol e.g. 'RELIANCE'
            start_date : Start of computation range
            end_date   : End of computation range
            save_to_db : Whether to upsert results into features_trend table

        Returns:
            DataFrame with all trend features and trend_score,
            or None if insufficient data.
        """
        # Load with enough history for EMA200 warm-up
        # EMA200 needs at least 200 bars, add 30-bar buffer
        history_start = start_date - timedelta(days=MIN_BARS_REQUIRED + 30)

        df = load_ohlcv(symbol, history_start, end_date, self.conn)

        if df.empty:
            logger.warning(f"{symbol}: No data available, skipping.")
            return None

        if len(df) < MIN_BARS_REQUIRED:
            logger.warning(
                f"{symbol}: Only {len(df)} bars available, "
                f"need {MIN_BARS_REQUIRED}. Skipping."
            )
            return None

        # ── Run all feature computations ──────────────────────────────────
        df = compute_ema_ribbon(df)
        df = compute_adx(df)
        df = compute_supertrend(df)
        df = compute_vwap_deviation(df)
        df = compute_swing_structure(df)
        df = compute_ichimoku(df)

        # ── Compute final trend score ─────────────────────────────────────
        df["trend_score"] = compute_trend_score(df)

        # ── Trim to requested date range (remove warm-up period) ──────────
        df = df[df.index >= pd.Timestamp(start_date)].copy()

        if df.empty:
            logger.warning(f"{symbol}: No data in requested range after warm-up trim.")
            return None

        # ── Save to database ──────────────────────────────────────────────
        if save_to_db:
            self._save(symbol, df)

        return df

    def _save(self, symbol: str, df: pd.DataFrame):
        """
        Upserts computed trend features into features_trend table.
        Safe to re-run — uses ON CONFLICT DO UPDATE.
        """
        cols = [
            "ema_9", "ema_21", "ema_50", "ema_100", "ema_200",
            "ema_9_21_gap", "ema_21_50_gap", "ema_50_200_gap",
            "ribbon_aligned_up", "ribbon_aligned_dn", "price_vs_ema200",
            "adx", "plus_di", "minus_di", "di_crossover", "adx_trending",
            "supertrend_dir", "supertrend_val", "price_vs_st",
            "typical_price", "vwap_20d", "vwap_deviation",
            "swing_structure", "last_swing_high", "last_swing_low",
            "tenkan", "kijun", "senkou_a", "senkou_b",
            "chikou_above", "price_above_cloud", "tk_cross",
            "trend_score",
        ]

        records = []
        for ts, row in df.iterrows():
            rec = {"date": ts.date(), "symbol": symbol}
            for col in cols:
                val = row.get(col, None)
                # Convert numpy types to Python natives for psycopg2
                if pd.isna(val) if not isinstance(val, bool) else False:
                    rec[col] = None
                elif isinstance(val, (np.integer,)):
                    rec[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    rec[col] = float(val)
                elif isinstance(val, (np.bool_,)):
                    rec[col] = bool(val)
                else:
                    rec[col] = val
            records.append(rec)

        if not records:
            return

        insert_sql = """
            INSERT INTO features_trend (
                date, symbol,
                ema_9, ema_21, ema_50, ema_100, ema_200,
                ema_9_21_gap, ema_21_50_gap, ema_50_200_gap,
                ribbon_aligned_up, ribbon_aligned_dn, price_vs_ema200,
                adx, plus_di, minus_di, di_crossover, adx_trending,
                supertrend_dir, supertrend_val, price_vs_st,
                typical_price, vwap_20d, vwap_deviation,
                swing_structure, last_swing_high, last_swing_low,
                tenkan, kijun, senkou_a, senkou_b,
                chikou_above, price_above_cloud, tk_cross,
                trend_score
            ) VALUES (
                %(date)s, %(symbol)s,
                %(ema_9)s, %(ema_21)s, %(ema_50)s, %(ema_100)s, %(ema_200)s,
                %(ema_9_21_gap)s, %(ema_21_50_gap)s, %(ema_50_200_gap)s,
                %(ribbon_aligned_up)s, %(ribbon_aligned_dn)s, %(price_vs_ema200)s,
                %(adx)s, %(plus_di)s, %(minus_di)s, %(di_crossover)s, %(adx_trending)s,
                %(supertrend_dir)s, %(supertrend_val)s, %(price_vs_st)s,
                %(typical_price)s, %(vwap_20d)s, %(vwap_deviation)s,
                %(swing_structure)s, %(last_swing_high)s, %(last_swing_low)s,
                %(tenkan)s, %(kijun)s, %(senkou_a)s, %(senkou_b)s,
                %(chikou_above)s, %(price_above_cloud)s, %(tk_cross)s,
                %(trend_score)s
            )
            ON CONFLICT (date, symbol) DO UPDATE SET
                trend_score       = EXCLUDED.trend_score,
                ema_9             = EXCLUDED.ema_9,
                ema_21            = EXCLUDED.ema_21,
                ema_50            = EXCLUDED.ema_50,
                ema_100           = EXCLUDED.ema_100,
                ema_200           = EXCLUDED.ema_200,
                ribbon_aligned_up = EXCLUDED.ribbon_aligned_up,
                ribbon_aligned_dn = EXCLUDED.ribbon_aligned_dn,
                adx               = EXCLUDED.adx,
                supertrend_dir    = EXCLUDED.supertrend_dir,
                swing_structure   = EXCLUDED.swing_structure,
                price_above_cloud = EXCLUDED.price_above_cloud;
        """

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=500)
            self.conn.commit()
            logger.success(f"{symbol}: {len(records)} trend feature rows saved.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"{symbol}: DB save failed — {e}")
            raise

    def run_all(
        self,
        end_date: Optional[date] = None,
        start_date: Optional[date] = None,
        symbols: Optional[list[str]] = None,
    ):
        """
        Computes and saves trend features for all (or specified) symbols.

        Args:
            end_date   : Compute up to this date (default: today)
            start_date : Compute from this date (default: 2019-01-01)
            symbols    : List of symbols to process (default: all in DB)

        Usage:
            extractor = TrendExtractor()

            # Full history for all stocks (run once in Phase 1)
            extractor.run_all(end_date=date(2024, 12, 31))

            # Daily update (run from Airflow nightly)
            extractor.run_all(
                start_date=date.today() - timedelta(days=5),
                end_date=date.today()
            )
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = date(2019, 1, 1)

        if symbols is None:
            symbols = load_all_symbols(self.conn)

        logger.info(
            f"TrendExtractor.run_all: {len(symbols)} symbols | "
            f"{start_date} → {end_date}"
        )

        success = 0
        failed  = 0
        skipped = 0

        for i, symbol in enumerate(symbols, 1):
            try:
                result = self.compute(symbol, start_date, end_date, save_to_db=True)
                if result is not None:
                    success += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"[{i}/{len(symbols)}] {symbol} failed: {e}")
                failed += 1

            if i % 50 == 0:
                logger.info(
                    f"Progress: {i}/{len(symbols)} | "
                    f"Success: {success} | Skipped: {skipped} | Failed: {failed}"
                )

        logger.info(
            f"run_all complete — "
            f"Success: {success} | Skipped: {skipped} | Failed: {failed}"
        )

    def get_latest_score(self, symbol: str) -> Optional[float]:
        """
        Returns the most recent trend_score for a given symbol.
        Used by the signal engine during live trading.

        Returns:
            float in [-1.0, +1.0] or None if no data
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT trend_score
                FROM features_trend
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 1;
            """, (symbol,))
            row = cur.fetchone()
        return float(row[0]) if row else None

    def get_scores_for_date(self, target_date: date) -> pd.DataFrame:
        """
        Returns trend_score for ALL symbols on a specific date.
        Used by the signal engine to rank all 500 stocks each morning.

        Returns:
            DataFrame with columns: symbol, trend_score
            Sorted by trend_score descending (strongest uptrends first)
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, trend_score
                FROM features_trend
                WHERE date = %s
                ORDER BY trend_score DESC;
            """, (target_date,))
            rows = cur.fetchall()

        return pd.DataFrame(rows, columns=["symbol", "trend_score"])

    def close(self):
        """Close the database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — Trend Feature Extractor")
    parser.add_argument("--mode",   choices=["all", "single", "score"], default="all")
    parser.add_argument("--symbol", type=str, help="NSE symbol for single/score mode")
    parser.add_argument("--start",  type=str, default="2019-01-01")
    parser.add_argument("--end",    type=str, default=str(date.today()))
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    with TrendExtractor() as extractor:
        if args.mode == "all":
            extractor.run_all(start_date=start, end_date=end)

        elif args.mode == "single":
            if not args.symbol:
                print("--symbol required for single mode")
            else:
                df = extractor.compute(args.symbol, start, end)
                if df is not None:
                    print(df[["trend_score", "ribbon_aligned_up",
                               "adx", "supertrend_dir"]].tail(10))

        elif args.mode == "score":
            if not args.symbol:
                print("--symbol required for score mode")
            else:
                score = extractor.get_latest_score(args.symbol)
                print(f"{args.symbol} latest trend_score: {score}")


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest features/trend.py -v
#  Or:       python features/trend.py (triggers __main__ block above)
# ══════════════════════════════════════════════════════════════════════════

def _make_sample_ohlcv(n: int = 300, trend: str = "up") -> pd.DataFrame:
    """
    Generates synthetic OHLCV data for unit testing.

    Args:
        n     : Number of bars
        trend : 'up', 'down', or 'sideways'
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    if trend == "up":
        close = 1000 + np.cumsum(np.random.randn(n) * 5 + 0.5)
    elif trend == "down":
        close = 1000 + np.cumsum(np.random.randn(n) * 5 - 0.5)
    else:  # sideways
        close = 1000 + np.cumsum(np.random.randn(n) * 5)

    close = np.maximum(close, 10)   # no negative prices
    high  = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low   = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)
    vol   = np.random.randint(100_000, 5_000_000, n).astype(float)

    return pd.DataFrame({
        "open"      : open_,
        "high"      : high,
        "low"       : low,
        "close"     : close,
        "prev_close": np.roll(close, 1),
        "volume"    : vol,
        "turnover"  : close * vol,
    }, index=dates)


class TestTrendFeatures:
    """Unit tests for all Pillar 1 feature computation functions."""

    def setup_method(self):
        self.df_up   = _make_sample_ohlcv(300, "up")
        self.df_down = _make_sample_ohlcv(300, "down")
        self.df_side = _make_sample_ohlcv(300, "sideways")

    # ── EMA Ribbon Tests ─────────────────────────────────────────────────

    def test_ema_ribbon_columns_exist(self):
        df = compute_ema_ribbon(self.df_up.copy())
        for period in EMA_PERIODS:
            assert f"ema_{period}" in df.columns, f"ema_{period} missing"
        assert "ribbon_aligned_up" in df.columns
        assert "ribbon_aligned_dn" in df.columns
        assert "price_vs_ema200"   in df.columns

    def test_ema_ribbon_uptrend_alignment(self):
        """In a strong uptrend, ribbon should align bullish."""
        df = compute_ema_ribbon(self.df_up.copy())
        # Last 50 bars should mostly be bullishly aligned
        recent_aligned = df["ribbon_aligned_up"].tail(50).sum()
        assert recent_aligned > 20, (
            f"Expected >20 bullish ribbon bars in uptrend, got {recent_aligned}"
        )

    def test_ema_ribbon_no_simultaneous_alignment(self):
        """ribbon_aligned_up and ribbon_aligned_dn cannot both be True."""
        df = compute_ema_ribbon(self.df_up.copy())
        both_true = (df["ribbon_aligned_up"] & df["ribbon_aligned_dn"]).sum()
        assert both_true == 0, "ribbon_aligned_up and _dn cannot both be True"

    def test_ema_gaps_sign_in_uptrend(self):
        """In uptrend: EMA9 > EMA21 > EMA50, so gaps should be positive."""
        df = compute_ema_ribbon(self.df_up.copy())
        recent = df.tail(30).dropna()
        pos_9_21 = (recent["ema_9_21_gap"] > 0).sum()
        assert pos_9_21 > 15, "EMA 9-21 gap should be mostly positive in uptrend"

    # ── ADX Tests ────────────────────────────────────────────────────────

    def test_adx_columns_exist(self):
        df = compute_adx(self.df_up.copy())
        assert "adx"          in df.columns
        assert "plus_di"      in df.columns
        assert "minus_di"     in df.columns
        assert "di_crossover" in df.columns
        assert "adx_trending" in df.columns

    def test_adx_range(self):
        """ADX must always be in [0, 100]."""
        df = compute_adx(self.df_up.copy())
        adx_valid = df["adx"].dropna()
        assert (adx_valid >= 0).all() and (adx_valid <= 100).all(), \
            "ADX out of [0, 100] range"

    def test_di_values_non_negative(self):
        """DI values are always >= 0."""
        df = compute_adx(self.df_up.copy())
        assert (df["plus_di"].dropna()  >= 0).all()
        assert (df["minus_di"].dropna() >= 0).all()

    def test_di_crossover_values(self):
        """di_crossover must only be -1, 0, or +1."""
        df = compute_adx(self.df_up.copy())
        valid = {-1, 0, 1}
        assert set(df["di_crossover"].unique()).issubset(valid)

    # ── Supertrend Tests ──────────────────────────────────────────────────

    def test_supertrend_columns_exist(self):
        df = compute_supertrend(self.df_up.copy())
        assert "supertrend_dir" in df.columns
        assert "supertrend_val" in df.columns
        assert "price_vs_st"    in df.columns

    def test_supertrend_direction_values(self):
        """supertrend_dir must only be -1, 0, or +1."""
        df = compute_supertrend(self.df_up.copy())
        valid = {-1, 0, 1}
        assert set(df["supertrend_dir"].unique()).issubset(valid)

    def test_supertrend_uptrend_bias(self):
        """In a clear uptrend, supertrend should be mostly +1."""
        df = compute_supertrend(self.df_up.copy())
        up_count = (df["supertrend_dir"] == 1).sum()
        total    = df["supertrend_dir"].notna().sum()
        assert up_count / total > 0.5, \
            f"Uptrend: expected >50% +1 supertrend, got {up_count/total:.1%}"

    # ── VWAP Tests ────────────────────────────────────────────────────────

    def test_vwap_columns_exist(self):
        df = compute_vwap_deviation(self.df_up.copy())
        assert "typical_price"  in df.columns
        assert "vwap_20d"       in df.columns
        assert "vwap_deviation" in df.columns

    def test_typical_price_formula(self):
        """Typical price = (H + L + C) / 3."""
        df = compute_vwap_deviation(self.df_up.copy())
        expected = (df["high"] + df["low"] + df["close"]) / 3
        np.testing.assert_array_almost_equal(
            df["typical_price"].values, expected.values, decimal=4
        )

    def test_vwap_deviation_reasonable_range(self):
        """VWAP deviation should rarely exceed ±30% on daily bars."""
        df = compute_vwap_deviation(self.df_up.copy())
        extreme = (df["vwap_deviation"].abs() > 0.30).sum()
        assert extreme < 5, f"Too many extreme VWAP deviations: {extreme}"

    # ── Swing Structure Tests ─────────────────────────────────────────────

    def test_swing_structure_columns_exist(self):
        df = compute_swing_structure(self.df_up.copy())
        assert "swing_structure" in df.columns
        assert "last_swing_high" in df.columns
        assert "last_swing_low"  in df.columns

    def test_swing_structure_valid_values(self):
        """swing_structure must only be -1, 0, or +1."""
        df = compute_swing_structure(self.df_up.copy())
        valid = {-1, 0, 1}
        assert set(df["swing_structure"].unique()).issubset(valid)

    def test_swing_uptrend_bias(self):
        """Strong uptrend should produce mostly +1 swing structure."""
        df = compute_swing_structure(self.df_up.copy())
        up_count   = (df["swing_structure"] == 1).sum()
        down_count = (df["swing_structure"] == -1).sum()
        assert up_count > down_count, \
            f"Uptrend: expected more HH/HL ({up_count}) than LH/LL ({down_count})"

    # ── Ichimoku Tests ────────────────────────────────────────────────────

    def test_ichimoku_columns_exist(self):
        df = compute_ichimoku(self.df_up.copy())
        for col in ["tenkan", "kijun", "senkou_a", "senkou_b",
                    "chikou_above", "price_above_cloud", "tk_cross"]:
            assert col in df.columns, f"Ichimoku column missing: {col}"

    def test_tk_cross_values(self):
        """tk_cross must only be -1, 0, or +1."""
        df = compute_ichimoku(self.df_up.copy())
        valid = {-1, 0, 1}
        assert set(df["tk_cross"].unique()).issubset(valid)

    def test_boolean_columns_are_bool(self):
        df = compute_ichimoku(self.df_up.copy())
        assert df["price_above_cloud"].dtype == bool or \
               df["price_above_cloud"].isin([True, False, np.nan]).all()

    # ── Trend Score Tests ─────────────────────────────────────────────────

    def test_trend_score_range(self):
        """Trend score must always be in [-1.0, +1.0]."""
        for df_raw in [self.df_up, self.df_down, self.df_side]:
            df = compute_ema_ribbon(df_raw.copy())
            df = compute_adx(df)
            df = compute_supertrend(df)
            df = compute_vwap_deviation(df)
            df = compute_swing_structure(df)
            df = compute_ichimoku(df)
            score = compute_trend_score(df)
            assert (score.dropna() >= -1.0).all() and \
                   (score.dropna() <= 1.0).all(), \
                "trend_score out of [-1, +1] range"

    def test_uptrend_score_positive(self):
        """Uptrend data should produce mostly positive trend scores."""
        df = compute_ema_ribbon(self.df_up.copy())
        df = compute_adx(df)
        df = compute_supertrend(df)
        df = compute_vwap_deviation(df)
        df = compute_swing_structure(df)
        df = compute_ichimoku(df)
        score = compute_trend_score(df).dropna()
        pos_pct = (score > 0).mean()
        assert pos_pct > 0.55, \
            f"Uptrend should have >55% positive scores, got {pos_pct:.1%}"

    def test_downtrend_score_negative(self):
        """Downtrend data should produce mostly negative trend scores."""
        df = compute_ema_ribbon(self.df_down.copy())
        df = compute_adx(df)
        df = compute_supertrend(df)
        df = compute_vwap_deviation(df)
        df = compute_swing_structure(df)
        df = compute_ichimoku(df)
        score = compute_trend_score(df).dropna()
        neg_pct = (score < 0).mean()
        assert neg_pct > 0.55, \
            f"Downtrend should have >55% negative scores, got {neg_pct:.1%}"

    def test_score_no_nan_after_warmup(self):
        """After 210-bar warm-up, there should be no NaN trend scores."""
        df = compute_ema_ribbon(self.df_up.copy())
        df = compute_adx(df)
        df = compute_supertrend(df)
        df = compute_vwap_deviation(df)
        df = compute_swing_structure(df)
        df = compute_ichimoku(df)
        score = compute_trend_score(df)
        post_warmup_nans = score.iloc[MIN_BARS_REQUIRED:].isna().sum()
        assert post_warmup_nans == 0, \
            f"{post_warmup_nans} NaN scores found after warm-up period"


# ── Run tests when file is executed directly ──────────────────────────────
if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))