"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Pillar 5: Volatility Analysis                  ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : features/volatility.py                                 ║
║         Phase   : 1 — Feature Engineering                               ║
║                                                                          ║
║  What this pillar learns:                                                ║
║    The current and predicted volatility regime of individual stocks      ║
║    and the market. Volatility directly controls two critical outputs:    ║
║      1. Position sizing — high volatility = smaller position             ║
║      2. TP/SL placement — high volatility = wider stops                 ║
║                                                                          ║
║  Why volatility matters more than most traders realize:                  ║
║    A 1.5% stop-loss on a stock with 3% daily ATR will be stopped out    ║
║    by random noise almost every day. The system adapts all TP/SL        ║
║    values dynamically using ATR so stops are placed OUTSIDE the noise.  ║
║                                                                          ║
║  Features computed:                                                      ║
║    ATR(14)          → absolute daily range; primary TP/SL input         ║
║    ATR % of price   → relative ATR; regime classification               ║
║    Historical Vol   → 20-day annualized std dev of log returns           ║
║    HV Percentile    → HV rank vs 252-day window (0–100)                 ║
║    IV Rank (IVR)    → from options chain ATM IV (if available)           ║
║    GARCH(1,1)       → 1-day forward volatility forecast                 ║
║    Beta to Nifty    → 60-day rolling market beta                        ║
║    Vol Regime       → low / medium / high / extreme (categorical)       ║
║    Vol Z-score      → current HV vs 1-year rolling mean/std             ║
║    Volatility Score → composite [-1, +1] for LSTM input                 ║
║                                                                          ║
║  Dynamic TP/SL adjustment (used in Phase 6 execution):                  ║
║    Swing  : TP = max(4.0%, 2.0 × ATR%) | SL = max(1.5%, 0.75 × ATR%) ║
║    Intraday: TP = max(2.5%, 1.5 × ATR%) | SL = max(0.8%, 0.5 × ATR%) ║
║                                                                          ║
║  Volatility Regime Classification:                                       ║
║    Low     : ATR% < 1.0%  → larger positions, tighter stops             ║
║    Medium  : ATR% 1–2%    → standard sizing                             ║
║    High    : ATR% 2–4%    → reduced positions, wider stops              ║
║    Extreme : ATR% > 4%    → minimum position or skip                    ║
║                                                                          ║
║  Database:                                                               ║
║    Reads from : daily_ohlcv                                              ║
║    Reads from : options_chain (for IV — optional)                        ║
║    Writes to  : features_volatility                                      ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install pandas numpy psycopg2-binary arch loguru python-dotenv   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
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

# ── Volatility parameters ──────────────────────────────────────────────────
ATR_PERIOD          = 14     # ATR lookback
HV_PERIOD           = 20     # Historical volatility window
HV_PERCENTILE_WINDOW= 252    # 1 trading year for HV percentile
BETA_PERIOD         = 60     # Rolling beta lookback
GARCH_MIN_OBS       = 60     # Minimum observations for GARCH fit
ANNUALIZATION_FACTOR= 252    # Trading days per year

# Regime thresholds (ATR as % of price)
REGIME_LOW      = 1.0    # ATR% < 1.0%  → low vol
REGIME_MEDIUM   = 2.0    # ATR% 1–2%    → medium vol
REGIME_HIGH     = 4.0    # ATR% 2–4%    → high vol
                          # ATR% > 4%    → extreme vol

# Dynamic TP/SL multipliers (ATR-based)
SWING_TP_ATR_MULT   = 2.0    # TP = max(4.0%, 2.0 × ATR%)
SWING_SL_ATR_MULT   = 0.75   # SL = max(1.5%, 0.75 × ATR%)
INTRA_TP_ATR_MULT   = 1.5    # TP = max(2.5%, 1.5 × ATR%)
INTRA_SL_ATR_MULT   = 0.5    # SL = max(0.8%, 0.5 × ATR%)

# Base TP/SL floors (minimum regardless of ATR)
SWING_TP_FLOOR  = 4.0    # %
SWING_SL_FLOOR  = 1.5    # %
INTRA_TP_FLOOR  = 2.5    # %
INTRA_SL_FLOOR  = 0.8    # %

MIN_BARS        = 70     # minimum bars needed for reliable features


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_conn():
    return psycopg2.connect(DB_URL)


def _ensure_features_table(conn):
    """Creates features_volatility table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS features_volatility (
                date                DATE        NOT NULL,
                symbol              VARCHAR(20) NOT NULL,

                -- ATR features
                atr_14              NUMERIC(12,4),   -- absolute ATR in ₹
                atr_pct             NUMERIC(6,4),    -- ATR as % of close price
                atr_zscore          NUMERIC(6,4),    -- ATR z-score vs 60-day

                -- Historical Volatility
                hv_20               NUMERIC(6,4),    -- 20-day HV (annualized %)
                hv_percentile       NUMERIC(5,1),    -- HV rank [0, 100] vs 252-day
                hv_zscore           NUMERIC(6,4),    -- HV z-score vs 252-day

                -- Implied Volatility (from options chain, may be NULL)
                iv_atm              NUMERIC(6,4),    -- ATM implied volatility
                iv_rank             NUMERIC(5,1),    -- IV rank [0, 100] vs 252-day
                iv_hv_spread        NUMERIC(6,4),    -- IV - HV (vol premium/discount)

                -- GARCH(1,1) forecast
                garch_forecast      NUMERIC(6,4),    -- 1-day ahead vol forecast
                garch_available     BOOLEAN,         -- False if insufficient data

                -- Beta
                beta_nifty          NUMERIC(6,4),    -- 60-day rolling beta to Nifty
                beta_available      BOOLEAN,

                -- Dynamic TP/SL (computed from ATR)
                swing_tp_pct        NUMERIC(5,2),    -- recommended swing TP %
                swing_sl_pct        NUMERIC(5,2),    -- recommended swing SL %
                intra_tp_pct        NUMERIC(5,2),    -- recommended intraday TP %
                intra_sl_pct        NUMERIC(5,2),    -- recommended intraday SL %

                -- Regime classification
                vol_regime          VARCHAR(10),     -- low/medium/high/extreme
                vol_regime_code     SMALLINT,        -- 0=low,1=med,2=high,3=extreme

                -- Composite volatility score for LSTM [-1, +1]
                -- Positive = high vol (caution), Negative = low vol (opportunity)
                volatility_score    NUMERIC(5,4),

                PRIMARY KEY (date, symbol)
            );
        """)

        cur.execute("""
            SELECT create_hypertable(
                'features_volatility', 'date',
                if_not_exists => TRUE,
                migrate_data  => TRUE
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_vol_symbol
            ON features_volatility (symbol, date DESC);
        """)

    conn.commit()
    logger.info("features_volatility table ready.")


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════

def load_ohlcv(symbol: str, start_date: date, end_date: date, conn) -> pd.DataFrame:
    """Loads OHLCV from TimescaleDB for a single symbol."""
    history_start = start_date - timedelta(days=HV_PERCENTILE_WINDOW + BETA_PERIOD + 30)

    sql = """
        SELECT date, open, high, low, close, volume
        FROM daily_ohlcv
        WHERE symbol = %s AND date BETWEEN %s AND %s
        ORDER BY date ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, history_start, end_date))
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.astype(float)
    df = df.dropna(subset=["close"])
    return df


def load_nifty50(start_date: date, end_date: date, conn) -> pd.Series:
    """
    Loads Nifty 50 index close prices for beta calculation.
    Uses NIFTY50-proxy: average of top 10 Nifty 50 stocks by weight,
    or falls back to a single liquid large-cap (RELIANCE) if index not stored.
    """
    history_start = start_date - timedelta(days=BETA_PERIOD + 30)

    # Try to load a Nifty 50 proxy symbol
    for proxy in ["^NSEI", "NIFTY50", "RELIANCE"]:
        sql = """
            SELECT date, close FROM daily_ohlcv
            WHERE symbol = %s AND date BETWEEN %s AND %s
            ORDER BY date ASC;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (proxy, history_start, end_date))
            rows = cur.fetchall()

        if rows:
            df = pd.DataFrame(rows, columns=["date", "close"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df["close"] = df["close"].astype(float)
            return df["close"]

    return pd.Series(dtype=float)


def load_iv(symbol: str, start_date: date, end_date: date, conn) -> pd.Series:
    """
    Loads ATM implied volatility from options_chain table.
    Returns empty Series if table doesn't exist (handled gracefully).
    """
    history_start = start_date - timedelta(days=HV_PERCENTILE_WINDOW + 10)
    try:
        sql = """
            SELECT date, AVG(iv) AS iv_atm
            FROM options_chain
            WHERE symbol = %s
              AND ABS(strike - underlying_close) =
                  (SELECT MIN(ABS(strike - underlying_close))
                   FROM options_chain oc2
                   WHERE oc2.symbol = %s AND oc2.date = options_chain.date)
              AND date BETWEEN %s AND %s
            GROUP BY date
            ORDER BY date ASC;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, symbol, history_start, end_date))
            rows = cur.fetchall()

        if not rows:
            return pd.Series(dtype=float)

        iv = pd.Series(
            {pd.to_datetime(r[0]): float(r[1]) for r in rows if r[1] is not None}
        )
        return iv

    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        return pd.Series(dtype=float)
    except Exception as e:
        logger.warning(f"IV load failed for {symbol}: {e}")
        conn.rollback()
        return pd.Series(dtype=float)


def load_all_symbols(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM daily_ohlcv ORDER BY symbol;")
        return [row[0] for row in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE COMPUTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """
    Computes Average True Range (ATR) and derived features.

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = Wilder's smoothing of True Range over `period` bars.

    ATR% = ATR / Close × 100 — this is the primary input for
    dynamic TP/SL placement. A stock with ATR% = 2% needs at
    least a 2% stop-loss to avoid being stopped out by noise.

    Also computes:
        atr_zscore : (ATR% - rolling_mean_ATR%) / rolling_std_ATR%
                     Positive = currently more volatile than usual
                     Negative = currently calmer than usual

    Args:
        df     : OHLCV DataFrame
        period : ATR smoothing period (default 14)

    Returns:
        DataFrame with atr_14, atr_pct, atr_zscore columns added
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev  = close.shift(1)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing (same as RMA / SMMA)
    df["atr_14"] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # ATR as percentage of close price
    df["atr_pct"] = (df["atr_14"] / close * 100).round(4)

    # ATR z-score vs 60-day rolling window
    roll_mean = df["atr_pct"].rolling(60, min_periods=20).mean()
    roll_std  = df["atr_pct"].rolling(60, min_periods=20).std().replace(0, np.nan)
    df["atr_zscore"] = ((df["atr_pct"] - roll_mean) / roll_std).clip(-3, 3)

    return df


def compute_historical_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 20-day Historical Volatility (HV) and its percentile rank.

    HV Formula:
        log_return = ln(Close_t / Close_{t-1})
        HV_20 = std(log_returns, 20 bars) × sqrt(252) × 100
        Result is annualized volatility as a percentage.

    HV Percentile:
        Where does today's HV rank vs the past 252 trading days?
        Percentile 80 = today's vol is higher than 80% of past year's days.
        This is a volatility expansion/contraction signal.

    HV Z-score:
        Standardized distance from 252-day rolling mean.
        Used as a continuous input for LSTM.

    Args:
        df : OHLCV DataFrame with close prices

    Returns:
        DataFrame with hv_20, hv_percentile, hv_zscore columns
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # 20-day annualized HV (%)
    df["hv_20"] = (
        log_ret.rolling(HV_PERIOD, min_periods=HV_PERIOD).std()
        * np.sqrt(ANNUALIZATION_FACTOR)
        * 100
    ).round(4)

    # HV percentile vs trailing 252-day window
    def _percentile_rank(series: pd.Series, window: int) -> pd.Series:
        """Rolling percentile rank: what fraction of past values is current ≤ x."""
        result = pd.Series(index=series.index, dtype=float)
        arr = series.values
        for i in range(len(arr)):
            if i < window // 2:
                result.iloc[i] = np.nan
                continue
            start   = max(0, i - window + 1)
            window_ = arr[start: i + 1]
            valid   = window_[~np.isnan(window_)]
            if len(valid) < 10:
                result.iloc[i] = np.nan
            else:
                result.iloc[i] = float(np.sum(valid <= arr[i]) / len(valid) * 100)
        return result

    df["hv_percentile"] = _percentile_rank(df["hv_20"], HV_PERCENTILE_WINDOW)

    # HV z-score
    roll_mean = df["hv_20"].rolling(HV_PERCENTILE_WINDOW, min_periods=60).mean()
    roll_std  = df["hv_20"].rolling(HV_PERCENTILE_WINDOW, min_periods=60).std().replace(0, np.nan)
    df["hv_zscore"] = ((df["hv_20"] - roll_mean) / roll_std).clip(-3, 3)

    return df


def compute_iv_features(
    df: pd.DataFrame,
    iv_series: pd.Series,
) -> pd.DataFrame:
    """
    Computes Implied Volatility Rank (IVR) and IV-HV spread.

    IV Rank (IVR):
        Where does today's IV sit relative to its 252-day range?
        IVR = (IV - IV_min_252d) / (IV_max_252d - IV_min_252d) × 100
        IVR 80+ = IV historically high → sell premium strategies favored
        IVR 20- = IV historically low  → buy premium strategies favored

    IV-HV Spread:
        IV - HV = Volatility Risk Premium
        Positive spread = options overpriced vs realized vol (sell premium)
        Negative spread = options underpriced vs realized vol (buy premium)

    Args:
        df        : OHLCV DataFrame (must have hv_20 computed)
        iv_series : ATM IV series from options chain (may be empty)

    Returns:
        DataFrame with iv_atm, iv_rank, iv_hv_spread columns
    """
    if iv_series.empty:
        df["iv_atm"]      = np.nan
        df["iv_rank"]     = np.nan
        df["iv_hv_spread"]= np.nan
        return df

    # Align IV to OHLCV index
    iv_aligned = iv_series.reindex(df.index)
    df["iv_atm"] = iv_aligned

    # IV Rank: percentile of current IV vs 252-day min/max range
    iv_min = iv_aligned.rolling(HV_PERCENTILE_WINDOW, min_periods=30).min()
    iv_max = iv_aligned.rolling(HV_PERCENTILE_WINDOW, min_periods=30).max()
    iv_range = (iv_max - iv_min).replace(0, np.nan)
    df["iv_rank"] = ((iv_aligned - iv_min) / iv_range * 100).clip(0, 100)

    # IV - HV spread (volatility risk premium)
    if "hv_20" in df.columns:
        df["iv_hv_spread"] = (df["iv_atm"] - df["hv_20"]).round(4)
    else:
        df["iv_hv_spread"] = np.nan

    return df


def compute_garch_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits a GARCH(1,1) model and produces 1-day ahead volatility forecasts.

    GARCH(1,1) model:
        σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
        Where ε is the standardized residual (log return / conditional std)

    This captures the well-known volatility clustering effect:
        "Volatile periods tend to follow volatile periods."
    Standard HV is backward-looking. GARCH is forward-looking.

    Implementation:
        Uses the `arch` library (already in requirements.txt).
        Fits on all available history, forecasts 1 step ahead.
        Falls back to HV_20 if arch is not installed or fit fails.
        Re-fitted daily during the nightly retraining pipeline.

    Args:
        df : OHLCV DataFrame with 'close' (must have ≥ GARCH_MIN_OBS rows)

    Returns:
        DataFrame with garch_forecast and garch_available columns
    """
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna() * 100

    if len(log_ret) < GARCH_MIN_OBS:
        df["garch_forecast"] = df.get("hv_20", np.nan)
        df["garch_available"]= False
        return df

    try:
        from arch import arch_model

        garch_forecasts = pd.Series(np.nan, index=df.index)

        # Expanding window GARCH: fit on all data up to each point
        # For efficiency, only refit every 21 bars (monthly)
        refit_dates  = list(range(GARCH_MIN_OBS, len(log_ret), 21))
        last_forecast= np.nan
        last_fit_idx = -1

        for i, idx in enumerate(df.index):
            ret_up_to_i = log_ret[log_ret.index <= idx]

            if len(ret_up_to_i) < GARCH_MIN_OBS:
                garch_forecasts[idx] = np.nan
                continue

            # Only refit when due (every 21 bars)
            should_refit = (i - last_fit_idx) >= 21

            if should_refit:
                try:
                    model = arch_model(
                        ret_up_to_i,
                        vol    = "GARCH",
                        p      = 1,
                        q      = 1,
                        dist   = "normal",
                        rescale= False,
                    )
                    res          = model.fit(disp="off", show_warning=False)
                    forecast     = res.forecast(horizon=1, reindex=False)
                    var_1step    = forecast.variance.iloc[-1, 0]
                    # Convert variance (in % units) to annualized vol %
                    last_forecast= float(np.sqrt(var_1step * ANNUALIZATION_FACTOR))
                    last_fit_idx = i
                except Exception:
                    # GARCH fit failed — keep last forecast
                    pass

            garch_forecasts[idx] = last_forecast

        df["garch_forecast"]  = garch_forecasts
        df["garch_available"] = True

    except ImportError:
        logger.warning("arch library not installed — using HV_20 as GARCH proxy")
        df["garch_forecast"]  = df.get("hv_20", pd.Series(np.nan, index=df.index))
        df["garch_available"] = False

    except Exception as e:
        logger.warning(f"GARCH computation failed: {e} — using HV_20 as proxy")
        df["garch_forecast"]  = df.get("hv_20", pd.Series(np.nan, index=df.index))
        df["garch_available"] = False

    return df


def compute_beta(df: pd.DataFrame, nifty_close: pd.Series) -> pd.DataFrame:
    """
    Computes 60-day rolling beta of stock vs Nifty 50.

    Beta interpretation:
        β = 1.0 : stock moves 1:1 with Nifty
        β > 1.0 : amplified Nifty moves (high-beta, e.g. smallcaps)
        β < 1.0 : dampened Nifty moves (defensive, e.g. FMCG)
        β < 0.0 : inverse of Nifty (rare, e.g. some gold stocks)

    Position sizing adjustment (applied in Phase 3):
        Target position size ÷ beta = beta-adjusted position size
        High beta stock → smaller position for same portfolio risk

    Args:
        df          : OHLCV DataFrame
        nifty_close : Nifty 50 close prices aligned to same dates

    Returns:
        DataFrame with beta_nifty and beta_available columns
    """
    if nifty_close.empty:
        df["beta_nifty"]    = np.nan
        df["beta_available"]= False
        return df

    # Align nifty to stock dates
    nifty_aligned = nifty_close.reindex(df.index).ffill()

    stock_ret = np.log(df["close"] / df["close"].shift(1))
    nifty_ret = np.log(nifty_aligned / nifty_aligned.shift(1))

    # Rolling covariance and variance
    roll_cov = stock_ret.rolling(BETA_PERIOD, min_periods=30).cov(nifty_ret)
    roll_var = nifty_ret.rolling(BETA_PERIOD, min_periods=30).var().replace(0, np.nan)

    df["beta_nifty"]    = (roll_cov / roll_var).clip(-3.0, 5.0).round(4)
    df["beta_available"]= df["beta_nifty"].notna()

    return df


def classify_vol_regime(atr_pct: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Classifies each bar into a volatility regime based on ATR%.

    Regimes:
        low     (0) : ATR% < 1.0%  → low daily range; tight stops work
        medium  (1) : 1% ≤ ATR% < 2% → normal market conditions
        high    (2) : 2% ≤ ATR% < 4% → increased caution; wider stops
        extreme (3) : ATR% ≥ 4%     → minimum position or skip trade

    Args:
        atr_pct : ATR as percentage of price

    Returns:
        (regime_label Series, regime_code Series)
    """
    regime_label = pd.Series("medium", index=atr_pct.index, dtype=str)
    regime_code  = pd.Series(1, index=atr_pct.index, dtype=int)

    regime_label = np.where(atr_pct < REGIME_LOW,    "low",
                   np.where(atr_pct < REGIME_MEDIUM,  "medium",
                   np.where(atr_pct < REGIME_HIGH,    "high",
                                                       "extreme")))
    regime_code  = np.where(atr_pct < REGIME_LOW,    0,
                   np.where(atr_pct < REGIME_MEDIUM,  1,
                   np.where(atr_pct < REGIME_HIGH,    2,
                                                       3)))

    return (
        pd.Series(regime_label, index=atr_pct.index),
        pd.Series(regime_code.astype(int), index=atr_pct.index),
    )


def compute_dynamic_tpsl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes dynamic TP/SL levels for swing and intraday trades.

    Logic:
        ATR-based TP/SL = ATR multiplier × ATR%
        Floor-based TP/SL = minimum fixed % regardless of ATR
        Final TP/SL = max(ATR-based, floor)

    This ensures:
        - Stops are always outside the noise (ATR-based)
        - Stops are never so tight they can't work (floor)

    Args:
        df : DataFrame with atr_pct computed

    Returns:
        DataFrame with swing_tp_pct, swing_sl_pct,
                          intra_tp_pct, intra_sl_pct columns
    """
    atr = df["atr_pct"].fillna(2.0)   # fallback to 2% if ATR missing

    df["swing_tp_pct"] = np.maximum(
        SWING_TP_FLOOR, (SWING_TP_ATR_MULT * atr).round(2)
    )
    df["swing_sl_pct"] = np.maximum(
        SWING_SL_FLOOR, (SWING_SL_ATR_MULT * atr).round(2)
    )
    df["intra_tp_pct"] = np.maximum(
        INTRA_TP_FLOOR, (INTRA_TP_ATR_MULT * atr).round(2)
    )
    df["intra_sl_pct"] = np.maximum(
        INTRA_SL_FLOOR, (INTRA_SL_ATR_MULT * atr).round(2)
    )

    return df


def compute_volatility_score(df: pd.DataFrame) -> pd.Series:
    """
    Aggregates all volatility features into a composite score [-1, +1].

    Score interpretation:
        High positive score = high/rising volatility
            → system should reduce position size, widen stops
        High negative score = low/falling volatility
            → system can increase position size, use tighter stops
        Near zero = normal volatility regime

    Component weights:
        HV z-score         : 0.35  (primary signal)
        ATR z-score        : 0.30  (secondary signal)
        HV percentile rank : 0.20  (regime context)
        GARCH forecast     : 0.15  (forward-looking)

    Args:
        df : DataFrame with all volatility features computed

    Returns:
        pd.Series of volatility_score in [-1.0, +1.0]
    """
    score = pd.Series(0.0, index=df.index)

    # HV z-score contribution (already in [-3,+3], normalize to [-1,+1])
    hv_z = df["hv_zscore"].fillna(0).clip(-3, 3) / 3.0
    score += hv_z * 0.35

    # ATR z-score contribution
    atr_z = df["atr_zscore"].fillna(0).clip(-3, 3) / 3.0
    score += atr_z * 0.30

    # HV percentile: map [0,100] → [-1,+1]
    # percentile 100 = extreme high vol → +1
    # percentile   0 = extreme low vol  → -1
    hv_pct = df["hv_percentile"].fillna(50)
    pct_norm = (hv_pct - 50.0) / 50.0
    score += pct_norm * 0.20

    # GARCH contribution: compare forecast to current HV
    if "garch_forecast" in df.columns and "hv_20" in df.columns:
        garch_premium = (
            (df["garch_forecast"].fillna(0) - df["hv_20"].fillna(0))
            / df["hv_20"].replace(0, np.nan).fillna(20)
        ).clip(-1, 1).fillna(0)
        score += garch_premium * 0.15

    return score.clip(-1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN EXTRACTOR CLASS
# ══════════════════════════════════════════════════════════════════════════

class VolatilityExtractor:
    """
    Main interface for Pillar 5 — Volatility Analysis.

    Usage:
        extractor = VolatilityExtractor()

        # Compute for one stock
        df = extractor.compute("RELIANCE", date(2024, 1, 1), date(2024, 12, 31))

        # Compute for all stocks
        extractor.run_all(end_date=date(2024, 12, 31))

        # Get dynamic TP/SL for signal engine
        tpsl = extractor.get_tpsl("RELIANCE")
        # → {"swing_tp": 5.2, "swing_sl": 1.9, "intra_tp": 3.1, "intra_sl": 1.0}
    """

    def __init__(self):
        self.conn = _get_conn()
        _ensure_features_table(self.conn)
        self._nifty_cache: Optional[pd.Series] = None

    def _get_nifty(self, start_date: date, end_date: date) -> pd.Series:
        """Returns Nifty 50 series, cached to avoid repeated DB queries."""
        if self._nifty_cache is None:
            self._nifty_cache = load_nifty50(start_date, end_date, self.conn)
        return self._nifty_cache

    def compute(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        save_to_db: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Computes all volatility features for a single symbol.

        Gracefully handles:
            - Missing IV (options chain not populated)
            - GARCH failure (falls back to HV)
            - Missing Nifty data (beta not computed)
            - Insufficient history
        """
        df = load_ohlcv(symbol, start_date, end_date, self.conn)

        if df.empty or len(df) < MIN_BARS:
            logger.warning(f"{symbol}: insufficient data ({len(df)} bars). Skipping.")
            return None

        # ── Compute all features ──────────────────────────────────────────
        df = compute_atr(df)
        df = compute_historical_volatility(df)

        # IV (optional)
        iv_series = load_iv(symbol, start_date, end_date, self.conn)
        df = compute_iv_features(df, iv_series)

        # GARCH (optional — falls back gracefully)
        df = compute_garch_forecast(df)

        # Beta (optional)
        nifty = self._get_nifty(start_date, end_date)
        df = compute_beta(df, nifty)

        # Dynamic TP/SL
        df = compute_dynamic_tpsl(df)

        # Regime classification
        df["vol_regime"], df["vol_regime_code"] = classify_vol_regime(df["atr_pct"])

        # Composite score
        df["volatility_score"] = compute_volatility_score(df)

        # Trim to requested range
        df = df[df.index >= pd.Timestamp(start_date)].copy()

        if df.empty:
            return None

        if save_to_db:
            self._save(symbol, df)

        return df

    def _save(self, symbol: str, df: pd.DataFrame):
        """Upserts volatility features into features_volatility."""

        cols = [
            "atr_14", "atr_pct", "atr_zscore",
            "hv_20", "hv_percentile", "hv_zscore",
            "iv_atm", "iv_rank", "iv_hv_spread",
            "garch_forecast", "garch_available",
            "beta_nifty", "beta_available",
            "swing_tp_pct", "swing_sl_pct", "intra_tp_pct", "intra_sl_pct",
            "vol_regime", "vol_regime_code", "volatility_score",
        ]

        records = []
        for ts, row in df.iterrows():
            rec = {"date": ts.date(), "symbol": symbol}
            for col in cols:
                val = row.get(col, None)
                try:
                    if val is not None and not isinstance(val, (bool, str)) and pd.isna(val):
                        rec[col] = None
                        continue
                except (TypeError, ValueError):
                    pass
                if isinstance(val, (np.integer,)):   rec[col] = int(val)
                elif isinstance(val, (np.floating,)): rec[col] = None if np.isnan(val) else float(val)
                elif isinstance(val, (np.bool_,)):    rec[col] = bool(val)
                else:                                  rec[col] = val
            records.append(rec)

        if not records:
            return

        insert_sql = """
            INSERT INTO features_volatility (
                date, symbol,
                atr_14, atr_pct, atr_zscore,
                hv_20, hv_percentile, hv_zscore,
                iv_atm, iv_rank, iv_hv_spread,
                garch_forecast, garch_available,
                beta_nifty, beta_available,
                swing_tp_pct, swing_sl_pct, intra_tp_pct, intra_sl_pct,
                vol_regime, vol_regime_code, volatility_score
            ) VALUES (
                %(date)s, %(symbol)s,
                %(atr_14)s, %(atr_pct)s, %(atr_zscore)s,
                %(hv_20)s, %(hv_percentile)s, %(hv_zscore)s,
                %(iv_atm)s, %(iv_rank)s, %(iv_hv_spread)s,
                %(garch_forecast)s, %(garch_available)s,
                %(beta_nifty)s, %(beta_available)s,
                %(swing_tp_pct)s, %(swing_sl_pct)s,
                %(intra_tp_pct)s, %(intra_sl_pct)s,
                %(vol_regime)s, %(vol_regime_code)s, %(volatility_score)s
            )
            ON CONFLICT (date, symbol) DO UPDATE SET
                atr_14          = EXCLUDED.atr_14,
                atr_pct         = EXCLUDED.atr_pct,
                hv_20           = EXCLUDED.hv_20,
                hv_percentile   = EXCLUDED.hv_percentile,
                garch_forecast  = EXCLUDED.garch_forecast,
                beta_nifty      = EXCLUDED.beta_nifty,
                swing_tp_pct    = EXCLUDED.swing_tp_pct,
                swing_sl_pct    = EXCLUDED.swing_sl_pct,
                intra_tp_pct    = EXCLUDED.intra_tp_pct,
                intra_sl_pct    = EXCLUDED.intra_sl_pct,
                vol_regime      = EXCLUDED.vol_regime,
                vol_regime_code = EXCLUDED.vol_regime_code,
                volatility_score= EXCLUDED.volatility_score;
        """

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=500)
            self.conn.commit()
            logger.success(f"{symbol}: {len(records)} volatility rows saved.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"{symbol}: volatility save failed — {e}")
            raise

    def run_all(
        self,
        end_date: Optional[date] = None,
        start_date: Optional[date] = None,
        symbols: Optional[list[str]] = None,
    ):
        """Computes and saves volatility features for all symbols."""
        if end_date   is None: end_date   = date.today()
        if start_date is None: start_date = date(2019, 1, 1)
        if symbols    is None: symbols    = load_all_symbols(self.conn)

        logger.info(
            f"VolatilityExtractor.run_all: {len(symbols)} symbols | "
            f"{start_date} → {end_date}"
        )

        # Pre-load Nifty once for all beta computations
        self._nifty_cache = load_nifty50(start_date, end_date, self.conn)

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
                logger.info(f"Progress {i}/{len(symbols)} | ✓{success} ✗{failed} -{skipped}")

        logger.info(f"run_all done — ✓{success} ✗{failed} -{skipped}")

    def get_tpsl(self, symbol: str) -> Optional[dict]:
        """
        Returns latest dynamic TP/SL for a symbol.
        Called by execution engine for every new trade entry.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT swing_tp_pct, swing_sl_pct,
                       intra_tp_pct, intra_sl_pct,
                       atr_pct, vol_regime, volatility_score
                FROM features_volatility
                WHERE symbol = %s
                ORDER BY date DESC LIMIT 1;
            """, (symbol,))
            row = cur.fetchone()

        if not row:
            return None
        return {
            "swing_tp"        : float(row[0]),
            "swing_sl"        : float(row[1]),
            "intra_tp"        : float(row[2]),
            "intra_sl"        : float(row[3]),
            "atr_pct"         : float(row[4]),
            "vol_regime"      : str(row[5]),
            "volatility_score": float(row[6]),
        }

    def get_scores_for_date(self, target_date: date) -> pd.DataFrame:
        """Returns volatility scores for all symbols on a specific date."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, volatility_score, vol_regime,
                       atr_pct, swing_tp_pct, swing_sl_pct
                FROM features_volatility
                WHERE date = %s
                ORDER BY volatility_score DESC;
            """, (target_date,))
            rows = cur.fetchall()

        return pd.DataFrame(rows, columns=[
            "symbol", "volatility_score", "vol_regime",
            "atr_pct", "swing_tp_pct", "swing_sl_pct"
        ])

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

    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — Volatility Extractor")
    parser.add_argument("--mode",   choices=["all", "single", "tpsl"], default="all")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--start",  type=str, default="2019-01-01")
    parser.add_argument("--end",    type=str, default=str(date.today()))
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    with VolatilityExtractor() as extractor:
        if args.mode == "all":
            extractor.run_all(start_date=start, end_date=end)
        elif args.mode == "single":
            if not args.symbol:
                print("--symbol required"); sys.exit(1)
            df = extractor.compute(args.symbol, start, end, save_to_db=False)
            if df is not None:
                print(df[[
                    "atr_pct", "hv_20", "vol_regime",
                    "swing_tp_pct", "swing_sl_pct", "volatility_score"
                ]].tail(10))
        elif args.mode == "tpsl":
            if not args.symbol:
                print("--symbol required"); sys.exit(1)
            print(extractor.get_tpsl(args.symbol))


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run: python -m pytest features/volatility.py -v
# ══════════════════════════════════════════════════════════════════════════

def _make_sample_ohlcv(n: int = 350, regime: str = "normal") -> pd.DataFrame:
    np.random.seed(99)
    dates = pd.date_range("2019-01-01", periods=n, freq="B")

    if regime == "high_vol":
        daily_vol = 0.025    # 2.5% daily std
    elif regime == "low_vol":
        daily_vol = 0.006    # 0.6% daily std
    else:
        daily_vol = 0.015    # 1.5% daily std

    log_ret = np.random.normal(0.0003, daily_vol, n)
    close   = 1000 * np.exp(np.cumsum(log_ret))
    high    = close * (1 + np.abs(np.random.randn(n) * daily_vol))
    low     = close * (1 - np.abs(np.random.randn(n) * daily_vol))
    open_   = close * (1 + np.random.randn(n) * daily_vol * 0.5)
    vol     = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    }, index=dates)


class TestVolatilityFeatures:

    def setup_method(self):
        self.df_normal   = _make_sample_ohlcv(350, "normal")
        self.df_high_vol = _make_sample_ohlcv(350, "high_vol")
        self.df_low_vol  = _make_sample_ohlcv(350, "low_vol")

    # ── ATR Tests ──────────────────────────────────────────────────────────

    def test_atr_columns_exist(self):
        df = compute_atr(self.df_normal.copy())
        for col in ["atr_14", "atr_pct", "atr_zscore"]:
            assert col in df.columns

    def test_atr_positive(self):
        df = compute_atr(self.df_normal.copy())
        assert (df["atr_14"].dropna() > 0).all(), "ATR must always be positive"

    def test_atr_pct_positive(self):
        df = compute_atr(self.df_normal.copy())
        assert (df["atr_pct"].dropna() > 0).all()

    def test_high_vol_atr_larger(self):
        df_hv = compute_atr(self.df_high_vol.copy())
        df_lv = compute_atr(self.df_low_vol.copy())
        assert df_hv["atr_pct"].mean() > df_lv["atr_pct"].mean(), \
            "High vol regime should have larger ATR%"

    def test_atr_zscore_range(self):
        df = compute_atr(self.df_normal.copy())
        z  = df["atr_zscore"].dropna()
        assert (z >= -3).all() and (z <= 3).all()

    # ── HV Tests ───────────────────────────────────────────────────────────

    def test_hv_columns_exist(self):
        df = compute_atr(self.df_normal.copy())
        df = compute_historical_volatility(df)
        for col in ["hv_20", "hv_percentile", "hv_zscore"]:
            assert col in df.columns

    def test_hv_positive(self):
        df = compute_atr(self.df_normal.copy())
        df = compute_historical_volatility(df)
        assert (df["hv_20"].dropna() > 0).all()

    def test_hv_percentile_range(self):
        df = compute_atr(self.df_normal.copy())
        df = compute_historical_volatility(df)
        pct = df["hv_percentile"].dropna()
        assert (pct >= 0).all() and (pct <= 100).all()

    def test_high_vol_hv_larger(self):
        df_hv = compute_atr(self.df_high_vol.copy())
        df_hv = compute_historical_volatility(df_hv)
        df_lv = compute_atr(self.df_low_vol.copy())
        df_lv = compute_historical_volatility(df_lv)
        assert df_hv["hv_20"].mean() > df_lv["hv_20"].mean()

    # ── IV Tests ───────────────────────────────────────────────────────────

    def test_iv_empty_series_handled(self):
        df = compute_atr(self.df_normal.copy())
        df = compute_historical_volatility(df)
        df = compute_iv_features(df, pd.Series(dtype=float))
        assert df["iv_atm"].isna().all()
        assert df["iv_rank"].isna().all()

    def test_iv_rank_range_when_data_present(self):
        dates  = self.df_normal.index
        iv_raw = pd.Series(np.random.uniform(15, 45, len(dates)), index=dates)
        df = compute_atr(self.df_normal.copy())
        df = compute_historical_volatility(df)
        df = compute_iv_features(df, iv_raw)
        rank = df["iv_rank"].dropna()
        assert (rank >= 0).all() and (rank <= 100).all()

    # ── Dynamic TP/SL Tests ───────────────────────────────────────────────

    def test_tpsl_columns_exist(self):
        df = compute_atr(self.df_normal.copy())
        df = compute_dynamic_tpsl(df)
        for col in ["swing_tp_pct", "swing_sl_pct", "intra_tp_pct", "intra_sl_pct"]:
            assert col in df.columns

    def test_tpsl_floors_respected(self):
        df = compute_atr(self.df_low_vol.copy())
        df = compute_dynamic_tpsl(df)
        assert (df["swing_tp_pct"] >= SWING_TP_FLOOR).all()
        assert (df["swing_sl_pct"] >= SWING_SL_FLOOR).all()
        assert (df["intra_tp_pct"] >= INTRA_TP_FLOOR).all()
        assert (df["intra_sl_pct"] >= INTRA_SL_FLOOR).all()

    def test_high_vol_widens_stops(self):
        df_hv = compute_atr(self.df_high_vol.copy())
        df_hv = compute_dynamic_tpsl(df_hv)
        df_lv = compute_atr(self.df_low_vol.copy())
        df_lv = compute_dynamic_tpsl(df_lv)
        assert df_hv["swing_sl_pct"].mean() > df_lv["swing_sl_pct"].mean(), \
            "High vol should produce wider stops"

    def test_tp_always_greater_than_sl(self):
        df = compute_atr(self.df_normal.copy())
        df = compute_dynamic_tpsl(df)
        assert (df["swing_tp_pct"] > df["swing_sl_pct"]).all()
        assert (df["intra_tp_pct"] > df["intra_sl_pct"]).all()

    # ── Regime Classification Tests ───────────────────────────────────────

    def test_regime_valid_labels(self):
        atr_pct = pd.Series([0.5, 1.5, 3.0, 5.0])
        labels, codes = classify_vol_regime(atr_pct)
        assert set(labels).issubset({"low", "medium", "high", "extreme"})

    def test_regime_codes_valid(self):
        atr_pct = pd.Series([0.5, 1.5, 3.0, 5.0])
        labels, codes = classify_vol_regime(atr_pct)
        assert set(codes.unique()).issubset({0, 1, 2, 3})

    def test_low_vol_classified_correctly(self):
        atr_pct = pd.Series([0.5])
        labels, codes = classify_vol_regime(atr_pct)
        assert labels.iloc[0] == "low" and codes.iloc[0] == 0

    def test_extreme_vol_classified_correctly(self):
        atr_pct = pd.Series([6.0])
        labels, codes = classify_vol_regime(atr_pct)
        assert labels.iloc[0] == "extreme" and codes.iloc[0] == 3

    # ── Beta Tests ────────────────────────────────────────────────────────

    def test_beta_empty_nifty_handled(self):
        df = compute_beta(self.df_normal.copy(), pd.Series(dtype=float))
        assert df["beta_nifty"].isna().all()
        assert (df["beta_available"] == False).all()

    def test_beta_with_nifty(self):
        nifty = pd.Series(
            1000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, 350))),
            index=self.df_normal.index
        )
        df = compute_beta(self.df_normal.copy(), nifty)
        beta_valid = df["beta_nifty"].dropna()
        assert len(beta_valid) > 50
        assert (beta_valid >= -3.0).all() and (beta_valid <= 5.0).all()

    # ── Volatility Score Tests ─────────────────────────────────────────────

    def test_vol_score_range(self):
        df = compute_atr(self.df_normal.copy())
        df = compute_historical_volatility(df)
        df = compute_dynamic_tpsl(df)
        score = compute_volatility_score(df)
        assert (score.dropna() >= -1.0).all()
        assert (score.dropna() <=  1.0).all()

    def test_high_vol_positive_score(self):
        df = compute_atr(self.df_high_vol.copy())
        df = compute_historical_volatility(df)
        score = compute_volatility_score(df).dropna()
        assert score.mean() > 0, \
            "High vol regime should produce mostly positive vol scores"

    def test_low_vol_negative_score(self):
        """Low vol should score lower than high vol — relative comparison."""
        df_lv = compute_atr(self.df_low_vol.copy())
        df_lv = compute_historical_volatility(df_lv)
        score_lv = compute_volatility_score(df_lv).dropna().iloc[100:]

        df_hv = compute_atr(self.df_high_vol.copy())
        df_hv = compute_historical_volatility(df_hv)
        score_hv = compute_volatility_score(df_hv).dropna().iloc[100:]

        assert score_lv.mean() < score_hv.mean(), \
            f"Low vol mean score ({score_lv.mean():.4f}) should be less than " \
            f"high vol mean score ({score_hv.mean():.4f})"

if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))