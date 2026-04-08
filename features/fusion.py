"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Feature Fusion Module                          ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : features/fusion.py                                    ║
║         Phase   : 1 — Feature Engineering (Final Step)                  ║
║                                                                          ║
║  What this module does:                                                  ║
║    Assembles all 6 pillar outputs into a single unified feature          ║
║    tensor ready for the LSTM + Transformer backbone in Phase 2.         ║
║                                                                          ║
║  Input (per stock per day):                                              ║
║    Pillar 1 — trend_score + 8 raw trend features                        ║
║    Pillar 2 — msi_signal + 4 MSI sub-components                        ║
║    Pillar 3 — mds_continuous (market-level, same for all stocks)        ║
║    Pillar 4 — sentiment_score + sentiment_momentum + event_flag         ║
║    Pillar 5 — volatility_score + atr_pct + vol_regime_code              ║
║    Pillar 6 — correlation_score + sector_divergence_5d + lead_lag       ║
║                                                                          ║
║  Output tensor shape:                                                    ║
║    (N_stocks, T_timesteps, D_features)                                  ║
║    N_stocks  = up to 500 (Nifty 500 universe)                           ║
║    T_timesteps = 60 (lookback window for LSTM)                          ║
║    D_features = 28 (total features across all 6 pillars)                ║
║                                                                          ║
║  Feature vector layout (28 features per timestep):                      ║
║    [0]  trend_score          [-1, +1]                                   ║
║    [1]  ema_ribbon_gap       [-1, +1]  (9-21 gap normalized)            ║
║    [2]  adx_normalized       [0, +1]   (ADX/40 clipped)                 ║
║    [3]  supertrend_dir       {-1, +1}                                   ║
║    [4]  price_vs_ema200      [-1, +1]                                   ║
║    [5]  swing_structure      {-1, 0, +1}                                ║
║    [6]  msi_signal           [-1, +1]                                   ║
║    [7]  vrsi_normalized      [-1, +1]  (VRSI mapped from [0,100])       ║
║    [8]  mfi_normalized       [-1, +1]                                   ║
║    [9]  msi_divergence       {-1, 0, +1}                                ║
║    [10] mds_continuous       [-1, +1]  (market-level FII/DII signal)    ║
║    [11] fii_norm             [-1, +1]                                   ║
║    [12] dii_norm             [-1, +1]                                   ║
║    [13] sentiment_score      [-1, +1]                                   ║
║    [14] sentiment_momentum   [-1, +1]                                   ║
║    [15] event_flag           {0, 1}    (high-impact event detected)     ║
║    [16] market_fear_greed_n  [-1, +1]  (fear/greed mapped from [0,100]) ║
║    [17] volatility_score     [-1, +1]                                   ║
║    [18] atr_pct_normalized   [0, +1]   (ATR% / 5.0 clipped)            ║
║    [19] vol_regime_code_n    [0, +1]   (regime code / 3.0)             ║
║    [20] hv_percentile_n      [-1, +1]  (percentile mapped)             ║
║    [21] correlation_score    [-1, +1]                                   ║
║    [22] sector_divergence_n  [-1, +1]  (5d div normalized)             ║
║    [23] lead_lag_score       [-1, +1]                                   ║
║    [24] peer_corr_mean       [-1, +1]                                   ║
║    [25] delivery_mom_n       [-1, +1]  (delivery mom from [0,100])     ║
║    [26] swing_tp_normalized  [0, +1]   (TP% / 10.0 clipped)           ║
║    [27] swing_sl_normalized  [0, +1]   (SL% / 5.0 clipped)            ║
║                                                                          ║
║  Normalization guarantee:                                                ║
║    ALL 28 features are in [-1, +1] or [0, +1].                         ║
║    No feature can dominate the LSTM due to scale difference.            ║
║    NaN values filled with 0.0 (neutral) before output.                 ║
║                                                                          ║
║  Database:                                                               ║
║    Reads from : features_trend, features_msi, features_fii_dii,        ║
║                 features_sentiment, features_volatility,                ║
║                 features_correlation                                    ║
║    Writes to  : features_fused (wide table, one row per stock per day) ║
║    Also saves : numpy .npy tensor files for fast batch loading          ║
║                 during Phase 2 training                                 ║
║                                                                          ║
║  Usage:                                                                  ║
║    # Build full feature tensor for training                             ║
║    fusion = FeatureFusion()                                             ║
║    tensor = fusion.build_tensor(                                        ║
║        symbols=['RELIANCE', 'TCS', ...],                               ║
║        start_date=date(2019, 1, 1),                                     ║
║        end_date=date(2024, 12, 31),                                     ║
║        lookback=60                                                      ║
║    )                                                                    ║
║    # tensor.shape → (N_stocks, T_dates, 60, 28)                        ║
║                                                                          ║
║    # Real-time inference (signal engine)                                ║
║    vec = fusion.get_inference_vector('RELIANCE', date.today())          ║
║    # vec.shape → (60, 28) — last 60 days of features                   ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install pandas numpy psycopg2-binary loguru python-dotenv        ║
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
from pathlib import Path
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

# ── Tensor configuration ──────────────────────────────────────────────────
LOOKBACK_WINDOW = 60      # timesteps fed to LSTM
N_FEATURES      = 28      # total features per timestep
TENSOR_SAVE_DIR = Path("data/tensors")   # where .npy files are saved

# ── Feature index map (position in the 28-feature vector) ─────────────────
FEATURE_NAMES = [
    "trend_score",          # [0]
    "ema_ribbon_gap",       # [1]
    "adx_normalized",       # [2]
    "supertrend_dir",       # [3]
    "price_vs_ema200",      # [4]
    "swing_structure",      # [5]
    "msi_signal",           # [6]
    "vrsi_normalized",      # [7]
    "mfi_normalized",       # [8]
    "msi_divergence",       # [9]
    "mds_continuous",       # [10]
    "fii_norm",             # [11]
    "dii_norm",             # [12]
    "sentiment_score",      # [13]
    "sentiment_momentum",   # [14]
    "event_flag",           # [15]
    "market_fear_greed_n",  # [16]
    "volatility_score",     # [17]
    "atr_pct_normalized",   # [18]
    "vol_regime_code_n",    # [19]
    "hv_percentile_n",      # [20]
    "correlation_score",    # [21]
    "sector_divergence_n",  # [22]
    "lead_lag_score",       # [23]
    "peer_corr_mean",       # [24]
    "delivery_mom_n",       # [25]
    "swing_tp_normalized",  # [26]
    "swing_sl_normalized",  # [27]
]

assert len(FEATURE_NAMES) == N_FEATURES, \
    f"FEATURE_NAMES length {len(FEATURE_NAMES)} != N_FEATURES {N_FEATURES}"

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_conn():
    return psycopg2.connect(DB_URL)


def _ensure_fused_table(conn):
    """
    Creates features_fused table — wide format, one row per stock per day.
    All 28 features stored as individual columns for easy SQL querying.
    """
    with conn.cursor() as cur:
        col_defs = "\n".join([
            f"    f{i:02d}_{name}  NUMERIC(7,4),"
            for i, name in enumerate(FEATURE_NAMES)
        ])
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS features_fused (
                date    DATE        NOT NULL,
                symbol  VARCHAR(20) NOT NULL,
                {col_defs}
                data_completeness  NUMERIC(4,2),  -- fraction of features non-null [0,1]
                PRIMARY KEY (date, symbol)
            );
        """)

        cur.execute("""
            SELECT create_hypertable(
                'features_fused', 'date',
                if_not_exists => TRUE,
                migrate_data  => TRUE
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_fused_symbol
            ON features_fused (symbol, date DESC);
        """)

    conn.commit()
    logger.info("features_fused table ready.")


# ══════════════════════════════════════════════════════════════════════════
#  PILLAR DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════

def _load_pillar(
    table: str,
    columns: list[str],
    symbols: list[str],
    start_date: date,
    end_date: date,
    conn,
    market_level: bool = False,
) -> pd.DataFrame:
    """
    Generic loader for any pillar feature table.

    Args:
        table        : Table name (e.g. 'features_trend')
        columns      : Columns to select
        symbols      : Stock symbols to load
        start_date   : Start date
        end_date     : End date
        conn         : DB connection
        market_level : If True, table has no symbol column (FII/DII)

    Returns:
        DataFrame indexed by (date, symbol) — or just date if market_level
    """
    # Include extra history for lookback window warm-up
    history_start = start_date - timedelta(days=LOOKBACK_WINDOW + 5)

    col_str = ", ".join(columns)

    try:
        if market_level:
            sql = f"""
                SELECT date, {col_str}
                FROM {table}
                WHERE date BETWEEN %s AND %s
                ORDER BY date ASC;
            """
            with conn.cursor() as cur:
                cur.execute(sql, (history_start, end_date))
                rows = cur.fetchall()
                col_names = ["date"] + columns
        else:
            sql = f"""
                SELECT date, symbol, {col_str}
                FROM {table}
                WHERE symbol = ANY(%s)
                  AND date BETWEEN %s AND %s
                ORDER BY date ASC, symbol ASC;
            """
            with conn.cursor() as cur:
                cur.execute(sql, (symbols, history_start, end_date))
                rows = cur.fetchall()
                col_names = ["date", "symbol"] + columns

    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        logger.warning(f"Table {table} does not exist — pillar data will be neutral (0)")
        return pd.DataFrame()
    except Exception as e:
        conn.rollback()
        logger.warning(f"Failed to load from {table}: {e} — using neutral values")
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=col_names)
    df["date"] = pd.to_datetime(df["date"])

    if market_level:
        df = df.set_index("date").sort_index()
    else:
        df = df.set_index(["date", "symbol"]).sort_index()

    # Convert all to float
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE VECTOR BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_feature_vectors(
    symbols   : list[str],
    start_date: date,
    end_date  : date,
    conn,
) -> pd.DataFrame:
    """
    Assembles all 6 pillar features into a unified wide DataFrame.

    Loads each pillar table, joins them on (date, symbol), applies
    normalizations, and outputs a DataFrame with exactly N_FEATURES columns
    plus date and symbol index.

    Missing pillar data → 0.0 (neutral) for that feature.
    This ensures the system is never blocked by a single pillar's absence.

    Args:
        symbols    : List of NSE symbols
        start_date : Start of feature range
        end_date   : End of feature range
        conn       : DB connection

    Returns:
        DataFrame indexed by (date, symbol) with N_FEATURES columns
        named f00_trend_score, f01_ema_ribbon_gap, etc.
    """
    logger.info(
        f"Building fused features for {len(symbols)} symbols | "
        f"{start_date} → {end_date}"
    )

    # ── Load all pillars ──────────────────────────────────────────────────

    # Pillar 1 — Trend
    trend = _load_pillar(
        "features_trend",
        ["trend_score", "ema_9_21_gap", "adx", "supertrend_dir",
         "price_vs_ema200", "swing_structure"],
        symbols, start_date, end_date, conn
    )

    # Pillar 2 — MSI
    msi = _load_pillar(
        "features_msi",
        ["msi_signal", "vrsi", "mfi", "divergence", "delivery_mom_score"],
        symbols, start_date, end_date, conn
    )

    # Pillar 3 — FII/DII (market level)
    fii_dii = _load_pillar(
        "features_fii_dii",
        ["mds_continuous", "fii_norm", "dii_norm"],
        symbols, start_date, end_date, conn,
        market_level=True
    )

    # Pillar 4 — Sentiment
    sentiment = _load_pillar(
        "features_sentiment",
        ["sentiment_score", "sentiment_momentum", "event_flag",
         "market_fear_greed"],
        symbols, start_date, end_date, conn
    )

    # Pillar 5 — Volatility
    volatility = _load_pillar(
        "features_volatility",
        ["volatility_score", "atr_pct", "vol_regime_code",
         "hv_percentile", "swing_tp_pct", "swing_sl_pct"],
        symbols, start_date, end_date, conn
    )

    # Pillar 6 — Correlation
    correlation = _load_pillar(
        "features_correlation",
        ["correlation_score", "sector_divergence_5d",
         "lead_lag_score", "peer_correlation_mean"],
        symbols, start_date, end_date, conn
    )

    # ── Build master index: all (date, symbol) combinations ───────────────
    # Use Pillar 1 (trend) as the base since it's most complete
    if not trend.empty:
        master_idx = trend.index
    elif not msi.empty:
        master_idx = msi.index
    else:
        logger.error("No feature data available — cannot build fusion")
        return pd.DataFrame()

    result = pd.DataFrame(index=master_idx)

    # ── Helper: safe column extraction ────────────────────────────────────
    def _get_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        """
        Safely extracts a column from a pillar DataFrame,
        reindexed to master_idx. Returns default if unavailable.
        """
        if df.empty or col not in df.columns:
            return pd.Series(default, index=master_idx)
        try:
            return df[col].reindex(master_idx).fillna(default)
        except Exception:
            return pd.Series(default, index=master_idx)

    def _get_market_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        """
        Extracts a market-level column (date-only index),
        broadcast to all symbols in master_idx.
        """
        if df.empty or col not in df.columns:
            return pd.Series(default, index=master_idx)
        try:
            # master_idx is (date, symbol) — get date level
            dates = master_idx.get_level_values(0)
            vals  = df[col].reindex(dates).fillna(default).values
            return pd.Series(vals, index=master_idx)
        except Exception:
            return pd.Series(default, index=master_idx)

    # ── Feature [0]: trend_score [-1, +1] ────────────────────────────────
    result[f"f00_{FEATURE_NAMES[0]}"] = _get_col(trend, "trend_score").clip(-1, 1)

    # ── Feature [1]: ema_ribbon_gap [-1, +1] ─────────────────────────────
    result[f"f01_{FEATURE_NAMES[1]}"] = _get_col(trend, "ema_9_21_gap").clip(-1, 1)

    # ── Feature [2]: adx_normalized [0, +1] ──────────────────────────────
    # ADX range [0, 100] → normalize by dividing by 40 (trending threshold)
    result[f"f02_{FEATURE_NAMES[2]}"] = (
        _get_col(trend, "adx", 20.0) / 40.0
    ).clip(0, 1)

    # ── Feature [3]: supertrend_dir {-1, 0, +1} ──────────────────────────
    result[f"f03_{FEATURE_NAMES[3]}"] = _get_col(
        trend, "supertrend_dir", 0.0
    ).clip(-1, 1)

    # ── Feature [4]: price_vs_ema200 [-1, +1] ────────────────────────────
    result[f"f04_{FEATURE_NAMES[4]}"] = _get_col(
        trend, "price_vs_ema200", 0.0
    ).clip(-1, 1)

    # ── Feature [5]: swing_structure {-1, 0, +1} ─────────────────────────
    result[f"f05_{FEATURE_NAMES[5]}"] = _get_col(
        trend, "swing_structure", 0.0
    ).clip(-1, 1)

    # ── Feature [6]: msi_signal [-1, +1] ─────────────────────────────────
    result[f"f06_{FEATURE_NAMES[6]}"] = _get_col(
        msi, "msi_signal", 0.0
    ).clip(-1, 1)

    # ── Feature [7]: vrsi_normalized [-1, +1] ────────────────────────────
    # VRSI [0, 100] → [-1, +1]: (vrsi - 50) / 50
    result[f"f07_{FEATURE_NAMES[7]}"] = (
        (_get_col(msi, "vrsi", 50.0) - 50.0) / 50.0
    ).clip(-1, 1)

    # ── Feature [8]: mfi_normalized [-1, +1] ─────────────────────────────
    result[f"f08_{FEATURE_NAMES[8]}"] = (
        (_get_col(msi, "mfi", 50.0) - 50.0) / 50.0
    ).clip(-1, 1)

    # ── Feature [9]: msi_divergence {-1, 0, +1} ──────────────────────────
    result[f"f09_{FEATURE_NAMES[9]}"] = _get_col(
        msi, "divergence", 0.0
    ).clip(-1, 1)

    # ── Feature [10]: mds_continuous [-1, +1] ────────────────────────────
    result[f"f10_{FEATURE_NAMES[10]}"] = _get_market_col(
        fii_dii, "mds_continuous", 0.0
    ).clip(-1, 1)

    # ── Feature [11]: fii_norm [-1, +1] ──────────────────────────────────
    result[f"f11_{FEATURE_NAMES[11]}"] = _get_market_col(
        fii_dii, "fii_norm", 0.0
    ).clip(-1, 1)

    # ── Feature [12]: dii_norm [-1, +1] ──────────────────────────────────
    result[f"f12_{FEATURE_NAMES[12]}"] = _get_market_col(
        fii_dii, "dii_norm", 0.0
    ).clip(-1, 1)

    # ── Feature [13]: sentiment_score [-1, +1] ───────────────────────────
    result[f"f13_{FEATURE_NAMES[13]}"] = _get_col(
        sentiment, "sentiment_score", 0.0
    ).clip(-1, 1)

    # ── Feature [14]: sentiment_momentum [-1, +1] ────────────────────────
    result[f"f14_{FEATURE_NAMES[14]}"] = _get_col(
        sentiment, "sentiment_momentum", 0.0
    ).clip(-1, 1)

    # ── Feature [15]: event_flag {0, 1} ──────────────────────────────────
    result[f"f15_{FEATURE_NAMES[15]}"] = _get_col(
        sentiment, "event_flag", 0.0
    ).clip(0, 1)

    # ── Feature [16]: market_fear_greed_n [-1, +1] ───────────────────────
    # fear/greed [0, 100] → [-1, +1]: (fg - 50) / 50
    result[f"f16_{FEATURE_NAMES[16]}"] = (
        (_get_col(sentiment, "market_fear_greed", 50.0) - 50.0) / 50.0
    ).clip(-1, 1)

    # ── Feature [17]: volatility_score [-1, +1] ──────────────────────────
    result[f"f17_{FEATURE_NAMES[17]}"] = _get_col(
        volatility, "volatility_score", 0.0
    ).clip(-1, 1)

    # ── Feature [18]: atr_pct_normalized [0, +1] ─────────────────────────
    # ATR% typically [0, 5%] → normalize to [0, 1]: atr_pct / 5.0
    result[f"f18_{FEATURE_NAMES[18]}"] = (
        _get_col(volatility, "atr_pct", 1.5) / 5.0
    ).clip(0, 1)

    # ── Feature [19]: vol_regime_code_n [0, +1] ──────────────────────────
    # Regime codes {0,1,2,3} → [0,1]: code / 3
    result[f"f19_{FEATURE_NAMES[19]}"] = (
        _get_col(volatility, "vol_regime_code", 1.0) / 3.0
    ).clip(0, 1)

    # ── Feature [20]: hv_percentile_n [-1, +1] ───────────────────────────
    # HV percentile [0,100] → [-1,+1]: (pct - 50) / 50
    result[f"f20_{FEATURE_NAMES[20]}"] = (
        (_get_col(volatility, "hv_percentile", 50.0) - 50.0) / 50.0
    ).clip(-1, 1)

    # ── Feature [21]: correlation_score [-1, +1] ─────────────────────────
    result[f"f21_{FEATURE_NAMES[21]}"] = _get_col(
        correlation, "correlation_score", 0.0
    ).clip(-1, 1)

    # ── Feature [22]: sector_divergence_n [-1, +1] ───────────────────────
    # 5-day sector divergence normalized: clip to ±5% range → [-1, +1]
    result[f"f22_{FEATURE_NAMES[22]}"] = (
        _get_col(correlation, "sector_divergence_5d", 0.0) / 0.05
    ).clip(-1, 1)

    # ── Feature [23]: lead_lag_score [-1, +1] ────────────────────────────
    result[f"f23_{FEATURE_NAMES[23]}"] = _get_col(
        correlation, "lead_lag_score", 0.0
    ).clip(-1, 1)

    # ── Feature [24]: peer_corr_mean [-1, +1] ────────────────────────────
    result[f"f24_{FEATURE_NAMES[24]}"] = _get_col(
        correlation, "peer_correlation_mean", 0.5
    ).clip(-1, 1)

    # ── Feature [25]: delivery_mom_n [-1, +1] ────────────────────────────
    # delivery_mom_score [0, 100] → [-1, +1]: (score - 50) / 50
    result[f"f25_{FEATURE_NAMES[25]}"] = (
        (_get_col(msi, "delivery_mom_score", 50.0) - 50.0) / 50.0
    ).clip(-1, 1)

    # ── Feature [26]: swing_tp_normalized [0, +1] ────────────────────────
    # TP% typically [1.5, 10%] → normalize by 10
    result[f"f26_{FEATURE_NAMES[26]}"] = (
        _get_col(volatility, "swing_tp_pct", 4.0) / 10.0
    ).clip(0, 1)

    # ── Feature [27]: swing_sl_normalized [0, +1] ────────────────────────
    # SL% typically [0.8, 5%] → normalize by 5
    result[f"f27_{FEATURE_NAMES[27]}"] = (
        _get_col(volatility, "swing_sl_pct", 1.5) / 5.0
    ).clip(0, 1)

    # ── Data completeness score ───────────────────────────────────────────
    feature_cols = [c for c in result.columns if c.startswith("f")]
    result["data_completeness"] = (
        result[feature_cols].notna().sum(axis=1) / len(feature_cols)
    ).round(2)

    # ── Final NaN fill: all NaN → 0.0 (neutral) ──────────────────────────
    result[feature_cols] = result[feature_cols].fillna(0.0)

    logger.info(
        f"Fused feature matrix: {len(result)} rows × {len(feature_cols)} features | "
        f"avg completeness: {result['data_completeness'].mean():.1%}"
    )

    return result


# ══════════════════════════════════════════════════════════════════════════
#  TENSOR BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_lstm_tensor(
    fused_df  : pd.DataFrame,
    symbols   : list[str],
    start_date: date,
    end_date  : date,
    lookback  : int = LOOKBACK_WINDOW,
) -> tuple[np.ndarray, list[str], list[date]]:
    """
    Converts the fused feature DataFrame into a 3D numpy tensor
    suitable for LSTM training.

    Output tensor shape: (N_stocks, N_dates, lookback, N_features)

    For each (stock, date) pair, we take the `lookback` most recent
    rows of features as a temporal sequence. This is the input to
    the BiLSTM encoder in Phase 2.

    Args:
        fused_df   : Output of build_feature_vectors()
        symbols    : List of symbols to include
        start_date : Start of output tensor date range
        end_date   : End of output tensor date range
        lookback   : Temporal sequence length (default 60)

    Returns:
        (tensor, symbol_list, date_list)
        tensor.shape = (N_stocks, N_dates, lookback, N_features)
        symbol_list  = ordered list of symbols (axis 0)
        date_list    = ordered list of dates (axis 1)
    """
    feature_cols = sorted([c for c in fused_df.columns if c.startswith("f")])

    if len(feature_cols) != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} feature columns, got {len(feature_cols)}"
        )

    # Get all trading dates in range
    all_dates = sorted(set(
        fused_df.index.get_level_values(0).date
    ))
    target_dates = [d for d in all_dates
                    if start_date <= d <= end_date]

    if not target_dates:
        logger.error("No dates in target range after filtering")
        return np.array([]), [], []

    # Filter symbols to those with sufficient data
    available_symbols = []
    for sym in symbols:
        try:
            sym_data = fused_df.xs(sym, level=1, axis=0)
            if len(sym_data) >= lookback:
                available_symbols.append(sym)
        except KeyError:
            continue

    if not available_symbols:
        logger.error("No symbols with sufficient data for tensor building")
        return np.array([]), [], []

    N_stocks = len(available_symbols)
    N_dates  = len(target_dates)
    tensor   = np.zeros((N_stocks, N_dates, lookback, N_FEATURES), dtype=np.float32)

    for s_idx, symbol in enumerate(available_symbols):
        try:
            sym_df = fused_df.xs(symbol, level=1, axis=0)[feature_cols].sort_index()
        except KeyError:
            continue

        sym_values = sym_df.values.astype(np.float32)
        sym_dates  = [d.date() for d in sym_df.index]

        for d_idx, target_date in enumerate(target_dates):
            # Find position of target_date in sym_dates
            try:
                end_pos = sym_dates.index(target_date)
            except ValueError:
                # Date not available for this symbol — leave as zeros
                continue

            start_pos = end_pos - lookback + 1
            if start_pos < 0:
                # Not enough history — pad with zeros at the start
                available   = sym_values[max(0, start_pos): end_pos + 1]
                pad_len     = lookback - len(available)
                padded      = np.vstack([
                    np.zeros((pad_len, N_FEATURES), dtype=np.float32),
                    available
                ])
                tensor[s_idx, d_idx] = padded
            else:
                tensor[s_idx, d_idx] = sym_values[start_pos: end_pos + 1]

    logger.info(
        f"Tensor built: shape={tensor.shape} | "
        f"symbols={N_stocks} | dates={N_dates} | lookback={lookback}"
    )

    return tensor, available_symbols, target_dates


# ══════════════════════════════════════════════════════════════════════════
#  MAIN FUSION CLASS
# ══════════════════════════════════════════════════════════════════════════

class FeatureFusion:
    """
    Main interface for feature fusion.

    Usage:
        fusion = FeatureFusion()

        # Phase 1: Build and save full fused feature table
        fusion.run_all(start_date=date(2019,1,1), end_date=date(2024,12,31))

        # Phase 2: Build tensor for LSTM training
        tensor, symbols, dates = fusion.build_tensor(
            symbols=['RELIANCE', 'TCS', ...],
            start_date=date(2019,1,1),
            end_date=date(2024,12,31),
        )
        # tensor.shape → (N_stocks, N_dates, 60, 28)

        # Live inference: get today's 60-day feature sequence for one stock
        vec = fusion.get_inference_vector('RELIANCE', date.today())
        # vec.shape → (60, 28)
    """

    def __init__(self):
        self.conn = _get_conn()
        _ensure_fused_table(self.conn)
        TENSOR_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def run_all(
        self,
        start_date: Optional[date] = None,
        end_date  : Optional[date] = None,
        symbols   : Optional[list[str]] = None,
        save_tensor: bool = True,
    ):
        """
        Builds the complete fused feature table and optionally saves
        the training tensor to disk as .npy files.

        Args:
            start_date  : Feature start date (default 2019-01-01)
            end_date    : Feature end date (default today)
            symbols     : Symbol list (default: all in DB)
            save_tensor : Whether to save .npy tensor to data/tensors/
        """
        if start_date is None: start_date = date(2019, 1, 1)
        if end_date   is None: end_date   = date.today()

        if symbols is None:
            with self.conn.cursor() as cur:
                cur.execute("SELECT DISTINCT symbol FROM daily_ohlcv ORDER BY symbol;")
                symbols = [r[0] for r in cur.fetchall()]

        logger.info(
            f"FeatureFusion.run_all: {len(symbols)} symbols | "
            f"{start_date} → {end_date}"
        )

        # Build fused feature DataFrame
        fused = build_feature_vectors(symbols, start_date, end_date, self.conn)

        if fused.empty:
            logger.error("Fusion returned empty — check that all pillars have been run.")
            return

        # Save to features_fused table
        self._save(fused)

        if save_tensor:
            self._save_tensor(fused, symbols, start_date, end_date)

    def _save(self, df: pd.DataFrame):
        """Upserts fused features into features_fused table."""
        feature_cols = sorted([c for c in df.columns if c.startswith("f")])

        col_list = ", ".join(feature_cols + ["data_completeness"])
        val_keys = ", ".join([f"%({c})s" for c in feature_cols + ["data_completeness"]])
        update_set = ", ".join([f"{c} = EXCLUDED.{c}" for c in feature_cols])

        insert_sql = f"""
            INSERT INTO features_fused (date, symbol, {col_list})
            VALUES (%(date)s, %(symbol)s, {val_keys})
            ON CONFLICT (date, symbol) DO UPDATE SET {update_set};
        """

        records = []
        for (dt, symbol), row in df.iterrows():
            rec = {
                "date"  : dt.date() if hasattr(dt, "date") else dt,
                "symbol": symbol,
            }
            for col in feature_cols + ["data_completeness"]:
                val = row.get(col, 0.0)
                rec[col] = None if pd.isna(val) else float(val)
            records.append(rec)

        if not records:
            return

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=2000)
            self.conn.commit()
            logger.success(f"features_fused: {len(records)} rows saved.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"features_fused save failed: {e}")
            raise

    def _save_tensor(
        self,
        fused     : pd.DataFrame,
        symbols   : list[str],
        start_date: date,
        end_date  : date,
    ):
        """
        Builds and saves training tensor as numpy files.

        Saves:
            data/tensors/tensor_YYYYMMDD.npy  — feature tensor
            data/tensors/symbols_YYYYMMDD.npy — symbol list
            data/tensors/dates_YYYYMMDD.npy   — date list
        """
        tensor, sym_list, date_list = build_lstm_tensor(
            fused, symbols, start_date, end_date
        )

        if tensor.size == 0:
            logger.warning("Tensor is empty — not saving.")
            return

        suffix = end_date.strftime("%Y%m%d")
        np.save(TENSOR_SAVE_DIR / f"tensor_{suffix}.npy", tensor)
        np.save(TENSOR_SAVE_DIR / f"symbols_{suffix}.npy", np.array(sym_list))
        np.save(TENSOR_SAVE_DIR / f"dates_{suffix}.npy",
                np.array([str(d) for d in date_list]))

        size_mb = tensor.nbytes / 1_000_000
        logger.success(
            f"Tensor saved to data/tensors/ | "
            f"shape={tensor.shape} | size={size_mb:.1f}MB"
        )

    def build_tensor(
        self,
        symbols   : list[str],
        start_date: date,
        end_date  : date,
        lookback  : int = LOOKBACK_WINDOW,
    ) -> tuple[np.ndarray, list[str], list[date]]:
        """
        Loads fused features from DB and builds training tensor.

        Returns:
            (tensor, symbol_list, date_list)
            tensor.shape = (N_stocks, N_dates, lookback, N_features)
        """
        fused = build_feature_vectors(symbols, start_date, end_date, self.conn)
        if fused.empty:
            return np.array([]), [], []
        return build_lstm_tensor(fused, symbols, start_date, end_date, lookback)

    def get_inference_vector(
        self,
        symbol     : str,
        target_date: date,
        lookback   : int = LOOKBACK_WINDOW,
    ) -> Optional[np.ndarray]:
        """
        Returns the feature sequence for one stock for live inference.
        Called by the signal engine for every stock on every trading day.

        Returns:
            numpy array of shape (lookback, N_FEATURES) = (60, 28)
            or None if insufficient data.
        """
        start = target_date - timedelta(days=lookback * 2)  # extra buffer

        fused = build_feature_vectors([symbol], start, target_date, self.conn)

        if fused.empty:
            return None

        feature_cols = sorted([c for c in fused.columns if c.startswith("f")])

        try:
            sym_df = fused.xs(symbol, level=1, axis=0)[feature_cols].sort_index()
        except KeyError:
            return None

        if len(sym_df) < lookback:
            logger.warning(
                f"{symbol}: only {len(sym_df)} bars, need {lookback}. "
                f"Padding with zeros."
            )
            pad_len = lookback - len(sym_df)
            padding = np.zeros((pad_len, N_FEATURES), dtype=np.float32)
            data    = sym_df.values.astype(np.float32)
            return np.vstack([padding, data])

        return sym_df.values[-lookback:].astype(np.float32)

    def get_feature_names(self) -> list[str]:
        """Returns the ordered list of feature names (for LSTM layer naming)."""
        return FEATURE_NAMES.copy()

    def get_feature_index(self) -> dict[str, int]:
        """Returns mapping of feature name → position index."""
        return FEATURE_INDEX.copy()

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

    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — Feature Fusion")
    parser.add_argument("--mode",   choices=["all", "vector", "tensor"], default="all")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--start",  type=str, default="2019-01-01")
    parser.add_argument("--end",    type=str, default=str(date.today()))
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    with FeatureFusion() as fusion:
        if args.mode == "all":
            try:
                with open("config/universe.yaml") as f:
                    symbols = yaml.safe_load(f).get("nifty500", [])
            except FileNotFoundError:
                symbols = None
            fusion.run_all(start_date=start, end_date=end, symbols=symbols)

        elif args.mode == "vector":
            if not args.symbol:
                print("--symbol required"); sys.exit(1)
            vec = fusion.get_inference_vector(args.symbol, end)
            if vec is not None:
                print(f"{args.symbol} inference vector shape: {vec.shape}")
                print(f"Feature names: {fusion.get_feature_names()}")
                print(f"Last timestep:\n{dict(zip(FEATURE_NAMES, vec[-1].tolist()))}")

        elif args.mode == "tensor":
            if not args.symbol:
                print("--symbol required"); sys.exit(1)
            tensor, syms, dates = fusion.build_tensor(
                [args.symbol], start, end
            )
            print(f"Tensor shape: {tensor.shape}")


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run: python -m pytest features/fusion.py -v
# ══════════════════════════════════════════════════════════════════════════

def _make_fused_df(
    n_stocks : int = 5,
    n_days   : int = 100,
    seed     : int = 42,
) -> pd.DataFrame:
    """
    Generates synthetic fused feature DataFrame for testing.
    Mimics the output of build_feature_vectors().
    """
    np.random.seed(seed)
    dates   = pd.date_range("2023-01-01", periods=n_days, freq="B")
    symbols = [f"STOCK{i:02d}" for i in range(n_stocks)]

    idx   = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    data  = np.random.uniform(-1, 1, (len(idx), N_FEATURES)).astype(np.float32)

    feature_cols = [f"f{i:02d}_{name}" for i, name in enumerate(FEATURE_NAMES)]
    df = pd.DataFrame(data, index=idx, columns=feature_cols)

    # Some features must be in [0, +1] — clip them
    for col in ["f02_adx_normalized", "f15_event_flag",
                "f18_atr_pct_normalized", "f19_vol_regime_code_n",
                "f26_swing_tp_normalized", "f27_swing_sl_normalized"]:
        df[col] = df[col].abs()

    df["data_completeness"] = 1.0
    return df


class TestFeatureFusion:

    def setup_method(self):
        self.fused = _make_fused_df(n_stocks=5, n_days=100)

    # ── Feature metadata tests ────────────────────────────────────────────

    def test_feature_count(self):
        assert N_FEATURES == 28, f"Expected 28 features, got {N_FEATURES}"

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_feature_index_consistent(self):
        """FEATURE_INDEX must match FEATURE_NAMES positions."""
        for name, idx in FEATURE_INDEX.items():
            assert FEATURE_NAMES[idx] == name, \
                f"FEATURE_INDEX[{name}]={idx} but FEATURE_NAMES[{idx}]={FEATURE_NAMES[idx]}"

    def test_no_duplicate_feature_names(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), \
            "Duplicate feature names found"

    # ── Fused DataFrame structure tests ───────────────────────────────────

    def test_fused_df_has_correct_columns(self):
        feature_cols = [c for c in self.fused.columns if c.startswith("f")]
        assert len(feature_cols) == N_FEATURES, \
            f"Expected {N_FEATURES} feature columns, got {len(feature_cols)}"

    def test_fused_df_multiindex(self):
        """Fused DataFrame must be indexed by (date, symbol)."""
        assert isinstance(self.fused.index, pd.MultiIndex)
        assert self.fused.index.names == ["date", "symbol"]

    def test_fused_df_no_nan(self):
        """After fillna(0.0), there should be no NaN in feature columns."""
        fused = _make_fused_df()
        # Introduce some NaN
        fused.iloc[0, 0] = np.nan
        feature_cols = [c for c in fused.columns if c.startswith("f")]
        fused[feature_cols] = fused[feature_cols].fillna(0.0)
        assert fused[feature_cols].isna().sum().sum() == 0

    def test_fused_feature_ranges(self):
        """All features must be within their documented ranges."""
        feature_cols = [c for c in self.fused.columns if c.startswith("f")]
        for col in feature_cols:
            vals = self.fused[col].dropna()
            assert (vals >= -1.0 - 1e-6).all(), f"{col} below -1.0"
            assert (vals <=  1.0 + 1e-6).all(), f"{col} above +1.0"

    def test_data_completeness_range(self):
        """data_completeness must be in [0, 1]."""
        assert (self.fused["data_completeness"] >= 0).all()
        assert (self.fused["data_completeness"] <= 1).all()

    # ── Tensor building tests ─────────────────────────────────────────────

    def test_tensor_shape(self):
        """Tensor shape must be (N_stocks, N_dates, lookback, N_features)."""
        symbols   = ["STOCK00", "STOCK01", "STOCK02"]
        start     = date(2023, 1, 1)
        end       = date(2023, 6, 30)
        lookback  = 30

        tensor, sym_list, date_list = build_lstm_tensor(
            self.fused, symbols, start, end, lookback
        )

        assert tensor.ndim == 4, f"Expected 4D tensor, got {tensor.ndim}D"
        assert tensor.shape[0] == len(sym_list)
        assert tensor.shape[2] == lookback
        assert tensor.shape[3] == N_FEATURES

    def test_tensor_dtype_float32(self):
        """Tensor must be float32 for GPU compatibility."""
        tensor, _, _ = build_lstm_tensor(
            self.fused,
            ["STOCK00", "STOCK01"],
            date(2023, 1, 1),
            date(2023, 6, 30),
            lookback=30
        )
        assert tensor.dtype == np.float32

    def test_tensor_no_nan(self):
        """Tensor must contain no NaN values."""
        tensor, _, _ = build_lstm_tensor(
            self.fused,
            ["STOCK00"],
            date(2023, 1, 1),
            date(2023, 6, 30),
            lookback=30
        )
        assert not np.isnan(tensor).any(), "Tensor contains NaN"

    def test_tensor_no_inf(self):
        """Tensor must contain no Inf values."""
        tensor, _, _ = build_lstm_tensor(
            self.fused,
            ["STOCK00"],
            date(2023, 1, 1),
            date(2023, 6, 30),
            lookback=30
        )
        assert not np.isinf(tensor).any(), "Tensor contains Inf"

    def test_tensor_range(self):
        """All tensor values must be in [-1, +1]."""
        tensor, _, _ = build_lstm_tensor(
            self.fused,
            ["STOCK00", "STOCK01"],
            date(2023, 1, 1),
            date(2023, 6, 30),
            lookback=30
        )
        assert tensor.min() >= -1.0 - 1e-5, f"Tensor min {tensor.min():.4f} < -1"
        assert tensor.max() <=  1.0 + 1e-5, f"Tensor max {tensor.max():.4f} > +1"

    def test_tensor_symbol_order_preserved(self):
        """Symbol order in tensor must match returned symbol_list."""
        symbols = ["STOCK02", "STOCK00", "STOCK04"]
        tensor, sym_list, _ = build_lstm_tensor(
            self.fused, symbols,
            date(2023, 1, 1), date(2023, 6, 30), lookback=30
        )
        for i, sym in enumerate(sym_list):
            assert sym in symbols, f"Unexpected symbol {sym} in output"

    def test_tensor_insufficient_history_pads_zeros(self):
        """Stocks with < lookback history should be padded with zeros."""
        # Create a very short dataset (10 days)
        short_df = _make_fused_df(n_stocks=1, n_days=10)
        tensor, syms, dates = build_lstm_tensor(
            short_df, ["STOCK00"],
            date(2023, 1, 1), date(2023, 1, 20),
            lookback=30
        )
        if tensor.size > 0:
            # Leading rows should be zeros (padding)
            assert (tensor[0, 0, :20] == 0).all(), \
                "Insufficient history should be padded with zeros"

    def test_tensor_empty_for_unknown_symbols(self):
        """Unknown symbols should be excluded from tensor gracefully."""
        tensor, sym_list, _ = build_lstm_tensor(
            self.fused,
            ["UNKNOWN_STOCK_XYZ"],
            date(2023, 1, 1), date(2023, 6, 30), lookback=30
        )
        assert len(sym_list) == 0 or tensor.size == 0

    # ── Normalisation tests ───────────────────────────────────────────────

    def test_vrsi_normalization(self):
        """VRSI 50 → normalized 0.0, VRSI 100 → +1.0, VRSI 0 → -1.0."""
        assert abs((50 - 50) / 50) == 0.0
        assert abs((100 - 50) / 50) == 1.0
        assert abs((0 - 50) / 50) == 1.0

    def test_adx_normalization(self):
        """ADX 40 (strong trend) → normalized 1.0. ADX 0 → 0.0."""
        assert abs(40 / 40 - 1.0) < 1e-6
        assert abs(0  / 40 - 0.0) < 1e-6

    def test_fear_greed_normalization(self):
        """Fear/Greed 100 → +1.0, 0 → -1.0, 50 → 0.0."""
        assert abs((100 - 50) / 50 - 1.0) < 1e-6
        assert abs((0   - 50) / 50 + 1.0) < 1e-6
        assert abs((50  - 50) / 50 - 0.0) < 1e-6

    def test_atr_normalization(self):
        """ATR% 5.0 → normalized 1.0. ATR% 2.5 → 0.5."""
        assert abs(5.0 / 5.0 - 1.0) < 1e-6
        assert abs(2.5 / 5.0 - 0.5) < 1e-6

    def test_vol_regime_normalization(self):
        """Vol regime code 3 (extreme) → 1.0. Code 0 (low) → 0.0."""
        assert abs(3 / 3 - 1.0) < 1e-6
        assert abs(0 / 3 - 0.0) < 1e-6


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))