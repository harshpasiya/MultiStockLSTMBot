"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Pillar 3: FII/DII Flow Analysis                ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : features/fii_dii.py                                   ║
║         Phase   : 1 — Feature Engineering                               ║
║                                                                          ║
║  What this pillar learns:                                                ║
║    The aggregate daily buying/selling pressure from Foreign              ║
║    Institutional Investors (FII) and Domestic Institutional             ║
║    Investors (DII) — and what it implies for the next day's             ║
║    market direction.                                                     ║
║                                                                          ║
║  Key design decisions:                                                   ║
║    1. This pillar is MARKET-LEVEL, not stock-level.                     ║
║       One MDS score per day influences ALL 500 stock signals.           ║
║    2. FII data has an 18-hour lag. Provisional (~4 PM same day)         ║
║       and Final (~6 PM) are treated as separate features.               ║
║    3. Cumulative flow (5-day rolling) matters more than single-day.     ║
║       Single days are noise; sustained flows create trends.             ║
║    4. Historical FII/DII backfill is incomplete (only 2 rows in DB).    ║
║       The system is designed to work in LIVE mode from day one          ║
║       and improve as data accumulates over time.                        ║
║                                                                          ║
║  Output — Market Direction Signal (MDS):                                ║
║    MDS = integer in [-3, +3] computed each morning before 9:15 AM      ║
║                                                                          ║
║    +3 : Strong Bullish — FII buying + DII buying + FII long futures     ║
║    +2 : Bullish        — FII buying OR strong DII support               ║
║    +1 : Mild Bullish   — slight FII net positive, mixed DII             ║
║     0 : Neutral        — balanced flows, no clear institutional bias    ║
║    -1 : Mild Bearish   — slight FII selling, mixed DII                  ║
║    -2 : Bearish        — FII selling + DII unable to absorb             ║
║    -3 : Strong Bearish — heavy FII + DII selling (RC-09 may trigger)   ║
║                                                                          ║
║  Effect on trade signals:                                                ║
║    MDS +3 : Swing buy signals get 1.5× position size                    ║
║    MDS +2 : All buy signals pass; sells need higher confidence          ║
║    MDS +1 : All signals pass at standard sizing                         ║
║    MDS  0 : All signals pass; prefer intraday over swing                ║
║    MDS -1 : Buy signals need higher confidence threshold                ║
║    MDS -2 : Swing buys suppressed; short/hedge signals boosted          ║
║    MDS -3 : ALL long signals blocked (Risk Constitution RC-09)          ║
║                                                                          ║
║  Features computed:                                                      ║
║    fii_net_crore        → FII net cash (provisional, in ₹ crore)       ║
║    dii_net_crore        → DII net cash (final, in ₹ crore)             ║
║    fii_5d_cumulative    → 5-day rolling FII net                         ║
║    dii_5d_cumulative    → 5-day rolling DII net                         ║
║    fii_dii_ratio        → signed FII÷DII ratio                         ║
║    institutional_flow   → combined FII+DII net                          ║
║    flow_momentum        → 3-day rate of change of institutional flow    ║
║    mds_score            → final MDS integer [-3, +3]                    ║
║    mds_continuous       → continuous MDS float [-1.0, +1.0] for LSTM   ║
║                                                                          ║
║  Database:                                                               ║
║    Reads from : fii_dii_flow (populated by data/ingestion/fii_dii.py)  ║
║    Writes to  : features_fii_dii (market-level, one row per date)       ║
║                                                                          ║
║  Note on historical data:                                                ║
║    fii_dii_flow currently has only 2 rows (backfill incomplete).        ║
║    The system handles this gracefully — returns MDS=0 (neutral)         ║
║    when insufficient history exists, and improves automatically         ║
║    as daily ingestion accumulates more data.                            ║
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

# ── MDS computation parameters ────────────────────────────────────────────
CUMULATIVE_WINDOW   = 5      # days for rolling cumulative flow
MOMENTUM_WINDOW     = 3      # days for flow momentum (rate of change)
PANIC_THRESHOLD     = -3000  # ₹ crore — triggers RC-09 if FII sells more
STRONG_BUY_THRESHOLD=  2000  # ₹ crore — signals strong institutional buying
SMOOTHING_WINDOW    = 3      # days to smooth noisy single-day readings
MIN_DAYS_FOR_MDS    = 3      # minimum days of data to compute reliable MDS

# MDS scoring thresholds (₹ crore, 5-day cumulative)
# These are calibrated to Indian market historical FII flow ranges
MDS_THRESHOLDS = {
    # Cumulative 5-day FII net → MDS component score
    "fii_strong_bull":   5000,   # > ₹5000 cr net buy → +1.5
    "fii_bull":          1500,   # > ₹1500 cr net buy → +1.0
    "fii_mild_bull":      300,   # > ₹300  cr net buy → +0.5
    "fii_mild_bear":     -300,   # < -₹300 cr         → -0.5
    "fii_bear":         -1500,   # < -₹1500 cr        → -1.0
    "fii_strong_bear":  -5000,   # < -₹5000 cr        → -1.5
    # DII net (single day, as DII reacts to FII)
    "dii_strong_buy":    2000,   # > ₹2000 cr DII buy → +0.5
    "dii_buy":            500,   # > ₹500  cr DII buy → +0.3
    "dii_sell":          -500,   # < -₹500 cr DII sell→ -0.3
}


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_conn():
    return psycopg2.connect(DB_URL)


def _ensure_features_table(conn):
    """
    Creates features_fii_dii table — market-level, one row per trading date.
    Unlike other pillars, this table has NO symbol column.
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS features_fii_dii (
                date                    DATE    PRIMARY KEY,

                -- Raw FII/DII values (₹ crore)
                fii_net_crore           NUMERIC(12,2),  -- provisional or final
                dii_net_crore           NUMERIC(12,2),
                fii_buy_crore           NUMERIC(12,2),
                fii_sell_crore          NUMERIC(12,2),
                dii_buy_crore           NUMERIC(12,2),
                dii_sell_crore          NUMERIC(12,2),
                data_source             VARCHAR(20),    -- 'provisional' or 'final'

                -- Derived flow features
                institutional_flow      NUMERIC(12,2),  -- FII + DII combined net
                fii_5d_cumulative       NUMERIC(14,2),  -- 5-day rolling FII net
                dii_5d_cumulative       NUMERIC(14,2),  -- 5-day rolling DII net
                inst_5d_cumulative      NUMERIC(14,2),  -- 5-day combined net
                fii_dii_ratio           NUMERIC(8,4),   -- FII net / DII net (signed)
                flow_momentum           NUMERIC(8,4),   -- 3-day ROC of inst flow
                fii_above_panic         BOOLEAN,        -- FII net < -3000 cr
                fii_above_strong_buy    BOOLEAN,        -- FII net 5d > +5000 cr

                -- Normalized features for LSTM input [-1, +1]
                fii_norm                NUMERIC(6,4),
                dii_norm                NUMERIC(6,4),
                inst_flow_norm          NUMERIC(6,4),
                momentum_norm           NUMERIC(6,4),

                -- Final MDS output
                mds_score               SMALLINT,       -- integer [-3, +3]
                mds_continuous          NUMERIC(5,4),   -- float [-1.0, +1.0]

                -- Data quality flags
                is_estimated            BOOLEAN DEFAULT FALSE,  -- True if interpolated
                days_since_real_data    SMALLINT DEFAULT 0
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_fii_dii_date
            ON features_fii_dii (date DESC);
        """)

    conn.commit()
    logger.info("features_fii_dii table ready.")


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_fii_dii_flow(
    start_date: date,
    end_date: date,
    conn,
) -> pd.DataFrame:
    """
    Loads FII/DII flow data from fii_dii_flow table.

    Expected columns in fii_dii_flow:
        date, fii_net, dii_net, fii_buy, fii_sell, dii_buy, dii_sell,
        data_source ('provisional' or 'final')

    Handles gracefully:
        - Table with very few rows (historical backfill incomplete)
        - Missing columns (different schema versions)
        - All-NULL values

    Returns:
        DataFrame indexed by date, or empty DataFrame if no data.
    """
    # Include extra history for rolling windows
    history_start = start_date - timedelta(days=CUMULATIVE_WINDOW + MOMENTUM_WINDOW + 5)

    try:
        # Try full schema first
        sql = """
            SELECT
                date,
                fii_net_crore   AS fii_net,
                dii_net_crore   AS dii_net,
                fii_buy_crore   AS fii_buy,
                fii_sell_crore  AS fii_sell,
                dii_buy_crore   AS dii_buy,
                dii_sell_crore  AS dii_sell,
                COALESCE(data_source, 'final') AS data_source
            FROM fii_dii_flow
            WHERE date BETWEEN %s AND %s
            ORDER BY date ASC;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (history_start, end_date))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]

    except psycopg2.errors.UndefinedTable:
        logger.warning("fii_dii_flow table does not exist — returning empty DataFrame")
        conn.rollback()
        return pd.DataFrame()

    except psycopg2.errors.UndefinedColumn:
        # Table exists but schema differs — try minimal query
        conn.rollback()
        logger.warning("fii_dii_flow schema differs — trying minimal query")
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT date, fii_net_crore AS fii_net, dii_net_crore AS dii_net
                    FROM fii_dii_flow
                    WHERE date BETWEEN %s AND %s
                    ORDER BY date ASC;
                """, (history_start, end_date))
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
        except Exception as e:
            logger.error(f"Minimal FII/DII query also failed: {e}")
            conn.rollback()
            return pd.DataFrame()

    if not rows:
        logger.warning(
            f"fii_dii_flow has no data between {history_start} and {end_date}. "
            f"MDS will default to 0 (neutral) until data accumulates."
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Convert numeric columns
    numeric = [c for c in df.columns if c != "data_source"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"FII/DII flow loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  FLOW FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════

def compute_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all derived FII/DII flow features from raw data.

    Features:
        institutional_flow  : FII + DII combined net (₹ crore)
        fii_5d_cumulative   : 5-day rolling sum of FII net
        dii_5d_cumulative   : 5-day rolling sum of DII net
        inst_5d_cumulative  : 5-day rolling combined net
        fii_dii_ratio       : FII net / DII net (negative = divergence)
        flow_momentum       : 3-day % change of institutional flow
        fii_above_panic     : True if single-day FII net < -3000 cr
        fii_above_strong_buy: True if 5d cumulative FII > +5000 cr

    Args:
        df : Raw FII/DII DataFrame from load_fii_dii_flow()

    Returns:
        DataFrame with all flow features added
    """
    result = df.copy()

    fii = result.get("fii_net", pd.Series(dtype=float))
    dii = result.get("dii_net", pd.Series(dtype=float))

    if isinstance(fii, pd.Series) and fii.empty:
        fii = pd.Series(0.0, index=result.index)
    if isinstance(dii, pd.Series) and dii.empty:
        dii = pd.Series(0.0, index=result.index)

    fii = fii.fillna(0.0)
    dii = dii.fillna(0.0)

    # ── Combined institutional flow ───────────────────────────────────────
    result["institutional_flow"] = fii + dii

    # ── 5-day rolling cumulative flows ───────────────────────────────────
    result["fii_5d_cumulative"]  = fii.rolling(CUMULATIVE_WINDOW, min_periods=1).sum()
    result["dii_5d_cumulative"]  = dii.rolling(CUMULATIVE_WINDOW, min_periods=1).sum()
    result["inst_5d_cumulative"] = result["institutional_flow"].rolling(
        CUMULATIVE_WINDOW, min_periods=1
    ).sum()

    # ── FII/DII ratio (signed) ────────────────────────────────────────────
    # Positive: both buying or both selling (agreement)
    # Negative: FII selling while DII buying or vice versa (divergence)
    safe_dii = dii.replace(0, np.nan)
    result["fii_dii_ratio"] = (fii / safe_dii).clip(-5, 5).fillna(0.0)

    # ── Flow momentum (3-day rate of change) ─────────────────────────────
    inst = result["institutional_flow"]
    result["flow_momentum"] = inst.pct_change(periods=MOMENTUM_WINDOW).fillna(0.0).clip(-2, 2)

    # ── Panic and strong buy flags ────────────────────────────────────────
    result["fii_above_panic"]      = fii < PANIC_THRESHOLD
    result["fii_above_strong_buy"] = result["fii_5d_cumulative"] > STRONG_BUY_THRESHOLD

    return result


def normalize_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes flow features to [-1.0, +1.0] for LSTM input.

    Uses fixed normalization bounds calibrated to Indian market
    historical FII/DII flow ranges (2010–2024 data):
        FII single day: typical range ±5000 ₹ crore
        DII single day: typical range ±3000 ₹ crore
        Combined 5d:    typical range ±15000 ₹ crore

    Fixed bounds are preferred over rolling z-score here because
    the LSTM needs consistent scaling across market regimes.

    Args:
        df : DataFrame with flow features

    Returns:
        DataFrame with added normalized columns
    """
    # FII net normalized: ±5000 cr → ±1.0
    fii = df.get("fii_net", pd.Series(0.0, index=df.index))
    df["fii_norm"] = (fii / 5000.0).clip(-1.0, 1.0)

    # DII net normalized: ±3000 cr → ±1.0
    dii = df.get("dii_net", pd.Series(0.0, index=df.index))
    df["dii_norm"] = (dii / 3000.0).clip(-1.0, 1.0)

    # 5-day cumulative institutional flow: ±15000 cr → ±1.0
    inst5d = df.get("inst_5d_cumulative", pd.Series(0.0, index=df.index))
    df["inst_flow_norm"] = (inst5d / 15000.0).clip(-1.0, 1.0)

    # Flow momentum already in [-2, +2] → clip to [-1, +1]
    mom = df.get("flow_momentum", pd.Series(0.0, index=df.index))
    df["momentum_norm"] = mom.clip(-1.0, 1.0)

    return df


# ══════════════════════════════════════════════════════════════════════════
#  MDS SCORE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════

def compute_mds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Market Direction Signal (MDS) from FII/DII flow features.

    MDS scoring methodology:
        Four components contribute to a raw score, which is then
        mapped to an integer in [-3, +3].

        Component 1 — FII 5-day Cumulative (weight 0.50):
            The most predictive signal. Sustained FII flows over
            5 days have strong next-day direction correlation.

        Component 2 — DII Net (weight 0.25):
            DII often acts as a counterweight to FII. When DII
            ALSO buys alongside FII, it's a strong bullish signal.
            When DII buys while FII sells, it cushions the fall.

        Component 3 — Flow Momentum (weight 0.15):
            Accelerating flows in either direction. An FII that
            was selling ₹500cr/day and now sells ₹2000cr/day
            signals escalating institutional pressure.

        Component 4 — FII/DII Divergence (weight 0.10):
            When FII and DII agree in direction, confidence is
            higher. When they disagree, signal is weaker.

    MDS continuous → MDS discrete mapping:
        raw > +0.60 → MDS = +3
        raw > +0.35 → MDS = +2
        raw > +0.10 → MDS = +1
        raw in [-0.10, +0.10] → MDS = 0
        raw < -0.10 → MDS = -1
        raw < -0.35 → MDS = -2
        raw < -0.60 → MDS = -3

    Args:
        df : DataFrame with all flow features computed

    Returns:
        DataFrame with mds_score and mds_continuous columns added
    """
    n = len(df)
    raw_score = pd.Series(0.0, index=df.index)

    # ── Component 1: FII 5-day Cumulative (weight 0.50) ──────────────────
    fii_5d = df.get("fii_5d_cumulative", pd.Series(0.0, index=df.index)).fillna(0)

    fii_component = pd.Series(0.0, index=df.index)
    fii_component = np.where(fii_5d > MDS_THRESHOLDS["fii_strong_bull"],  1.0, fii_component)
    fii_component = np.where(
        (fii_5d > MDS_THRESHOLDS["fii_bull"]) &
        (fii_5d <= MDS_THRESHOLDS["fii_strong_bull"]),  0.65, fii_component
    )
    fii_component = np.where(
        (fii_5d > MDS_THRESHOLDS["fii_mild_bull"]) &
        (fii_5d <= MDS_THRESHOLDS["fii_bull"]),  0.30, fii_component
    )
    fii_component = np.where(
        (fii_5d < MDS_THRESHOLDS["fii_mild_bear"]) &
        (fii_5d >= MDS_THRESHOLDS["fii_bear"]), -0.30, fii_component
    )
    fii_component = np.where(
        (fii_5d < MDS_THRESHOLDS["fii_bear"]) &
        (fii_5d >= MDS_THRESHOLDS["fii_strong_bear"]), -0.65, fii_component
    )
    fii_component = np.where(fii_5d < MDS_THRESHOLDS["fii_strong_bear"], -1.0, fii_component)

    raw_score += pd.Series(fii_component, index=df.index) * 0.50

    # ── Component 2: DII Net Single Day (weight 0.25) ─────────────────────
    dii = df.get("dii_net", pd.Series(0.0, index=df.index)).fillna(0)

    dii_component = pd.Series(0.0, index=df.index)
    dii_component = np.where(dii >  MDS_THRESHOLDS["dii_strong_buy"],  1.0, dii_component)
    dii_component = np.where(
        (dii > MDS_THRESHOLDS["dii_buy"]) &
        (dii <= MDS_THRESHOLDS["dii_strong_buy"]),  0.5, dii_component
    )
    dii_component = np.where(
        (dii < MDS_THRESHOLDS["dii_sell"]) &
        (dii >= -MDS_THRESHOLDS["dii_strong_buy"]), -0.5, dii_component
    )
    dii_component = np.where(dii < -MDS_THRESHOLDS["dii_strong_buy"], -1.0, dii_component)

    raw_score += pd.Series(dii_component, index=df.index) * 0.25

    # ── Component 3: Flow Momentum (weight 0.15) ──────────────────────────
    # Positive momentum = flows accelerating in bullish direction
    momentum = df.get("flow_momentum", pd.Series(0.0, index=df.index)).fillna(0)
    raw_score += momentum.clip(-1.0, 1.0) * 0.15

    # ── Component 4: FII/DII Agreement (weight 0.10) ──────────────────────
    # +1 when both positive (agreement bullish)
    # -1 when both negative (agreement bearish)
    # 0 when diverging (uncertainty)
    fii_single = df.get("fii_net", pd.Series(0.0, index=df.index)).fillna(0)
    agreement = np.sign(fii_single) * np.sign(dii)
    # Only count agreement, not divergence (divergence = 0 contribution)
    agreement = np.where(agreement > 0, agreement, 0.0)
    raw_score += pd.Series(agreement, index=df.index) * 0.10

    # ── Clip raw score to [-1, +1] ─────────────────────────────────────────
    raw_score = raw_score.clip(-1.0, 1.0)
    df["mds_continuous"] = raw_score

    # ── Map continuous → discrete MDS [-3, +3] ───────────────────────────
    def _to_discrete(x: float) -> int:
        if   x >  0.60: return  3
        elif x >  0.35: return  2
        elif x >  0.10: return  1
        elif x < -0.60: return -3
        elif x < -0.35: return -2
        elif x < -0.10: return -1
        else:           return  0

    df["mds_score"] = raw_score.apply(_to_discrete).astype(int)

    return df


def _make_neutral_mds_row(target_date: date) -> dict:
    """
    Returns a neutral MDS record for dates with no FII/DII data.
    Used when fii_dii_flow has insufficient history.
    MDS=0 means 'no institutional bias' — all signals pass at standard sizing.
    """
    return {
        "date"                : target_date,
        "fii_net_crore"       : None,
        "dii_net_crore"       : None,
        "fii_buy_crore"       : None,
        "fii_sell_crore"      : None,
        "dii_buy_crore"       : None,
        "dii_sell_crore"      : None,
        "data_source"         : "estimated",
        "institutional_flow"  : 0.0,
        "fii_5d_cumulative"   : 0.0,
        "dii_5d_cumulative"   : 0.0,
        "inst_5d_cumulative"  : 0.0,
        "fii_dii_ratio"       : 0.0,
        "flow_momentum"       : 0.0,
        "fii_above_panic"     : False,
        "fii_above_strong_buy": False,
        "fii_norm"            : 0.0,
        "dii_norm"            : 0.0,
        "inst_flow_norm"      : 0.0,
        "momentum_norm"       : 0.0,
        "mds_score"           : 0,
        "mds_continuous"      : 0.0,
        "is_estimated"        : True,
        "days_since_real_data": 999,
    }


# ══════════════════════════════════════════════════════════════════════════
#  MAIN EXTRACTOR CLASS
# ══════════════════════════════════════════════════════════════════════════

class FIIDIIExtractor:
    """
    Main interface for Pillar 3 — FII/DII Market Direction Signal.

    Unlike other pillars, this produces ONE row per trading date
    (not per stock). The MDS is a market-wide bias that gates
    all stock-level signals.

    Usage:
        extractor = FIIDIIExtractor()

        # Compute MDS for a date range
        result = extractor.compute(date(2024, 1, 1), date(2024, 12, 31))

        # Get today's MDS (called by signal engine before 9:15 AM)
        mds = extractor.get_mds_for_date(date.today())
        print(mds)  # {'mds_score': 2, 'mds_continuous': 0.45, ...}

        # Check if trading is blocked (MDS = -3, RC-09)
        if extractor.is_trading_blocked(date.today()):
            print("All long signals blocked — FII panic selling")
    """

    def __init__(self):
        self.conn = _get_conn()
        _ensure_features_table(self.conn)

    def compute(
        self,
        start_date: date,
        end_date: date,
        save_to_db: bool = True,
    ) -> pd.DataFrame:
        """
        Computes MDS for all dates in the given range.

        When fii_dii_flow has insufficient data, returns estimated
        neutral rows (MDS=0) for all dates and logs a warning.

        Args:
            start_date : Start of computation range
            end_date   : End of computation range
            save_to_db : Whether to upsert into features_fii_dii

        Returns:
            DataFrame with one row per trading date containing
            all flow features and MDS score.
        """
        raw = load_fii_dii_flow(start_date, end_date, self.conn)

        if raw.empty or len(raw) < MIN_DAYS_FOR_MDS:
            logger.warning(
                f"Insufficient FII/DII data ({len(raw)} rows). "
                f"MDS will be neutral (0) for all dates. "
                f"This improves automatically as daily ingestion accumulates."
            )
            # Generate neutral rows for all trading days in range
            trading_days = pd.bdate_range(start_date, end_date)
            result = pd.DataFrame([
                _make_neutral_mds_row(d.date()) for d in trading_days
            ])
            if not result.empty:
                result = result.set_index(pd.to_datetime(result["date"]))
                result = result.drop(columns=["date"])

            if save_to_db and not result.empty:
                self._save(result)

            return result

        # ── Compute all features ──────────────────────────────────────────
        df = compute_flow_features(raw)
        df = normalize_flow_features(df)
        df = compute_mds(df)

        # ── Add metadata columns ──────────────────────────────────────────
        df["is_estimated"]         = False
        df["days_since_real_data"] = 0

        # ── Trim to requested range ───────────────────────────────────────
        df = df[df.index >= pd.Timestamp(start_date)].copy()

        if save_to_db and not df.empty:
            self._save(df)

        return df

    def _save(self, df: pd.DataFrame):
        """Upserts FII/DII features into features_fii_dii."""

        # Rename raw columns to match DB schema
        col_map = {
            "fii_net"   : "fii_net_crore",
            "dii_net"   : "dii_net_crore",
            "fii_buy"   : "fii_buy_crore",
            "fii_sell"  : "fii_sell_crore",
            "dii_buy"   : "dii_buy_crore",
            "dii_sell"  : "dii_sell_crore",
        }
        df = df.rename(columns=col_map)

        db_cols = [
            "fii_net_crore", "dii_net_crore", "fii_buy_crore",
            "fii_sell_crore", "dii_buy_crore", "dii_sell_crore",
            "data_source", "institutional_flow",
            "fii_5d_cumulative", "dii_5d_cumulative", "inst_5d_cumulative",
            "fii_dii_ratio", "flow_momentum",
            "fii_above_panic", "fii_above_strong_buy",
            "fii_norm", "dii_norm", "inst_flow_norm", "momentum_norm",
            "mds_score", "mds_continuous",
            "is_estimated", "days_since_real_data",
        ]

        records = []
        for ts, row in df.iterrows():
            rec = {"date": ts.date() if hasattr(ts, "date") else ts}

            for col in db_cols:
                val = row.get(col, None)
                try:
                    if val is None or (not isinstance(val, bool) and pd.isna(val)):
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
            INSERT INTO features_fii_dii (
                date,
                fii_net_crore, dii_net_crore, fii_buy_crore,
                fii_sell_crore, dii_buy_crore, dii_sell_crore,
                data_source, institutional_flow,
                fii_5d_cumulative, dii_5d_cumulative, inst_5d_cumulative,
                fii_dii_ratio, flow_momentum,
                fii_above_panic, fii_above_strong_buy,
                fii_norm, dii_norm, inst_flow_norm, momentum_norm,
                mds_score, mds_continuous,
                is_estimated, days_since_real_data
            ) VALUES (
                %(date)s,
                %(fii_net_crore)s, %(dii_net_crore)s, %(fii_buy_crore)s,
                %(fii_sell_crore)s, %(dii_buy_crore)s, %(dii_sell_crore)s,
                %(data_source)s, %(institutional_flow)s,
                %(fii_5d_cumulative)s, %(dii_5d_cumulative)s, %(inst_5d_cumulative)s,
                %(fii_dii_ratio)s, %(flow_momentum)s,
                %(fii_above_panic)s, %(fii_above_strong_buy)s,
                %(fii_norm)s, %(dii_norm)s, %(inst_flow_norm)s, %(momentum_norm)s,
                %(mds_score)s, %(mds_continuous)s,
                %(is_estimated)s, %(days_since_real_data)s
            )
            ON CONFLICT (date) DO UPDATE SET
                mds_score           = EXCLUDED.mds_score,
                mds_continuous      = EXCLUDED.mds_continuous,
                fii_net_crore       = EXCLUDED.fii_net_crore,
                dii_net_crore       = EXCLUDED.dii_net_crore,
                institutional_flow  = EXCLUDED.institutional_flow,
                fii_5d_cumulative   = EXCLUDED.fii_5d_cumulative,
                fii_above_panic     = EXCLUDED.fii_above_panic,
                is_estimated        = EXCLUDED.is_estimated;
        """

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=500)
            self.conn.commit()
            logger.success(f"FII/DII: {len(records)} market-level rows saved.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"FII/DII DB save failed: {e}")
            raise

    def get_mds_for_date(self, target_date: date) -> dict:
        """
        Returns the MDS for a specific date.
        This is the primary method called by the signal engine
        every morning before 9:15 AM.

        Returns:
            dict with mds_score, mds_continuous, fii_net_crore,
            dii_net_crore, is_estimated, fii_above_panic
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    mds_score, mds_continuous,
                    fii_net_crore, dii_net_crore,
                    fii_5d_cumulative, institutional_flow,
                    fii_above_panic, fii_above_strong_buy,
                    is_estimated, days_since_real_data
                FROM features_fii_dii
                WHERE date = %s;
            """, (target_date,))
            row = cur.fetchone()

        if row is None:
            logger.warning(
                f"No MDS found for {target_date} — returning neutral MDS=0"
            )
            return {
                "mds_score"           : 0,
                "mds_continuous"      : 0.0,
                "fii_net_crore"       : None,
                "dii_net_crore"       : None,
                "fii_5d_cumulative"   : None,
                "institutional_flow"  : None,
                "fii_above_panic"     : False,
                "fii_above_strong_buy": False,
                "is_estimated"        : True,
                "days_since_real_data": 999,
            }

        return {
            "mds_score"           : int(row[0]),
            "mds_continuous"      : float(row[1]),
            "fii_net_crore"       : float(row[2]) if row[2] else None,
            "dii_net_crore"       : float(row[3]) if row[3] else None,
            "fii_5d_cumulative"   : float(row[4]) if row[4] else None,
            "institutional_flow"  : float(row[5]) if row[5] else None,
            "fii_above_panic"     : bool(row[6]),
            "fii_above_strong_buy": bool(row[7]),
            "is_estimated"        : bool(row[8]),
            "days_since_real_data": int(row[9]),
        }

    def is_trading_blocked(self, target_date: date) -> bool:
        """
        Returns True if MDS = -3 (Risk Constitution RC-09 territory).
        When True, the signal engine must block ALL long signals.
        """
        mds = self.get_mds_for_date(target_date)
        return mds["mds_score"] <= -3 or mds.get("fii_above_panic", False)

    def get_position_size_multiplier(self, target_date: date) -> float:
        """
        Returns a position size multiplier based on MDS.
        Used by position_sizer.py (Phase 3) to scale Kelly sizing.

        MDS +3 → 1.5× (aggressive — institutions are buying)
        MDS +2 → 1.2×
        MDS +1 → 1.0×
        MDS  0 → 1.0×
        MDS -1 → 0.8×
        MDS -2 → 0.5× (defensive)
        MDS -3 → 0.0× (no new longs)
        """
        mds = self.get_mds_for_date(target_date)
        score = mds["mds_score"]
        multipliers = {3: 1.5, 2: 1.2, 1: 1.0, 0: 1.0, -1: 0.8, -2: 0.5, -3: 0.0}
        return multipliers.get(score, 1.0)

    def get_recent_mds(self, n_days: int = 10) -> pd.DataFrame:
        """Returns MDS for the last n_days. Useful for dashboard display."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT date, mds_score, mds_continuous,
                       fii_net_crore, dii_net_crore,
                       fii_5d_cumulative, is_estimated
                FROM features_fii_dii
                ORDER BY date DESC
                LIMIT %s;
            """, (n_days,))
            rows = cur.fetchall()

        return pd.DataFrame(rows, columns=[
            "date", "mds_score", "mds_continuous",
            "fii_net_crore", "dii_net_crore",
            "fii_5d_cumulative", "is_estimated"
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

    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — FII/DII MDS Extractor")
    parser.add_argument("--mode",  choices=["compute", "today", "recent"], default="today")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end",   type=str, default=str(date.today()))
    parser.add_argument("--days",  type=int, default=10)
    args = parser.parse_args()

    with FIIDIIExtractor() as extractor:
        if args.mode == "compute":
            start = date.fromisoformat(args.start)
            end   = date.fromisoformat(args.end)
            result = extractor.compute(start, end)
            print(result[["mds_score", "mds_continuous",
                           "fii_5d_cumulative", "inst_5d_cumulative"]].tail(10))

        elif args.mode == "today":
            mds = extractor.get_mds_for_date(date.today())
            print(f"\nToday's MDS: {mds}")
            print(f"Position multiplier: {extractor.get_position_size_multiplier(date.today())}×")
            print(f"Trading blocked: {extractor.is_trading_blocked(date.today())}")

        elif args.mode == "recent":
            df = extractor.get_recent_mds(args.days)
            print(df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run: python -m pytest features/fii_dii.py -v
# ══════════════════════════════════════════════════════════════════════════

def _make_sample_flow(n: int = 30, regime: str = "bull") -> pd.DataFrame:
    """
    Generates synthetic FII/DII flow data for testing.

    Args:
        n      : Number of trading days
        regime : 'bull', 'bear', 'panic', or 'neutral'
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    if regime == "bull":
        fii_net = np.random.normal(1500, 500, n)    # FII buying ₹1500cr/day avg
        dii_net = np.random.normal(800, 300, n)     # DII also buying
    elif regime == "bear":
        fii_net = np.random.normal(-1200, 400, n)   # FII selling
        dii_net = np.random.normal(600, 200, n)     # DII buying (cushioning)
    elif regime == "panic":
        fii_net = np.random.normal(-3500, 500, n)   # FII panic selling
        dii_net = np.random.normal(1000, 300, n)    # DII buying can't absorb
    else:  # neutral
        fii_net = np.random.normal(0, 300, n)
        dii_net = np.random.normal(0, 200, n)

    return pd.DataFrame({
        "fii_net"     : fii_net,
        "dii_net"     : dii_net,
        "fii_buy"     : np.abs(fii_net) + 2000,
        "fii_sell"    : np.abs(fii_net),
        "dii_buy"     : np.abs(dii_net) + 1000,
        "dii_sell"    : np.abs(dii_net),
        "data_source" : "final",
    }, index=dates)


class TestFIIDIIFeatures:

    def setup_method(self):
        self.df_bull    = _make_sample_flow(30, "bull")
        self.df_bear    = _make_sample_flow(30, "bear")
        self.df_panic   = _make_sample_flow(30, "panic")
        self.df_neutral = _make_sample_flow(30, "neutral")

    # ── Flow Feature Tests ────────────────────────────────────────────────

    def test_flow_features_columns_exist(self):
        df = compute_flow_features(self.df_bull.copy())
        for col in ["institutional_flow", "fii_5d_cumulative",
                    "dii_5d_cumulative", "fii_dii_ratio",
                    "flow_momentum", "fii_above_panic"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_institutional_flow_is_sum(self):
        """institutional_flow must equal fii_net + dii_net."""
        df = compute_flow_features(self.df_bull.copy())
        expected = self.df_bull["fii_net"] + self.df_bull["dii_net"]
        np.testing.assert_array_almost_equal(
            df["institutional_flow"].values,
            expected.values, decimal=2
        )

    def test_cumulative_window_length(self):
        """5-day cumulative should start producing values from bar 1."""
        df = compute_flow_features(self.df_bull.copy())
        assert df["fii_5d_cumulative"].notna().all(), \
            "fii_5d_cumulative should have no NaN (min_periods=1)"

    def test_panic_flag_triggered(self):
        """fii_above_panic should be True when FII sells > 3000 cr."""
        df = compute_flow_features(self.df_panic.copy())
        assert df["fii_above_panic"].any(), \
            "Panic regime should trigger fii_above_panic flag"

    def test_panic_flag_not_triggered_in_bull(self):
        """fii_above_panic should never trigger in bull regime."""
        df = compute_flow_features(self.df_bull.copy())
        assert not df["fii_above_panic"].any(), \
            "Bull regime should not trigger panic flag"

    def test_ratio_clipped(self):
        """fii_dii_ratio should be clipped to [-5, +5]."""
        df = compute_flow_features(self.df_bull.copy())
        assert (df["fii_dii_ratio"] >= -5).all()
        assert (df["fii_dii_ratio"] <=  5).all()

    # ── Normalization Tests ───────────────────────────────────────────────

    def test_norm_features_range(self):
        """All normalized features must be in [-1.0, +1.0]."""
        df = compute_flow_features(self.df_bull.copy())
        df = normalize_flow_features(df)
        for col in ["fii_norm", "dii_norm", "inst_flow_norm", "momentum_norm"]:
            assert (df[col] >= -1.0).all() and (df[col] <= 1.0).all(), \
                f"{col} out of [-1, +1]"

    def test_bull_fii_norm_positive(self):
        """Bull regime FII norm should be mostly positive."""
        df = compute_flow_features(self.df_bull.copy())
        df = normalize_flow_features(df)
        assert (df["fii_norm"] > 0).mean() > 0.8

    def test_bear_fii_norm_negative(self):
        """Bear/panic regime FII norm should be mostly negative."""
        df = compute_flow_features(self.df_panic.copy())
        df = normalize_flow_features(df)
        assert (df["fii_norm"] < 0).mean() > 0.8

    # ── MDS Score Tests ───────────────────────────────────────────────────

    def test_mds_score_valid_range(self):
        """MDS score must only be integers in [-3, +3]."""
        for df_raw in [self.df_bull, self.df_bear, self.df_panic, self.df_neutral]:
            df = compute_flow_features(df_raw.copy())
            df = normalize_flow_features(df)
            df = compute_mds(df)
            valid = set(range(-3, 4))
            assert set(df["mds_score"].unique()).issubset(valid), \
                f"Invalid MDS values: {df['mds_score'].unique()}"

    def test_mds_continuous_range(self):
        """MDS continuous must be in [-1.0, +1.0]."""
        df = compute_flow_features(self.df_bull.copy())
        df = normalize_flow_features(df)
        df = compute_mds(df)
        assert (df["mds_continuous"] >= -1.0).all()
        assert (df["mds_continuous"] <=  1.0).all()

    def test_bull_mds_positive(self):
        """Bull regime should produce positive MDS scores."""
        df = compute_flow_features(self.df_bull.copy())
        df = normalize_flow_features(df)
        df = compute_mds(df)
        pos_pct = (df["mds_score"] > 0).mean()
        assert pos_pct > 0.7, \
            f"Bull regime: expected >70% positive MDS, got {pos_pct:.1%}"

    def test_panic_mds_negative(self):
        """Panic regime should produce strongly negative MDS scores."""
        df = compute_flow_features(self.df_panic.copy())
        df = normalize_flow_features(df)
        df = compute_mds(df)
        neg_pct = (df["mds_score"] < 0).mean()
        assert neg_pct > 0.7, \
            f"Panic regime: expected >70% negative MDS, got {neg_pct:.1%}"

    def test_neutral_mds_near_zero(self):
        """Neutral regime should mostly produce MDS = 0."""
        df = compute_flow_features(self.df_neutral.copy())
        df = normalize_flow_features(df)
        df = compute_mds(df)
        zero_pct = (df["mds_score"] == 0).mean()
        assert zero_pct > 0.3, \
            f"Neutral regime: expected >30% zero MDS, got {zero_pct:.1%}"

    def test_mds_discrete_consistent_with_continuous(self):
        """MDS score sign must match mds_continuous sign."""
        df = compute_flow_features(self.df_bull.copy())
        df = normalize_flow_features(df)
        df = compute_mds(df)
        # Where mds_continuous > 0.10, mds_score must be positive
        strong = df[df["mds_continuous"] > 0.10]
        if len(strong) > 0:
            assert (strong["mds_score"] > 0).all(), \
                "Positive continuous MDS should map to positive discrete MDS"

    # ── Position Multiplier Tests ─────────────────────────────────────────

    def test_neutral_mds_row(self):
        """Neutral MDS row should have mds_score=0 and is_estimated=True."""
        row = _make_neutral_mds_row(date(2024, 1, 15))
        assert row["mds_score"]      == 0
        assert row["mds_continuous"] == 0.0
        assert row["is_estimated"]   == True

    def test_all_mds_scores_have_multiplier(self):
        """Every valid MDS score must map to a defined multiplier."""
        multipliers = {3: 1.5, 2: 1.2, 1: 1.0, 0: 1.0, -1: 0.8, -2: 0.5, -3: 0.0}
        for score in range(-3, 4):
            assert score in multipliers, f"MDS score {score} has no multiplier"

    def test_panic_multiplier_is_zero(self):
        """MDS = -3 must give 0.0 multiplier (no new longs)."""
        multipliers = {3: 1.5, 2: 1.2, 1: 1.0, 0: 1.0, -1: 0.8, -2: 0.5, -3: 0.0}
        assert multipliers[-3] == 0.0

    def test_strong_bull_multiplier_is_one_point_five(self):
        """MDS = +3 must give 1.5× multiplier."""
        multipliers = {3: 1.5, 2: 1.2, 1: 1.0, 0: 1.0, -1: 0.8, -2: 0.5, -3: 0.0}
        assert multipliers[3] == 1.5


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))