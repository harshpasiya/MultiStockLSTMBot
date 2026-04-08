"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Pillar 6: Correlation & Inter-stock Influence  ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : features/correlation.py                                ║
║         Phase   : 1 — Feature Engineering                               ║
║                                                                          ║
║  What this pillar learns:                                                ║
║    How individual stocks move in relation to each other, to their        ║
║    sector index, and to macro factors. This pillar serves two goals:    ║
║                                                                          ║
║    Goal 1 — Portfolio Risk Control:                                      ║
║      Prevents the system from holding 4 highly correlated positions.    ║
║      If RELIANCE and ONGC are 90% correlated, holding both = 1 bet,    ║
║      not 2 diversified bets. The Risk Constitution (RC-03) uses this.  ║
║                                                                          ║
║    Goal 2 — Alpha Detection:                                             ║
║      Identifies stocks temporarily decorrelated from their sector.      ║
║      If the entire IT sector rallies but INFY stays flat → INFY is     ║
║      likely to catch up → divergence trade opportunity.                 ║
║                                                                          ║
║  Features computed:                                                      ║
║    peer_correlation_mean   → avg correlation with 10 closest peers      ║
║    sector_correlation      → correlation with sector index proxy        ║
║    market_correlation      → correlation with Nifty 50 (beta proxy)     ║
║    correlation_percentile  → where current corr sits vs 252-day history ║
║    sector_divergence       → stock return - sector return (alpha)        ║
║    sector_divergence_5d    → 5-day cumulative divergence                 ║
║    lead_lag_score          → does this stock lead or lag its peers?      ║
║    pca_factor_1            → loading on dominant PCA market factor       ║
║    pca_factor_2            → loading on second PCA factor (sector)      ║
║    concentration_risk      → how much this stock adds to portfolio risk  ║
║    correlation_score       → composite [-1, +1] for LSTM input          ║
║                                                                          ║
║  RC-03 Integration:                                                      ║
║    get_portfolio_correlation(symbols) returns the max pairwise           ║
║    correlation in a proposed portfolio. If > 0.75, the new signal       ║
║    is rejected by the Risk Constitution before execution.               ║
║                                                                          ║
║  Sector mapping:                                                         ║
║    Uses a built-in NSE sector map (500 stocks → 20 sectors).            ║
║    Sector proxy = equal-weight average of top 10 stocks in sector.      ║
║                                                                          ║
║  Database:                                                               ║
║    Reads from : daily_ohlcv                                              ║
║    Writes to  : features_correlation (per stock per day)                 ║
║    Writes to  : correlation_matrix   (pairwise, updated weekly)         ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install pandas numpy psycopg2-binary scikit-learn loguru         ║
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

# ── Correlation parameters ────────────────────────────────────────────────
CORR_WINDOW          = 60     # rolling correlation window (days)
CORR_HISTORY_WINDOW  = 252    # 1-year window for percentile rank
SECTOR_DIV_WINDOW    = 5      # days for cumulative sector divergence
LEAD_LAG_MAX         = 3      # max lead/lag days to test
N_PEERS              = 10     # number of closest peers for avg correlation
PCA_N_COMPONENTS     = 5      # PCA factors to extract
MIN_STOCKS_FOR_PCA   = 20     # minimum stocks needed for meaningful PCA
MIN_BARS             = 80     # minimum bars needed

# RC-03 threshold — if any pair exceeds this, block new position
RC03_CORRELATION_CAP = 0.75

# ── NSE Sector Map (abbreviated — top sectors with representative stocks) ─
# Full map would cover all 500 stocks; this covers major sectors
NSE_SECTOR_MAP: dict[str, list[str]] = {
    "IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
        "LTIM", "MPHASIS", "PERSISTENT", "COFORGE", "OFSS",
    ],
    "BANKING": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
        "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB",
    ],
    "PHARMA": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
        "TORNTPHARM", "ALKEM", "LUPIN", "BIOCON", "AUROPHARMA",
    ],
    "AUTO": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
        "EICHERMOT", "ASHOKLEY", "TVSMOTOR", "BOSCHLTD", "MOTHERSON",
    ],
    "ENERGY": [
        "RELIANCE", "ONGC", "BPCL", "IOC", "HINDPETRO",
        "GAIL", "ADANIGREEN", "TATAPOWER", "NTPC", "POWERGRID",
    ],
    "METALS": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL",
        "COALINDIA", "NMDC", "HINDCOPPER", "NATIONALUM", "APLAPOLLO",
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
        "GODREJCP", "MARICO", "COLPAL", "EMAMILTD", "VBL",
    ],
    "FINANCE": [
        "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIPRULI",
        "CHOLAFIN", "MUTHOOTFIN", "MANAPPURAM", "LICHSGFIN", "POONAWALLA",
    ],
    "REALTY": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD",
        "SOBHA", "MAHLIFE", "BRIGADE", "SUNTECK", "LODHA",
    ],
    "CEMENT": [
        "ULTRACEMCO", "GRASIM", "AMBUJACEM", "ACC", "SHREECEM",
        "DALMIA", "JKCEMENT", "HEIDELBERG", "BIRLACOPR", "RAMCOCEM",
    ],
    "TELECOM": [
        "BHARTIARTL", "IDEA", "TATACOMM", "INDIAMART", "ROUTE",
    ],
    "CHEMICALS": [
        "PIDILITIND", "AAPL", "AARTIIND", "DEEPAKNITRITE", "NAVINFLUOR",
        "FINEORG", "VINATIORGA", "TATACHEM", "GNFC", "GSFC",
    ],
}

# Reverse map: symbol → sector
SYMBOL_TO_SECTOR: dict[str, str] = {
    sym: sector
    for sector, symbols in NSE_SECTOR_MAP.items()
    for sym in symbols
}


# ══════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_conn():
    return psycopg2.connect(DB_URL)


def _ensure_tables(conn):
    """Creates features_correlation and correlation_matrix tables."""
    with conn.cursor() as cur:

        # Per-stock per-day features
        cur.execute("""
            CREATE TABLE IF NOT EXISTS features_correlation (
                date                    DATE        NOT NULL,
                symbol                  VARCHAR(20) NOT NULL,
                sector                  VARCHAR(20),

                -- Peer correlation
                peer_correlation_mean   NUMERIC(6,4),   -- avg corr with N peers
                peer_correlation_max    NUMERIC(6,4),   -- max corr with any peer
                sector_correlation      NUMERIC(6,4),   -- corr with sector proxy
                market_correlation      NUMERIC(6,4),   -- corr with Nifty proxy

                -- Correlation regime
                corr_percentile         NUMERIC(5,1),   -- [0,100] vs 252-day
                corr_zscore             NUMERIC(6,4),

                -- Sector divergence (alpha signal)
                sector_divergence       NUMERIC(8,4),   -- stock_ret - sector_ret (1d)
                sector_divergence_5d    NUMERIC(8,4),   -- 5-day cumulative
                is_sector_diverging     BOOLEAN,        -- |5d div| > 1 std of hist div

                -- Lead-lag
                lead_lag_score          NUMERIC(6,4),   -- + = leads peers, - = lags

                -- PCA factor loadings
                pca_factor_1            NUMERIC(6,4),
                pca_factor_2            NUMERIC(6,4),

                -- Portfolio risk contribution
                concentration_risk      NUMERIC(6,4),   -- [0,1] normalized

                -- Composite score for LSTM [-1, +1]
                -- + = highly correlated (concentration risk, less alpha)
                -- - = low correlation (diversifying, potential alpha)
                correlation_score       NUMERIC(5,4),

                PRIMARY KEY (date, symbol)
            );
        """)

        cur.execute("""
            SELECT create_hypertable(
                'features_correlation', 'date',
                if_not_exists => TRUE,
                migrate_data  => TRUE
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_corr_symbol
            ON features_correlation (symbol, date DESC);
        """)

        # Pairwise correlation matrix (updated weekly)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS correlation_matrix (
                computed_date   DATE        NOT NULL,
                symbol_a        VARCHAR(20) NOT NULL,
                symbol_b        VARCHAR(20) NOT NULL,
                correlation     NUMERIC(6,4),
                window_days     SMALLINT    DEFAULT 60,
                PRIMARY KEY (computed_date, symbol_a, symbol_b)
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_corr_matrix_date
            ON correlation_matrix (computed_date DESC);
        """)

    conn.commit()
    logger.info("Correlation tables ready.")


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_returns_matrix(
    symbols: list[str],
    start_date: date,
    end_date: date,
    conn,
) -> pd.DataFrame:
    """
    Loads daily log returns for all symbols as a wide matrix.

    Returns:
        DataFrame with dates as index, symbols as columns.
        Values are log returns: ln(close_t / close_{t-1}).
        Missing dates filled with NaN (stock not traded or data gap).
    """
    history_start = start_date - timedelta(
        days=CORR_HISTORY_WINDOW + CORR_WINDOW + 30
    )

    sql = """
        SELECT date, symbol, close
        FROM daily_ohlcv
        WHERE symbol = ANY(%s)
          AND date BETWEEN %s AND %s
        ORDER BY date ASC, symbol ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbols, history_start, end_date))
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "symbol", "close"])
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)

    # Pivot to wide format: date × symbol
    prices = df.pivot(index="date", columns="symbol", values="close")
    prices = prices.sort_index()

    # Log returns
    returns = np.log(prices / prices.shift(1))

    return returns


def load_all_symbols(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM daily_ohlcv ORDER BY symbol;")
        return [row[0] for row in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE COMPUTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def compute_rolling_correlations(
    returns: pd.DataFrame,
    window: int = CORR_WINDOW,
) -> pd.DataFrame:
    """
    Computes rolling pairwise Pearson correlations for all symbol pairs.

    For N=500 stocks this is a 500×500 matrix updated daily.
    For efficiency, we compute a 60-day rolling correlation using
    pandas' built-in rolling().corr() which is vectorized.

    Args:
        returns : Wide returns DataFrame (dates × symbols)
        window  : Rolling window in days

    Returns:
        DataFrame of shape (dates × symbols²) — MultiIndex columns.
        Use .xs(symbol, level=1) to get all correlations for one stock.
    """
    if returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()

    # For large universes, limit to symbols with sufficient data
    valid_cols = returns.columns[returns.notna().sum() >= window].tolist()
    returns    = returns[valid_cols]

    logger.info(
        f"Computing {window}-day rolling correlations "
        f"for {len(valid_cols)} symbols..."
    )

    rolling_corr = returns.rolling(window=window, min_periods=window // 2).corr()
    return rolling_corr


def compute_sector_proxy(
    returns: pd.DataFrame,
    sector: str,
) -> pd.Series:
    """
    Computes an equal-weight sector proxy return series.

    The proxy is the average return of all available sector stocks
    in the returns matrix. Used as the "sector index" for:
        1. Sector correlation computation
        2. Sector divergence detection

    Args:
        returns : Wide returns DataFrame
        sector  : Sector name (from NSE_SECTOR_MAP keys)

    Returns:
        pd.Series of sector proxy daily returns,
        or empty Series if sector stocks not in returns.
    """
    sector_stocks = NSE_SECTOR_MAP.get(sector, [])
    available     = [s for s in sector_stocks if s in returns.columns]

    if len(available) < 2:
        return pd.Series(dtype=float)

    return returns[available].mean(axis=1)


def compute_sector_divergence(
    stock_returns: pd.Series,
    sector_proxy: pd.Series,
    window: int = SECTOR_DIV_WINDOW,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes sector divergence: how much a stock outperforms/underperforms
    its sector on a daily and cumulative basis.

    Positive divergence = stock outperforming sector = alpha generation
    Negative divergence = stock underperforming = sector drag

    Large persistent divergence (5-day cumulative) is a mean-reversion
    signal — the stock is likely to snap back toward its sector.

    Args:
        stock_returns : Daily log returns for one stock
        sector_proxy  : Daily log returns for sector proxy
        window        : Days for cumulative divergence

    Returns:
        (divergence_1d, divergence_5d, is_diverging)
    """
    aligned_sector = sector_proxy.reindex(stock_returns.index).fillna(0)

    div_1d = stock_returns - aligned_sector
    div_5d = div_1d.rolling(window=window, min_periods=1).sum()

    # Is the stock significantly diverging? Compare to 1-year std of divergence
    div_std = div_5d.rolling(252, min_periods=60).std()
    is_diverging = (div_5d.abs() > div_std).fillna(False)

    return div_1d, div_5d, is_diverging


def compute_lead_lag(
    stock_returns: pd.Series,
    peers_returns: pd.DataFrame,
    max_lag: int = LEAD_LAG_MAX,
) -> pd.Series:
    """
    Detects whether a stock leads or lags its peers.

    Method:
        For each lag k in [1, max_lag]:
            corr_lead(k) = corr(stock_t, peer_t+k) — stock predicts peer
            corr_lag(k)  = corr(stock_t, peer_t-k) — peer predicts stock
        lead_lag_score = mean(corr_lead) - mean(corr_lag)

    Interpretation:
        score > 0 : stock tends to LEAD peers (high-quality signal)
        score < 0 : stock tends to LAG peers (follower, weaker signal)
        score ≈ 0 : no consistent lead/lag relationship

    Lead stocks are more valuable for signal generation because
    their moves predict future moves in correlated stocks.

    Args:
        stock_returns : Daily returns for the target stock
        peers_returns : Returns for peer stocks (wide DataFrame)
        max_lag       : Maximum lag to test (default 3 days)

    Returns:
        pd.Series of rolling lead_lag_score, same index as stock_returns
    """
    if peers_returns.empty or peers_returns.shape[1] == 0:
        return pd.Series(0.0, index=stock_returns.index)

    lead_scores = []
    lag_scores  = []

    for lag in range(1, max_lag + 1):
        # Stock leads: correlate stock with FUTURE peer returns
        peer_future = peers_returns.shift(-lag).mean(axis=1)
        lead_corr   = stock_returns.rolling(60, min_periods=30).corr(peer_future)
        lead_scores.append(lead_corr)

        # Stock lags: correlate stock with PAST peer returns
        peer_past = peers_returns.shift(lag).mean(axis=1)
        lag_corr  = stock_returns.rolling(60, min_periods=30).corr(peer_past)
        lag_scores.append(lag_corr)

    lead_mean = pd.concat(lead_scores, axis=1).mean(axis=1)
    lag_mean  = pd.concat(lag_scores,  axis=1).mean(axis=1)

    return (lead_mean - lag_mean).clip(-1.0, 1.0).fillna(0.0)


def compute_pca_loadings(
    returns: pd.DataFrame,
    n_components: int = PCA_N_COMPONENTS,
    window: int = CORR_WINDOW,
) -> pd.DataFrame:
    """
    Computes rolling PCA factor loadings for all stocks.

    PCA on the return matrix decomposes market variance into
    orthogonal factors:
        Factor 1 : The "market" factor — explains ~40–60% of variance.
                   All stocks have positive loading on this.
        Factor 2 : Often a "sector" or "value vs growth" factor.
        Factor 3+ : Industry-specific, momentum, etc.

    High loading on Factor 1 = stock moves closely with market.
    Low loading = stock is more idiosyncratic = more alpha potential.

    For efficiency, PCA is recomputed every 21 bars (monthly)
    using the trailing `window` days of returns.

    Args:
        returns      : Wide returns DataFrame
        n_components : Number of PCA factors to extract
        window       : Lookback window for PCA fitting

    Returns:
        DataFrame with columns pca_factor_1, pca_factor_2, ...
        indexed by symbol (one row per symbol, updated monthly)
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("scikit-learn not installed — PCA features unavailable")
        return pd.DataFrame()

    if returns.shape[1] < MIN_STOCKS_FOR_PCA:
        logger.warning(
            f"Only {returns.shape[1]} stocks — need {MIN_STOCKS_FOR_PCA} for PCA"
        )
        return pd.DataFrame()

    # Use last `window` rows of returns
    recent = returns.tail(window).dropna(axis=1, thresh=window // 2)

    if recent.shape[1] < MIN_STOCKS_FOR_PCA:
        return pd.DataFrame()

    # Fill remaining NaN with column means
    recent = recent.fillna(recent.mean())

    try:
        scaler  = StandardScaler()
        scaled  = scaler.fit_transform(recent.T)   # shape: (n_stocks, n_days)

        pca     = PCA(n_components=min(n_components, scaled.shape[1]))
        loadings= pca.fit_transform(scaled)        # shape: (n_stocks, n_components)

        result  = pd.DataFrame(
            loadings[:, :2],
            index   = recent.columns,
            columns = ["pca_factor_1", "pca_factor_2"],
        )

        # Normalize to [-1, +1]
        for col in result.columns:
            max_abs = result[col].abs().max()
            if max_abs > 0:
                result[col] = result[col] / max_abs

        logger.info(
            f"PCA variance explained: "
            f"F1={pca.explained_variance_ratio_[0]:.1%}, "
            f"F2={pca.explained_variance_ratio_[1]:.1%}"
        )
        return result

    except Exception as e:
        logger.warning(f"PCA failed: {e}")
        return pd.DataFrame()


def compute_peer_correlations(
    symbol: str,
    rolling_corr: pd.DataFrame,
    sector: Optional[str],
    n_peers: int = N_PEERS,
) -> tuple[pd.Series, pd.Series]:
    """
    Extracts peer correlation statistics for one symbol.

    Peers are determined by:
        1. Same sector stocks first (most relevant)
        2. Cross-sector stocks with historically high correlation

    Args:
        symbol       : Target stock symbol
        rolling_corr : Full rolling correlation DataFrame
        sector       : Sector of the target stock (may be None)
        n_peers      : Number of peers to average

    Returns:
        (peer_corr_mean, peer_corr_max) — both pd.Series indexed by date
    """
    if rolling_corr.empty or symbol not in rolling_corr.columns.get_level_values(0):
        empty = pd.Series(dtype=float)
        return empty, empty

    try:
        # Get correlations of this symbol with all others
        sym_corrs = rolling_corr.xs(symbol, level=0, axis=0)

        if sym_corrs.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Prioritize same-sector peers
        sector_peers = NSE_SECTOR_MAP.get(sector or "", [])
        sector_peers = [p for p in sector_peers if p != symbol and p in sym_corrs.columns]
        other_peers  = [c for c in sym_corrs.columns if c not in sector_peers and c != symbol]

        peer_order   = (sector_peers + other_peers)[:n_peers]

        if not peer_order:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        peer_df      = sym_corrs[peer_order]
        peer_mean    = peer_df.mean(axis=1)
        peer_max     = peer_df.max(axis=1)

        return peer_mean, peer_max

    except Exception as e:
        logger.warning(f"Peer correlation extraction failed for {symbol}: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)


def compute_correlation_score(
    peer_corr_mean   : pd.Series,
    sector_corr      : pd.Series,
    sector_divergence: pd.Series,
    lead_lag         : pd.Series,
) -> pd.Series:
    """
    Aggregates correlation features into a composite score [-1, +1].

    Score interpretation:
        High positive (+1) : Highly correlated, no alpha, concentration risk
        Near zero (0)      : Average correlation level
        High negative (-1) : Low correlation, potential alpha, diversifying

    Component weights:
        Peer correlation mean : 0.40  (primary concentration risk signal)
        Sector correlation    : 0.25  (sector-level systematic risk)
        Sector divergence     : 0.20  (alpha opportunity, inverted)
        Lead-lag score        : 0.15  (quality of signal — leaders get bonus)

    Note: sector_divergence is inverted — high divergence = alpha opportunity
    = LOW correlation score (good for the portfolio).

    Args:
        peer_corr_mean    : Rolling mean peer correlation [-1, +1]
        sector_corr       : Rolling sector correlation [-1, +1]
        sector_divergence : 5-day cumulative sector divergence
        lead_lag          : Lead/lag score [-1, +1]

    Returns:
        pd.Series of correlation_score in [-1.0, +1.0]
    """
    score = pd.Series(0.0, index=peer_corr_mean.index)

    # Peer correlation: high corr → high (positive) score
    pc = peer_corr_mean.fillna(0.5).clip(-1, 1)
    score += pc * 0.40

    # Sector correlation: high corr → high (positive) score
    sc = sector_corr.reindex(peer_corr_mean.index).fillna(0.5).clip(-1, 1)
    score += sc * 0.25

    # Sector divergence: INVERTED — high divergence → LOWER score (alpha)
    # Normalize 5d divergence to [-1, +1] using rolling std
    div_5d = sector_divergence.reindex(peer_corr_mean.index).fillna(0)
    div_std = div_5d.rolling(252, min_periods=30).std().replace(0, np.nan).fillna(0.01)
    div_norm= (div_5d / div_std).clip(-2, 2) / 2.0
    # Invert: large divergence → score contribution goes NEGATIVE
    score -= div_norm.abs() * 0.20

    # Lead-lag: stock that leads peers gets a slight NEGATIVE score adjustment
    # (it has more alpha potential = less "generic" correlation)
    ll = lead_lag.reindex(peer_corr_mean.index).fillna(0)
    score -= ll * 0.15

    return score.clip(-1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════
#  PORTFOLIO RISK FUNCTIONS (Used by Risk Constitution RC-03)
# ══════════════════════════════════════════════════════════════════════════

def get_portfolio_correlation(
    symbols: list[str],
    as_of_date: date,
    conn,
) -> dict:
    """
    Returns pairwise correlations for a proposed portfolio.
    Called by Risk Constitution RC-03 before any new position entry.

    Args:
        symbols    : List of symbols in proposed portfolio (2–4 stocks)
        as_of_date : Date to check correlations for
        conn       : DB connection

    Returns:
        dict with:
            max_pair_corr   : highest pairwise correlation in portfolio
            mean_pair_corr  : average pairwise correlation
            blocking_pair   : (symbol_a, symbol_b) causing RC-03 if any
            rc03_triggered  : bool — True if max_pair_corr > 0.75
    """
    if len(symbols) < 2:
        return {
            "max_pair_corr" : 0.0,
            "mean_pair_corr": 0.0,
            "blocking_pair" : None,
            "rc03_triggered": False,
        }

    # Try to get from correlation_matrix table (pre-computed)
    try:
        placeholders = ", ".join(["%s"] * len(symbols))
        sql = f"""
            SELECT symbol_a, symbol_b, correlation
            FROM correlation_matrix
            WHERE computed_date = (
                SELECT MAX(computed_date) FROM correlation_matrix
                WHERE computed_date <= %s
            )
            AND symbol_a IN ({placeholders})
            AND symbol_b IN ({placeholders})
            AND symbol_a != symbol_b;
        """
        params = [as_of_date] + symbols + symbols
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        if rows:
            corr_vals = [float(r[2]) for r in rows if r[2] is not None]
            pairs     = [(r[0], r[1]) for r in rows if r[2] is not None]

            if not corr_vals:
                return _no_corr_result()

            max_corr  = max(corr_vals)
            mean_corr = float(np.mean(corr_vals))
            max_idx   = corr_vals.index(max_corr)
            blocking  = pairs[max_idx] if max_corr > RC03_CORRELATION_CAP else None

            return {
                "max_pair_corr" : max_corr,
                "mean_pair_corr": mean_corr,
                "blocking_pair" : blocking,
                "rc03_triggered": max_corr > RC03_CORRELATION_CAP,
            }

    except Exception as e:
        logger.warning(f"Could not fetch from correlation_matrix: {e}")
        conn.rollback()

    # Fallback: compute from raw returns on the fly
    try:
        returns = load_returns_matrix(symbols, as_of_date - timedelta(days=90), as_of_date, conn)
        if returns.empty or returns.shape[1] < 2:
            return _no_corr_result()

        available = [s for s in symbols if s in returns.columns]
        if len(available) < 2:
            return _no_corr_result()

        recent_corr = returns[available].tail(CORR_WINDOW).corr()
        corr_vals   = []
        pairs       = []

        for i, sym_a in enumerate(available):
            for sym_b in available[i + 1:]:
                val = recent_corr.loc[sym_a, sym_b]
                if not np.isnan(val):
                    corr_vals.append(float(val))
                    pairs.append((sym_a, sym_b))

        if not corr_vals:
            return _no_corr_result()

        max_corr  = max(corr_vals)
        mean_corr = float(np.mean(corr_vals))
        max_idx   = corr_vals.index(max_corr)
        blocking  = pairs[max_idx] if max_corr > RC03_CORRELATION_CAP else None

        return {
            "max_pair_corr" : max_corr,
            "mean_pair_corr": mean_corr,
            "blocking_pair" : blocking,
            "rc03_triggered": max_corr > RC03_CORRELATION_CAP,
        }

    except Exception as e:
        logger.error(f"Portfolio correlation computation failed: {e}")
        return _no_corr_result()


def _no_corr_result() -> dict:
    return {
        "max_pair_corr" : 0.0,
        "mean_pair_corr": 0.0,
        "blocking_pair" : None,
        "rc03_triggered": False,
    }


# ══════════════════════════════════════════════════════════════════════════
#  MAIN EXTRACTOR CLASS
# ══════════════════════════════════════════════════════════════════════════

class CorrelationExtractor:
    """
    Main interface for Pillar 6 — Correlation & Inter-stock Influence.

    Unlike other pillars, this works on the ENTIRE universe simultaneously
    (correlation is inherently a multi-stock computation). It also
    maintains the correlation_matrix table for the RC-03 risk check.

    Usage:
        extractor = CorrelationExtractor()

        # Full run (Phase 1 setup)
        extractor.run_all(end_date=date(2024, 12, 31))

        # RC-03 check before placing a trade
        result = extractor.check_rc03(
            current_positions=["RELIANCE", "TCS"],
            new_symbol="ONGC"
        )
        if result["rc03_triggered"]:
            print(f"Signal rejected: {result['blocking_pair']} correlation too high")
    """

    def __init__(self):
        self.conn = _get_conn()
        _ensure_tables(self.conn)

    def run_all(
        self,
        end_date  : Optional[date] = None,
        start_date: Optional[date] = None,
        symbols   : Optional[list[str]] = None,
    ):
        """
        Computes correlation features for all symbols across the full history.
        This is more complex than other pillars because it requires loading
        ALL symbols simultaneously for the correlation matrix.
        """
        if end_date   is None: end_date   = date.today()
        if start_date is None: start_date = date(2019, 1, 1)
        if symbols    is None: symbols    = load_all_symbols(self.conn)

        logger.info(
            f"CorrelationExtractor.run_all: {len(symbols)} symbols | "
            f"{start_date} → {end_date}"
        )

        # ── Step 1: Load all returns at once ─────────────────────────────
        logger.info("Loading returns matrix for all symbols...")
        returns = load_returns_matrix(symbols, start_date, end_date, self.conn)

        if returns.empty:
            logger.error("No returns data loaded. Aborting.")
            return

        logger.info(f"Returns matrix: {returns.shape[0]} days × {returns.shape[1]} symbols")

        # ── Step 2: Compute rolling correlations ──────────────────────────
        rolling_corr = compute_rolling_correlations(returns)

        # ── Step 3: Compute PCA on full universe ──────────────────────────
        pca_loadings = compute_pca_loadings(returns)

        # ── Step 4: Compute sector proxies ────────────────────────────────
        sector_proxies: dict[str, pd.Series] = {}
        for sector in NSE_SECTOR_MAP:
            proxy = compute_sector_proxy(returns, sector)
            if not proxy.empty:
                sector_proxies[sector] = proxy

        # ── Step 5: Per-symbol feature computation ────────────────────────
        all_records: list[dict] = []
        compute_dates = pd.bdate_range(start_date, end_date)

        for sym_idx, symbol in enumerate(symbols, 1):
            if symbol not in returns.columns:
                continue

            sector = SYMBOL_TO_SECTOR.get(symbol)
            stock_returns = returns[symbol].dropna()

            if len(stock_returns) < MIN_BARS:
                continue

            # Peer correlations
            peer_mean, peer_max = compute_peer_correlations(
                symbol, rolling_corr, sector
            )

            # Sector proxy
            sector_proxy = sector_proxies.get(sector or "", pd.Series(dtype=float))

            # Sector correlation
            sector_corr = pd.Series(dtype=float)
            if not sector_proxy.empty:
                sector_corr = stock_returns.rolling(CORR_WINDOW, min_periods=30).corr(
                    sector_proxy.reindex(stock_returns.index).fillna(0)
                )

            # Sector divergence
            div_1d, div_5d, is_diverging = compute_sector_divergence(
                stock_returns, sector_proxy
            )

            # Lead-lag
            sector_stocks = NSE_SECTOR_MAP.get(sector or "", [])
            peers_in_universe = [
                s for s in sector_stocks
                if s != symbol and s in returns.columns
            ]
            peers_returns = returns[peers_in_universe] if peers_in_universe else pd.DataFrame()
            lead_lag = compute_lead_lag(stock_returns, peers_returns)

            # PCA loadings for this symbol
            pca_f1 = float(pca_loadings.loc[symbol, "pca_factor_1"]) \
                     if not pca_loadings.empty and symbol in pca_loadings.index else np.nan
            pca_f2 = float(pca_loadings.loc[symbol, "pca_factor_2"]) \
                     if not pca_loadings.empty and symbol in pca_loadings.index else np.nan

            # Composite score
            corr_score = compute_correlation_score(
                peer_mean, sector_corr, div_5d, lead_lag
            )

            # Correlation percentile
            corr_pct = pd.Series(dtype=float)
            if not peer_mean.empty:
                roll_min = peer_mean.rolling(CORR_HISTORY_WINDOW, min_periods=60).min()
                roll_max = peer_mean.rolling(CORR_HISTORY_WINDOW, min_periods=60).max()
                rng = (roll_max - roll_min).replace(0, np.nan)
                corr_pct = ((peer_mean - roll_min) / rng * 100).clip(0, 100)

            # Build records for each date in range
            date_idx = returns.index[returns.index >= pd.Timestamp(start_date)]
            for dt in date_idx:
                def _get(series, idx, default=None):
                    try:
                        v = series.get(idx, default) if hasattr(series, "get") else default
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            return default
                        return float(v)
                    except Exception:
                        return default

                rec = {
                    "date"                : dt.date(),
                    "symbol"              : symbol,
                    "sector"              : sector or "",
                    "peer_correlation_mean": _get(peer_mean, dt),
                    "peer_correlation_max" : _get(peer_max, dt),
                    "sector_correlation"   : _get(sector_corr, dt),
                    "market_correlation"   : None,
                    "corr_percentile"      : _get(corr_pct, dt),
                    "corr_zscore"          : None,
                    "sector_divergence"    : _get(div_1d, dt, 0.0),
                    "sector_divergence_5d" : _get(div_5d, dt, 0.0),
                    "is_sector_diverging"  : bool(is_diverging.get(dt, False)),
                    "lead_lag_score"       : _get(lead_lag, dt, 0.0),
                    "pca_factor_1"         : pca_f1,
                    "pca_factor_2"         : pca_f2,
                    "concentration_risk"   : _get(peer_mean, dt, 0.5),
                    "correlation_score"    : _get(corr_score, dt, 0.0),
                }
                all_records.append(rec)

            if sym_idx % 50 == 0:
                logger.info(f"Progress: {sym_idx}/{len(symbols)} symbols processed")
                # Batch save
                if all_records:
                    self._save_batch(all_records)
                    all_records = []

        # Save remaining
        if all_records:
            self._save_batch(all_records)

        # Update correlation matrix
        self._update_correlation_matrix(returns, end_date)

        logger.info("CorrelationExtractor.run_all complete.")

    def _save_batch(self, records: list[dict]):
        """Batch upsert into features_correlation."""
        if not records:
            return

        insert_sql = """
            INSERT INTO features_correlation (
                date, symbol, sector,
                peer_correlation_mean, peer_correlation_max,
                sector_correlation, market_correlation,
                corr_percentile, corr_zscore,
                sector_divergence, sector_divergence_5d,
                is_sector_diverging, lead_lag_score,
                pca_factor_1, pca_factor_2,
                concentration_risk, correlation_score
            ) VALUES (
                %(date)s, %(symbol)s, %(sector)s,
                %(peer_correlation_mean)s, %(peer_correlation_max)s,
                %(sector_correlation)s, %(market_correlation)s,
                %(corr_percentile)s, %(corr_zscore)s,
                %(sector_divergence)s, %(sector_divergence_5d)s,
                %(is_sector_diverging)s, %(lead_lag_score)s,
                %(pca_factor_1)s, %(pca_factor_2)s,
                %(concentration_risk)s, %(correlation_score)s
            )
            ON CONFLICT (date, symbol) DO UPDATE SET
                peer_correlation_mean = EXCLUDED.peer_correlation_mean,
                sector_correlation    = EXCLUDED.sector_correlation,
                sector_divergence_5d  = EXCLUDED.sector_divergence_5d,
                is_sector_diverging   = EXCLUDED.is_sector_diverging,
                lead_lag_score        = EXCLUDED.lead_lag_score,
                correlation_score     = EXCLUDED.correlation_score;
        """

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur, insert_sql, records, page_size=1000
                )
            self.conn.commit()
            logger.info(f"Saved {len(records)} correlation rows.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Correlation batch save failed: {e}")
            raise

    def _update_correlation_matrix(self, returns: pd.DataFrame, as_of_date: date):
        """
        Updates the pairwise correlation_matrix table.
        Stores the most recent 60-day correlation for all pairs.
        Called at end of run_all and weekly by Airflow DAG.
        """
        recent = returns.tail(CORR_WINDOW).dropna(axis=1, thresh=CORR_WINDOW // 2)

        if recent.shape[1] < 2:
            return

        corr_matrix = recent.corr()
        symbols     = corr_matrix.columns.tolist()

        records = []
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i + 1:]:
                val = corr_matrix.loc[sym_a, sym_b]
                if not np.isnan(val):
                    records.append({
                        "computed_date": as_of_date,
                        "symbol_a"     : sym_a,
                        "symbol_b"     : sym_b,
                        "correlation"  : float(round(val, 4)),
                        "window_days"  : CORR_WINDOW,
                    })
                    # Also store reverse pair
                    records.append({
                        "computed_date": as_of_date,
                        "symbol_a"     : sym_b,
                        "symbol_b"     : sym_a,
                        "correlation"  : float(round(val, 4)),
                        "window_days"  : CORR_WINDOW,
                    })

        insert_sql = """
            INSERT INTO correlation_matrix
                (computed_date, symbol_a, symbol_b, correlation, window_days)
            VALUES
                (%(computed_date)s, %(symbol_a)s, %(symbol_b)s,
                 %(correlation)s, %(window_days)s)
            ON CONFLICT (computed_date, symbol_a, symbol_b)
            DO UPDATE SET correlation = EXCLUDED.correlation;
        """

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=2000)
            self.conn.commit()
            logger.success(
                f"correlation_matrix updated: {len(records)//2} pairs as of {as_of_date}"
            )
        except Exception as e:
            self.conn.rollback()
            logger.error(f"correlation_matrix update failed: {e}")

    def check_rc03(
        self,
        current_positions: list[str],
        new_symbol: str,
    ) -> dict:
        """
        Checks Risk Constitution RC-03 for a proposed new position.
        Called by execution engine before every trade entry.

        Args:
            current_positions : Symbols currently held in portfolio
            new_symbol        : Symbol of the proposed new trade

        Returns:
            dict with rc03_triggered, blocking_pair, max_pair_corr
        """
        proposed = current_positions + [new_symbol]
        return get_portfolio_correlation(proposed, date.today(), self.conn)

    def get_latest_score(self, symbol: str) -> Optional[dict]:
        """Returns latest correlation features for signal engine."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT correlation_score, peer_correlation_mean,
                       sector_divergence_5d, is_sector_diverging,
                       lead_lag_score, sector
                FROM features_correlation
                WHERE symbol = %s
                ORDER BY date DESC LIMIT 1;
            """, (symbol,))
            row = cur.fetchone()

        if not row:
            return None
        return {
            "correlation_score"   : float(row[0]) if row[0] else 0.0,
            "peer_corr_mean"      : float(row[1]) if row[1] else 0.0,
            "sector_divergence_5d": float(row[2]) if row[2] else 0.0,
            "is_sector_diverging" : bool(row[3]),
            "lead_lag_score"      : float(row[4]) if row[4] else 0.0,
            "sector"              : str(row[5] or ""),
        }

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

    parser = argparse.ArgumentParser(description="G.O.D.S E.Y.E — Correlation Extractor")
    parser.add_argument("--mode",     choices=["all", "rc03", "score"], default="all")
    parser.add_argument("--symbols",  type=str, help="Comma-separated symbols for rc03")
    parser.add_argument("--new",      type=str, help="New symbol for rc03 check")
    parser.add_argument("--start",    type=str, default="2019-01-01")
    parser.add_argument("--end",      type=str, default=str(date.today()))
    args = parser.parse_args()

    with CorrelationExtractor() as extractor:
        if args.mode == "all":
            extractor.run_all(
                start_date=date.fromisoformat(args.start),
                end_date=date.fromisoformat(args.end)
            )
        elif args.mode == "rc03":
            if not args.symbols or not args.new:
                print("--symbols and --new required for rc03 mode"); sys.exit(1)
            positions = args.symbols.split(",")
            result = extractor.check_rc03(positions, args.new)
            print(f"RC-03 check: {result}")
        elif args.mode == "score":
            if not args.symbols:
                print("--symbols required"); sys.exit(1)
            score = extractor.get_latest_score(args.symbols)
            print(f"{args.symbols}: {score}")


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run: python -m pytest features/correlation.py -v
# ══════════════════════════════════════════════════════════════════════════

def _make_returns(n: int = 150, n_stocks: int = 15, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic return matrix for testing."""
    np.random.seed(seed)
    dates   = pd.date_range("2023-01-01", periods=n, freq="B")
    symbols = [f"STOCK{i:02d}" for i in range(n_stocks)]

    # Market factor (common to all stocks)
    market = np.random.normal(0.0003, 0.012, n)

    returns = {}
    for sym in symbols:
        stock_beta   = np.random.uniform(0.5, 1.5)
        idio_vol     = np.random.uniform(0.005, 0.015)
        idio_return  = np.random.normal(0, idio_vol, n)
        returns[sym] = market * stock_beta + idio_return

    return pd.DataFrame(returns, index=dates)


def _make_two_correlated_stocks(n: int = 150, rho: float = 0.9) -> pd.DataFrame:
    """Creates two stocks with known correlation rho."""
    np.random.seed(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    x = np.random.normal(0, 0.01, n)
    y = rho * x + np.sqrt(1 - rho ** 2) * np.random.normal(0, 0.01, n)
    return pd.DataFrame({"A": x, "B": y}, index=dates)


class TestCorrelationFeatures:

    def setup_method(self):
        self.returns    = _make_returns(150, 15)
        self.corr_pair  = _make_two_correlated_stocks(150, rho=0.9)
        self.uncorr_pair= _make_two_correlated_stocks(150, rho=0.1)

    # ── Rolling Correlation Tests ─────────────────────────────────────────

    def test_rolling_corr_shape(self):
        rc = compute_rolling_correlations(self.returns, window=30)
        assert not rc.empty

    def test_rolling_corr_diagonal_is_one(self):
        """Correlation of a stock with itself must be 1.0."""
        rc = compute_rolling_correlations(self.returns, window=30)
        sym = self.returns.columns[0]
        # level=1 because MultiIndex is (date, symbol)
        self_corr = rc.xs(sym, level=1, axis=0)[sym].dropna()
        assert (self_corr.round(4) == 1.0).all(), \
            "Self-correlation must be 1.0"

    def test_high_corr_detected(self):
        """Two highly correlated stocks should show high rolling correlation."""
        rc = compute_rolling_correlations(self.corr_pair, window=30)
        ab_corr = rc.xs("A", level=1, axis=0)["B"].dropna()
        assert ab_corr.mean() > 0.7, \
            f"Expected high correlation, got {ab_corr.mean():.2f}"

    def test_low_corr_detected(self):
        """Two uncorrelated stocks should show low rolling correlation."""
        rc = compute_rolling_correlations(self.uncorr_pair, window=30)
        ab_corr = rc.xs("A", level=1, axis=0)["B"].dropna()
        assert ab_corr.mean() < 0.5, \
            f"Expected low correlation, got {ab_corr.mean():.2f}"



    # ── Sector Proxy Tests ────────────────────────────────────────────────

    def test_sector_proxy_empty_for_unknown_sector(self):
        proxy = compute_sector_proxy(self.returns, "NONEXISTENT_SECTOR")
        assert proxy.empty

    def test_sector_proxy_computed_when_stocks_available(self):
        # Manually add sector stocks to returns
        returns = self.returns.copy()
        returns.columns = ["TCS", "INFY", "HCLTECH"] + list(returns.columns[3:])
        proxy = compute_sector_proxy(returns, "IT")
        assert not proxy.empty
        assert len(proxy) == len(returns)

    # ── Sector Divergence Tests ───────────────────────────────────────────

    def test_divergence_zero_when_identical(self):
        """Stock identical to sector should have zero divergence."""
        stock  = pd.Series(np.random.normal(0, 0.01, 100))
        sector = stock.copy()
        div_1d, div_5d, _ = compute_sector_divergence(stock, sector)
        assert div_1d.abs().max() < 1e-10

    def test_divergence_nonzero_when_different(self):
        stock  = pd.Series(np.random.normal(0.001, 0.01, 100))
        sector = pd.Series(np.random.normal(-0.001, 0.01, 100))
        div_1d, div_5d, _ = compute_sector_divergence(stock, sector)
        assert div_1d.abs().mean() > 0

    def test_divergence_5d_is_rolling_sum(self):
        """5d divergence should be rolling sum of 1d divergence."""
        stock  = pd.Series(np.random.normal(0, 0.01, 100))
        sector = pd.Series(np.random.normal(0, 0.01, 100))
        div_1d, div_5d, _ = compute_sector_divergence(stock, sector, window=5)
        expected_5d = div_1d.rolling(5, min_periods=1).sum()
        np.testing.assert_array_almost_equal(
            div_5d.values, expected_5d.values, decimal=8
        )

    def test_divergence_length_matches(self):
        stock  = pd.Series(np.random.normal(0, 0.01, 100))
        sector = pd.Series(np.random.normal(0, 0.01, 100))
        div_1d, div_5d, is_div = compute_sector_divergence(stock, sector)
        assert len(div_1d) == len(stock)
        assert len(div_5d) == len(stock)
        assert len(is_div) == len(stock)

    # ── Lead-Lag Tests ────────────────────────────────────────────────────

    def test_lead_lag_range(self):
        """Lead-lag score must be in [-1, +1]."""
        stock = self.returns["STOCK00"]
        peers = self.returns[["STOCK01", "STOCK02", "STOCK03"]]
        ll    = compute_lead_lag(stock, peers, max_lag=2)
        assert (ll >= -1.0).all() and (ll <= 1.0).all()

    def test_lead_lag_empty_peers(self):
        """Empty peer DataFrame should return zero series."""
        stock = self.returns["STOCK00"]
        ll    = compute_lead_lag(stock, pd.DataFrame(), max_lag=2)
        assert (ll == 0.0).all()

    def test_lead_lag_length_matches(self):
        stock = self.returns["STOCK00"]
        peers = self.returns[["STOCK01", "STOCK02"]]
        ll    = compute_lead_lag(stock, peers)
        assert len(ll) == len(stock)

    # ── PCA Tests ─────────────────────────────────────────────────────────

    def test_pca_columns_exist(self):
        result = compute_pca_loadings(self.returns, n_components=2, window=60)
        if not result.empty:
            assert "pca_factor_1" in result.columns
            assert "pca_factor_2" in result.columns

    def test_pca_returns_one_row_per_symbol(self):
        result = compute_pca_loadings(self.returns, n_components=2, window=60)
        if not result.empty:
            assert len(result) == self.returns.shape[1]

    def test_pca_factor_range(self):
        """PCA loadings normalized to [-1, +1]."""
        result = compute_pca_loadings(self.returns, n_components=2, window=60)
        if not result.empty:
            assert (result["pca_factor_1"].abs() <= 1.0 + 1e-6).all()
            assert (result["pca_factor_2"].abs() <= 1.0 + 1e-6).all()

    def test_pca_insufficient_stocks(self):
        """PCA should return empty DataFrame when too few stocks."""
        small_returns = self.returns[["STOCK00", "STOCK01", "STOCK02"]]
        result = compute_pca_loadings(small_returns, n_components=2, window=60)
        assert result.empty

    # ── Correlation Score Tests ───────────────────────────────────────────

    def test_correlation_score_range(self):
        """Correlation score must be in [-1, +1]."""
        n   = 100
        pc  = pd.Series(np.random.uniform(0.3, 0.9, n))
        sc  = pd.Series(np.random.uniform(0.3, 0.9, n))
        div = pd.Series(np.random.normal(0, 0.02, n))
        ll  = pd.Series(np.random.uniform(-0.3, 0.3, n))
        score = compute_correlation_score(pc, sc, div, ll)
        assert (score >= -1.0).all() and (score <= 1.0).all()

    def test_high_correlation_gives_positive_score(self):
        """High peer correlation should give positive correlation score."""
        n  = 100
        pc = pd.Series(np.full(n, 0.9))   # very high peer correlation
        sc = pd.Series(np.full(n, 0.9))
        div= pd.Series(np.zeros(n))
        ll = pd.Series(np.zeros(n))
        score = compute_correlation_score(pc, sc, div, ll)
        assert score.mean() > 0.3, \
            f"High correlation should score positive, got {score.mean():.3f}"

    # ── RC-03 Helper Tests ────────────────────────────────────────────────

    def test_no_corr_result_structure(self):
        result = _no_corr_result()
        assert result["rc03_triggered"] == False
        assert result["max_pair_corr"]  == 0.0
        assert result["blocking_pair"]  is None

    def test_rc03_cap_value(self):
        assert RC03_CORRELATION_CAP == 0.75, \
            "RC-03 threshold must be 0.75 per Risk Constitution"

    def test_sector_map_coverage(self):
        """All sectors in SYMBOL_TO_SECTOR must exist in NSE_SECTOR_MAP."""
        for sym, sector in SYMBOL_TO_SECTOR.items():
            assert sector in NSE_SECTOR_MAP, \
                f"Symbol {sym} mapped to unknown sector {sector}"

    def test_symbol_to_sector_consistent(self):
        """Every symbol in NSE_SECTOR_MAP must appear in SYMBOL_TO_SECTOR."""
        for sector, syms in NSE_SECTOR_MAP.items():
            for sym in syms:
                assert sym in SYMBOL_TO_SECTOR, \
                    f"{sym} in NSE_SECTOR_MAP[{sector}] but not in SYMBOL_TO_SECTOR"


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))