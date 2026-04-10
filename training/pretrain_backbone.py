"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Backbone Supervised Pre-training                ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : training/pretrain_backbone.py                          ║
║         Phase   : 2 — AI Backbone (Supervised Pre-training)             ║
║                                                                          ║
║  What this file does:                                                    ║
║    Trains the GodsEyeBackbone (LSTM + Transformer + FusionHead) on      ║
║    5 years of historical data using supervised learning.                 ║
║                                                                          ║
║    The task: given the last 60 days of 28 features for each stock,      ║
║    predict:                                                              ║
║      1. Direction  — will the stock be UP or DOWN in 5 trading days?    ║
║         (binary classification → BCE loss, weight 0.6)                  ║
║      2. Magnitude  — what is the 5-day forward return in %?             ║
║         (regression → MSE loss, weight 0.4)                             ║
║                                                                          ║
║  Training splits:                                                        ║
║    Train      : 2019-01-01 → 2023-06-30  (4.5 years)                   ║
║    Validation : 2023-07-01 → 2024-01-31  (7 months)                    ║
║    Test       : 2024-02-01 → 2024-12-31  (UNTOUCHED until Phase 4)     ║
║                                                                          ║
║  Training config:                                                        ║
║    Optimizer  : AdamW (lr=1e-4, weight_decay=0.01)                      ║
║    Scheduler  : Cosine annealing with warm restart (T_0=10)             ║
║    Batch size : 64 stocks per step                                       ║
║    Max epochs : 100 (early stopping patience=10 on val IC)              ║
║    Grad clip  : max_norm=1.0                                             ║
║                                                                          ║
║  Gate criterion (Phase 2 → Phase 3 promotion):                          ║
║    validation IC  > 0.05  (Information Coefficient vs 5d forward return)║
║    direction accuracy > 57%                                              ║
║                                                                          ║
║  Outputs:                                                                ║
║    checkpoints/pretrain_best.pt    ← best validation IC checkpoint      ║
║    checkpoints/pretrain_last.pt    ← latest epoch checkpoint            ║
║    logs/pretrain_metrics.csv       ← epoch-by-epoch metrics log         ║
║                                                                          ║
║  Usage:                                                                  ║
║    # Full training run                                                   ║
║    python -m training.pretrain_backbone                                  ║
║                                                                          ║
║    # Resume from checkpoint                                              ║
║    python -m training.pretrain_backbone --resume checkpoints/last.pt    ║
║                                                                          ║
║    # Quick smoke test (5 epochs, small data)                             ║
║    python -m training.pretrain_backbone --smoke-test                    ║
║                                                                          ║
║  Dependencies:                                                           ║
║    models/backbone.py, features/fusion.py                               ║
║    pip install torch>=2.1.0 psycopg2-binary pandas numpy loguru         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import csv
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn

from datetime  import date, timedelta
from pathlib   import Path
from typing    import Dict, Iterator, List, Optional, Tuple
from loguru    import logger
from dotenv    import load_dotenv

from models.backbone import GodsEyeBackbone, build_backbone, BACKBONE_OUT_DIM

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR        = Path("logs")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Logger ────────────────────────────────────────────────────────────────
logger.add(
    LOG_DIR / "pretrain_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# ── Database ──────────────────────────────────────────────────────────────
DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)

# ── Training constants ────────────────────────────────────────────────────
SEQ_LEN        = 60     # lookback window (days)
N_FEATURES     = 28     # fused feature vector length (from fusion.py)
FORWARD_DAYS   = 5      # predict return N days ahead
BATCH_STOCKS   = 64     # stocks per training step
MAX_EPOCHS     = 100
PATIENCE       = 10     # early stopping patience (epochs)
LR             = 1e-4
WEIGHT_DECAY   = 0.01
GRAD_CLIP      = 1.0
DIR_WEIGHT     = 0.6    # BCE loss weight
RET_WEIGHT     = 0.4    # MSE loss weight
IC_GATE        = 0.05   # minimum validation IC to pass Phase 2
ACC_GATE       = 0.57   # minimum direction accuracy to pass Phase 2

# ── Training splits ───────────────────────────────────────────────────────
TRAIN_START = date(2019, 1,  1)
TRAIN_END   = date(2023, 6, 30)
VAL_START   = date(2023, 7,  1)
VAL_END     = date(2024, 1, 31)
# TEST split: 2024-02-01 → 2024-12-31 — DO NOT TOUCH until Phase 4


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING FROM TIMESCALEDB
# ══════════════════════════════════════════════════════════════════════════

def get_db_connection():
    """Returns a psycopg2 connection to TimescaleDB."""
    return psycopg2.connect(DB_URL)


def load_fused_features(
    conn,
    start_date: date,
    end_date  : date,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[date]]:
    """
    Loads the fused feature matrix from features_fused table in TimescaleDB.

    The features_fused table is populated by features/fusion.py (Phase 1).
    Each row = one stock × one date × 28 features.

    Args:
        conn       : psycopg2 DB connection
        start_date : First date to load
        end_date   : Last date to load (inclusive)

    Returns:
        features   : np.ndarray of shape (N_dates, N_stocks, N_features)
                     All values in [-1, +1] (already normalized by fusion.py)
        returns_5d : np.ndarray of shape (N_dates, N_stocks)
                     5-day forward return in % for each stock/date
        symbols    : List[str] of stock symbols in order (length N_stocks)
        dates      : List[date] of trading dates in order (length N_dates)

    Note:
        Returns a fallback synthetic dataset if features_fused table is empty.
        This allows Phase 2 training to begin before Phase 1 is fully complete.
    """
    # ── Try to load from features_fused table ────────────────────────────
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM features_fused
                WHERE date BETWEEN %s AND %s
            """, (start_date, end_date))
            row_count = cur.fetchone()[0]

        if row_count > 0:
            return _load_from_features_table(conn, start_date, end_date)

    except psycopg2.errors.UndefinedTable:
        logger.warning("features_fused table not found — using OHLCV fallback")
        conn.rollback()

    # ── Fallback: compute basic features directly from daily_ohlcv ───────
    logger.warning(
        "features_fused table empty — computing basic features from daily_ohlcv. "
        "Run features/fusion.py first for full 28-feature training."
    )
    return _load_basic_features_from_ohlcv(conn, start_date, end_date)


def _load_from_features_table(
    conn,
    start_date: date,
    end_date  : date,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[date]]:
    """
    Loads full 28-feature fused vectors from features_fused table.
    Also computes 5-day forward returns from daily_ohlcv.
    """
    # Feature columns from fusion.py (features [0]–[27])
    feature_cols = [
        "trend_score", "ema_ribbon_gap", "adx_normalized",
        "supertrend_dir", "price_vs_ema200", "swing_structure",
        "msi_signal", "vrsi_normalized", "mfi_normalized",
        "msi_divergence", "mds_continuous", "fii_norm",
        "dii_norm", "sentiment_score", "sentiment_momentum",
        "event_flag", "market_fear_greed_n", "volatility_score",
        "atr_pct_normalized", "vol_regime_code_n", "hv_percentile_n",
        "correlation_score", "sector_divergence_n", "lead_lag_score",
        "peer_corr_mean", "delivery_mom_n", "swing_tp_normalized",
        "swing_sl_normalized",
    ]

    cols_sql = ", ".join(f"COALESCE({c}, 0.0)" for c in feature_cols)

    sql = f"""
        SELECT date, symbol, {cols_sql}
        FROM features_fused
        WHERE date BETWEEN %s AND %s
        ORDER BY date ASC, symbol ASC
    """

    with conn.cursor() as cur:
        cur.execute(sql, (start_date, end_date))
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(f"No fused features found between {start_date} and {end_date}")

    df = pd.DataFrame(rows, columns=[
        "date", "symbol", "open", "high", "low", "close", "prev_close", "volume"
    ])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["open", "high", "low", "close", "prev_close", "volume"]:
        df[col] = df[col].astype(float)

    return _pivot_and_compute_returns(df, feature_cols, conn, start_date, end_date)


def _load_basic_features_from_ohlcv(
    conn,
    start_date: date,
    end_date  : date,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[date]]:
    """
    Fallback: computes 6 basic features directly from daily_ohlcv.
    Used when features_fused table is not yet populated.

    Features computed:
        [0] return_1d    : 1-day return
        [1] return_5d    : 5-day return
        [2] volume_norm  : log-normalized volume
        [3] hl_range     : (high-low)/close (daily range)
        [4] gap_pct      : (open-prev_close)/prev_close
        [5] close_pos    : (close-low)/(high-low) close position in range
        [6-27]           : zeros (placeholders for missing pillars)
    """
    # Load extra history for rolling windows
    history_start = start_date - timedelta(days=30)

    sql = """
            SELECT date, symbol,
                   open::float, high::float, low::float,
                   close::float, prev_close::float, volume::float
            FROM daily_ohlcv
            WHERE date BETWEEN %s AND %s
              AND close > 0 AND volume > 0
            ORDER BY date ASC, symbol ASC
        """
    with conn.cursor() as cur:
        cur.execute(sql, (history_start, end_date))
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=[
        "date", "symbol", "open", "high", "low", "close", "prev_close", "volume"
    ])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # ── Per-symbol feature computation ────────────────────────────────────
    feature_records = []

    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("date").copy()
        grp["return_1d"]   = grp["close"].pct_change()
        grp["return_5d"]   = grp["close"].pct_change(5)
        grp["volume_norm"] = np.log1p(grp["volume"].astype(float))
        grp["hl_range"]    = (grp["high"] - grp["low"]) / grp["close"].clip(0.01)
        grp["gap_pct"]     = (grp["open"] - grp["prev_close"]) / grp["prev_close"].clip(0.01)
        denom = (grp["high"] - grp["low"]).clip(0.001)
        grp["close_pos"]   = (grp["close"] - grp["low"]) / denom

        # Clip to requested date range
        grp = grp[grp["date"] >= start_date]

        for _, row in grp.iterrows():
            feat = np.zeros(N_FEATURES, dtype=np.float32)
            feat[0] = float(row["return_1d"])  if pd.notna(row["return_1d"])  else 0.0
            feat[1] = float(row["return_5d"])  if pd.notna(row["return_5d"])  else 0.0
            feat[2] = float(row["volume_norm"])/ 20.0   # rough normalization
            feat[3] = float(row["hl_range"])   if pd.notna(row["hl_range"])   else 0.0
            feat[4] = float(row["gap_pct"])    if pd.notna(row["gap_pct"])    else 0.0
            feat[5] = float(row["close_pos"])  if pd.notna(row["close_pos"])  else 0.5
            feat    = np.clip(feat, -1.0, 1.0)
            feature_records.append({
                "date": row["date"], "symbol": symbol,
                **{f"f{i}": feat[i] for i in range(N_FEATURES)}
            })

    feat_cols = [f"f{i}" for i in range(N_FEATURES)]
    feat_df   = pd.DataFrame(feature_records)

    return _pivot_and_compute_returns(feat_df, feat_cols, conn, start_date, end_date)


def _pivot_and_compute_returns(
    df        : pd.DataFrame,
    feat_cols : List[str],
    conn,
    start_date: date,
    end_date  : date,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[date]]:
    """
    Pivots long-format feature DataFrame into (N_dates, N_stocks, N_features),
    computes 5-day forward returns, and aligns both arrays.
    """
    # ── Get all symbols and dates ─────────────────────────────────────────
    symbols = sorted(df["symbol"].unique().tolist())
    dates   = sorted(df["date"].unique().tolist())

    N_dates   = len(dates)
    N_stocks  = len(symbols)
    sym_idx   = {s: i for i, s in enumerate(symbols)}
    date_idx  = {d: i for i, d in enumerate(dates)}

    # ── Build feature tensor ──────────────────────────────────────────────
    features = np.zeros((N_dates, N_stocks, N_FEATURES), dtype=np.float32)

    for _, row in df.iterrows():
        di = date_idx.get(row["date"])
        si = sym_idx.get(row["symbol"])
        if di is not None and si is not None:
            features[di, si, :] = [float(row[c]) for c in feat_cols]

    # Replace any NaN/Inf that slipped through
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    features = np.clip(features, -1.0, 1.0)

    # ── Compute 5-day forward returns from daily_ohlcv ────────────────────
    # We need FORWARD_DAYS extra days beyond end_date for the last labels
    fwd_end = end_date + timedelta(days=FORWARD_DAYS * 2)

    sql = """
        SELECT date, symbol, close
        FROM daily_ohlcv
        WHERE date BETWEEN %s AND %s
          AND close > 0
        ORDER BY date ASC, symbol ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (start_date, fwd_end))
        close_rows = cur.fetchall()

    close_df = pd.DataFrame(close_rows, columns=["date", "symbol", "close"])
    close_df["date"]  = pd.to_datetime(close_df["date"]).dt.date
    close_df["close"] = close_df["close"].astype(float)

    # Pivot close prices: index=date, columns=symbol
    close_pivot = close_df.pivot(index="date", columns="symbol", values="close")
    close_pivot = close_pivot.reindex(columns=symbols, fill_value=np.nan)

    # 5-day forward return for date d = (close[d+5] - close[d]) / close[d]
    all_dates_sorted = sorted(close_pivot.index.tolist())
    date_to_idx_full = {d: i for i, d in enumerate(all_dates_sorted)}

    returns_5d = np.zeros((N_dates, N_stocks), dtype=np.float32)

    for di, d in enumerate(dates):
        d_pos = date_to_idx_full.get(d)
        if d_pos is None:
            continue
        # Find the trading day FORWARD_DAYS ahead
        future_dates = [
            all_dates_sorted[j]
            for j in range(d_pos + 1, min(d_pos + FORWARD_DAYS + 5, len(all_dates_sorted)))
        ]
        if len(future_dates) < FORWARD_DAYS:
            continue
        d_future = future_dates[FORWARD_DAYS - 1]

        if d in close_pivot.index and d_future in close_pivot.index:
            close_now    = close_pivot.loc[d].values.astype(float)
            close_future = close_pivot.loc[d_future].values.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = np.where(
                    close_now > 0,
                    (close_future - close_now) / close_now * 100,
                    0.0
                )
            returns_5d[di] = np.nan_to_num(ret, nan=0.0, posinf=5.0, neginf=-5.0)

    logger.info(
        f"Dataset built: {N_dates} dates × {N_stocks} stocks "
        f"| features {features.shape} | returns {returns_5d.shape}"
    )
    return features, returns_5d, symbols, dates


# ══════════════════════════════════════════════════════════════════════════
#  DATASET & DATALOADER
# ══════════════════════════════════════════════════════════════════════════

class PretrainDataset:
    """
    Sliding-window dataset for supervised backbone pre-training.

    For each (stock, date) pair where we have at least SEQ_LEN prior days
    of features AND a valid 5-day forward return label, produces:
        x       : (SEQ_LEN, N_FEATURES) feature sequence
        dir_tgt : 1 if 5d return > 0 else 0  (direction label)
        ret_tgt : 5-day forward return in %   (magnitude label)

    Args:
        features   : (N_dates, N_stocks, N_features) float32 array
        returns_5d : (N_dates, N_stocks) float32 array (% returns)
        seq_len    : Lookback window size (default: 60)
        min_return : Minimum absolute return to include as training sample.
                     Filters out near-zero returns where direction is noise.
                     Default: 0.1% (drops very flat days)
    """

    def __init__(
        self,
        features   : np.ndarray,
        returns_5d : np.ndarray,
        seq_len    : int   = SEQ_LEN,
        min_return : float = 0.1,
    ):
        self.features   = features     # (N_dates, N_stocks, N_feat)
        self.returns_5d = returns_5d   # (N_dates, N_stocks)
        self.seq_len    = seq_len
        self.min_return = min_return

        # Build index of valid (date_idx, stock_idx) pairs
        self.samples = self._build_sample_index()
        logger.info(f"PretrainDataset: {len(self.samples):,} valid samples")

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """
        Identifies all (date_idx, stock_idx) pairs where:
          1. date_idx >= seq_len (enough history for the window)
          2. return is not NaN and |return| >= min_return
        """
        N_dates, N_stocks, _ = self.features.shape
        samples = []

        for si in range(N_stocks):
            for di in range(self.seq_len, N_dates):
                ret = self.returns_5d[di, si]
                if np.isnan(ret) or np.isinf(ret):
                    continue
                if abs(ret) < self.min_return:
                    continue
                # Check feature window has no all-zero rows
                # (indicates missing data for that stock/date)
                window = self.features[di - self.seq_len:di, si, :]
                if np.all(window == 0):
                    continue
                samples.append((di, si))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        di, si = self.samples[idx]

        # Feature window: (seq_len, N_features)
        x = self.features[di - self.seq_len:di, si, :].copy()

        # Direction target: 1 if return > 0 else 0
        ret     = float(self.returns_5d[di, si])
        dir_tgt = 1.0 if ret > 0 else 0.0

        # Return target: clipped to ±15% to reduce outlier influence
        ret_tgt = float(np.clip(ret, -15.0, 15.0))

        return (
            torch.from_numpy(x).float(),
            torch.tensor(dir_tgt, dtype=torch.float32),
            torch.tensor(ret_tgt, dtype=torch.float32),
        )


def batch_iterator(
    dataset   : PretrainDataset,
    batch_size: int,
    shuffle   : bool = True,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Simple batch iterator (avoids DataLoader multiprocessing issues on Windows).

    Yields:
        x_batch   : (batch_size, seq_len, N_features)
        dir_batch : (batch_size,) — direction labels
        ret_batch : (batch_size,) — return labels
    """
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start: start + batch_size]
        if len(batch_idx) == 0:
            continue

        xs, dirs, rets = [], [], []
        for i in batch_idx:
            x, d, r = dataset[i]
            xs.append(x)
            dirs.append(d)
            rets.append(r)

        yield (
            torch.stack(xs),     # (batch, seq, features)
            torch.stack(dirs),   # (batch,)
            torch.stack(rets),   # (batch,)
        )


# ══════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_ic(
    predictions: np.ndarray,
    targets    : np.ndarray,
) -> float:
    """
    Computes Information Coefficient (IC) = Spearman rank correlation
    between predicted returns and actual 5-day forward returns.

    Uses numpy ranking to avoid scipy dependency.

    IC > 0.05 : Weak but consistent signal (institutional quality floor)
    IC > 0.10 : Good signal
    IC > 0.15 : Excellent signal (rare in liquid markets)
    """
    if len(predictions) < 3:
        return 0.0

    def _rank(arr: np.ndarray) -> np.ndarray:
        """Returns rank array handling ties with average method."""
        arr   = np.asarray(arr, dtype=float)
        order = arr.argsort()
        ranks = np.empty(len(arr), dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        # Fix ties — find groups of equal values and assign average rank
        arr_sorted = arr[order]
        i = 0
        while i < len(arr_sorted):
            j = i + 1
            while j < len(arr_sorted) and arr_sorted[j] == arr_sorted[i]:
                j += 1
            if j > i + 1:
                avg_rank = ranks[order[i:j]].mean()
                ranks[order[i:j]] = avg_rank
            i = j
        return ranks

    r_pred = _rank(np.asarray(predictions, dtype=float))
    r_tgt  = _rank(np.asarray(targets,     dtype=float))

    # Pearson correlation of ranks = Spearman correlation
    r_pred -= r_pred.mean()
    r_tgt  -= r_tgt.mean()

    denom = (np.sqrt((r_pred**2).sum()) * np.sqrt((r_tgt**2).sum()))
    if denom < 1e-10:
        return 0.0

    ic = float((r_pred * r_tgt).sum() / denom)
    return ic if not math.isnan(ic) else 0.0


# ══════════════════════════════════════════════════════════════════════════
#  TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════

class PretrainEngine:
    """
    Manages the full supervised pre-training loop for GodsEyeBackbone.

    Handles:
        - Training / validation epoch loops
        - AdamW optimizer + cosine LR scheduling
        - Gradient clipping
        - Early stopping on validation IC
        - Checkpoint saving (best + last)
        - Metrics logging to CSV

    Args:
        backbone       : GodsEyeBackbone in pretrain_mode=True
        train_dataset  : PretrainDataset for training split
        val_dataset    : PretrainDataset for validation split
        batch_size     : Stocks per training step (default: 64)
        lr             : Initial learning rate (default: 1e-4)
        weight_decay   : AdamW weight decay (default: 0.01)
        max_epochs     : Maximum training epochs (default: 100)
        patience       : Early stopping patience (default: 10)
        device         : torch.device to train on
        dir_weight     : BCE loss weight (default: 0.6)
        ret_weight     : MSE loss weight (default: 0.4)
    """

    def __init__(
        self,
        backbone     : GodsEyeBackbone,
        train_dataset: PretrainDataset,
        val_dataset  : PretrainDataset,
        batch_size   : int   = BATCH_STOCKS,
        lr           : float = LR,
        weight_decay : float = WEIGHT_DECAY,
        max_epochs   : int   = MAX_EPOCHS,
        patience     : int   = PATIENCE,
        device       : Optional[torch.device] = None,
        dir_weight   : float = DIR_WEIGHT,
        ret_weight   : float = RET_WEIGHT,
    ):
        self.backbone      = backbone
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.batch_size    = batch_size
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.dir_weight    = dir_weight
        self.ret_weight    = ret_weight

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.backbone = self.backbone.to(self.device)

        # ── Optimizer ─────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr           = lr,
            weight_decay = weight_decay,
            betas        = (0.9, 0.999),
            eps          = 1e-8,
        )

        # ── LR Scheduler: Cosine Annealing with Warm Restarts ─────────────
        # Restarts every T_0=10 epochs — helps escape local minima
        # T_mult=2 doubles the restart period after each restart
        steps_per_epoch = max(1, len(train_dataset) // batch_size)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0    = 10 * steps_per_epoch,
            T_mult = 2,
            eta_min= lr * 0.01,
        )

        # ── State tracking ────────────────────────────────────────────────
        self.best_val_ic   = -float("inf")
        self.best_epoch    = 0
        self.epochs_no_imp = 0   # epochs without validation IC improvement
        self.history       : List[Dict] = []

        # ── Metrics CSV ───────────────────────────────────────────────────
        self.metrics_path = LOG_DIR / "pretrain_metrics.csv"
        self._init_metrics_csv()

        logger.info(
            f"PretrainEngine initialized | "
            f"device={self.device} | "
            f"train_samples={len(train_dataset):,} | "
            f"val_samples={len(val_dataset):,} | "
            f"batch_size={batch_size}"
        )

    def _init_metrics_csv(self):
        """Creates metrics CSV with header if it doesn't exist."""
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "epoch", "train_loss", "train_dir_loss", "train_ret_loss",
                    "train_dir_acc", "val_loss", "val_dir_loss", "val_ret_loss",
                    "val_dir_acc", "val_ic", "lr", "is_best",
                ])
                writer.writeheader()

    def _log_metrics(self, metrics: dict):
        """Appends one epoch's metrics to the CSV log."""
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writerow(metrics)

    # ── Single epoch training ─────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Runs one full training epoch.

        Returns:
            Dict with train_loss, train_dir_loss, train_ret_loss, train_dir_acc
        """
        self.backbone.train()

        total_loss     = 0.0
        total_dir_loss = 0.0
        total_ret_loss = 0.0
        total_dir_acc  = 0.0
        n_batches      = 0

        for x_batch, dir_batch, ret_batch in batch_iterator(
            self.train_dataset, self.batch_size, shuffle=True
        ):
            x_batch   = x_batch.to(self.device)
            dir_batch = dir_batch.to(self.device)
            ret_batch = ret_batch.to(self.device)

            # Forward pass
            emb, dir_logit, ret_pred = self.backbone(x_batch)

            # Loss computation
            losses = self.backbone.compute_loss(
                emb, dir_logit, ret_pred,
                dir_batch, ret_batch,
                self.dir_weight, self.ret_weight,
            )

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.backbone.parameters(), max_norm=GRAD_CLIP
            )

            self.optimizer.step()
            self.scheduler.step()

            # Accumulate metrics
            total_loss     += losses["total_loss"].item()
            total_dir_loss += losses["direction_loss"].item()
            total_ret_loss += losses["return_loss"].item()
            total_dir_acc  += losses["direction_acc"].item()
            n_batches      += 1

        n_batches = max(n_batches, 1)
        return {
            "train_loss"    : total_loss     / n_batches,
            "train_dir_loss": total_dir_loss / n_batches,
            "train_ret_loss": total_ret_loss / n_batches,
            "train_dir_acc" : total_dir_acc  / n_batches,
        }

    # ── Validation epoch ──────────────────────────────────────────────────

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """
        Runs one full validation epoch.
        Also computes Information Coefficient (IC) which is the gate metric.

        Returns:
            Dict with val_loss, val_dir_loss, val_ret_loss, val_dir_acc, val_ic
        """
        self.backbone.eval()

        total_loss     = 0.0
        total_dir_loss = 0.0
        total_ret_loss = 0.0
        total_dir_acc  = 0.0
        n_batches      = 0

        all_ret_preds  = []   # for IC computation
        all_ret_targets= []

        for x_batch, dir_batch, ret_batch in batch_iterator(
            self.val_dataset, self.batch_size, shuffle=False
        ):
            x_batch   = x_batch.to(self.device)
            dir_batch = dir_batch.to(self.device)
            ret_batch = ret_batch.to(self.device)

            emb, dir_logit, ret_pred = self.backbone(x_batch)

            losses = self.backbone.compute_loss(
                emb, dir_logit, ret_pred,
                dir_batch, ret_batch,
                self.dir_weight, self.ret_weight,
            )

            total_loss     += losses["total_loss"].item()
            total_dir_loss += losses["direction_loss"].item()
            total_ret_loss += losses["return_loss"].item()
            total_dir_acc  += losses["direction_acc"].item()
            n_batches      += 1

            all_ret_preds.append(ret_pred.cpu().numpy())
            all_ret_targets.append(ret_batch.cpu().numpy())

        n_batches = max(n_batches, 1)

        # Information Coefficient
        if all_ret_preds:
            preds = np.concatenate(all_ret_preds)
            targets = np.concatenate(all_ret_targets)
            val_ic = compute_ic(preds, targets)
        else:
            val_ic = 0.0

        return {
            "val_loss"    : total_loss     / n_batches,
            "val_dir_loss": total_dir_loss / n_batches,
            "val_ret_loss": total_ret_loss / n_batches,
            "val_dir_acc" : total_dir_acc  / n_batches,
            "val_ic"      : val_ic,
        }

    # ── Main training loop ────────────────────────────────────────────────

    def train(self, resume_from: Optional[str] = None) -> Dict:
        """
        Runs the full pre-training loop with early stopping.

        Args:
            resume_from : Optional path to checkpoint to resume from

        Returns:
            Dict with best metrics and whether gate criteria were met
        """
        start_epoch = 0

        # ── Resume from checkpoint if provided ────────────────────────────
        if resume_from and Path(resume_from).exists():
            logger.info(f"Resuming from checkpoint: {resume_from}")
            _, meta = GodsEyeBackbone.load_checkpoint(
                resume_from, map_location=str(self.device)
            )
            start_epoch        = meta.get("epoch", 0) + 1
            self.best_val_ic   = meta.get("metrics", {}).get("val_ic", -float("inf"))
            logger.info(
                f"Resumed at epoch {start_epoch} | "
                f"best val_ic so far: {self.best_val_ic:.4f}"
            )

        logger.info("=" * 60)
        logger.info("G.O.D.S E.Y.E — Phase 2 Backbone Pre-training")
        logger.info(f"Device    : {self.device}")
        logger.info(f"Epochs    : {start_epoch} → {self.max_epochs}")
        logger.info(f"Gate IC   : {IC_GATE} | Gate Acc: {ACC_GATE}")
        logger.info("=" * 60)

        for epoch in range(start_epoch, self.max_epochs):

            # ── Training ──────────────────────────────────────────────────
            train_metrics = self.train_epoch(epoch)

            # ── Validation ────────────────────────────────────────────────
            val_metrics = self.validate_epoch()

            # ── Current LR ───────────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── Check for improvement ─────────────────────────────────────
            val_ic  = val_metrics["val_ic"]
            is_best = val_ic > self.best_val_ic

            if is_best:
                self.best_val_ic   = val_ic
                self.best_epoch    = epoch
                self.epochs_no_imp = 0
                # Save best checkpoint
                self.backbone.save_checkpoint(
                    CHECKPOINT_DIR / "pretrain_best.pt",
                    epoch     = epoch,
                    optimizer = self.optimizer,
                    metrics   = {**train_metrics, **val_metrics},
                )
                logger.success(
                    f"Epoch {epoch:3d} ✓ NEW BEST "
                    f"| val_ic={val_ic:.4f} "
                    f"| val_acc={val_metrics['val_dir_acc']:.3f} "
                    f"| train_loss={train_metrics['train_loss']:.4f}"
                )
            else:
                self.epochs_no_imp += 1
                logger.info(
                    f"Epoch {epoch:3d} "
                    f"| val_ic={val_ic:.4f} "
                    f"| val_acc={val_metrics['val_dir_acc']:.3f} "
                    f"| train_loss={train_metrics['train_loss']:.4f} "
                    f"| no_imp={self.epochs_no_imp}/{self.patience}"
                )

            # ── Save last checkpoint (always) ─────────────────────────────
            self.backbone.save_checkpoint(
                CHECKPOINT_DIR / "pretrain_last.pt",
                epoch     = epoch,
                optimizer = self.optimizer,
                metrics   = {**train_metrics, **val_metrics},
            )

            # ── Log to CSV ────────────────────────────────────────────────
            row = {
                "epoch"         : epoch,
                "lr"            : f"{current_lr:.6f}",
                "is_best"       : int(is_best),
                **train_metrics,
                **val_metrics,
            }
            self._log_metrics(row)
            self.history.append(row)

            # ── Early stopping ────────────────────────────────────────────
            if self.epochs_no_imp >= self.patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        # ── Final report ──────────────────────────────────────────────────
        best_metrics = {}
        if self.history:
            best_row = max(self.history, key=lambda r: r.get("val_ic", -999))
            best_metrics = best_row

        gate_ic_pass  = self.best_val_ic >= IC_GATE
        gate_acc_pass = best_metrics.get("val_dir_acc", 0) >= ACC_GATE
        gate_passed   = gate_ic_pass and gate_acc_pass

        logger.info("=" * 60)
        logger.info("PRE-TRAINING COMPLETE")
        logger.info(f"Best epoch   : {self.best_epoch}")
        logger.info(f"Best val_ic  : {self.best_val_ic:.4f}  (gate: {IC_GATE}) {'✓ PASS' if gate_ic_pass  else '✗ FAIL'}")
        logger.info(f"Best val_acc : {best_metrics.get('val_dir_acc',0):.4f}  (gate: {ACC_GATE}) {'✓ PASS' if gate_acc_pass else '✗ FAIL'}")
        logger.info(f"Phase 2 Gate : {'✓ PASSED — ready for Phase 3 RL' if gate_passed else '✗ FAILED — retrain or adjust config'}")
        logger.info(f"Best model   : {CHECKPOINT_DIR / 'pretrain_best.pt'}")
        logger.info("=" * 60)

        return {
            "best_val_ic"     : self.best_val_ic,
            "best_epoch"      : self.best_epoch,
            "gate_passed"     : gate_passed,
            "gate_ic_pass"    : gate_ic_pass,
            "gate_acc_pass"   : gate_acc_pass,
            "best_val_dir_acc": best_metrics.get("val_dir_acc", 0.0),
        }


# ══════════════════════════════════════════════════════════════════════════
#  SMOKE TEST DATASET (no DB required)
# ══════════════════════════════════════════════════════════════════════════

def make_smoke_test_data(
    n_dates : int = 100,
    n_stocks: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic features + returns for smoke testing.
    No database connection required.

    Args:
        n_dates  : Number of trading days
        n_stocks : Number of stocks

    Returns:
        features   : (n_dates, n_stocks, N_FEATURES) float32
        returns_5d : (n_dates, n_stocks) float32
    """
    np.random.seed(42)
    features   = np.random.randn(n_dates, n_stocks, N_FEATURES).astype(np.float32)
    features   = np.clip(features, -1.0, 1.0)
    returns_5d = np.random.randn(n_dates, n_stocks).astype(np.float32) * 2.0
    return features, returns_5d


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="G.O.D.S E.Y.E — Phase 2 Backbone Pre-training"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick 3-epoch smoke test with synthetic data (no DB required)"
    )
    parser.add_argument(
        "--epochs", type=int, default=MAX_EPOCHS,
        help=f"Maximum epochs (default: {MAX_EPOCHS})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_STOCKS,
        help=f"Stocks per training step (default: {BATCH_STOCKS})"
    )
    parser.add_argument(
        "--lr", type=float, default=LR,
        help=f"Learning rate (default: {LR})"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Smoke test mode ───────────────────────────────────────────────────
    if args.smoke_test:
        logger.info("SMOKE TEST MODE — synthetic data, 3 epochs")
        features, returns_5d = make_smoke_test_data(n_dates=400, n_stocks=30)

        # 80/20 split — need >60 val rows for SEQ_LEN window
        split = int(len(features) * 0.8)
        train_feat, val_feat = features[:split], features[split:]
        train_ret,  val_ret  = returns_5d[:split], returns_5d[split:]

        train_ds = PretrainDataset(train_feat, train_ret, seq_len=SEQ_LEN)
        val_ds   = PretrainDataset(val_feat,   val_ret,   seq_len=SEQ_LEN)

        backbone = build_backbone(pretrain_mode=True)
        engine = PretrainEngine(
            backbone      = backbone,
            train_dataset = train_ds,
            val_dataset   = val_ds,
            batch_size    = 32,
            max_epochs    = 3,
            patience      = 3,
            device        = device,
        )
        results = engine.train()
        logger.info(f"Smoke test complete: {results}")
        return

    # ── Full training mode ────────────────────────────────────────────────
    logger.info("Connecting to TimescaleDB...")
    conn = get_db_connection()

    try:
        logger.info(f"Loading TRAIN features: {TRAIN_START} → {TRAIN_END}")
        train_feat, train_ret, train_syms, train_dates = load_fused_features(
            conn, TRAIN_START, TRAIN_END
        )

        logger.info(f"Loading VAL features: {VAL_START} → {VAL_END}")
        val_feat, val_ret, val_syms, val_dates = load_fused_features(
            conn, VAL_START, VAL_END
        )
    finally:
        conn.close()

    # Build datasets
    train_ds = PretrainDataset(train_feat, train_ret, seq_len=SEQ_LEN)
    val_ds   = PretrainDataset(val_feat,   val_ret,   seq_len=SEQ_LEN)

    # Build backbone
    backbone = build_backbone(pretrain_mode=True)
    logger.info(f"Backbone parameters: {backbone.num_parameters:,}")

    # Build engine and train
    engine = PretrainEngine(
        backbone      = backbone,
        train_dataset = train_ds,
        val_dataset   = val_ds,
        batch_size    = args.batch_size,
        lr            = args.lr,
        max_epochs    = args.epochs,
        patience      = PATIENCE,
        device        = device,
    )

    results = engine.train(resume_from=args.resume)

    if results["gate_passed"]:
        logger.success("Phase 2 complete. Ready to begin Phase 3 RL training.")
    else:
        logger.warning(
            "Phase 2 gate NOT passed. "
            "Consider: more epochs, lower LR, or check feature quality."
        )


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest training/pretrain_backbone.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestPretrainComponents:
    """
    Unit tests for all pre-training components.
    All tests use synthetic data — no database required.
    """

    def setup_method(self):
        torch.manual_seed(42)
        np.random.seed(42)

        self.n_dates  = 120
        self.n_stocks = 15
        self.n_feat   = N_FEATURES
        self.seq_len  = SEQ_LEN

        self.features, self.returns_5d = make_smoke_test_data(
            n_dates=self.n_dates, n_stocks=self.n_stocks
        )
        self.device = torch.device("cpu")

    # ── Dataset tests ─────────────────────────────────────────────────────

    def test_dataset_builds(self):
        """PretrainDataset must build without errors."""
        ds = PretrainDataset(self.features, self.returns_5d, seq_len=self.seq_len)
        assert len(ds) > 0, "Dataset has no samples"

    def test_dataset_sample_shapes(self):
        """Each sample must have correct shapes."""
        ds = PretrainDataset(self.features, self.returns_5d, seq_len=self.seq_len)
        x, dir_tgt, ret_tgt = ds[0]
        assert x.shape      == (self.seq_len, self.n_feat), \
            f"Feature shape wrong: {x.shape}"
        assert dir_tgt.shape == torch.Size([]), "Direction target must be scalar"
        assert ret_tgt.shape == torch.Size([]), "Return target must be scalar"

    def test_direction_target_binary(self):
        """Direction target must be exactly 0.0 or 1.0."""
        ds = PretrainDataset(self.features, self.returns_5d, seq_len=self.seq_len)
        for i in range(min(50, len(ds))):
            _, dir_tgt, _ = ds[i]
            assert dir_tgt.item() in {0.0, 1.0}, \
                f"Direction target {dir_tgt.item()} is not 0 or 1"

    def test_return_target_clipped(self):
        """Return target must be clipped to [-15, +15]."""
        ds = PretrainDataset(self.features, self.returns_5d, seq_len=self.seq_len)
        for i in range(min(50, len(ds))):
            _, _, ret_tgt = ds[i]
            assert -15.0 <= ret_tgt.item() <= 15.0, \
                f"Return target {ret_tgt.item()} outside [-15, +15]"

    def test_feature_values_in_range(self):
        """All feature values must be in [-1, +1]."""
        ds = PretrainDataset(self.features, self.returns_5d, seq_len=self.seq_len)
        x, _, _ = ds[0]
        assert x.min().item() >= -1.0 - 1e-5, "Feature below -1"
        assert x.max().item() <=  1.0 + 1e-5, "Feature above +1"

    def test_min_return_filter(self):
        """Samples with |return| < min_return must be excluded."""
        # Make returns all near-zero
        tiny_returns = np.ones((self.n_dates, self.n_stocks)) * 0.001
        ds = PretrainDataset(
            self.features, tiny_returns,
            seq_len=self.seq_len, min_return=0.1
        )
        assert len(ds) == 0, "Near-zero returns should be filtered out"

    # ── Batch iterator tests ──────────────────────────────────────────────

    def test_batch_iterator_shapes(self):
        """Batch iterator must yield correct shapes."""
        ds         = PretrainDataset(self.features, self.returns_5d)
        batch_size = 16
        for x_b, dir_b, ret_b in batch_iterator(ds, batch_size, shuffle=False):
            assert x_b.shape[1]  == self.seq_len, "Wrong seq_len in batch"
            assert x_b.shape[2]  == self.n_feat,  "Wrong n_feat in batch"
            assert dir_b.shape   == (x_b.shape[0],)
            assert ret_b.shape   == (x_b.shape[0],)
            break   # just check first batch

    def test_batch_iterator_covers_all_samples(self):
        """All samples must appear exactly once per epoch."""
        ds     = PretrainDataset(self.features, self.returns_5d)
        total  = sum(
            x_b.shape[0]
            for x_b, _, _ in batch_iterator(ds, 16, shuffle=False)
        )
        assert total == len(ds), \
            f"Iterator covered {total} samples, expected {len(ds)}"

    # ── IC metric test ────────────────────────────────────────────────────

    def test_ic_perfect_correlation(self):
        """IC of identical arrays must be 1.0."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic  = compute_ic(arr, arr)
        assert abs(ic - 1.0) < 1e-5, f"Perfect IC should be 1.0, got {ic}"

    def test_ic_perfect_anticorrelation(self):
        """IC of reversed arrays must be -1.0."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic  = compute_ic(arr, arr[::-1])
        assert abs(ic + 1.0) < 1e-5, f"Anti-correlated IC should be -1.0, got {ic}"

    def test_ic_random_near_zero(self):
        """IC of independent random arrays must be near 0."""
        np.random.seed(0)
        a  = np.random.randn(500)
        b  = np.random.randn(500)
        ic = compute_ic(a, b)
        assert abs(ic) < 0.15, f"Random IC too high: {ic:.4f}"

    def test_ic_too_few_samples(self):
        """IC must return 0.0 for fewer than 10 samples."""
        ic = compute_ic(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert ic == 0.0

    # ── Engine tests ──────────────────────────────────────────────────────

    def test_engine_one_epoch(self):
        """Engine must complete one training epoch without errors."""
        features, returns = make_smoke_test_data(n_dates=250, n_stocks=20)
        split = 200
        train_ds = PretrainDataset(features[:split],  returns[:split])
        val_ds   = PretrainDataset(features[split:],  returns[split:])

        backbone = GodsEyeBackbone(pretrain_mode=True)
        engine   = PretrainEngine(
            backbone      = backbone,
            train_dataset = train_ds,
            val_dataset   = val_ds,
            batch_size    = 32,
            max_epochs    = 1,
            patience      = 1,
            device        = self.device,
        )

        train_metrics = engine.train_epoch(epoch=0)
        assert "train_loss"     in train_metrics
        assert "train_dir_acc"  in train_metrics
        assert not math.isnan(train_metrics["train_loss"]), "Train loss is NaN"
        assert train_metrics["train_loss"] > 0

    def test_engine_validation(self):
        """Validation epoch must return val_ic metric."""
        features, returns = make_smoke_test_data(n_dates=250, n_stocks=20)
        split = 200
        train_ds = PretrainDataset(features[:split], returns[:split])
        val_ds   = PretrainDataset(features[split:], returns[split:])

        backbone = GodsEyeBackbone(pretrain_mode=True)
        engine   = PretrainEngine(
            backbone      = backbone,
            train_dataset = train_ds,
            val_dataset   = val_ds,
            batch_size    = 32,
            device        = self.device,
        )

        val_metrics = engine.validate_epoch()
        assert "val_ic"      in val_metrics
        assert "val_dir_acc" in val_metrics
        assert -1.0 <= val_metrics["val_ic"]      <= 1.0
        assert  0.0 <= val_metrics["val_dir_acc"] <= 1.0

    def test_engine_early_stopping(self):
        """Engine must stop early when patience exhausted."""
        features, returns = make_smoke_test_data(n_dates=250, n_stocks=20)
        split = 200
        train_ds = PretrainDataset(features[:split], returns[:split])
        val_ds   = PretrainDataset(features[split:], returns[split:])

        backbone = GodsEyeBackbone(pretrain_mode=True)
        engine   = PretrainEngine(
            backbone      = backbone,
            train_dataset = train_ds,
            val_dataset   = val_ds,
            batch_size    = 32,
            max_epochs    = 50,
            patience      = 3,    # stop after 3 epochs without improvement
            device        = self.device,
        )
        results = engine.train()

        # Should stop well before 50 epochs with patience=3
        assert engine.best_epoch < 45, \
            f"Expected early stopping before epoch 45, stopped at {engine.best_epoch}"

    def test_checkpoint_saved(self, tmp_path):
        """Training must save best and last checkpoints."""
        import shutil
        # Temporarily redirect checkpoint dir
        orig = CHECKPOINT_DIR

        features, returns = make_smoke_test_data(n_dates=250, n_stocks=20)
        split = 200
        train_ds = PretrainDataset(features[:split], returns[:split])
        val_ds   = PretrainDataset(features[split:], returns[split:])

        backbone = GodsEyeBackbone(pretrain_mode=True)
        engine   = PretrainEngine(
            backbone      = backbone,
            train_dataset = train_ds,
            val_dataset   = val_ds,
            batch_size    = 32,
            max_epochs    = 2,
            patience      = 2,
            device        = self.device,
        )
        engine.train()

        assert (CHECKPOINT_DIR / "pretrain_best.pt").exists(), \
            "pretrain_best.pt not saved"
        assert (CHECKPOINT_DIR / "pretrain_last.pt").exists(), \
            "pretrain_last.pt not saved"

    def test_smoke_test_data_shape(self):
        """make_smoke_test_data must return correct shapes."""
        feat, ret = make_smoke_test_data(n_dates=80, n_stocks=10)
        assert feat.shape == (80, 10, N_FEATURES)
        assert ret.shape  == (80, 10)
        assert feat.dtype == np.float32
        assert ret.dtype  == np.float32


# ── Run when executed directly ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--pytest" in sys.argv or any("pytest" in a for a in sys.argv):
        import pytest
        sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))
    else:
        main()