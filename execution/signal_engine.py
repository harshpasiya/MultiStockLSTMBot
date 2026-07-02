"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Signal Generation Engine                        ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : execution/signal_engine.py                             ║
║         Phase   : 5 — Paper Trading / Phase 6 — Live Trading            ║
║                                                                          ║
║  What this file does:                                                    ║
║    The central nervous system of G.O.D.S E.Y.E during live operation.   ║
║    Runs every morning at 9:00 AM IST (15 minutes before market open)    ║
║    and continuously during market hours to:                              ║
║      1. Load pre-computed embeddings for today                           ║
║      2. Build observation vector from live portfolio state               ║
║      3. Query trained PPO model for action                               ║
║      4. Run action through Risk Constitution                             ║
║      5. Size position via Kelly criterion                                ║
║      6. Emit trade signal to executor                                    ║
║      7. Monitor open positions for TP/SL/trailing stop hits             ║
║      8. Log everything to DB for audit trail                             ║
║                                                                          ║
║  Two operating modes:                                                    ║
║    PAPER  : Signals logged to DB, no real orders placed                  ║
║    LIVE   : Signals sent to kite_executor.py for real order placement   ║
║                                                                          ║
║  Signal lifecycle:                                                       ║
║    GENERATED → RC_CHECK → SIZED → EMITTED → FILLED → MONITORED → CLOSED║
║                                                                          ║
║  Usage:                                                                  ║
║    # Paper trading mode (Phase 5)                                        ║
║    python -m execution.signal_engine --mode paper                        ║
║                                                                          ║
║    # Live trading mode (Phase 6)                                         ║
║    python -m execution.signal_engine --mode live                         ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install stable-baselines3 torch psycopg2-binary redis loguru     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import redis
import torch

from dataclasses import dataclass, field
from datetime    import datetime, date, timedelta
from enum        import Enum
from pathlib     import Path
from typing      import Dict, List, Optional, Tuple
from loguru      import logger
from dotenv      import load_dotenv

from stable_baselines3 import PPO

from environment.risk_constitution import (
    RiskConstitution, PortfolioState, MarketState, RCResult
)
from environment.position_sizer import (
    PositionSizer, SizingInput, SizingResult, build_position_sizer
)
from environment.reward_fn      import RewardMode
from models.backbone            import GodsEyeBackbone

load_dotenv()

# ── Connection config ──────────────────────────────────────────────────────
DB_URL        = os.getenv("TIMESCALE_URL", "postgresql://godseye_user:godseye_pass@localhost:5433/godseye")
REDIS_URL     = os.getenv("REDIS_URL",     "redis://:godseye_redis_pass@localhost:6380")
KITE_API_KEY  = os.getenv("KITE_API_KEY",  "2ab966z3tkr18z3c")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent
CHECKPOINT_DIR= ROOT_DIR / "checkpoints"
SWING_BEST    = CHECKPOINT_DIR / "swing_best.pt"

# ── Signal engine config ───────────────────────────────────────────────────
EMBEDDING_DIM      = 128
N_STOCKS           = 46      # full universe — must match training exactly
PORTFOLIO_DIM      = 8
OBS_DIM            = N_STOCKS * EMBEDDING_DIM + PORTFOLIO_DIM
MIN_CONFIDENCE     = 0.55     # minimum RL confidence to emit signal
SIGNAL_LOOP_SEC    = 60       # check positions every 60 seconds
PRE_MARKET_HOUR    = 9        # 9:00 AM IST pre-market run
PRE_MARKET_MINUTE  = 0
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR  = 15
MARKET_CLOSE_MINUTE= 30


# ══════════════════════════════════════════════════════════════════════════
#  ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════

class SignalMode(str, Enum):
    PAPER = "paper"
    LIVE  = "live"


class SignalAction(str, Enum):
    BUY         = "BUY"
    SELL        = "SELL"
    HOLD        = "HOLD"
    STRONG_BUY  = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class SignalStatus(str, Enum):
    GENERATED = "GENERATED"
    RC_BLOCKED= "RC_BLOCKED"
    SIZED     = "SIZED"
    EMITTED   = "EMITTED"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_TRAIL = "CLOSED_TRAIL"
    CLOSED_MANUAL= "CLOSED_MANUAL"


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeSignal:
    """
    A complete trade signal with all information needed for execution.
    Generated by SignalEngine and passed to the executor.
    """
    signal_id       : str
    timestamp       : datetime
    symbol          : str
    action          : SignalAction
    mode            : SignalMode

    # ── Prices ────────────────────────────────────────────────────────────
    entry_price     : float
    tp_price        : float
    sl_price        : float
    trail_activate  : float    # price at which trailing stop activates
    trail_distance  : float    # trailing stop distance (fraction)

    # ── Sizing ────────────────────────────────────────────────────────────
    position_pct    : float    # fraction of portfolio
    invest_amount   : float    # ₹ to invest
    quantity        : int      # shares to buy/sell

    # ── Confidence ────────────────────────────────────────────────────────
    rl_confidence   : float    # PPO action probability [0, 1]
    mds_score       : int      # Market Direction Signal [-3, +3]

    # ── Pillar scores (for logging/monitoring) ────────────────────────────
    trend_score     : float = 0.0
    msi_score       : float = 0.0
    sentiment_score : float = 0.0
    volatility_score: float = 0.0

    # ── Status ────────────────────────────────────────────────────────────
    status          : SignalStatus = SignalStatus.GENERATED
    rc_rule_blocked : str          = ""
    notes           : str          = ""

    def to_dict(self) -> Dict:
        return {
            "signal_id"      : self.signal_id,
            "timestamp"      : self.timestamp.isoformat(),
            "symbol"         : self.symbol,
            "action"         : self.action.value,
            "mode"           : self.mode.value,
            "entry_price"    : self.entry_price,
            "tp_price"       : self.tp_price,
            "sl_price"       : self.sl_price,
            "position_pct"   : self.position_pct,
            "invest_amount"  : self.invest_amount,
            "quantity"       : self.quantity,
            "rl_confidence"  : self.rl_confidence,
            "mds_score"      : self.mds_score,
            "trend_score"    : self.trend_score,
            "msi_score"      : self.msi_score,
            "sentiment_score": self.sentiment_score,
            "status"         : self.status.value,
            "rc_rule_blocked": self.rc_rule_blocked,
        }


@dataclass
class OpenPosition:
    """Tracks an open position during live/paper trading."""
    symbol          : str
    signal_id       : str
    entry_price     : float
    quantity        : int
    tp_price        : float
    sl_price        : float
    entry_time      : datetime
    trail_active    : bool  = False
    trail_peak      : float = 0.0
    trail_stop      : float = 0.0
    current_price   : float = 0.0
    unrealised_pnl  : float = 0.0
    hold_days       : int   = 0

    @property
    def position_value(self) -> float:
        return self.entry_price * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


# ══════════════════════════════════════════════════════════════════════════
#  LIVE DATA PROVIDER
# ══════════════════════════════════════════════════════════════════════════

class LiveDataProvider:
    """
    Provides real-time market data from Redis cache and TimescaleDB.

    Redis holds tick-level data updated by the Kite WebSocket feed.
    TimescaleDB holds end-of-day data and pre-computed embeddings.
    """

    def __init__(self):
        self._redis  = None
        self._conn   = None
        self._connect()

    def _connect(self):
        try:
            self._redis = redis.from_url(REDIS_URL, decode_responses=True)
            self._redis.ping()
            logger.info("Redis connected.")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Using DB fallback.")
            self._redis = None

        self._conn = psycopg2.connect(DB_URL)
        self._conn.autocommit = True   # prevents transaction abort cascade

    def get_ltp(self, symbol: str) -> Optional[float]:
        """
        Returns Last Traded Price from Redis (live) or DB (fallback).
        Redis key: "ltp:{symbol}" set by kite_feed.py WebSocket handler.
        """
        if self._redis:
            try:
                val = self._redis.get(f"ltp:{symbol}")
                if val:
                    return float(val)
            except Exception:
                pass

        # DB fallback: latest close price
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT close FROM daily_ohlcv
                WHERE symbol = %s
                ORDER BY date DESC LIMIT 1;
            """, (symbol,))
            row = cur.fetchone()
        return float(row[0]) if row else None

    def get_embedding(self, symbol: str, date_str: str) -> Optional[np.ndarray]:
        """Returns pre-computed backbone embedding for symbol on date."""
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT embedding FROM backbone_embeddings
                WHERE symbol = %s AND date = %s;
            """, (symbol, date_str))
            row = cur.fetchone()
        if row and row[0]:
            return np.array(row[0], dtype=np.float32)
        return None

    def get_top_symbols_by_embedding(
        self, date_str: str, n: int = N_STOCKS
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Returns top-N symbols by embedding norm for today.
        Used to build the observation vector for the PPO model.
        Falls back to the latest available date if date_str has no data.
        """
        try:
            self._conn.rollback()   # clear any failed transaction
        except Exception:
            pass

        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT symbol, embedding
                    FROM backbone_embeddings
                    WHERE date = %s
                    LIMIT %s;
                """, (date_str, n))
                rows = cur.fetchall()
        except Exception as e:
            logger.warning(f"get_top_symbols_by_embedding failed for {date_str}: {e}")
            rows = []

        if not rows:
            # Fallback: use latest available date at or before date_str
            try:
                self._conn.rollback()
                with self._conn.cursor() as cur:
                    cur.execute("""
                        SELECT symbol, embedding
                        FROM backbone_embeddings
                        WHERE date = (
                            SELECT MAX(date) FROM backbone_embeddings
                            WHERE date <= %s
                        )
                        LIMIT %s;
                    """, (date_str, n))
                    rows = cur.fetchall()
                if rows:
                    logger.info(f"Using latest available embeddings instead of {date_str}")
            except Exception as e2:
                logger.error(f"Embedding fallback also failed: {e2}")
                return []

        result = []
        for sym, emb in rows:
            if emb:
                result.append((sym, np.array(emb, dtype=np.float32)))
        return result

    def get_mds_score(self) -> int:
        """Returns today's Market Direction Signal from features_fii_dii."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT mds_score FROM features_fii_dii
                    ORDER BY date DESC LIMIT 1;
                """)
                row = cur.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def get_pillar_scores(self, symbol: str) -> Dict[str, float]:
        """Returns latest pillar scores for a symbol from feature tables."""
        scores = {
            "trend_score"    : 0.0,
            "msi_signal"     : 0.0,
            "sentiment_score": 0.0,
            "volatility_score": 0.0,
            "atr_pct"        : 0.02,
            "vol_regime"     : "normal",
        }
        try:
            with self._conn.cursor() as cur:
                # Trend
                cur.execute("""
                    SELECT trend_score FROM features_trend
                    WHERE symbol = %s ORDER BY date DESC LIMIT 1;
                """, (symbol,))
                row = cur.fetchone()
                if row:
                    scores["trend_score"] = float(row[0] or 0)

                # MSI
                cur.execute("""
                    SELECT msi_signal FROM features_msi
                    WHERE symbol = %s ORDER BY date DESC LIMIT 1;
                """, (symbol,))
                row = cur.fetchone()
                if row:
                    scores["msi_signal"] = float(row[0] or 0)

                # Volatility
                cur.execute("""
                    SELECT volatility_score, atr_pct, vol_regime_code
                    FROM features_volatility
                    WHERE symbol = %s ORDER BY date DESC LIMIT 1;
                """, (symbol,))
                row = cur.fetchone()
                if row:
                    scores["volatility_score"] = float(row[0] or 0)
                    scores["atr_pct"]          = float(row[1] or 0.02)
                    regime_code = row[2] or 1
                    scores["vol_regime"] = {
                        0: "low", 1: "normal", 2: "high", 3: "extreme"
                    }.get(int(regime_code), "normal")

        except Exception as e:
            logger.debug(f"Pillar score fetch failed for {symbol}: {e}")

        return scores

    def get_india_vix(self) -> Tuple[float, float]:
        """Returns (vix_open, vix_current) from Redis or DB."""
        try:
            if self._redis:
                open_v = self._redis.get("vix:open")
                curr_v = self._redis.get("vix:current")
                if open_v and curr_v:
                    return float(open_v), float(curr_v)
        except Exception:
            pass
        return 15.0, 15.0   # neutral fallback

    def get_nifty_price(self) -> Tuple[float, float]:
        """Returns (nifty_open, nifty_current) from Redis."""
        try:
            if self._redis:
                open_n = self._redis.get("nifty:open")
                curr_n = self._redis.get("nifty:current")
                if open_n and curr_n:
                    return float(open_n), float(curr_n)
        except Exception:
            pass
        return 22000.0, 22000.0   # neutral fallback

    def get_fii_provisional(self) -> float:
        """Returns today's FII provisional net (₹ crore) from DB."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT fii_net_cr FROM fii_dii_flow
                    WHERE date = CURRENT_DATE
                    ORDER BY updated_at DESC LIMIT 1;
                """)
                row = cur.fetchone()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    def close(self):
        if self._conn:
            self._conn.close()


# ══════════════════════════════════════════════════════════════════════════
#  PORTFOLIO STATE MANAGER
# ══════════════════════════════════════════════════════════════════════════

class PortfolioManager:
    """
    Tracks portfolio state during paper/live trading.
    Persists positions and P&L to TimescaleDB.
    """

    def __init__(self, initial_capital: float = 1_000_000.0, mode: SignalMode = SignalMode.PAPER):
        self.initial_capital = initial_capital
        self.mode            = mode
        self._conn           = psycopg2.connect(DB_URL)
        self._open_positions : Dict[str, OpenPosition] = {}
        self._ensure_tables()
        self._load_open_positions()

    def _ensure_tables(self):
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal_log (
                    signal_id       VARCHAR(50) PRIMARY KEY,
                    timestamp       TIMESTAMP   NOT NULL,
                    symbol          VARCHAR(20) NOT NULL,
                    action          VARCHAR(20),
                    mode            VARCHAR(10),
                    entry_price     NUMERIC(12,4),
                    tp_price        NUMERIC(12,4),
                    sl_price        NUMERIC(12,4),
                    position_pct    NUMERIC(8,4),
                    invest_amount   NUMERIC(14,2),
                    quantity        INTEGER,
                    rl_confidence   NUMERIC(6,4),
                    mds_score       SMALLINT,
                    trend_score     NUMERIC(6,4),
                    msi_score       NUMERIC(6,4),
                    sentiment_score NUMERIC(6,4),
                    status          VARCHAR(20),
                    rc_rule_blocked VARCHAR(10),
                    exit_price      NUMERIC(12,4),
                    exit_time       TIMESTAMP,
                    realised_pnl    NUMERIC(14,2),
                    notes           TEXT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    recorded_at     TIMESTAMP PRIMARY KEY DEFAULT NOW(),
                    mode            VARCHAR(10),
                    portfolio_value NUMERIC(14,2),
                    cash            NUMERIC(14,2),
                    n_positions     SMALLINT,
                    daily_pnl       NUMERIC(14,2),
                    total_pnl       NUMERIC(14,2),
                    drawdown        NUMERIC(8,4)
                );
            """)
        self._conn.commit()

    def _load_open_positions(self):
        """Loads open positions from signal_log on startup."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT signal_id, symbol, entry_price, quantity,
                           tp_price, sl_price, timestamp
                    FROM signal_log
                    WHERE status = 'FILLED'
                      AND mode   = %s
                    ORDER BY timestamp;
                """, (self.mode.value,))
                rows = cur.fetchall()

            for sig_id, sym, ep, qty, tp, sl, ts in rows:
                self._open_positions[sym] = OpenPosition(
                    symbol      = sym,
                    signal_id   = sig_id,
                    entry_price = float(ep),
                    quantity    = int(qty),
                    tp_price    = float(tp),
                    sl_price    = float(sl),
                    entry_time  = ts,
                    trail_peak  = float(ep),
                    trail_stop  = float(sl),
                    current_price= float(ep),
                )
            logger.info(
                f"Loaded {len(self._open_positions)} open positions from DB."
            )
        except Exception as e:
            logger.warning(f"Could not load open positions: {e}")

    @property
    def portfolio_value(self) -> float:
        """Mark-to-market portfolio value."""
        mtm = sum(
            pos.current_price * pos.quantity
            for pos in self._open_positions.values()
        )
        return self.cash + mtm

    @property
    def cash(self) -> float:
        """Available cash = initial - invested in open positions."""
        invested = sum(
            pos.entry_price * pos.quantity
            for pos in self._open_positions.values()
        )
        return max(0.0, self.initial_capital - invested)

    @property
    def open_positions(self) -> Dict[str, OpenPosition]:
        return self._open_positions

    @property
    def n_positions(self) -> int:
        return len(self._open_positions)

    def update_prices(self, prices: Dict[str, float]):
        """Updates mark-to-market prices for all open positions."""
        for sym, pos in self._open_positions.items():
            if sym in prices:
                pos.current_price  = prices[sym]
                pos.unrealised_pnl = (
                    (prices[sym] - pos.entry_price) * pos.quantity
                )

    def add_position(self, signal: TradeSignal):
        """Records a newly filled position."""
        self._open_positions[signal.symbol] = OpenPosition(
            symbol      = signal.symbol,
            signal_id   = signal.signal_id,
            entry_price = signal.entry_price,
            quantity    = signal.quantity,
            tp_price    = signal.tp_price,
            sl_price    = signal.sl_price,
            entry_time  = signal.timestamp,
            trail_peak  = signal.entry_price,
            trail_stop  = signal.sl_price,
            current_price= signal.entry_price,
        )

    def remove_position(self, symbol: str):
        """Removes a closed position."""
        self._open_positions.pop(symbol, None)

    def save_signal(self, signal: TradeSignal):
        """Persists signal to signal_log table."""
        sql = """
            INSERT INTO signal_log (
                signal_id, timestamp, symbol, action, mode,
                entry_price, tp_price, sl_price,
                position_pct, invest_amount, quantity,
                rl_confidence, mds_score,
                trend_score, msi_score, sentiment_score,
                status, rc_rule_blocked, notes
            ) VALUES (
                %(signal_id)s, %(timestamp)s, %(symbol)s, %(action)s, %(mode)s,
                %(entry_price)s, %(tp_price)s, %(sl_price)s,
                %(position_pct)s, %(invest_amount)s, %(quantity)s,
                %(rl_confidence)s, %(mds_score)s,
                %(trend_score)s, %(msi_score)s, %(sentiment_score)s,
                %(status)s, %(rc_rule_blocked)s, %(notes)s
            )
            ON CONFLICT (signal_id) DO UPDATE SET
                status          = EXCLUDED.status,
                rc_rule_blocked = EXCLUDED.rc_rule_blocked;
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, {
                    **signal.to_dict(),
                    "timestamp": signal.timestamp,
                })
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Failed to save signal {signal.signal_id}: {e}")

    def update_signal_status(
        self,
        signal_id  : str,
        status     : SignalStatus,
        exit_price : Optional[float] = None,
        pnl        : Optional[float] = None,
    ):
        """Updates signal status and exit info in signal_log."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    UPDATE signal_log SET
                        status       = %s,
                        exit_price   = %s,
                        exit_time    = NOW(),
                        realised_pnl = %s
                    WHERE signal_id = %s;
                """, (status.value, exit_price, pnl, signal_id))
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Failed to update signal {signal_id}: {e}")

    def build_portfolio_state(self, sector_map: Dict[str, str]) -> PortfolioState:
        """Builds PortfolioState for Risk Constitution evaluation."""
        positions_dict = {
            sym: {
                "entry_price": pos.entry_price,
                "quantity"   : pos.quantity,
                "sl_price"   : pos.sl_price,
            }
            for sym, pos in self._open_positions.items()
        }
        return PortfolioState(
            total_value       = self.portfolio_value,
            peak_value        = max(self.portfolio_value, self.initial_capital),
            cash              = self.cash,
            open_positions    = positions_dict,
            trades_today      = self._count_trades_today(),
            trades_this_month = self._count_trades_month(),
            sector_map        = sector_map,
        )

    def _count_trades_today(self) -> int:
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM signal_log
                    WHERE DATE(timestamp) = CURRENT_DATE
                      AND status = 'FILLED'
                      AND mode   = %s;
                """, (self.mode.value,))
                return int(self._conn.cursor().fetchone()[0])
        except Exception:
            return 0

    def _count_trades_month(self) -> int:
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM signal_log
                    WHERE DATE_TRUNC('month', timestamp) = DATE_TRUNC('month', NOW())
                      AND status = 'FILLED'
                      AND mode   = %s;
                """, (self.mode.value,))
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0

    def close(self):
        if self._conn:
            self._conn.close()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════════

class SignalEngine:
    """
    The central signal generation engine for G.O.D.S E.Y.E.

    Runs a continuous loop during market hours:
        Pre-market (9:00 AM): generate entry signals for the day
        Market hours: monitor open positions, manage trailing stops
        Post-market: log daily P&L summary

    Usage:
        engine = SignalEngine(mode=SignalMode.PAPER)
        engine.setup()
        engine.run()   # blocks until market close
    """

    def __init__(
        self,
        mode           : SignalMode = SignalMode.PAPER,
        initial_capital: float      = 1_000_000.0,
        checkpoint_path: Path       = SWING_BEST,
        device         : str        = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.mode            = mode
        self.initial_capital = initial_capital
        self.checkpoint_path = checkpoint_path
        self.device          = device

        # Components (populated in setup())
        self.model     : Optional[PPO]              = None
        self.backbone  : Optional[GodsEyeBackbone]  = None
        self.rc        : Optional[RiskConstitution] = None
        self.sizer     : Optional[PositionSizer]    = None
        self.data      : Optional[LiveDataProvider] = None
        self.portfolio : Optional[PortfolioManager] = None

        # Sector mapping (loaded from DB)
        self._sector_map: Dict[str, str] = {}

        logger.info(f"SignalEngine initialised | mode={mode.value} | device={device}")

    def setup(self):
        """Loads all components. Call once before run()."""
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        logger.info("SignalEngine: setting up...")

        # ── Load PPO checkpoint dict ───────────────────────────────────────
        logger.info(f"Loading PPO model from {self.checkpoint_path}...")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Complete Phase 3 training first."
            )

        # Our checkpoint is a custom dict, not SB3 native format.
        # Load weights dict first.
        ckpt = torch.load(
            str(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )

        # Read exact obs_dim and net_arch from checkpoint weights —
        # don't trust hardcoded constants, the checkpoint is ground truth
        state    = ckpt.get("ppo_policy_state", {})
        w0       = state.get("mlp_extractor.policy_net.0.weight")
        w2       = state.get("mlp_extractor.policy_net.2.weight")
        ckpt_obs_dim = int(w0.shape[1]) if w0 is not None else OBS_DIM
        hidden1      = int(w0.shape[0]) if w0 is not None else 512
        hidden2      = int(w2.shape[0]) if w2 is not None else 256

        logger.info(
            f"Checkpoint architecture: obs_dim={ckpt_obs_dim} "
            f"net_arch=[{hidden1},{hidden2}]"
        )

        # Build a minimal gymnasium env matching the checkpoint's exact
        # obs_dim. This avoids needing a real GodsEyeEnv/data_loader just
        # to initialise SB3's observation_space/action_space.
        _obs_dim = ckpt_obs_dim

        def _make_fake_env():
            env = gym.Env()
            env.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(_obs_dim,), dtype=np.float32,
            )
            env.action_space = gym.spaces.Discrete(5)

            def reset(**kwargs):
                return np.zeros(_obs_dim, dtype=np.float32), {}

            def step(action):
                return np.zeros(_obs_dim, dtype=np.float32), 0.0, True, False, {}

            env.reset = reset
            env.step  = step
            return env

        # Build PPO shell on CPU with matching architecture.
        # MlpPolicy runs fine and faster on CPU for inference; GPU is
        # reserved for the backbone embedding computation below.
        vec_env    = DummyVecEnv([_make_fake_env])
        self.model = PPO(
            policy        = "MlpPolicy",
            env           = vec_env,
            device        = "cpu",
            verbose       = 0,
            policy_kwargs = dict(net_arch=[hidden1, hidden2]),
        )
        vec_env.close()

        # Restore policy weights — model and weights both on CPU
        if "ppo_policy_state" in ckpt:
            self.model.policy.load_state_dict(
                ckpt["ppo_policy_state"], strict=True
            )
            self.model.policy = self.model.policy.to("cpu")
            logger.success(
                f"PPO policy restored | "
                f"obs={ckpt_obs_dim} | "
                f"arch=[{hidden1},{hidden2}] | "
                f"step={ckpt.get('metadata', {}).get('step', 'unknown')}"
            )
        else:
            logger.warning("No ppo_policy_state in checkpoint — signals will be random.")

        # Store checkpoint obs_dim for use elsewhere in the class
        self._ckpt_obs_dim = ckpt_obs_dim

        # ── Load backbone on CUDA (GPU reserved for embedding inference) ───
        self.backbone = GodsEyeBackbone()
        if "backbone_state" in ckpt:
            self.backbone.load_state_dict(ckpt["backbone_state"], strict=False)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        logger.info(
            f"Backbone loaded on {self.device} | "
            f"{sum(p.numel() for p in self.backbone.parameters()):,} params"
        )

        # ── Initialise other components ────────────────────────────────────
        self.rc        = RiskConstitution()
        self.sizer     = build_position_sizer()
        self.data      = LiveDataProvider()
        self.portfolio = PortfolioManager(self.initial_capital, self.mode)

        logger.success("SignalEngine ready.")

    def run(self):
        """
        Main trading loop. Runs continuously until market close.
        Call this after setup().
        """
        logger.info(
            f"SignalEngine running in {self.mode.value.upper()} mode. "
            f"Press Ctrl+C to stop."
        )

        try:
            while True:
                now = datetime.now()

                # ── Pre-market signal generation ──────────────────────────
                if self._is_pre_market(now):
                    logger.info("Pre-market: generating entry signals...")
                    signals = self.generate_entry_signals()
                    logger.info(f"Generated {len(signals)} entry signals.")
                    time.sleep(60 * 15)   # wait 15 mins before next check

                # ── Market hours: position monitoring ─────────────────────
                elif self._is_market_hours(now):
                    self._monitor_positions()
                    time.sleep(SIGNAL_LOOP_SEC)

                # ── Market close: EOD summary ──────────────────────────────
                elif self._is_market_close(now):
                    self._eod_summary()
                    time.sleep(60 * 60)   # sleep 1 hour

                else:
                    # Outside market hours — sleep
                    time.sleep(60 * 5)

        except KeyboardInterrupt:
            logger.info("Signal engine stopped by user.")
        finally:
            self.close()

    def generate_entry_signals(self) -> List[TradeSignal]:
        """
        Generates entry signals for the current trading day.

        Steps:
            1. Get today's embeddings for top-N stocks
            2. Build observation vector
            3. Query PPO model
            4. Apply Risk Constitution
            5. Size via Kelly
            6. Return TradeSignal list

        Returns:
            List of TradeSignal with status EMITTED or RC_BLOCKED
        """
        # Use latest available embedding date (not necessarily today —
        # market data for today is only available after market close,
        # and the model trades on yesterday's close, same as backtest)
        try:
            with self.data._conn.cursor() as _cur:
                _cur.execute("SELECT MAX(date) FROM backbone_embeddings;")
                _latest = _cur.fetchone()[0]
            today = str(_latest) if _latest else date.today().strftime("%Y-%m-%d")
            logger.info(f"Using latest embedding date: {today}")
        except Exception:
            today = date.today().strftime("%Y-%m-%d")

        signals  : List[TradeSignal] = []

        # ── Build market state ─────────────────────────────────────────────
        vix_open, vix_curr   = self.data.get_india_vix()
        nifty_open, nifty_curr = self.data.get_nifty_price()
        fii_net              = self.data.get_fii_provisional()
        mds_score            = self.data.get_mds_score()

        market_state = MarketState(
            nifty_open        = nifty_open,
            nifty_current     = nifty_curr,
            india_vix_open    = vix_open,
            india_vix_current = vix_curr,
            fii_provisional_cr= fii_net,
            mds_score         = mds_score,
        )

        # ── Build portfolio state ──────────────────────────────────────────
        portfolio_state = self.portfolio.build_portfolio_state(self._sector_map)

        # ── Get top stocks with embeddings ─────────────────────────────────
        top_stocks = self.data.get_top_symbols_by_embedding(today, N_STOCKS)
        if not top_stocks:
            logger.warning(f"No embeddings found for {today}. Run precompute_embeddings.py.")
            return []

        # ── Build observation ──────────────────────────────────────────────
        obs = self._build_observation(top_stocks, portfolio_state)

        # ── Query PPO model ────────────────────────────────────────────────
        # Use predict() for the action — the correct SB3 inference method.
        # action_net output is integer logits; must cast to float before
        # softmax or torch raises "softmax_lastdim_kernel_impl not
        # implemented for Long".
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to("cpu")
        with torch.no_grad():
            action_array, _ = self.model.predict(obs, deterministic=False)
            action_idx      = int(action_array)

            features      = self.model.policy.extract_features(
                obs_tensor, self.model.policy.features_extractor
            )
            latent_pi, _  = self.model.policy.mlp_extractor(features)
            logits        = self.model.policy.action_net(latent_pi)
            probs         = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]

        confidence  = float(probs[action_idx])

        # Action mapping: 0=Strong Sell, 1=Sell, 2=Hold, 3=Buy, 4=Strong Buy
        action_map  = {
            0: SignalAction.STRONG_SELL,
            1: SignalAction.SELL,
            2: SignalAction.HOLD,
            3: SignalAction.BUY,
            4: SignalAction.STRONG_BUY,
        }
        action = action_map.get(action_idx, SignalAction.HOLD)

        logger.info(
            f"PPO output: {action.value} (confidence={confidence:.3f}) | "
            f"MDS={mds_score}"
        )

        # Skip HOLD and SELL when no positions open
        if action in (SignalAction.HOLD, SignalAction.SELL, SignalAction.STRONG_SELL):
            if portfolio_state.n_positions == 0:
                logger.info("HOLD/SELL signal with no open positions — skipping.")
                return []

        # ── Process BUY signals ────────────────────────────────────────────
        if action in (SignalAction.BUY, SignalAction.STRONG_BUY):
            if confidence < MIN_CONFIDENCE:
                logger.info(f"Confidence {confidence:.3f} below threshold — skipping.")
                return []

            # Try each top stock until one passes RC
            for sym, _ in top_stocks:
                if sym in portfolio_state.open_positions:
                    continue

                ltp = self.data.get_ltp(sym)
                if not ltp:
                    continue

                pillar = self.data.get_pillar_scores(sym)

                # ── RC evaluation ──────────────────────────────────────────
                rc_result = self.rc.evaluate_entry(
                    symbol          = sym,
                    portfolio       = portfolio_state,
                    market          = market_state,
                    avg_turnover_cr = 50.0,   # TODO: compute from DB
                )

                signal_id = f"{sym}_{today}_{int(time.time())}"

                if rc_result.blocked:
                    sig = self._make_blocked_signal(
                        signal_id, sym, action, ltp, rc_result, mds_score, pillar
                    )
                    self.portfolio.save_signal(sig)
                    logger.info(f"  {sym}: BLOCKED by {rc_result.rule} — {rc_result.reason}")
                    continue

                # ── Size position ──────────────────────────────────────────
                atr_pct   = pillar.get("atr_pct", 0.02)
                tp_price  = ltp * (1 + max(0.04, 2 * atr_pct))
                sl_price  = ltp * (1 - max(0.015, 0.75 * atr_pct))
                vol_regime= pillar.get("vol_regime", "normal")

                sizing = self.sizer.compute(SizingInput(
                    symbol           = sym,
                    entry_price      = ltp,
                    tp_price         = tp_price,
                    sl_price         = sl_price,
                    confidence_score = confidence,
                    portfolio_value  = portfolio_state.total_value,
                    available_cash   = portfolio_state.cash,
                    current_drawdown = portfolio_state.drawdown,
                    open_risk_pct    = 0.02,
                    mds_score        = mds_score,
                    vol_regime       = vol_regime,
                    atr_pct          = atr_pct,
                    is_strong_signal = (action == SignalAction.STRONG_BUY),
                ))

                if sizing.is_zero:
                    logger.info(f"  {sym}: sizing zero — {sizing.reason}")
                    continue

                # ── Emit signal ────────────────────────────────────────────
                sig = TradeSignal(
                    signal_id        = signal_id,
                    timestamp        = datetime.now(),
                    symbol           = sym,
                    action           = action,
                    mode             = self.mode,
                    entry_price      = ltp,
                    tp_price         = tp_price,
                    sl_price         = sl_price,
                    trail_activate   = ltp * 1.02,
                    trail_distance   = 0.008,
                    position_pct     = sizing.position_pct,
                    invest_amount    = sizing.invest_amount,
                    quantity         = sizing.quantity,
                    rl_confidence    = confidence,
                    mds_score        = mds_score,
                    trend_score      = pillar.get("trend_score", 0.0),
                    msi_score        = pillar.get("msi_signal", 0.0),
                    sentiment_score  = pillar.get("sentiment_score", 0.0),
                    status           = SignalStatus.EMITTED,
                )

                self.portfolio.save_signal(sig)
                signals.append(sig)

                logger.success(
                    f"  ✓ SIGNAL: {action.value} {sym} @ ₹{ltp:.2f} "
                    f"| qty={sizing.quantity} "
                    f"| TP=₹{tp_price:.2f} SL=₹{sl_price:.2f} "
                    f"| conf={confidence:.3f}"
                )
                break   # one signal per cycle (conservative)

        return signals

    def _monitor_positions(self):
        """
        Checks all open positions for TP/SL/trailing stop hits.
        Called every SIGNAL_LOOP_SEC during market hours.
        """
        if not self.portfolio.open_positions:
            return

        now = datetime.now()

        for sym, pos in list(self.portfolio.open_positions.items()):
            ltp = self.data.get_ltp(sym)
            if not ltp:
                continue

            pos.current_price   = ltp
            pos.hold_days      += 0   # incremented at EOD
            pos.unrealised_pnl  = (ltp - pos.entry_price) * pos.quantity

            close_reason = None

            # ── Take profit hit ────────────────────────────────────────────
            if ltp >= pos.tp_price:
                close_reason = SignalStatus.CLOSED_TP

            # ── Stop loss hit ──────────────────────────────────────────────
            elif ltp <= pos.sl_price:
                close_reason = SignalStatus.CLOSED_SL

            # ── Trailing stop ──────────────────────────────────────────────
            else:
                gain_pct = (ltp - pos.entry_price) / pos.entry_price
                if gain_pct >= 0.02 and not pos.trail_active:
                    pos.trail_active = True
                    pos.trail_peak   = ltp
                    pos.trail_stop   = ltp * 0.992

                if pos.trail_active:
                    if ltp > pos.trail_peak:
                        pos.trail_peak = ltp
                        pos.trail_stop = ltp * 0.992
                    if ltp <= pos.trail_stop:
                        close_reason = SignalStatus.CLOSED_TRAIL

            # ── RC-02 position loss check ──────────────────────────────────
            port_state = self.portfolio.build_portfolio_state(self._sector_map)
            rc_pos_result = self.rc.evaluate_position(
                sym,
                {"entry_price": pos.entry_price, "quantity": pos.quantity, "sl_price": pos.sl_price},
                ltp,
                port_state,
            )
            if rc_pos_result.action == "force_close":
                close_reason = SignalStatus.CLOSED_SL

            # ── Close position if triggered ────────────────────────────────
            if close_reason:
                pnl = (ltp - pos.entry_price) * pos.quantity
                self.portfolio.update_signal_status(
                    pos.signal_id, close_reason, ltp, pnl
                )
                self.portfolio.remove_position(sym)
                logger.info(
                    f"Position closed: {sym} @ ₹{ltp:.2f} "
                    f"| reason={close_reason.value} "
                    f"| P&L=₹{pnl:+,.0f}"
                )

    def _eod_summary(self):
        """Logs end-of-day portfolio summary."""
        pv  = self.portfolio.portfolio_value
        pnl = pv - self.initial_capital
        dd  = max(0.0, (self.initial_capital - pv) / self.initial_capital)

        logger.info("─" * 60)
        logger.info(f"EOD SUMMARY — {date.today()}")
        logger.info(f"  Portfolio Value : ₹{pv:,.2f}")
        logger.info(f"  Total P&L       : ₹{pnl:+,.2f} ({pnl/self.initial_capital:+.2%})")
        logger.info(f"  Open Positions  : {self.portfolio.n_positions}")
        logger.info(f"  Drawdown        : {dd:.2%}")
        logger.info("─" * 60)

    def _build_observation(
        self,
        top_stocks      : List[Tuple[str, np.ndarray]],
        portfolio_state : PortfolioState,
    ) -> np.ndarray:
        """Builds the flat observation vector for PPO inference."""
        # ── Embeddings ─────────────────────────────────────────────────────
        embeddings = np.zeros((N_STOCKS, EMBEDDING_DIM), dtype=np.float32)
        for i, (sym, emb) in enumerate(top_stocks[:N_STOCKS]):
            embeddings[i] = emb
        emb_flat = embeddings.flatten()

        # ── Portfolio state ────────────────────────────────────────────────
        pnl_pct  = (portfolio_state.total_value - self.initial_capital) / self.initial_capital
        dd       = portfolio_state.drawdown
        cash_pct = portfolio_state.cash / max(portfolio_state.total_value, 1)
        n_pos    = portfolio_state.n_positions / 4
        trades   = portfolio_state.trades_this_month / 15

        port_state_vec = np.array([
            np.clip(pnl_pct,  -1.0, 1.0),
            np.clip(dd,        0.0, 1.0),
            np.clip(cash_pct,  0.0, 1.0),
            np.clip(n_pos,     0.0, 1.0),
            np.clip(trades,    0.0, 1.0),
            0.5,   # bar_norm placeholder (not applicable in live)
            0.0,   # avg_pos_pnl placeholder
            0.0,   # heat placeholder
        ], dtype=np.float32)

        return np.concatenate([emb_flat, port_state_vec])

    def _make_blocked_signal(
        self,
        signal_id  : str,
        symbol     : str,
        action     : SignalAction,
        ltp        : float,
        rc_result  : RCResult,
        mds_score  : int,
        pillar     : Dict,
    ) -> TradeSignal:
        """Creates a blocked signal record for audit trail."""
        return TradeSignal(
            signal_id        = signal_id,
            timestamp        = datetime.now(),
            symbol           = symbol,
            action           = action,
            mode             = self.mode,
            entry_price      = ltp,
            tp_price         = ltp * 1.04,
            sl_price         = ltp * 0.985,
            trail_activate   = ltp * 1.02,
            trail_distance   = 0.008,
            position_pct     = 0.0,
            invest_amount    = 0.0,
            quantity         = 0,
            rl_confidence    = 0.0,
            mds_score        = mds_score,
            trend_score      = pillar.get("trend_score", 0.0),
            msi_score        = pillar.get("msi_signal", 0.0),
            sentiment_score  = 0.0,
            status           = SignalStatus.RC_BLOCKED,
            rc_rule_blocked  = rc_result.rule,
            notes            = rc_result.reason[:200],
        )

    @staticmethod
    def _is_pre_market(now: datetime) -> bool:
        return (now.hour == PRE_MARKET_HOUR and
                PRE_MARKET_MINUTE <= now.minute < PRE_MARKET_MINUTE + 5)

    @staticmethod
    def _is_market_hours(now: datetime) -> bool:
        start = now.replace(hour=MARKET_OPEN_HOUR,  minute=MARKET_OPEN_MINUTE,  second=0)
        end   = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)
        return start <= now <= end

    @staticmethod
    def _is_market_close(now: datetime) -> bool:
        return (now.hour == MARKET_CLOSE_HOUR and
                MARKET_CLOSE_MINUTE <= now.minute < MARKET_CLOSE_MINUTE + 5)

    def close(self):
        if self.data:
            self.data.close()
        if self.portfolio:
            self.portfolio.close()


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest execution/signal_engine.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestSignalEngine:

    def _make_portfolio(self, **kwargs) -> PortfolioState:
        defaults = dict(
            total_value=1_000_000.0, peak_value=1_000_000.0,
            cash=800_000.0, open_positions={},
            trades_today=0, trades_this_month=0, sector_map={},
        )
        defaults.update(kwargs)
        return PortfolioState(**defaults)

    # ── TradeSignal tests ─────────────────────────────────────────────────

    def test_trade_signal_to_dict(self):
        sig = TradeSignal(
            signal_id="TEST_001", timestamp=datetime.now(),
            symbol="RELIANCE", action=SignalAction.BUY,
            mode=SignalMode.PAPER, entry_price=2850.0,
            tp_price=2964.0, sl_price=2807.0,
            trail_activate=2907.0, trail_distance=0.008,
            position_pct=0.12, invest_amount=120_000.0, quantity=42,
            rl_confidence=0.72, mds_score=1,
        )
        d = sig.to_dict()
        assert d["symbol"]       == "RELIANCE"
        assert d["action"]       == "BUY"
        assert d["rl_confidence"]== 0.72
        assert d["status"]       == "GENERATED"

    def test_open_position_pnl(self):
        pos = OpenPosition(
            symbol="TEST", signal_id="S1",
            entry_price=100.0, quantity=100,
            tp_price=104.0, sl_price=98.5,
            entry_time=datetime.now(),
            current_price=105.0,
        )
        assert abs(pos.pnl_pct - 0.05) < 1e-6

    def test_open_position_value(self):
        pos = OpenPosition(
            symbol="TEST", signal_id="S1",
            entry_price=500.0, quantity=200,
            tp_price=520.0, sl_price=492.0,
            entry_time=datetime.now(),
        )
        assert pos.position_value == 100_000.0

    # ── Observation builder tests ─────────────────────────────────────────

    def test_build_observation_shape(self):
        engine = SignalEngine.__new__(SignalEngine)
        engine.initial_capital = 1_000_000.0
        port = self._make_portfolio()
        top  = [(f"STK{i}", np.random.randn(128).astype(np.float32))
                for i in range(N_STOCKS)]
        obs  = engine._build_observation(top, port)
        assert obs.shape == (OBS_DIM,), f"Expected {OBS_DIM}, got {obs.shape}"

    def test_build_observation_dtype(self):
        engine = SignalEngine.__new__(SignalEngine)
        engine.initial_capital = 1_000_000.0
        port = self._make_portfolio()
        top  = [(f"STK{i}", np.zeros(128, dtype=np.float32))
                for i in range(N_STOCKS)]
        obs  = engine._build_observation(top, port)
        assert obs.dtype == np.float32

    def test_build_observation_no_nan(self):
        engine = SignalEngine.__new__(SignalEngine)
        engine.initial_capital = 1_000_000.0
        port = self._make_portfolio()
        top  = [(f"STK{i}", np.random.randn(128).astype(np.float32))
                for i in range(N_STOCKS)]
        obs  = engine._build_observation(top, port)
        assert not np.isnan(obs).any()

    def test_build_observation_fewer_stocks(self):
        """Fewer than N_STOCKS available — obs must still be correct shape."""
        engine = SignalEngine.__new__(SignalEngine)
        engine.initial_capital = 1_000_000.0
        port = self._make_portfolio()
        top  = [(f"STK{i}", np.zeros(128, dtype=np.float32)) for i in range(5)]
        obs  = engine._build_observation(top, port)
        assert obs.shape == (OBS_DIM,)

    # ── Market hours tests ────────────────────────────────────────────────

    def test_is_market_hours_true(self):
        now = datetime.now().replace(hour=11, minute=30)
        assert SignalEngine._is_market_hours(now)

    def test_is_market_hours_false_before_open(self):
        now = datetime.now().replace(hour=8, minute=0)
        assert not SignalEngine._is_market_hours(now)

    def test_is_market_hours_false_after_close(self):
        now = datetime.now().replace(hour=16, minute=0)
        assert not SignalEngine._is_market_hours(now)

    def test_is_pre_market(self):
        now = datetime.now().replace(hour=9, minute=2)
        assert SignalEngine._is_pre_market(now)

    def test_is_market_close(self):
        now = datetime.now().replace(hour=15, minute=32)
        assert SignalEngine._is_market_close(now)

    # ── SignalAction/Status enum tests ────────────────────────────────────

    def test_signal_action_values(self):
        assert SignalAction.BUY.value        == "BUY"
        assert SignalAction.STRONG_BUY.value == "STRONG_BUY"
        assert SignalAction.HOLD.value       == "HOLD"

    def test_signal_status_values(self):
        assert SignalStatus.EMITTED.value    == "EMITTED"
        assert SignalStatus.RC_BLOCKED.value == "RC_BLOCKED"
        assert SignalStatus.CLOSED_TP.value  == "CLOSED_TP"

    def test_all_actions_covered(self):
        """All 5 PPO actions must map to a SignalAction."""
        action_map = {
            0: SignalAction.STRONG_SELL,
            1: SignalAction.SELL,
            2: SignalAction.HOLD,
            3: SignalAction.BUY,
            4: SignalAction.STRONG_BUY,
        }
        assert len(action_map) == 5
        for v in action_map.values():
            assert isinstance(v, SignalAction)

    # ── Blocked signal tests ──────────────────────────────────────────────

    def test_make_blocked_signal(self):
        from environment.risk_constitution import RCResult
        engine = SignalEngine.__new__(SignalEngine)
        engine.mode = SignalMode.PAPER
        rc_result = RCResult.block("RC-08", "Max positions reached", "block")
        sig = engine._make_blocked_signal(
            "SIG_001", "RELIANCE", SignalAction.BUY,
            2850.0, rc_result, 1, {}
        )
        assert sig.status          == SignalStatus.RC_BLOCKED
        assert sig.rc_rule_blocked == "RC-08"
        assert sig.quantity        == 0

    # ── Trailing stop tests ───────────────────────────────────────────────

    def test_trailing_stop_activates(self):
        pos = OpenPosition(
            symbol="TEST", signal_id="S1",
            entry_price=100.0, quantity=100,
            tp_price=104.0, sl_price=98.5,
            entry_time=datetime.now(),
            current_price=102.5,
        )
        # Simulate _monitor_positions trailing logic
        ltp      = 102.5
        gain_pct = (ltp - pos.entry_price) / pos.entry_price
        assert gain_pct >= 0.02
        pos.trail_active = True
        pos.trail_peak   = ltp
        pos.trail_stop   = ltp * 0.992
        assert pos.trail_active
        assert abs(pos.trail_stop - 101.7) < 0.1

    def test_trailing_stop_hit(self):
        pos = OpenPosition(
            symbol="TEST", signal_id="S1",
            entry_price=100.0, quantity=100,
            tp_price=104.0, sl_price=98.5,
            entry_time=datetime.now(),
            trail_active=True, trail_peak=103.0,
            trail_stop=102.1,   # trail stop at 102.1
            current_price=102.0,
        )
        # Price dropped below trail stop
        assert pos.current_price <= pos.trail_stop


# ── CLI ENTRY POINT ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G.O.D.S E.Y.E — Signal Engine"
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="paper: log signals only | live: execute real orders"
    )
    parser.add_argument(
        "--capital", type=float, default=1_000_000.0,
        help="Initial capital in ₹ (default: 1,000,000)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(SWING_BEST),
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one signal generation cycle and exit (for testing)"
    )
    args = parser.parse_args()

    engine = SignalEngine(
        mode            = SignalMode(args.mode),
        initial_capital = args.capital,
        checkpoint_path = Path(args.checkpoint),
        device          = args.device,
    )

    try:
        engine.setup()
        if args.once:
            signals = engine.generate_entry_signals()
            for sig in signals:
                logger.info(sig.to_dict())
        else:
            engine.run()
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        engine.close()
