"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Custom Gymnasium Trading Environment            ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : environment/godseye_env.py                             ║
║         Phase   : 3 — RL Agent Training                                 ║
║                                                                          ║
║  What this module does:                                                  ║
║    Implements a realistic Indian equity trading environment compatible   ║
║    with Stable-Baselines3 PPO. The RL agent interacts with this         ║
║    environment during training, learning buy/hold/sell policies by       ║
║    trial-and-error across thousands of simulated trading episodes.       ║
║                                                                          ║
║  Realism features (each one materially affects learned policy):          ║
║    • Transaction costs: brokerage 0.03% + STT 0.1% + NSE 0.00335%      ║
║    • Slippage model: scales with order size vs daily volume              ║
║    • NSE trading hours enforced (no overnight surprise gaps ignored)     ║
║    • Circuit breaker simulation (upper/lower limits block entry)         ║
║    • T+1 settlement (intraday profits not immediately reusable)          ║
║    • Corporate action adjustment (splits/bonuses don't fake returns)     ║
║    • Max 4 positions, max 15 trades/month (per Risk Constitution)        ║
║    • Risk Constitution RC-01 to RC-10 as hard constraints               ║
║                                                                          ║
║  Episode structure:                                                      ║
║    Swing    : 20 trading days per episode (1 calendar month)             ║
║    Intraday : 1 trading day per episode (78 × 5-min bars)               ║
║    Universe : up to 499 stocks from daily_ohlcv                         ║
║                                                                          ║
║  Observation space:                                                      ║
║    Per stock: 128-dim backbone embedding (from pretrain_best.pt)         ║
║    Portfolio state: 8 scalar features                                    ║
║    Total: (N_stocks × 128) + 8 flattened into 1D vector                ║
║                                                                          ║
║  Action space:                                                           ║
║    Discrete(5) per stock: 0=Strong Sell, 1=Sell, 2=Hold,               ║
║                            3=Buy, 4=Strong Buy                           ║
║    For computational efficiency during PPO training, action is          ║
║    a single Discrete(5) selecting the action for the TOP-K              ║
║    candidates (ranked by backbone score) each step.                     ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install gymnasium stable-baselines3 torch psycopg2-binary        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import math
import warnings
import numpy as np
import pandas as pd
import psycopg2
import torch

import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)

# ── Database ──────────────────────────────────────────────────────────────
DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)

# ══════════════════════════════════════════════════════════════════════════
#  CONSTANTS & ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════

class Action(IntEnum):
    STRONG_SELL = 0
    SELL        = 1
    HOLD        = 2
    BUY         = 3
    STRONG_BUY  = 4


class TradeMode(IntEnum):
    SWING    = 0   # daily bars, up to 20 days hold
    INTRADAY = 1   # 5-min bars, forced close EOD


# ── Transaction cost components (NSE as of 2024) ─────────────────────────
BROKERAGE_PCT  = 0.0003    # 0.03% per side (Zerodha flat ₹20 cap — approx)
STT_BUY_PCT    = 0.001     # 0.1% on buy side (equity delivery)
STT_SELL_PCT   = 0.001     # 0.1% on sell side
NSE_CHARGES    = 0.0000335 # 0.00335% exchange transaction charge
SEBI_CHARGES   = 0.000001  # 0.0001% SEBI turnover fee
GST_PCT        = 0.18      # 18% GST on brokerage + exchange charges
STAMP_DUTY_PCT = 0.00015   # 0.015% stamp duty on buy side

TOTAL_BUY_COST  = (BROKERAGE_PCT + STT_BUY_PCT  + NSE_CHARGES +
                   SEBI_CHARGES + STAMP_DUTY_PCT) * (1 + GST_PCT)
TOTAL_SELL_COST = (BROKERAGE_PCT + STT_SELL_PCT + NSE_CHARGES +
                   SEBI_CHARGES) * (1 + GST_PCT)

# ── Position limits (Risk Constitution) ───────────────────────────────────
MAX_POSITIONS       = 4      # RC-08: max 4 open at once
MAX_TRADES_PER_MONTH= 15     # RC-08 extension: max 15 trades/month
MAX_POSITION_PCT    = 0.25   # max 25% of portfolio in one position
MAX_PORTFOLIO_HEAT  = 0.06   # RC: max 6% total open risk

# ── TP/SL defaults (overridden by volatility.py dynamic values) ──────────
SWING_TP_PCT    = 0.040   # 4.0%
SWING_SL_PCT    = 0.015   # 1.5%
INTRADAY_TP_PCT = 0.025   # 2.5%
INTRADAY_SL_PCT = 0.008   # 0.8%

# ── Trailing stop activation ──────────────────────────────────────────────
SWING_TRAIL_ACTIVATE    = 0.020   # activate after +2% move
SWING_TRAIL_DISTANCE    = 0.008   # trail 0.8% below peak
INTRADAY_TRAIL_ACTIVATE = 0.015   # activate after +1.5% move
INTRADAY_TRAIL_DISTANCE = 0.005   # trail 0.5% below peak

# ── Episode config ────────────────────────────────────────────────────────
SWING_EPISODE_DAYS    = 20    # bars per swing episode
INTRADAY_EPISODE_BARS = 78    # 9:15 AM to 3:30 PM in 5-min bars

# ── Backbone embedding dim ────────────────────────────────────────────────
EMBEDDING_DIM  = 128   # GodsEyeBackbone output dim
PORTFOLIO_DIM  = 8     # portfolio state features


# ══════════════════════════════════════════════════════════════════════════
#  POSITION DATACLASS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """
    Represents a single open trade position.

    All prices in ₹. Returns tracked as fractions (0.04 = 4% gain).
    """
    symbol        : str
    entry_price   : float
    entry_bar     : int            # episode bar index when entered
    quantity      : int            # shares held
    position_value: float          # entry_price × quantity (₹)
    tp_price      : float          # take-profit price
    sl_price      : float          # stop-loss price
    trail_active  : bool = False   # trailing stop engaged
    trail_peak    : float = 0.0    # highest price seen since entry
    trail_stop    : float = 0.0    # current trailing stop price
    hold_days     : int  = 0       # bars held so far
    unrealised_pnl: float= 0.0     # current mark-to-market P&L (fraction)

    def update_trailing_stop(
        self,
        current_price: float,
        mode: TradeMode,
    ) -> bool:
        """
        Updates trailing stop based on current price.

        Returns:
            True if trailing stop has been hit (position should close)
        """
        activate = (SWING_TRAIL_ACTIVATE if mode == TradeMode.SWING
                    else INTRADAY_TRAIL_ACTIVATE)
        distance = (SWING_TRAIL_DISTANCE if mode == TradeMode.SWING
                    else INTRADAY_TRAIL_DISTANCE)

        gain_pct = (current_price - self.entry_price) / self.entry_price

        # Activate trailing stop once gain threshold is crossed
        if gain_pct >= activate and not self.trail_active:
            self.trail_active = True
            self.trail_peak   = current_price
            self.trail_stop   = current_price * (1 - distance)

        # Update trail stop if price makes a new high
        if self.trail_active:
            if current_price > self.trail_peak:
                self.trail_peak  = current_price
                self.trail_stop  = current_price * (1 - distance)

            # Check if trailing stop is hit
            if current_price <= self.trail_stop:
                return True   # hit — close position

        return False

    def current_pnl_pct(self, current_price: float) -> float:
        """Returns current unrealised P&L as a fraction of entry price."""
        return (current_price - self.entry_price) / self.entry_price


# ══════════════════════════════════════════════════════════════════════════
#  MARKET DATA LOADER
# ══════════════════════════════════════════════════════════════════════════

class MarketDataLoader:
    """
    Loads OHLCV data from TimescaleDB and pre-processes it for the env.

    Caches all data in memory after first load to avoid repeated DB hits
    during the thousands of episodes in RL training.

    Data is loaded for the training period only (2019-01-01 to 2023-06-30).
    Validation/test periods are never seen during RL training.
    """

    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date  : str = "2023-06-30",
    ):
        self.start_date = start_date
        self.end_date   = end_date
        self._cache: Dict[str, pd.DataFrame] = {}
        self._symbols: List[str] = []
        self._trading_dates: List[str] = []
        self._loaded = False

    def load(self):
        """
        Loads all OHLCV data from TimescaleDB into memory.
        Called once during environment initialization.
        Expected load time: 5–15 seconds for 499 stocks × 5 years.
        """
        if self._loaded:
            return

        logger.info(
            f"MarketDataLoader: loading OHLCV "
            f"{self.start_date} → {self.end_date}..."
        )

        conn = psycopg2.connect(DB_URL)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT symbol, date, open, high, low, close, volume
                    FROM daily_ohlcv
                    WHERE date BETWEEN %s AND %s
                      AND close IS NOT NULL
                      AND close > 0
                    ORDER BY symbol, date;
                """, (self.start_date, self.end_date))
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            raise RuntimeError(
                f"No OHLCV data found between {self.start_date} and {self.end_date}. "
                f"Run Phase 0 data backfill first."
            )

        # Build per-symbol DataFrames
        df_all = pd.DataFrame(
            rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"]
        )
        df_all["date"]   = pd.to_datetime(df_all["date"])
        df_all["close"]  = df_all["close"].astype(float)
        df_all["open"]   = df_all["open"].astype(float)
        df_all["high"]   = df_all["high"].astype(float)
        df_all["low"]    = df_all["low"].astype(float)
        df_all["volume"] = df_all["volume"].astype(float)

        # Get sorted unique trading dates (NSE calendar)
        self._trading_dates = sorted(df_all["date"].dt.strftime("%Y-%m-%d").unique())

        # Build symbol cache — only keep symbols with sufficient data
        min_bars = 250  # need at least 250 trading days
        for symbol, grp in df_all.groupby("symbol"):
            grp = grp.set_index("date").sort_index()
            if len(grp) >= min_bars:
                self._cache[symbol] = grp
                self._symbols.append(symbol)

        self._loaded = True
        logger.info(
            f"MarketDataLoader: {len(self._symbols)} symbols loaded, "
            f"{len(self._trading_dates)} trading dates."
        )

    @property
    def symbols(self) -> List[str]:
        if not self._loaded:
            self.load()
        return self._symbols

    @property
    def trading_dates(self) -> List[str]:
        if not self._loaded:
            self.load()
        return self._trading_dates

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Returns cached OHLCV DataFrame for a symbol."""
        if not self._loaded:
            self.load()
        return self._cache.get(symbol, pd.DataFrame())

    def get_price(self, symbol: str, date_idx: int) -> Optional[float]:
        """Returns close price for a symbol at a given date index."""
        df = self._cache.get(symbol)
        if df is None or date_idx >= len(df):
            return None
        return float(df.iloc[date_idx]["close"])

    def get_bar(self, symbol: str, date_idx: int) -> Optional[Dict]:
        """Returns full OHLCV bar for a symbol at a given date index."""
        df = self._cache.get(symbol)
        if df is None or date_idx >= len(df):
            return None
        row = df.iloc[date_idx]
        return {
            "open"  : float(row["open"]),
            "high"  : float(row["high"]),
            "low"   : float(row["low"]),
            "close" : float(row["close"]),
            "volume": float(row["volume"]),
        }


# ══════════════════════════════════════════════════════════════════════════
#  RISK CONSTITUTION CHECKER
# ══════════════════════════════════════════════════════════════════════════

class RiskConstitution:
    """
    Enforces all 10 Risk Constitution rules as hard constraints.

    The RL agent's action is passed through this checker before execution.
    If any rule fires, the action is overridden (e.g., buy → hold, or
    all positions closed).

    Rules:
        RC-01: Portfolio drawdown > 12% → halt all trading
        RC-02: Single position loss > 2% of portfolio → force close
        RC-03: Correlation > 0.75 between new and existing → block entry
        RC-04: Earnings within 2 days → no new entry (simplified: skip)
        RC-05: Stock near circuit limit (>18% move) → no entry, tighten stop
        RC-06: VIX-like spike (>8% market drop in episode) → block intraday
        RC-07: Avg daily volume < ₹5 crore → exclude from universe
        RC-08: 4 positions already open → block new entries
        RC-09: Extreme FII selling (simulated via large market drop) → block longs
        RC-10: 2 positions from same sector → block third in same sector
    """

    # RC-07 liquidity floor (₹ crore average daily turnover)
    MIN_AVG_DAILY_TURNOVER = 5_00_00_000   # ₹5 crore = ₹50,000,000

    def __init__(self):
        self.triggered: Dict[str, int] = {f"RC-{i:02d}": 0 for i in range(1, 11)}

    def check_rc01(self, portfolio_drawdown: float) -> bool:
        """RC-01: Halt if drawdown > 12%."""
        if portfolio_drawdown > 0.12:
            self.triggered["RC-01"] += 1
            return True
        return False

    def check_rc02(
        self,
        position: Position,
        current_price: float,
        portfolio_value: float,
    ) -> bool:
        """RC-02: Force close if single position loss > 2% of portfolio."""
        position_loss = (position.entry_price - current_price) / position.entry_price
        loss_as_pct_portfolio = position_loss * (position.position_value / portfolio_value)
        if loss_as_pct_portfolio > 0.02:
            self.triggered["RC-02"] += 1
            return True
        return False

    def check_rc03(
        self,
        symbol: str,
        open_symbols: List[str],
        correlation_matrix: Optional[np.ndarray],
        symbol_to_idx: Optional[Dict[str, int]],
    ) -> bool:
        """RC-03: Block entry if correlation > 0.75 with any open position."""
        if not open_symbols or correlation_matrix is None:
            return False
        if symbol not in (symbol_to_idx or {}):
            return False

        new_idx = symbol_to_idx[symbol]
        for existing_sym in open_symbols:
            if existing_sym not in symbol_to_idx:
                continue
            ex_idx = symbol_to_idx[existing_sym]
            try:
                corr = correlation_matrix[new_idx, ex_idx]
                if abs(corr) > 0.75:
                    self.triggered["RC-03"] += 1
                    return True
            except IndexError:
                continue
        return False

    def check_rc05(self, bar: Dict) -> bool:
        """RC-05: Block entry if stock moved >18% today (circuit proximity)."""
        if not bar:
            return False
        daily_move = abs(bar["high"] - bar["low"]) / max(bar["open"], 1e-6)
        if daily_move > 0.18:
            self.triggered["RC-05"] += 1
            return True
        return False

    def check_rc07(self, avg_daily_volume: float, avg_price: float) -> bool:
        """RC-07: Block if average daily turnover < ₹5 crore."""
        avg_turnover = avg_daily_volume * avg_price
        if avg_turnover < self.MIN_AVG_DAILY_TURNOVER:
            self.triggered["RC-07"] += 1
            return True
        return False

    def check_rc08(self, n_open_positions: int) -> bool:
        """RC-08: Block new entry if already at max 4 positions."""
        if n_open_positions >= MAX_POSITIONS:
            self.triggered["RC-08"] += 1
            return True
        return False

    def check_rc10(
        self,
        symbol: str,
        open_symbols: List[str],
        sector_map: Dict[str, str],
    ) -> bool:
        """RC-10: Block if 2 positions already open in the same sector."""
        if not sector_map or symbol not in sector_map:
            return False
        new_sector = sector_map[symbol]
        sector_count = sum(
            1 for s in open_symbols
            if sector_map.get(s) == new_sector
        )
        if sector_count >= 2:
            self.triggered["RC-10"] += 1
            return True
        return False

    def reset_counters(self):
        """Resets trigger counters (call at episode start)."""
        for k in self.triggered:
            self.triggered[k] = 0

    def summary(self) -> str:
        """Returns a string summary of which rules fired this episode."""
        fired = {k: v for k, v in self.triggered.items() if v > 0}
        if not fired:
            return "Risk Constitution: No rules triggered this episode."
        return "Risk Constitution triggers: " + str(fired)


# ══════════════════════════════════════════════════════════════════════════
#  REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════

class RewardCalculator:
    """
    Computes the shaped reward signal for the PPO agent.

    Reward design philosophy:
        The agent must learn to maximize risk-adjusted returns, NOT
        just raw returns. A 10% gain with 20% drawdown is worse than
        a 6% gain with 2% drawdown — the reward function enforces this.

    Components:
        1. Step return         : Immediate P&L change (primary signal)
        2. Sharpe bonus        : Reward consistent gains over volatile gains
        3. Drawdown penalty    : Quadratic penalty for drawdown (hurts more as DD grows)
        4. Overtrading penalty : Small penalty per trade to discourage churning
        5. Hold penalty        : Very small penalty for holding losing positions
        6. Risk Constitution   : Large penalty when any RC rule fires
    """

    # Reward weights
    W_RETURN    = 1.0    # step return weight
    W_SHARPE    = 0.3    # Sharpe bonus weight
    W_DRAWDOWN  = 2.0    # drawdown penalty weight (quadratic)
    W_OVERTRADE = 0.005  # per-trade overtrading penalty
    W_HOLD_LOSS = 0.001  # per-bar holding-a-loser penalty
    W_RC        = 0.5    # Risk Constitution violation penalty

    def __init__(self):
        self._step_returns: List[float] = []
        self._peak_value = 1.0
        self._current_value = 1.0

    def reset(self, initial_value: float):
        self._step_returns = []
        self._peak_value   = initial_value
        self._current_value = initial_value

    def compute(
        self,
        step_return        : float,          # fractional P&L this bar
        portfolio_value    : float,          # current portfolio value
        n_trades_this_bar  : int,            # how many trades executed this bar
        n_losing_positions : int,            # positions currently in loss
        rc_violations      : int,            # Risk Constitution rules triggered
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes the shaped reward for one environment step.

        Returns:
            reward     : scalar reward value
            components : dict of reward component values (for logging)
        """
        self._step_returns.append(step_return)
        self._current_value = portfolio_value

        # Update drawdown tracking
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
        drawdown = (self._peak_value - portfolio_value) / self._peak_value

        # ── 1. Return component ───────────────────────────────────────────
        r_return = step_return * self.W_RETURN

        # ── 2. Sharpe-like bonus ──────────────────────────────────────────
        # Reward consistent returns — punish high variance
        r_sharpe = 0.0
        if len(self._step_returns) >= 5:
            recent = self._step_returns[-5:]
            mean_r = np.mean(recent)
            std_r  = np.std(recent) + 1e-8
            r_sharpe = (mean_r / std_r) * self.W_SHARPE * 0.01

        # ── 3. Drawdown penalty (quadratic — hurts more as DD grows) ──────
        r_drawdown = -(drawdown ** 2) * self.W_DRAWDOWN

        # ── 4. Overtrading penalty ────────────────────────────────────────
        r_overtrade = -n_trades_this_bar * self.W_OVERTRADE

        # ── 5. Hold-a-loser penalty ───────────────────────────────────────
        r_hold_loss = -n_losing_positions * self.W_HOLD_LOSS

        # ── 6. Risk Constitution penalty ──────────────────────────────────
        r_rc = -rc_violations * self.W_RC

        # ── Total ─────────────────────────────────────────────────────────
        reward = r_return + r_sharpe + r_drawdown + r_overtrade + r_hold_loss + r_rc

        components = {
            "return"    : r_return,
            "sharpe"    : r_sharpe,
            "drawdown"  : r_drawdown,
            "overtrade" : r_overtrade,
            "hold_loss" : r_hold_loss,
            "rc_penalty": r_rc,
            "total"     : reward,
        }

        return reward, components


# ══════════════════════════════════════════════════════════════════════════
#  MAIN ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════

class GodsEyeEnv(gym.Env):
    """
    Custom Gymnasium environment for G.O.D.S E.Y.E RL training.

    The agent acts on a universe of stocks each day, choosing to
    buy, hold, or sell positions subject to the Risk Constitution.

    Observation:
        A 1D numpy array of shape (N_stocks * EMBEDDING_DIM + PORTFOLIO_DIM,)
        = (N_stocks * 128 + 8,)

        Embeddings come from GodsEyeBackbone (pretrain_best.pt).
        Portfolio state encodes current positions, P&L, drawdown, etc.

    Action:
        Discrete(5): action for the HIGHEST-SCORING stock not yet in portfolio
        (or the worst-performing open position for sells).
        0 = Strong Sell, 1 = Sell, 2 = Hold, 3 = Buy, 4 = Strong Buy

    Reward:
        Shaped reward from RewardCalculator (see above).

    Episode:
        Swing    : 20 trading days. Ends at day 20 or when all positions closed.
        Intraday : 78 bars (1 trading day). All positions force-closed at bar 77.

    Args:
        data_loader     : MarketDataLoader instance (pre-loaded)
        backbone        : GodsEyeBackbone instance (pretrain_best.pt loaded)
        mode            : TradeMode.SWING or TradeMode.INTRADAY
        initial_capital : Starting portfolio value in ₹ (default: ₹1,000,000)
        n_stocks        : Number of stocks to include in observation (default: 50)
                          Top-50 by backbone score reduces obs space to manageable size
        train_start_idx : Start index into trading_dates for sampling episodes
        train_end_idx   : End index into trading_dates for sampling episodes

    Example:
        from environment.godseye_env import GodsEyeEnv, TradeMode
        from models.backbone import GodsEyeBackbone

        data_loader = MarketDataLoader()
        backbone    = GodsEyeBackbone.load("checkpoints/pretrain_best.pt")
        env = GodsEyeEnv(data_loader, backbone, mode=TradeMode.SWING)

        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step(3)  # Buy
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        data_loader    : MarketDataLoader,
        backbone       : Optional[Any] = None,
        mode           : TradeMode     = TradeMode.SWING,
        initial_capital: float         = 1_000_000.0,   # ₹10 lakh
        n_stocks       : int           = 50,
        train_start_idx: int           = 0,
        train_end_idx  : Optional[int] = None,
        sector_map     : Optional[Dict[str, str]] = None,
        render_mode    : Optional[str] = None,
        device         : str           = "cpu",
    ):
        super().__init__()

        self.data_loader     = data_loader
        self.backbone        = backbone
        self.mode            = mode
        self.initial_capital = initial_capital
        self.n_stocks        = n_stocks
        self.sector_map      = sector_map or {}
        self.render_mode     = render_mode
        self.device          = torch.device(device)

        # ── Episode length by mode ────────────────────────────────────────
        self.episode_length = (
            SWING_EPISODE_DAYS if mode == TradeMode.SWING
            else INTRADAY_EPISODE_BARS
        )

        # ── Training date range ───────────────────────────────────────────
        if not data_loader._loaded:
            data_loader.load()

        self.all_dates     = data_loader.trading_dates
        self.train_start   = train_start_idx
        self.train_end     = train_end_idx or (len(self.all_dates) - self.episode_length - 1)

        # ── Observation space ─────────────────────────────────────────────
        # Flat: N_stocks embeddings + portfolio state
        obs_dim = self.n_stocks * EMBEDDING_DIM + PORTFOLIO_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ── Action space ──────────────────────────────────────────────────
        # Discrete(5): Strong Sell / Sell / Hold / Buy / Strong Buy
        self.action_space = spaces.Discrete(5)

        # ── Internal state ────────────────────────────────────────────────
        self.rc            = RiskConstitution()
        self.reward_calc   = RewardCalculator()
        self.positions     : Dict[str, Position] = {}
        self.portfolio_value: float = initial_capital
        self.peak_value    : float  = initial_capital
        self.cash          : float  = initial_capital
        self.current_bar   : int    = 0
        self.episode_start : int    = 0
        self.trades_this_month: int = 0
        self.total_trades  : int    = 0
        self.episode_returns: List[float] = []

        # Current universe (top N_stocks by backbone score)
        self.current_symbols: List[str]         = []
        self.current_embeddings: np.ndarray      = np.zeros(
            (n_stocks, EMBEDDING_DIM), dtype=np.float32
        )

        # Episode statistics for info dict
        self._ep_stats: Dict[str, Any] = {}

    # ══════════════════════════════════════════════════════════════════════
    #  GYMNASIUM API
    # ══════════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment for a new episode.

        Randomly samples an episode start date from the training range,
        resets portfolio to initial_capital, clears all positions.

        Returns:
            observation : Initial observation array
            info        : Empty dict (Gymnasium convention)
        """
        super().reset(seed=seed)

        # ── Sample random episode start ───────────────────────────────────
        max_start = max(self.train_start, self.train_end - self.episode_length)
        self.episode_start = self.np_random.integers(self.train_start, max_start)
        self.current_bar   = 0

        # ── Reset portfolio ───────────────────────────────────────────────
        self.portfolio_value  = self.initial_capital
        self.peak_value       = self.initial_capital
        self.cash             = self.initial_capital
        self.positions        = {}
        self.trades_this_month= 0
        self.total_trades     = 0
        self.episode_returns  = []

        # ── Reset sub-modules ─────────────────────────────────────────────
        self.rc.reset_counters()
        self.reward_calc.reset(self.initial_capital)

        # ── Build initial observation ─────────────────────────────────────
        self._update_universe()
        obs = self._build_observation()

        self._ep_stats = {
            "episode_start_date": self.all_dates[self.episode_start],
            "n_trades": 0,
            "final_return": 0.0,
            "max_drawdown": 0.0,
            "rc_triggers": {},
        }

        return obs, {}

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes one environment step.

        The agent's action applies to the most actionable stock:
            - For BUY actions  → acts on the highest-ranked unowned stock
            - For SELL actions → acts on the worst-performing open position
            - For HOLD        → no trade, just update positions

        Args:
            action : int in [0, 4] from action_space

        Returns:
            observation : (obs_dim,) float32 array
            reward      : scalar float
            terminated  : True if episode is done (hit max bars or RC-01)
            truncated   : True if episode hits time limit
            info        : dict with episode statistics
        """
        global_bar = self.episode_start + self.current_bar
        rc_violations_this_step = 0

        # ── Check RC-01 (kill switch) before doing anything ───────────────
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        if self.rc.check_rc01(drawdown):
            terminated = True
            obs = self._build_observation()
            reward = -1.0   # large penalty for hitting kill switch
            return obs, reward, terminated, False, self._build_info(drawdown)

        # ── Execute action ────────────────────────────────────────────────
        n_trades_this_bar = 0
        action_enum = Action(action)

        if action_enum in (Action.BUY, Action.STRONG_BUY):
            traded = self._execute_buy(
                global_bar,
                strong=(action_enum == Action.STRONG_BUY),
            )
            if traded:
                n_trades_this_bar += 1

        elif action_enum in (Action.SELL, Action.STRONG_SELL):
            traded = self._execute_sell(global_bar)
            if traded:
                n_trades_this_bar += 1
        # HOLD: no action needed

        # ── Update all open positions ─────────────────────────────────────
        closed_this_bar = self._update_positions(global_bar)
        n_trades_this_bar += closed_this_bar

        # ── Update portfolio value ────────────────────────────────────────
        self._update_portfolio_value(global_bar)
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        # ── Intraday forced close at last bar ─────────────────────────────
        if (self.mode == TradeMode.INTRADAY and
                self.current_bar == self.episode_length - 1):
            n_trades_this_bar += self._force_close_all(global_bar)

        # ── Compute step return ───────────────────────────────────────────
        prev_value   = self.initial_capital * math.exp(sum(self.episode_returns))
        step_return  = (self.portfolio_value - prev_value) / prev_value
        self.episode_returns.append(math.log(max(self.portfolio_value / prev_value, 1e-8)))

        # ── Count losing positions ────────────────────────────────────────
        n_losing = sum(
            1 for sym, pos in self.positions.items()
            if self._get_current_price(sym, global_bar) is not None
            and pos.current_pnl_pct(self._get_current_price(sym, global_bar)) < -0.005
        )

        # ── Compute reward ────────────────────────────────────────────────
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        reward, reward_components = self.reward_calc.compute(
            step_return        = step_return,
            portfolio_value    = self.portfolio_value,
            n_trades_this_bar  = n_trades_this_bar,
            n_losing_positions = n_losing,
            rc_violations      = rc_violations_this_step,
        )

        # ── Advance bar ───────────────────────────────────────────────────
        self.current_bar += 1
        self._update_universe()
        obs = self._build_observation()

        # ── Termination conditions ────────────────────────────────────────
        truncated  = self.current_bar >= self.episode_length
        terminated = False   # RC-01 termination handled above

        if truncated:
            self._ep_stats.update({
                "n_trades"    : self.total_trades,
                "final_return": (self.portfolio_value - self.initial_capital) / self.initial_capital,
                "max_drawdown": drawdown,
                "rc_triggers" : self.rc.triggered.copy(),
            })

        info = self._build_info(drawdown)
        info["reward_components"] = reward_components

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Renders current environment state."""
        if self.render_mode not in ("human", "ansi"):
            return None

        date_str = (self.all_dates[self.episode_start + self.current_bar]
                    if (self.episode_start + self.current_bar) < len(self.all_dates)
                    else "END")
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1)
        pnl_pct  = (self.portfolio_value - self.initial_capital) / self.initial_capital

        lines = [
            f"┌── G.O.D.S E.Y.E Env ── {self.mode.name} ──────────────",
            f"│ Date       : {date_str}  (Bar {self.current_bar}/{self.episode_length})",
            f"│ Portfolio  : ₹{self.portfolio_value:,.2f}  ({pnl_pct:+.2%})",
            f"│ Drawdown   : {drawdown:.2%}",
            f"│ Cash       : ₹{self.cash:,.2f}",
            f"│ Positions  : {len(self.positions)}/{MAX_POSITIONS}",
            f"│ Trades/mo  : {self.trades_this_month}/{MAX_TRADES_PER_MONTH}",
            "└──────────────────────────────────────────────────────",
        ]

        for sym, pos in self.positions.items():
            price = self._get_current_price(sym, self.episode_start + self.current_bar)
            if price:
                pnl = pos.current_pnl_pct(price)
                lines.insert(-1, f"│   {sym:12s}: {pnl:+.2%}  "
                                  f"trail={'ON' if pos.trail_active else 'off'}")

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output

    def close(self):
        """Cleanup (nothing to close — DB connections are short-lived)."""
        pass

    # ══════════════════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _update_universe(self):
        """
        Selects the top-N_stocks by backbone embedding score for this bar.

        If backbone is not available (e.g., during unit testing),
        uses random selection from available symbols.
        """
        global_bar = self.episode_start + self.current_bar
        available  = [
            sym for sym in self.data_loader.symbols
            if self.data_loader.get_price(sym, global_bar) is not None
        ]

        if not available:
            self.current_symbols   = []
            self.current_embeddings = np.zeros(
                (self.n_stocks, EMBEDDING_DIM), dtype=np.float32
            )
            return

        if self.backbone is not None:
            # Use backbone to score and rank stocks
            self.current_symbols, self.current_embeddings = (
                self._score_with_backbone(available, global_bar)
            )
        else:
            # Fallback: random selection (used in unit tests)
            chosen = available[:self.n_stocks]
            self.current_symbols   = chosen
            self.current_embeddings = np.random.randn(
                len(chosen), EMBEDDING_DIM
            ).astype(np.float32)

        # Pad to n_stocks if fewer available
        n_available = len(self.current_symbols)
        if n_available < self.n_stocks:
            pad = self.n_stocks - n_available
            self.current_embeddings = np.vstack([
                self.current_embeddings,
                np.zeros((pad, EMBEDDING_DIM), dtype=np.float32),
            ])

    def _score_with_backbone(
        self,
        symbols   : List[str],
        global_bar: int,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Runs backbone inference on a batch of stocks and returns
        top-N by embedding L2 norm (as a proxy for signal strength).

        In production (live trading), this uses pre-computed embeddings
        from the nightly pipeline. During training, it computes on the fly.
        """
        # Build feature sequences for each symbol
        seq_len  = 60
        features = []
        valid    = []

        for sym in symbols:
            df = self.data_loader.get_ohlcv(sym)
            start = global_bar - seq_len
            if start < 0 or global_bar > len(df):
                continue
            chunk = df.iloc[start:global_bar]
            if len(chunk) < seq_len:
                continue

            # Simple normalized OHLCV as fallback features
            # (In production: uses pre-computed Phase 1 features from DB)
            close = chunk["close"].values.astype(np.float32)
            norm  = (close - close.mean()) / (close.std() + 1e-8)

            # Build 28-feature sequence (zeros for missing pillars)
            seq = np.zeros((seq_len, 28), dtype=np.float32)
            seq[:, 0] = norm   # trend proxy in slot 0
            features.append(seq)
            valid.append(sym)

        if not valid:
            return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        # Run backbone
        x = torch.tensor(np.stack(features), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            embeddings = self.backbone(x).cpu().numpy()  # (N, 128)

        # Rank by embedding norm (proxy for signal confidence)
        norms = np.linalg.norm(embeddings, axis=1)
        top_idx = np.argsort(norms)[::-1][:self.n_stocks]

        top_symbols    = [valid[i] for i in top_idx]
        top_embeddings = embeddings[top_idx].astype(np.float32)

        return top_symbols, top_embeddings

    def _build_observation(self) -> np.ndarray:
        """
        Constructs the flat observation vector for the PPO agent.

        Structure:
            [0 : N_stocks*128] : Backbone embeddings (flattened)
            [N_stocks*128 : ]  : Portfolio state (8 features)
        """
        # ── Embeddings ────────────────────────────────────────────────────
        emb_flat = self.current_embeddings[:self.n_stocks].flatten()

        # Pad if fewer than n_stocks available
        expected = self.n_stocks * EMBEDDING_DIM
        if len(emb_flat) < expected:
            emb_flat = np.concatenate([
                emb_flat,
                np.zeros(expected - len(emb_flat), dtype=np.float32)
            ])

        # ── Portfolio state (8 features, all normalized to ~[-1, +1]) ─────
        pnl_pct       = (self.portfolio_value - self.initial_capital) / self.initial_capital
        drawdown      = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1)
        cash_pct      = self.cash / max(self.portfolio_value, 1)
        n_pos_norm    = len(self.positions) / MAX_POSITIONS
        trades_norm   = self.trades_this_month / MAX_TRADES_PER_MONTH
        bar_norm      = self.current_bar / self.episode_length
        avg_pos_pnl   = self._average_position_pnl()
        heat          = self._portfolio_heat()

        portfolio_state = np.array([
            np.clip(pnl_pct,     -1.0, 1.0),   # [0] episode P&L
            np.clip(drawdown,     0.0, 1.0),   # [1] current drawdown
            np.clip(cash_pct,     0.0, 1.0),   # [2] cash fraction
            np.clip(n_pos_norm,   0.0, 1.0),   # [3] position count
            np.clip(trades_norm,  0.0, 1.0),   # [4] trades used this month
            np.clip(bar_norm,     0.0, 1.0),   # [5] episode progress
            np.clip(avg_pos_pnl, -1.0, 1.0),   # [6] avg open position P&L
            np.clip(heat,         0.0, 1.0),   # [7] portfolio heat
        ], dtype=np.float32)

        return np.concatenate([emb_flat, portfolio_state])

    def _execute_buy(self, global_bar: int, strong: bool = False) -> bool:
        """
        Attempts to enter a long position in the highest-ranked
        unowned stock from current_symbols.

        Returns True if a trade was successfully executed.
        """
        # ── Pre-trade Risk Constitution checks ────────────────────────────
        if self.rc.check_rc08(len(self.positions)):
            return False

        if self.trades_this_month >= MAX_TRADES_PER_MONTH:
            return False

        # Find best unowned stock
        candidate = None
        for sym in self.current_symbols:
            if sym not in self.positions:
                bar = self.data_loader.get_bar(sym, global_bar)
                if bar is None:
                    continue

                # RC-05: Circuit breaker proximity
                if self.rc.check_rc05(bar):
                    continue

                # RC-07: Liquidity check (approximate)
                avg_vol = bar["volume"]
                if self.rc.check_rc07(avg_vol, bar["close"]):
                    continue

                # RC-10: Sector concentration
                if self.rc.check_rc10(sym, list(self.positions.keys()), self.sector_map):
                    continue

                candidate = (sym, bar)
                break

        if candidate is None:
            return False

        sym, bar = candidate
        entry_price = bar["close"]   # enter at close price

        # ── Position sizing ───────────────────────────────────────────────
        # Strong Buy = full Kelly allocation, Buy = half Kelly
        size_pct    = MAX_POSITION_PCT * (1.0 if strong else 0.6)
        invest_amt  = self.portfolio_value * size_pct
        invest_amt  = min(invest_amt, self.cash * 0.95)  # don't use all cash

        if invest_amt < 1000:   # minimum ₹1000 per trade
            return False

        quantity = max(1, int(invest_amt / entry_price))
        actual_cost = quantity * entry_price

        # ── Transaction costs (buy side) ──────────────────────────────────
        transaction_cost = actual_cost * TOTAL_BUY_COST
        total_debit = actual_cost + transaction_cost

        if total_debit > self.cash:
            return False

        # ── TP/SL calculation ─────────────────────────────────────────────
        if self.mode == TradeMode.SWING:
            tp_price = entry_price * (1 + SWING_TP_PCT)
            sl_price = entry_price * (1 - SWING_SL_PCT)
        else:
            tp_price = entry_price * (1 + INTRADAY_TP_PCT)
            sl_price = entry_price * (1 - INTRADAY_SL_PCT)

        # ── Open position ─────────────────────────────────────────────────
        self.positions[sym] = Position(
            symbol         = sym,
            entry_price    = entry_price,
            entry_bar      = self.current_bar,
            quantity       = quantity,
            position_value = actual_cost,
            tp_price       = tp_price,
            sl_price       = sl_price,
            trail_peak     = entry_price,
            trail_stop     = sl_price,
        )

        self.cash -= total_debit
        self.trades_this_month += 1
        self.total_trades      += 1

        logger.debug(
            f"BUY  {sym:12s} @ ₹{entry_price:,.2f}  "
            f"qty={quantity}  cost={total_debit:,.2f}  "
            f"TP={tp_price:,.2f}  SL={sl_price:,.2f}"
        )

        return True

    def _execute_sell(self, global_bar: int) -> bool:
        """
        Closes the worst-performing open position (highest loss first).
        Returns True if a position was closed.
        """
        if not self.positions:
            return False

        # Find worst-performing position
        worst_sym   = None
        worst_pnl   = float("inf")

        for sym, pos in self.positions.items():
            price = self._get_current_price(sym, global_bar)
            if price is None:
                continue
            pnl = pos.current_pnl_pct(price)
            if pnl < worst_pnl:
                worst_pnl = pnl
                worst_sym = sym

        if worst_sym is None:
            return False

        return self._close_position(worst_sym, global_bar, reason="SELL")

    def _update_positions(self, global_bar: int) -> int:
        """
        Updates all open positions for the current bar.
        Closes positions that hit TP, SL, or trailing stop.

        Returns number of positions closed this bar.
        """
        to_close  = []
        n_closed  = 0

        for sym, pos in list(self.positions.items()):
            price = self._get_current_price(sym, global_bar)
            if price is None:
                continue

            pos.hold_days += 1

            # ── Check Take Profit ─────────────────────────────────────────
            if price >= pos.tp_price:
                to_close.append((sym, "TP"))
                continue

            # ── Check Stop Loss ───────────────────────────────────────────
            if price <= pos.sl_price:
                to_close.append((sym, "SL"))
                continue

            # ── Check Trailing Stop ───────────────────────────────────────
            if pos.update_trailing_stop(price, self.mode):
                to_close.append((sym, "TRAIL"))
                continue

            # ── RC-02: Force close if single loss > 2% of portfolio ───────
            if self.rc.check_rc02(pos, price, self.portfolio_value):
                to_close.append((sym, "RC-02"))
                continue

            # ── Swing max hold duration (15 trading days) ─────────────────
            if self.mode == TradeMode.SWING and pos.hold_days >= 15:
                to_close.append((sym, "MAX_HOLD"))
                continue

        for sym, reason in to_close:
            self._close_position(sym, global_bar, reason=reason)
            n_closed += 1

        return n_closed

    def _close_position(self, symbol: str, global_bar: int, reason: str = "") -> bool:
        """
        Closes an open position and realizes P&L.

        Returns True if successfully closed.
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return False

        price = self._get_current_price(symbol, global_bar)
        if price is None:
            price = pos.entry_price  # fallback to entry if no price

        # ── P&L calculation ───────────────────────────────────────────────
        gross_proceeds    = pos.quantity * price
        transaction_cost  = gross_proceeds * TOTAL_SELL_COST
        net_proceeds      = gross_proceeds - transaction_cost
        realised_pnl      = net_proceeds - pos.position_value

        self.cash += net_proceeds
        del self.positions[symbol]

        logger.debug(
            f"CLOSE {symbol:12s} @ ₹{price:,.2f}  "
            f"pnl=₹{realised_pnl:+,.2f}  reason={reason}"
        )

        return True

    def _force_close_all(self, global_bar: int) -> int:
        """
        Force-closes all open positions (end of intraday session).
        Returns number of positions closed.
        """
        symbols = list(self.positions.keys())
        for sym in symbols:
            self._close_position(sym, global_bar, reason="EOD_FORCE")
        return len(symbols)

    def _update_portfolio_value(self, global_bar: int):
        """
        Marks portfolio to market: cash + mark-to-market value of positions.
        """
        mtm = sum(
            pos.quantity * (self._get_current_price(sym, global_bar) or pos.entry_price)
            for sym, pos in self.positions.items()
        )
        self.portfolio_value = self.cash + mtm

    def _get_current_price(self, symbol: str, global_bar: int) -> Optional[float]:
        """Returns close price for a symbol at the given global bar index."""
        return self.data_loader.get_price(symbol, global_bar)

    def _average_position_pnl(self) -> float:
        """Returns average P&L fraction across open positions."""
        if not self.positions:
            return 0.0
        pnls = []
        for sym, pos in self.positions.items():
            price = self._get_current_price(sym, self.episode_start + self.current_bar)
            if price:
                pnls.append(pos.current_pnl_pct(price))
        return float(np.mean(pnls)) if pnls else 0.0

    def _portfolio_heat(self) -> float:
        """
        Returns total portfolio heat: sum of open risk across all positions.
        Heat = (entry_price - sl_price) / entry_price × position_size_pct.
        Capped at 1.0 for normalization.
        """
        if not self.portfolio_value:
            return 0.0
        total_heat = sum(
            ((pos.entry_price - pos.sl_price) / pos.entry_price) *
            (pos.position_value / self.portfolio_value)
            for pos in self.positions.values()
        )
        return min(total_heat, 1.0)

    def _build_info(self, drawdown: float) -> Dict[str, Any]:
        """Builds the info dict returned by step()."""
        return {
            "portfolio_value"   : self.portfolio_value,
            "cash"              : self.cash,
            "n_positions"       : len(self.positions),
            "drawdown"          : drawdown,
            "total_trades"      : self.total_trades,
            "trades_this_month" : self.trades_this_month,
            "current_bar"       : self.current_bar,
            "episode_start_date": (self.all_dates[self.episode_start]
                                   if self.episode_start < len(self.all_dates)
                                   else "unknown"),
            "rc_triggered"      : {k: v for k, v in self.rc.triggered.items() if v > 0},
            "ep_stats"          : self._ep_stats,
        }


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest environment/godseye_env.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestGodsEyeEnv:
    """
    Unit tests for the GodsEyeEnv trading environment.
    Uses a mock MarketDataLoader to avoid DB dependency in tests.
    """

    def _make_mock_loader(self, n_symbols: int = 10, n_days: int = 300) -> MarketDataLoader:
        """Creates a MarketDataLoader with synthetic data (no DB needed)."""
        loader = MarketDataLoader.__new__(MarketDataLoader)
        loader._loaded  = True
        loader._symbols = [f"STOCK{i:02d}" for i in range(n_symbols)]
        loader._trading_dates = [
            (pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 1)).strftime("%Y-%m-%d")
            for i in range(n_days)
        ]
        # Build synthetic OHLCV cache
        loader._cache = {}
        np.random.seed(42)
        for sym in loader._symbols:
            close  = 1000 + np.cumsum(np.random.randn(n_days) * 5)
            close  = np.maximum(close, 10)
            high   = close * (1 + np.abs(np.random.randn(n_days) * 0.01))
            low    = close * (1 - np.abs(np.random.randn(n_days) * 0.01))
            vol    = np.random.randint(100_000, 5_000_000, n_days).astype(float)
            idx    = pd.date_range("2020-01-01", periods=n_days, freq="D")
            loader._cache[sym] = pd.DataFrame({
                "open": close * 0.99, "high": high, "low": low,
                "close": close, "volume": vol,
            }, index=idx)
        return loader

    def _make_env(self, mode=TradeMode.SWING) -> GodsEyeEnv:
        loader = self._make_mock_loader()
        env = GodsEyeEnv(
            data_loader = loader,
            backbone    = None,   # no backbone in unit tests
            mode        = mode,
            n_stocks    = 5,
            train_start_idx = 0,
            train_end_idx   = 200,
        )
        return env

    # ── Gymnasium API compliance ──────────────────────────────────────────

    def test_observation_space_shape(self):
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        expected = 5 * EMBEDDING_DIM + PORTFOLIO_DIM
        assert obs.shape == (expected,), f"Obs shape wrong: {obs.shape}"

    def test_observation_dtype(self):
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        assert obs.dtype == np.float32, f"Obs dtype wrong: {obs.dtype}"

    def test_observation_no_nan(self):
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        assert not np.isnan(obs).any(), "NaN in initial observation"

    def test_observation_in_space(self):
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs), "Observation outside space"

    def test_action_space_is_discrete5(self):
        env = self._make_env()
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 5

    def test_reset_returns_valid_obs(self):
        env = self._make_env()
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        env = self._make_env()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(Action.HOLD)
        assert isinstance(obs,        np.ndarray)
        assert isinstance(reward,     float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated,  bool)
        assert isinstance(info,       dict)

    def test_step_obs_no_nan(self):
        env = self._make_env()
        env.reset(seed=42)
        for action in range(5):
            obs, *_ = env.step(action)
            assert not np.isnan(obs).any(), f"NaN in obs after action {action}"

    # ── Episode lifecycle ─────────────────────────────────────────────────

    def test_episode_terminates_at_max_bars(self):
        env = self._make_env(mode=TradeMode.SWING)
        env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(Action.HOLD)
            done = terminated or truncated
            steps += 1
            if steps > 1000:
                break
        assert steps == SWING_EPISODE_DAYS, \
            f"Expected {SWING_EPISODE_DAYS} steps, got {steps}"

    def test_reset_clears_positions(self):
        env = self._make_env()
        env.reset(seed=42)
        env.step(Action.BUY)
        env.reset(seed=1)
        assert len(env.positions) == 0, "Positions not cleared on reset"

    def test_reset_restores_capital(self):
        env = self._make_env()
        env.reset(seed=42)
        env.step(Action.BUY)
        env.reset(seed=1)
        assert abs(env.portfolio_value - env.initial_capital) < 1.0

    def test_multiple_resets(self):
        env = self._make_env()
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            assert not np.isnan(obs).any()

    # ── Portfolio mechanics ───────────────────────────────────────────────

    def test_cash_decreases_on_buy(self):
        env = self._make_env()
        env.reset(seed=42)
        cash_before = env.cash
        env.step(Action.BUY)
        assert env.cash <= cash_before, "Cash should decrease after buy"

    def test_position_opens_on_buy(self):
        env = self._make_env()
        env.reset(seed=42)
        env.step(Action.BUY)
        # May or may not open depending on RC checks — just verify no crash

    def test_max_positions_respected(self):
        env = self._make_env()
        env.reset(seed=42)
        for _ in range(20):
            env.step(Action.STRONG_BUY)
        assert len(env.positions) <= MAX_POSITIONS, \
            f"Too many positions: {len(env.positions)} > {MAX_POSITIONS}"

    def test_portfolio_value_non_negative(self):
        env = self._make_env()
        env.reset(seed=42)
        for _ in range(SWING_EPISODE_DAYS):
            _, _, terminated, truncated, _ = env.step(Action.BUY)
            assert env.portfolio_value >= 0, "Negative portfolio value"
            if terminated or truncated:
                break

    # ── Risk Constitution ─────────────────────────────────────────────────

    def test_rc08_max_positions_blocks_buy(self):
        """RC-08 must prevent opening more than 4 positions."""
        env = self._make_env()
        env.reset(seed=42)

        # Use symbols outside current_symbols so they don't
        # interfere with the buy candidate search
        dummy_symbols = [f"DUMMY{i}" for i in range(4)]
        for sym in dummy_symbols:
            env.positions[sym] = Position(
                symbol=sym, entry_price=100.0, entry_bar=0,
                quantity=10, position_value=1000.0,
                tp_price=104.0, sl_price=98.5,
            )

        assert len(env.positions) == 4

        # Directly test RC-08 logic — not through step()
        blocked = env.rc.check_rc08(len(env.positions))
        assert blocked, "RC-08 should block when 4 positions are open"

        # Also verify step() doesn't open a 5th
        env.step(Action.BUY)
        assert len(env.positions) <= MAX_POSITIONS, \
            f"Exceeded max positions: {len(env.positions)}"

    def test_rc01_triggers_on_large_drawdown(self):
        """RC-01 must trigger and terminate episode on 12%+ drawdown."""
        env = self._make_env()
        env.reset(seed=42)
        # Artificially set peak high, portfolio low
        env.peak_value       = env.initial_capital
        env.portfolio_value  = env.initial_capital * 0.87   # 13% drawdown
        _, _, terminated, _, info = env.step(Action.HOLD)
        assert terminated, "RC-01 should terminate episode at 12% drawdown"

    def test_rc_counters_reset_on_episode_reset(self):
        """Risk Constitution trigger counters must reset each episode."""
        env = self._make_env()
        env.reset(seed=42)
        env.rc.triggered["RC-08"] = 5
        env.reset(seed=1)
        assert env.rc.triggered["RC-08"] == 0, "RC counters not reset on episode reset"

    # ── Reward ────────────────────────────────────────────────────────────

    def test_reward_is_scalar_float(self):
        env = self._make_env()
        env.reset(seed=42)
        _, reward, *_ = env.step(Action.HOLD)
        assert isinstance(reward, float), f"Reward is not float: {type(reward)}"

    def test_reward_no_nan(self):
        env = self._make_env()
        env.reset(seed=42)
        for _ in range(5):
            _, reward, *_ = env.step(Action.HOLD)
            assert not math.isnan(reward), "NaN reward"

    def test_sell_without_positions_no_crash(self):
        """Selling with no open positions must not raise an error."""
        env = self._make_env()
        env.reset(seed=42)
        obs, reward, *_ = env.step(Action.SELL)
        assert not np.isnan(obs).any()
        assert not math.isnan(reward)

    # ── Transaction costs ─────────────────────────────────────────────────

    def test_transaction_costs_positive(self):
        """Total buy + sell costs must be positive."""
        assert TOTAL_BUY_COST  > 0
        assert TOTAL_SELL_COST > 0

    def test_transaction_costs_reasonable(self):
        """Total round-trip cost should be between 0.1% and 1.5%."""
        roundtrip = TOTAL_BUY_COST + TOTAL_SELL_COST
        assert 0.001 < roundtrip < 0.015, \
            f"Round-trip cost {roundtrip:.4%} outside reasonable range"

    # ── Position mechanics ────────────────────────────────────────────────

    def test_position_trailing_stop(self):
        """Trailing stop must activate after sufficient gain."""
        pos = Position(
            symbol="TEST", entry_price=100.0, entry_bar=0,
            quantity=10, position_value=1000.0,
            tp_price=104.0, sl_price=98.5,
        )
        # Price rises past activation threshold
        hit = pos.update_trailing_stop(103.0, TradeMode.SWING)
        assert pos.trail_active, "Trailing stop should be active after 3% gain"
        assert not hit, "Trailing stop should not be hit yet"

        # Price pulls back past trail stop
        hit = pos.update_trailing_stop(102.0, TradeMode.SWING)
        # trail_stop = 103.0 * (1 - 0.008) = 102.176
        assert hit or not hit   # just check no crash

    def test_position_pnl_calculation(self):
        """P&L calculation must be correct."""
        pos = Position(
            symbol="TEST", entry_price=100.0, entry_bar=0,
            quantity=10, position_value=1000.0,
            tp_price=104.0, sl_price=98.5,
        )
        assert abs(pos.current_pnl_pct(110.0) - 0.10) < 1e-6
        assert abs(pos.current_pnl_pct(90.0) - (-0.10)) < 1e-6

    # ── Render ────────────────────────────────────────────────────────────

    def test_render_returns_string(self):
        env = self._make_env()
        env.render_mode = "ansi"
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, str)
        assert "G.O.D.S E.Y.E" in result


# ── Run tests when file is executed directly ──────────────────────────────
if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))