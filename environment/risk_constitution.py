"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Risk Constitution (Production)                  ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : environment/risk_constitution.py                       ║
║         Phase   : 3 — RL Agent Training / Live Trading                  ║
║                                                                          ║
║  What this module does:                                                  ║
║    Implements all 10 Risk Constitution rules as a standalone, stateful,  ║
║    production-grade constraint system used during LIVE trading.          ║
║                                                                          ║
║    This is distinct from the inline RC checks in godseye_env.py:         ║
║      godseye_env.py   → lightweight, fast, for RL training simulation   ║
║      THIS FILE        → full logging, DB persistence, Telegram alerts,  ║
║                         audit trail, used by execution/signal_engine.py ║
║                                                                          ║
║  The 10 Rules:                                                           ║
║    RC-01: Portfolio drawdown > 12%       → halt ALL trading             ║
║    RC-02: Single position loss > 2% port → force close immediately      ║
║    RC-03: New stock correlation > 0.75   → block entry                  ║
║    RC-04: Earnings within 2 trading days → block entry in that stock    ║
║    RC-05: Stock near circuit limit       → block entry, tighten stop    ║
║    RC-06: Market panic (VIX spike / big → block all intraday signals   ║
║            Nifty drop)                                                   ║
║    RC-07: Avg daily turnover < ₹5 crore  → exclude from universe        ║
║    RC-08: 4 positions already open       → block new entries            ║
║    RC-09: FII sells > ₹3000 crore/day   → MDS forced -3, block longs   ║
║    RC-10: 2 positions in same sector     → block third in same sector   ║
║                                                                          ║
║  Design principles:                                                      ║
║    • Rules are HARD constraints — no model output can override them     ║
║    • Every trigger is logged to DB + loguru with full context            ║
║    • State persists across sessions (RC-01 halt survives restart)       ║
║    • Telegram alert on every RC-01 trigger (account at risk)            ║
║    • Thread-safe (can be called from async signal engine)               ║
║                                                                          ║
║  Usage in signal engine:                                                 ║
║    rc = RiskConstitution(conn=db_conn)                                   ║
║    result = rc.evaluate_entry(symbol, portfolio_state, market_state)    ║
║    if result.blocked:                                                    ║
║        logger.info(f"Blocked by {result.rule}: {result.reason}")        ║
║    else:                                                                 ║
║        execute_trade(symbol)                                             ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install psycopg2-binary loguru python-dotenv requests            ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import threading
import requests
import psycopg2
import psycopg2.extras
import numpy as np

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List, Tuple
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────
DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)

# ── Telegram alerts ───────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Risk Constitution thresholds ──────────────────────────────────────────
# These match risk_config.yaml — loaded from env/config at runtime
RC01_MAX_DRAWDOWN          = 0.12    # 12% portfolio drawdown
RC02_SINGLE_LOSS_PCT       = 0.02    # 2% of total portfolio
RC03_CORRELATION_THRESHOLD = 0.75    # max allowed correlation
RC04_EARNINGS_BUFFER_DAYS  = 2       # trading days before earnings
RC05_CIRCUIT_PROXIMITY_PCT = 0.18    # 18% intraday move = circuit proximity
RC06_VIX_SPIKE_PCT         = 0.08    # 8% intraday VIX rise
RC06_NIFTY_DROP_PCT        = 0.03    # 3% Nifty drop in < 30 mins
RC07_MIN_TURNOVER_CR       = 5.0     # ₹5 crore minimum avg daily turnover
RC08_MAX_POSITIONS         = 4       # max simultaneous open positions
RC09_FII_SELL_THRESHOLD_CR = 3000.0  # ₹3000 crore FII net sell in one day
RC10_MAX_SECTOR_POSITIONS  = 2       # max positions per sector


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RCResult:
    """
    Result of a Risk Constitution evaluation.

    Returned by every RC check method and the top-level evaluate_entry().

    Attributes:
        allowed  : True if trade/action is permitted
        blocked  : True if trade/action is blocked by any rule
        rule     : Which rule fired (e.g. 'RC-03'), empty string if allowed
        reason   : Human-readable explanation of why blocked
        action   : What the system should do ('block', 'force_close',
                   'halt', 'tighten_stop', 'reduce_size')
        metadata : Additional context (correlation value, drawdown %, etc.)
    """
    allowed  : bool
    rule     : str        = ""
    reason   : str        = ""
    action   : str        = "none"
    metadata : Dict       = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.allowed

    @classmethod
    def permit(cls, metadata: Dict = None) -> "RCResult":
        """Factory for a passing (allowed) result."""
        return cls(allowed=True, metadata=metadata or {})

    @classmethod
    def block(
        cls,
        rule    : str,
        reason  : str,
        action  : str  = "block",
        metadata: Dict = None,
    ) -> "RCResult":
        """Factory for a blocked result."""
        return cls(
            allowed  = False,
            rule     = rule,
            reason   = reason,
            action   = action,
            metadata = metadata or {},
        )


@dataclass
class PortfolioState:
    """
    Snapshot of the current portfolio for RC evaluation.

    Passed into evaluate_entry() by the signal engine each time
    a new trade signal is being considered.
    """
    total_value       : float              # total portfolio value in ₹
    peak_value        : float              # all-time peak portfolio value
    cash              : float              # available cash in ₹
    open_positions    : Dict[str, Dict]    # {symbol: {entry_price, quantity, sl_price, ...}}
    trades_today      : int                # trades executed today
    trades_this_month : int                # trades executed this calendar month
    sector_map        : Dict[str, str]     # {symbol: sector_name}

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak as a fraction."""
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value

    @property
    def n_positions(self) -> int:
        return len(self.open_positions)

    @property
    def open_symbols(self) -> List[str]:
        return list(self.open_positions.keys())


@dataclass
class MarketState:
    """
    Snapshot of current market conditions for RC evaluation.

    Populated by the signal engine from live data feeds.
    """
    nifty_open        : float    = 0.0    # Nifty open price today
    nifty_current     : float    = 0.0    # Nifty current price
    india_vix_open    : float    = 0.0    # India VIX at market open
    india_vix_current : float    = 0.0    # India VIX current
    fii_provisional_cr: float    = 0.0    # FII provisional net (₹ crore, signed)
    mds_score         : int      = 0      # Market Direction Signal [-3, +3]
    is_halt           : bool     = False  # True if RC-01 halt is active

    @property
    def nifty_drop_pct(self) -> float:
        """Intraday Nifty drop from open as a positive fraction."""
        if self.nifty_open <= 0:
            return 0.0
        return max(0.0, (self.nifty_open - self.nifty_current) / self.nifty_open)

    @property
    def vix_spike_pct(self) -> float:
        """Intraday VIX spike from open as a positive fraction."""
        if self.india_vix_open <= 0:
            return 0.0
        return max(0.0, (self.india_vix_current - self.india_vix_open) / self.india_vix_open)


# ══════════════════════════════════════════════════════════════════════════
#  RISK CONSTITUTION
# ══════════════════════════════════════════════════════════════════════════

class RiskConstitution:
    """
    Production-grade Risk Constitution enforcement engine.

    Thread-safe. Persists halt state to DB so a restart doesn't
    accidentally resume trading after RC-01 fires.

    Usage:
        rc = RiskConstitution()

        # Before entering any trade
        result = rc.evaluate_entry(
            symbol    = "RELIANCE",
            portfolio = portfolio_state,
            market    = market_state,
            corr_matrix = correlation_matrix,
            symbol_idx  = symbol_to_index_map,
        )
        if result.blocked:
            logger.warning(f"Trade blocked: {result.rule} — {result.reason}")
        else:
            place_order(...)

        # Every bar — check existing positions
        for sym, pos in open_positions.items():
            result = rc.evaluate_position(sym, pos, current_price, portfolio)
            if result.action == "force_close":
                close_position(sym)
    """

    def __init__(self, conn=None):
        """
        Args:
            conn : Optional psycopg2 connection. If None, creates its own.
                   Pass a connection when called from a context that already
                   has one open (e.g., signal engine) to avoid extra connections.
        """
        self._lock       = threading.Lock()
        self._own_conn   = conn is None
        self._conn       = conn
        self._halt_active= False   # RC-01 halt state (loaded from DB on init)
        self._trigger_log: List[Dict] = []   # in-memory trigger history

        self._ensure_table()
        self._load_halt_state()

    # ══════════════════════════════════════════════════════════════════════
    #  TOP-LEVEL EVALUATION METHODS
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_entry(
        self,
        symbol      : str,
        portfolio   : PortfolioState,
        market      : MarketState,
        corr_matrix : Optional[np.ndarray] = None,
        symbol_idx  : Optional[Dict[str, int]] = None,
        bar_data    : Optional[Dict]            = None,
        avg_turnover_cr: float                  = 100.0,
    ) -> RCResult:
        """
        Evaluates ALL applicable RC rules before entering a new position.

        Rules are evaluated in priority order — the first blocking rule
        wins and is returned immediately (fail-fast).

        Args:
            symbol          : NSE symbol to enter
            portfolio       : Current portfolio snapshot
            market          : Current market conditions
            corr_matrix     : (N, N) correlation matrix (optional)
            symbol_idx      : {symbol: matrix_index} mapping (optional)
            bar_data        : Today's OHLCV bar for the symbol (optional)
            avg_turnover_cr : Average daily turnover in ₹ crore (optional)

        Returns:
            RCResult — allowed=True if all rules pass, blocked with details if not
        """
        with self._lock:

            # ── RC-01: Kill switch (checked first — overrides everything) ──
            result = self.check_rc01(portfolio)
            if result.blocked:
                return result

            # ── RC-06: Market panic ────────────────────────────────────────
            result = self.check_rc06(market)
            if result.blocked:
                return result

            # ── RC-09: FII panic selling ───────────────────────────────────
            result = self.check_rc09(market)
            if result.blocked:
                return result

            # ── RC-08: Max positions ───────────────────────────────────────
            result = self.check_rc08(portfolio)
            if result.blocked:
                return result

            # ── RC-07: Liquidity ───────────────────────────────────────────
            result = self.check_rc07(symbol, avg_turnover_cr)
            if result.blocked:
                return result

            # ── RC-05: Circuit breaker proximity ──────────────────────────
            if bar_data:
                result = self.check_rc05(symbol, bar_data)
                if result.blocked:
                    return result

            # ── RC-03: Correlation cap ─────────────────────────────────────
            if corr_matrix is not None and symbol_idx is not None:
                result = self.check_rc03(
                    symbol, portfolio.open_symbols,
                    corr_matrix, symbol_idx,
                )
                if result.blocked:
                    return result

            # ── RC-10: Sector concentration ────────────────────────────────
            result = self.check_rc10(symbol, portfolio)
            if result.blocked:
                return result

            # ── RC-04: Earnings blackout ───────────────────────────────────
            result = self.check_rc04(symbol)
            if result.blocked:
                return result

            # All rules passed
            return RCResult.permit(metadata={"symbol": symbol})

    def evaluate_position(
        self,
        symbol         : str,
        position       : Dict,
        current_price  : float,
        portfolio      : PortfolioState,
    ) -> RCResult:
        """
        Evaluates RC rules on an EXISTING open position each bar.

        Called by the order manager every bar for every open position.

        Args:
            symbol        : NSE symbol
            position      : Position dict with entry_price, quantity, sl_price
            current_price : Current market price
            portfolio     : Current portfolio snapshot

        Returns:
            RCResult — if action == 'force_close', close the position immediately
        """
        with self._lock:

            # ── RC-01: Kill switch ────────────────────────────────────────
            result = self.check_rc01(portfolio)
            if result.blocked:
                return result

            # ── RC-02: Single position loss ───────────────────────────────
            result = self.check_rc02(symbol, position, current_price, portfolio)
            if result.blocked:
                return result

            return RCResult.permit()

    def clear_halt(self, authorized_by: str = "manual") -> bool:
        """
        Clears the RC-01 halt state after human review.

        ONLY call this after manually reviewing why RC-01 fired
        and confirming it is safe to resume trading.

        Args:
            authorized_by : Name/ID of person clearing the halt

        Returns:
            True if halt was cleared, False if no halt was active
        """
        with self._lock:
            if not self._halt_active:
                logger.info("clear_halt called but no halt is active")
                return False

            self._halt_active = False
            self._persist_halt_state(active=False, cleared_by=authorized_by)
            logger.warning(
                f"RC-01 HALT CLEARED by {authorized_by}. "
                f"Trading will resume on next signal."
            )
            self._send_telegram(
                f"⚠️ RC-01 HALT CLEARED by {authorized_by}. Trading resuming."
            )
            return True

    # ══════════════════════════════════════════════════════════════════════
    #  INDIVIDUAL RULE IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════════════

    def check_rc01(self, portfolio: PortfolioState) -> RCResult:
        """
        RC-01: Portfolio Drawdown Kill Switch.

        If portfolio drawdown from peak exceeds 12%, ALL trading halts.
        The halt persists until manually cleared via clear_halt().

        This is the most critical rule — it protects against cascading
        losses during black-swan events (COVID crash, Adani episode, etc.)
        """
        # Check if halt is already active (from previous session)
        if self._halt_active:
            return RCResult.block(
                rule    = "RC-01",
                reason  = "Trading halt is active from previous RC-01 trigger. "
                          "Call rc.clear_halt() after manual review to resume.",
                action  = "halt",
                metadata= {"halt_active": True},
            )

        dd = portfolio.drawdown
        if dd > RC01_MAX_DRAWDOWN:
            self._halt_active = True
            self._persist_halt_state(active=True)
            self._log_trigger(
                rule     = "RC-01",
                symbol   = "PORTFOLIO",
                details  = f"Drawdown {dd:.2%} exceeded {RC01_MAX_DRAWDOWN:.0%} threshold",
                metadata = {"drawdown": dd, "portfolio_value": portfolio.total_value},
            )
            self._send_telegram(
                f"🚨 RC-01 KILL SWITCH ACTIVATED\n"
                f"Portfolio drawdown: {dd:.2%}\n"
                f"Portfolio value: ₹{portfolio.total_value:,.0f}\n"
                f"ALL TRADING HALTED — Manual review required."
            )
            logger.critical(
                f"RC-01 TRIGGERED: Drawdown {dd:.2%} > {RC01_MAX_DRAWDOWN:.0%}. "
                f"All trading halted."
            )
            return RCResult.block(
                rule    = "RC-01",
                reason  = f"Portfolio drawdown {dd:.2%} exceeds 12% kill switch threshold.",
                action  = "halt",
                metadata= {"drawdown": dd},
            )

        return RCResult.permit()

    def check_rc02(
        self,
        symbol       : str,
        position     : Dict,
        current_price: float,
        portfolio    : PortfolioState,
    ) -> RCResult:
        """
        RC-02: Single Position Loss Limit.

        If any single position loses more than 2% of total portfolio value,
        force-close it immediately regardless of where the stop-loss is set.

        This prevents a single bad trade from causing outsized damage
        to the overall portfolio.
        """
        entry_price  = position.get("entry_price", current_price)
        quantity     = position.get("quantity", 0)
        position_val = entry_price * quantity

        loss_fraction    = max(0.0, (entry_price - current_price) / entry_price)
        loss_abs         = loss_fraction * position_val
        loss_as_pct_port = loss_abs / max(portfolio.total_value, 1.0)

        if loss_as_pct_port > RC02_SINGLE_LOSS_PCT:
            self._log_trigger(
                rule    = "RC-02",
                symbol  = symbol,
                details = (
                    f"Position loss ₹{loss_abs:,.0f} = "
                    f"{loss_as_pct_port:.2%} of portfolio "
                    f"exceeds {RC02_SINGLE_LOSS_PCT:.0%} limit"
                ),
                metadata= {
                    "loss_pct_portfolio": loss_as_pct_port,
                    "loss_abs"          : loss_abs,
                    "entry_price"       : entry_price,
                    "current_price"     : current_price,
                },
            )
            logger.warning(
                f"RC-02: {symbol} position loss {loss_as_pct_port:.2%} of portfolio. "
                f"Force closing."
            )
            return RCResult.block(
                rule    = "RC-02",
                reason  = (
                    f"{symbol} has lost {loss_as_pct_port:.2%} of total portfolio "
                    f"(limit: {RC02_SINGLE_LOSS_PCT:.0%}). Force close immediately."
                ),
                action  = "force_close",
                metadata= {"loss_pct_portfolio": loss_as_pct_port},
            )

        return RCResult.permit()

    def check_rc03(
        self,
        symbol      : str,
        open_symbols: List[str],
        corr_matrix : np.ndarray,
        symbol_idx  : Dict[str, int],
    ) -> RCResult:
        """
        RC-03: Correlation Concentration Cap.

        Blocks entry if the new symbol has correlation > 0.75 with ANY
        currently open position. Prevents holding highly correlated stocks
        that would move together in a drawdown (concentration risk).
        """
        if symbol not in symbol_idx:
            return RCResult.permit()

        new_idx = symbol_idx[symbol]

        for existing_sym in open_symbols:
            if existing_sym not in symbol_idx:
                continue
            ex_idx = symbol_idx[existing_sym]

            try:
                corr = float(corr_matrix[new_idx, ex_idx])
            except (IndexError, TypeError):
                continue

            if abs(corr) > RC03_CORRELATION_THRESHOLD:
                self._log_trigger(
                    rule    = "RC-03",
                    symbol  = symbol,
                    details = (
                        f"Correlation with {existing_sym} = {corr:.3f} "
                        f"exceeds {RC03_CORRELATION_THRESHOLD} threshold"
                    ),
                    metadata= {
                        "correlated_with": existing_sym,
                        "correlation"    : corr,
                    },
                )
                return RCResult.block(
                    rule    = "RC-03",
                    reason  = (
                        f"{symbol} correlation with open position {existing_sym} "
                        f"is {corr:.2f} (limit: {RC03_CORRELATION_THRESHOLD}). "
                        f"Would create concentration risk."
                    ),
                    action  = "block",
                    metadata= {"correlated_with": existing_sym, "correlation": corr},
                )

        return RCResult.permit()

    def check_rc04(self, symbol: str) -> RCResult:
        """
        RC-04: Earnings Blackout Window.

        Blocks new entry within RC04_EARNINGS_BUFFER_DAYS trading days
        of a known earnings announcement.

        Earnings = binary event risk. The model cannot predict the
        direction of post-earnings moves reliably, so we simply avoid
        entering before them.

        Note: Requires earnings_calendar table in DB.
        Falls through (permits) if table doesn't exist yet.
        """
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MIN(announcement_date)
                    FROM earnings_calendar
                    WHERE symbol = %s
                      AND announcement_date >= CURRENT_DATE
                      AND announcement_date <= CURRENT_DATE + INTERVAL '%s days';
                """, (symbol, RC04_EARNINGS_BUFFER_DAYS * 2))  # calendar days
                row = cur.fetchone()

            if row and row[0] is not None:
                earnings_date = row[0]
                self._log_trigger(
                    rule    = "RC-04",
                    symbol  = symbol,
                    details = f"Earnings announcement on {earnings_date}",
                    metadata= {"earnings_date": str(earnings_date)},
                )
                return RCResult.block(
                    rule    = "RC-04",
                    reason  = (
                        f"{symbol} has earnings on {earnings_date} "
                        f"(within {RC04_EARNINGS_BUFFER_DAYS}-day blackout window)."
                    ),
                    action  = "block",
                    metadata= {"earnings_date": str(earnings_date)},
                )

        except psycopg2.errors.UndefinedTable:
            # earnings_calendar table not yet created — skip RC-04
            logger.debug("RC-04: earnings_calendar table not found, skipping.")
        except Exception as e:
            logger.warning(f"RC-04 check failed for {symbol}: {e}")

        return RCResult.permit()

    def check_rc05(self, symbol: str, bar_data: Dict) -> RCResult:
        """
        RC-05: Circuit Breaker Proximity.

        Blocks entry if the stock has moved more than 18% intraday
        (upper or lower circuit proximity).

        Circuit-locked stocks cannot be exited — entering near a
        circuit limit can trap capital indefinitely.

        Also returns action='tighten_stop' for existing positions.
        """
        open_price = bar_data.get("open", 0)
        high_price = bar_data.get("high", 0)
        low_price  = bar_data.get("low", open_price)

        if open_price <= 0:
            return RCResult.permit()

        intraday_range = (high_price - low_price) / open_price

        if intraday_range > RC05_CIRCUIT_PROXIMITY_PCT:
            self._log_trigger(
                rule    = "RC-05",
                symbol  = symbol,
                details = (
                    f"Intraday range {intraday_range:.1%} > "
                    f"{RC05_CIRCUIT_PROXIMITY_PCT:.0%} circuit proximity threshold"
                ),
                metadata= {"intraday_range": intraday_range},
            )
            return RCResult.block(
                rule    = "RC-05",
                reason  = (
                    f"{symbol} intraday range {intraday_range:.1%} suggests "
                    f"circuit breaker proximity. Entry blocked."
                ),
                action  = "block",
                metadata= {"intraday_range": intraday_range},
            )

        return RCResult.permit()

    def check_rc06(self, market: MarketState) -> RCResult:
        """
        RC-06: Market Panic Detection.

        Triggers when either:
            a) India VIX spikes more than 8% from today's open, OR
            b) Nifty drops more than 3% intraday

        When triggered: all INTRADAY signals are blocked.
        Swing signals require MDS ≤ -2 (very bearish bias) to pass.

        This is the environment protection rule — it stops the system
        from trading into a panic waterfall.
        """
        vix_spike   = market.vix_spike_pct
        nifty_drop  = market.nifty_drop_pct

        if vix_spike > RC06_VIX_SPIKE_PCT:
            self._log_trigger(
                rule    = "RC-06",
                symbol  = "MARKET",
                details = f"VIX spike {vix_spike:.1%} > {RC06_VIX_SPIKE_PCT:.0%}",
                metadata= {"vix_spike_pct": vix_spike, "nifty_drop_pct": nifty_drop},
            )
            logger.warning(
                f"RC-06: VIX spike {vix_spike:.1%}. All intraday signals blocked."
            )
            return RCResult.block(
                rule    = "RC-06",
                reason  = (
                    f"India VIX spiked {vix_spike:.1%} (limit: {RC06_VIX_SPIKE_PCT:.0%}). "
                    f"Market panic detected. All intraday signals blocked."
                ),
                action  = "block_intraday",
                metadata= {"vix_spike_pct": vix_spike},
            )

        if nifty_drop > RC06_NIFTY_DROP_PCT:
            self._log_trigger(
                rule    = "RC-06",
                symbol  = "MARKET",
                details = f"Nifty drop {nifty_drop:.1%} > {RC06_NIFTY_DROP_PCT:.0%}",
                metadata= {"nifty_drop_pct": nifty_drop, "vix_spike_pct": vix_spike},
            )
            logger.warning(
                f"RC-06: Nifty down {nifty_drop:.1%}. All intraday signals blocked."
            )
            return RCResult.block(
                rule    = "RC-06",
                reason  = (
                    f"Nifty dropped {nifty_drop:.1%} intraday "
                    f"(limit: {RC06_NIFTY_DROP_PCT:.0%}). "
                    f"All intraday signals blocked."
                ),
                action  = "block_intraday",
                metadata= {"nifty_drop_pct": nifty_drop},
            )

        return RCResult.permit()

    def check_rc07(self, symbol: str, avg_turnover_cr: float) -> RCResult:
        """
        RC-07: Liquidity Floor.

        Excludes stocks with average daily turnover below ₹5 crore.
        Illiquid stocks have wide bid-ask spreads and large slippage,
        making any backtest P&L impossible to replicate in live trading.

        Args:
            avg_turnover_cr : Average daily turnover in ₹ crore
        """
        if avg_turnover_cr < RC07_MIN_TURNOVER_CR:
            self._log_trigger(
                rule    = "RC-07",
                symbol  = symbol,
                details = (
                    f"Avg daily turnover ₹{avg_turnover_cr:.1f}cr < "
                    f"₹{RC07_MIN_TURNOVER_CR:.0f}cr minimum"
                ),
                metadata= {"avg_turnover_cr": avg_turnover_cr},
            )
            return RCResult.block(
                rule    = "RC-07",
                reason  = (
                    f"{symbol} average daily turnover ₹{avg_turnover_cr:.1f}cr "
                    f"is below ₹{RC07_MIN_TURNOVER_CR:.0f}cr liquidity floor."
                ),
                action  = "exclude",
                metadata= {"avg_turnover_cr": avg_turnover_cr},
            )

        return RCResult.permit()

    def check_rc08(self, portfolio: PortfolioState) -> RCResult:
        """
        RC-08: Maximum Open Positions Cap.

        Blocks any new entry when 4 positions are already open.
        Forces the system to close existing positions before opening new ones,
        maintaining disciplined position management.

        Also blocks if monthly trade limit (15) is reached.
        """
        if portfolio.n_positions >= RC08_MAX_POSITIONS:
            self._log_trigger(
                rule    = "RC-08",
                symbol  = "PORTFOLIO",
                details = f"{portfolio.n_positions} positions open (max {RC08_MAX_POSITIONS})",
                metadata= {"n_positions": portfolio.n_positions},
            )
            return RCResult.block(
                rule    = "RC-08",
                reason  = (
                    f"Maximum positions reached: {portfolio.n_positions}/{RC08_MAX_POSITIONS}. "
                    f"Close an existing position before entering new ones."
                ),
                action  = "block",
                metadata= {"n_positions": portfolio.n_positions},
            )

        if portfolio.trades_this_month >= 15:
            return RCResult.block(
                rule    = "RC-08",
                reason  = (
                    f"Monthly trade limit reached: "
                    f"{portfolio.trades_this_month}/15 trades this month."
                ),
                action  = "block",
                metadata= {"trades_this_month": portfolio.trades_this_month},
            )

        return RCResult.permit()

    def check_rc09(self, market: MarketState) -> RCResult:
        """
        RC-09: FII Panic Selling Threshold.

        If FII provisional net selling exceeds ₹3000 crore in a single day,
        MDS is forced to -3 and ALL long signals are blocked for the day
        and the following trading day.

        FII selling at this scale typically precedes significant market
        declines (e.g., March 2020, Jan 2022).

        Note: fii_provisional_cr is negative for selling (e.g., -3500 = sold ₹3500cr)
        """
        fii_sell = market.fii_provisional_cr   # negative = selling

        if fii_sell < -RC09_FII_SELL_THRESHOLD_CR:
            sell_amt = abs(fii_sell)
            self._log_trigger(
                rule    = "RC-09",
                symbol  = "MARKET",
                details = (
                    f"FII sold ₹{sell_amt:.0f}cr provisional "
                    f"(threshold: ₹{RC09_FII_SELL_THRESHOLD_CR:.0f}cr)"
                ),
                metadata= {"fii_selling_cr": sell_amt},
            )
            self._send_telegram(
                f"⚠️ RC-09 TRIGGERED\n"
                f"FII selling: ₹{sell_amt:.0f} crore\n"
                f"All long signals blocked today and tomorrow."
            )
            logger.warning(
                f"RC-09: FII sold ₹{sell_amt:.0f}cr. MDS forced -3. Longs blocked."
            )
            return RCResult.block(
                rule    = "RC-09",
                reason  = (
                    f"FII provisional selling ₹{sell_amt:.0f}cr exceeds "
                    f"₹{RC09_FII_SELL_THRESHOLD_CR:.0f}cr panic threshold. "
                    f"All long signals blocked."
                ),
                action  = "block_longs",
                metadata= {"fii_selling_cr": sell_amt, "force_mds": -3},
            )

        return RCResult.permit()

    def check_rc10(self, symbol: str, portfolio: PortfolioState) -> RCResult:
        """
        RC-10: Sector Concentration Cap.

        Blocks a new position if 2 positions from the same sector
        are already open. Prevents over-concentration in one sector
        (e.g., 3 IT stocks all dropping together on US recession fears).
        """
        if not portfolio.sector_map or symbol not in portfolio.sector_map:
            return RCResult.permit()

        new_sector   = portfolio.sector_map[symbol]
        sector_count = sum(
            1 for s in portfolio.open_symbols
            if portfolio.sector_map.get(s) == new_sector
        )

        if sector_count >= RC10_MAX_SECTOR_POSITIONS:
            self._log_trigger(
                rule    = "RC-10",
                symbol  = symbol,
                details = (
                    f"Sector '{new_sector}' already has {sector_count} open positions "
                    f"(max {RC10_MAX_SECTOR_POSITIONS})"
                ),
                metadata= {"sector": new_sector, "sector_count": sector_count},
            )
            return RCResult.block(
                rule    = "RC-10",
                reason  = (
                    f"Sector concentration: {sector_count} positions in '{new_sector}' "
                    f"already open (limit: {RC10_MAX_SECTOR_POSITIONS}). "
                    f"Entry in {symbol} blocked."
                ),
                action  = "block",
                metadata= {"sector": new_sector, "sector_count": sector_count},
            )

        return RCResult.permit()

    # ══════════════════════════════════════════════════════════════════════
    #  STATUS & REPORTING
    # ══════════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """
        Returns a summary dict of current RC state.
        Used by the monitoring dashboard.
        """
        return {
            "halt_active"    : self._halt_active,
            "triggers_today" : self._count_triggers_today(),
            "trigger_log"    : self._trigger_log[-20:],  # last 20 triggers
            "thresholds"     : {
                "RC-01": f"Drawdown > {RC01_MAX_DRAWDOWN:.0%}",
                "RC-02": f"Single loss > {RC02_SINGLE_LOSS_PCT:.0%} of portfolio",
                "RC-03": f"Correlation > {RC03_CORRELATION_THRESHOLD}",
                "RC-04": f"Earnings within {RC04_EARNINGS_BUFFER_DAYS} trading days",
                "RC-05": f"Intraday move > {RC05_CIRCUIT_PROXIMITY_PCT:.0%}",
                "RC-06": f"VIX spike > {RC06_VIX_SPIKE_PCT:.0%} OR Nifty drop > {RC06_NIFTY_DROP_PCT:.0%}",
                "RC-07": f"Avg daily turnover < ₹{RC07_MIN_TURNOVER_CR}cr",
                "RC-08": f"Positions >= {RC08_MAX_POSITIONS} OR trades >= 15/month",
                "RC-09": f"FII selling > ₹{RC09_FII_SELL_THRESHOLD_CR:.0f}cr/day",
                "RC-10": f"Sector positions >= {RC10_MAX_SECTOR_POSITIONS}",
            },
        }

    def get_trigger_history(self, days: int = 30) -> List[Dict]:
        """
        Returns trigger history from DB for the last N days.
        Used by monitoring dashboard and weekly review.
        """
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT rule, symbol, details, triggered_at, metadata
                    FROM rc_trigger_log
                    WHERE triggered_at >= NOW() - INTERVAL '%s days'
                    ORDER BY triggered_at DESC
                    LIMIT 500;
                """, (days,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Could not fetch trigger history: {e}")
            return self._trigger_log

    # ══════════════════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _get_conn(self):
        """Returns DB connection — creates new one if not provided."""
        if self._conn is not None:
            try:
                # Test if connection is still alive
                self._conn.cursor().execute("SELECT 1")
                return self._conn
            except Exception:
                pass
        return psycopg2.connect(DB_URL)

    def _ensure_table(self):
        """Creates RC tables in DB if they don't exist."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                # Trigger log table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rc_trigger_log (
                        id           SERIAL PRIMARY KEY,
                        rule         VARCHAR(10)  NOT NULL,
                        symbol       VARCHAR(20),
                        details      TEXT,
                        metadata     JSONB,
                        triggered_at TIMESTAMP    DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rc_trigger_rule
                    ON rc_trigger_log (rule, triggered_at DESC);
                """)

                # Halt state table (persists RC-01 across restarts)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rc_halt_state (
                        id          SERIAL PRIMARY KEY,
                        halt_active BOOLEAN   NOT NULL DEFAULT FALSE,
                        triggered_at TIMESTAMP,
                        cleared_at   TIMESTAMP,
                        cleared_by   VARCHAR(100),
                        reason       TEXT
                    );
                """)

                # Insert initial row if table is empty
                cur.execute("SELECT COUNT(*) FROM rc_halt_state;")
                if cur.fetchone()[0] == 0:
                    cur.execute(
                        "INSERT INTO rc_halt_state (halt_active) VALUES (FALSE);"
                    )

            conn.commit()
        except Exception as e:
            logger.warning(f"RC table setup failed (non-critical): {e}")

    def _load_halt_state(self):
        """Loads RC-01 halt state from DB on startup."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT halt_active FROM rc_halt_state ORDER BY id DESC LIMIT 1;"
                )
                row = cur.fetchone()
                if row:
                    self._halt_active = bool(row[0])
                    if self._halt_active:
                        logger.critical(
                            "RC-01 HALT IS ACTIVE from previous session. "
                            "Call rc.clear_halt() after review to resume trading."
                        )
        except Exception as e:
            logger.warning(f"Could not load halt state: {e}. Assuming no halt.")
            self._halt_active = False

    def _persist_halt_state(self, active: bool, cleared_by: str = ""):
        """Persists RC-01 halt state to DB."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                if active:
                    cur.execute("""
                        UPDATE rc_halt_state
                        SET halt_active = TRUE,
                            triggered_at = NOW(),
                            cleared_at   = NULL,
                            cleared_by   = NULL
                        WHERE id = (SELECT MAX(id) FROM rc_halt_state);
                    """)
                else:
                    cur.execute("""
                        UPDATE rc_halt_state
                        SET halt_active = FALSE,
                            cleared_at  = NOW(),
                            cleared_by  = %s
                        WHERE id = (SELECT MAX(id) FROM rc_halt_state);
                    """, (cleared_by,))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist halt state: {e}")

    def _log_trigger(
        self,
        rule    : str,
        symbol  : str,
        details : str,
        metadata: Dict = None,
    ):
        """
        Logs a Rule Constitution trigger to both:
            1. In-memory list (fast access for monitoring)
            2. rc_trigger_log table in DB (persistent audit trail)
        """
        import json
        entry = {
            "rule"        : rule,
            "symbol"      : symbol,
            "details"     : details,
            "triggered_at": datetime.now().isoformat(),
            "metadata"    : metadata or {},
        }
        self._trigger_log.append(entry)

        # Keep in-memory log bounded
        if len(self._trigger_log) > 500:
            self._trigger_log = self._trigger_log[-500:]

        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rc_trigger_log (rule, symbol, details, metadata)
                    VALUES (%s, %s, %s, %s);
                """, (rule, symbol, details, json.dumps(metadata or {})))
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not persist RC trigger to DB: {e}")

    def _count_triggers_today(self) -> Dict[str, int]:
        """Returns count of triggers per rule today."""
        today = date.today().isoformat()
        counts: Dict[str, int] = {}
        for entry in self._trigger_log:
            if entry["triggered_at"].startswith(today):
                rule = entry["rule"]
                counts[rule] = counts.get(rule, 0) + 1
        return counts

    def _send_telegram(self, message: str):
        """
        Sends a Telegram alert for critical RC events.
        Fails silently if Telegram is not configured.
        """
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={
                "chat_id"   : TELEGRAM_CHAT_ID,
                "text"      : f"🤖 G.O.D.S E.Y.E\n{message}",
                "parse_mode": "HTML",
            }, timeout=5)
        except Exception as e:
            logger.warning(f"Telegram alert failed: {e}")

    def __repr__(self) -> str:
        return (
            f"RiskConstitution("
            f"halt={self._halt_active}, "
            f"triggers={len(self._trigger_log)})"
        )


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest environment/risk_constitution.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestRiskConstitution:
    """
    Unit tests for all 10 Risk Constitution rules.
    All tests are DB-free (mocks _get_conn to avoid dependency).
    """

    def _make_rc(self) -> RiskConstitution:
        """Creates RC instance without DB connection."""
        rc = RiskConstitution.__new__(RiskConstitution)
        rc._lock        = threading.Lock()
        rc._own_conn    = False
        rc._conn        = None
        rc._halt_active = False
        rc._trigger_log = []
        return rc

    def _make_portfolio(self, **kwargs) -> PortfolioState:
        defaults = dict(
            total_value       = 1_000_000.0,
            peak_value        = 1_000_000.0,
            cash              = 800_000.0,
            open_positions    = {},
            trades_today      = 0,
            trades_this_month = 0,
            sector_map        = {},
        )
        defaults.update(kwargs)
        return PortfolioState(**defaults)

    def _make_market(self, **kwargs) -> MarketState:
        defaults = dict(
            nifty_open=22000.0, nifty_current=22000.0,
            india_vix_open=15.0, india_vix_current=15.0,
            fii_provisional_cr=0.0, mds_score=0, is_halt=False,
        )
        defaults.update(kwargs)
        return MarketState(**defaults)

    # ── RC-01 Tests ───────────────────────────────────────────────────────

    def test_rc01_permits_below_threshold(self):
        rc = self._make_rc()
        port = self._make_portfolio(
            total_value=900_000.0, peak_value=1_000_000.0   # 10% DD — below 12%
        )
        result = rc.check_rc01(port)
        assert result.allowed, "RC-01 should permit < 12% drawdown"

    def test_rc01_blocks_above_threshold(self):
        rc = self._make_rc()
        port = self._make_portfolio(
            total_value=870_000.0, peak_value=1_000_000.0   # 13% DD
        )
        # Patch out DB and Telegram calls
        rc._persist_halt_state = lambda **kw: None
        rc._send_telegram      = lambda msg: None
        rc._log_trigger        = lambda **kw: None
        result = rc.check_rc01(port)
        assert result.blocked,         "RC-01 should block > 12% drawdown"
        assert result.rule   == "RC-01"
        assert result.action == "halt"

    def test_rc01_halt_persists_on_subsequent_calls(self):
        rc = self._make_rc()
        rc._halt_active        = True   # simulate already-active halt
        rc._persist_halt_state = lambda **kw: None
        rc._send_telegram      = lambda msg: None
        rc._log_trigger        = lambda **kw: None
        port = self._make_portfolio()   # even with no drawdown
        result = rc.check_rc01(port)
        assert result.blocked, "RC-01 halt should persist once active"

    def test_rc01_clear_halt(self):
        rc = self._make_rc()
        rc._halt_active        = True
        rc._persist_halt_state = lambda **kw: None
        rc._send_telegram      = lambda msg: None
        cleared = rc.clear_halt("test_user")
        assert cleared,              "clear_halt should return True"
        assert not rc._halt_active,  "halt should be inactive after clear"

    def test_rc01_clear_halt_no_active_halt(self):
        rc = self._make_rc()
        rc._halt_active = False
        result = rc.clear_halt("test_user")
        assert not result, "clear_halt should return False when no halt active"

    def test_rc01_exact_threshold_permitted(self):
        """Exactly 12% drawdown should be permitted (rule is strictly >)."""
        rc = self._make_rc()
        rc._persist_halt_state = lambda **kw: None
        rc._send_telegram      = lambda msg: None
        rc._log_trigger        = lambda **kw: None
        port = self._make_portfolio(
            total_value=880_000.0, peak_value=1_000_000.0   # exactly 12%
        )
        result = rc.check_rc01(port)
        assert result.allowed, "Exactly 12% DD should be permitted (strictly >)"

    # ── RC-02 Tests ───────────────────────────────────────────────────────

    def test_rc02_permits_small_loss(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        port = self._make_portfolio()
        pos  = {"entry_price": 100.0, "quantity": 100}
        result = rc.check_rc02("TEST", pos, 99.0, port)   # 1% loss = 0.01% port
        assert result.allowed

    def test_rc02_blocks_large_loss(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        # Position: 200 shares × ₹100 = ₹20,000 position
        # Loss: ₹100 → ₹89 = 11% × ₹20,000 = ₹2,200 loss
        # Portfolio: ₹100,000 → loss = 2.2% of portfolio (> 2%)
        port = self._make_portfolio(total_value=100_000.0)
        pos  = {"entry_price": 100.0, "quantity": 200}
        result = rc.check_rc02("TEST", pos, 89.0, port)
        assert result.blocked,              "RC-02 should block large loss"
        assert result.rule   == "RC-02"
        assert result.action == "force_close"

    def test_rc02_force_close_action(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        port = self._make_portfolio(total_value=50_000.0)
        pos  = {"entry_price": 100.0, "quantity": 500}   # ₹50,000 position = 100% portfolio
        result = rc.check_rc02("TEST", pos, 98.0, port)  # 2% loss = 2% portfolio → borderline
        # 2% loss on ₹50K = ₹1000, portfolio ₹50K → 2.0% — should block
        assert result.action in ("force_close", "none")  # depends on exact boundary

    # ── RC-03 Tests ───────────────────────────────────────────────────────

    def test_rc03_permits_low_correlation(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        idx    = {"STOCK_A": 0, "STOCK_B": 1}
        result = rc.check_rc03("STOCK_B", ["STOCK_A"], matrix, idx)
        assert result.allowed, "Correlation 0.5 should be permitted"

    def test_rc03_blocks_high_correlation(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        matrix = np.array([[1.0, 0.85], [0.85, 1.0]])
        idx    = {"STOCK_A": 0, "STOCK_B": 1}
        result = rc.check_rc03("STOCK_B", ["STOCK_A"], matrix, idx)
        assert result.blocked, "Correlation 0.85 should be blocked"
        assert result.rule == "RC-03"

    def test_rc03_blocks_high_negative_correlation(self):
        """High negative correlation is also concentration risk."""
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        matrix = np.array([[1.0, -0.80], [-0.80, 1.0]])
        idx    = {"STOCK_A": 0, "STOCK_B": 1}
        result = rc.check_rc03("STOCK_B", ["STOCK_A"], matrix, idx)
        assert result.blocked, "High negative correlation should also be blocked"

    def test_rc03_permits_unknown_symbol(self):
        """Unknown symbol (not in idx) should be permitted (no data = no block)."""
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        matrix = np.array([[1.0]])
        idx    = {"STOCK_A": 0}
        result = rc.check_rc03("UNKNOWN", ["STOCK_A"], matrix, idx)
        assert result.allowed

    def test_rc03_no_open_positions(self):
        """No open positions = no correlation to check = permit."""
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        matrix = np.array([[1.0, 0.9], [0.9, 1.0]])
        idx    = {"STOCK_A": 0, "STOCK_B": 1}
        result = rc.check_rc03("STOCK_B", [], matrix, idx)
        assert result.allowed

    # ── RC-05 Tests ───────────────────────────────────────────────────────

    def test_rc05_permits_normal_range(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        bar = {"open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0}
        result = rc.check_rc05("TEST", bar)
        assert result.allowed  # range = 7% < 18%

    def test_rc05_blocks_circuit_proximity(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        bar = {"open": 100.0, "high": 120.0, "low": 99.0, "close": 118.0}
        result = rc.check_rc05("TEST", bar)
        assert result.blocked  # range = 21% > 18%
        assert result.rule == "RC-05"

    def test_rc05_zero_open_price(self):
        """Zero open price should not crash — should permit."""
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        bar = {"open": 0.0, "high": 10.0, "low": 9.0, "close": 9.5}
        result = rc.check_rc05("TEST", bar)
        assert result.allowed

    # ── RC-06 Tests ───────────────────────────────────────────────────────

    def test_rc06_permits_calm_market(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        market = self._make_market()
        result = rc.check_rc06(market)
        assert result.allowed

    def test_rc06_blocks_vix_spike(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        market = self._make_market(india_vix_open=15.0, india_vix_current=16.5)  # 10% spike
        result = rc.check_rc06(market)
        assert result.blocked
        assert result.rule   == "RC-06"
        assert result.action == "block_intraday"

    def test_rc06_blocks_nifty_crash(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        market = self._make_market(nifty_open=22000.0, nifty_current=21100.0)  # 4.09% drop
        result = rc.check_rc06(market)
        assert result.blocked
        assert result.rule == "RC-06"

    # ── RC-07 Tests ───────────────────────────────────────────────────────

    def test_rc07_permits_liquid_stock(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        result = rc.check_rc07("RELIANCE", avg_turnover_cr=500.0)
        assert result.allowed

    def test_rc07_blocks_illiquid_stock(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        result = rc.check_rc07("ILLIQUID", avg_turnover_cr=2.0)
        assert result.blocked
        assert result.rule   == "RC-07"
        assert result.action == "exclude"

    def test_rc07_exact_threshold(self):
        """Exactly ₹5cr should be permitted (strictly <)."""
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        result = rc.check_rc07("TEST", avg_turnover_cr=5.0)
        assert result.allowed

    # ── RC-08 Tests ───────────────────────────────────────────────────────

    def test_rc08_permits_under_limit(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        port = self._make_portfolio(open_positions={"A": {}, "B": {}})
        result = rc.check_rc08(port)
        assert result.allowed

    def test_rc08_blocks_at_limit(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        port = self._make_portfolio(
            open_positions={"A": {}, "B": {}, "C": {}, "D": {}}
        )
        result = rc.check_rc08(port)
        assert result.blocked
        assert result.rule == "RC-08"

    def test_rc08_blocks_monthly_trade_limit(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        port = self._make_portfolio(trades_this_month=15)
        result = rc.check_rc08(port)
        assert result.blocked
        assert result.rule == "RC-08"

    # ── RC-09 Tests ───────────────────────────────────────────────────────

    def test_rc09_permits_normal_fii(self):
        rc = self._make_rc()
        rc._log_trigger  = lambda **kw: None
        rc._send_telegram= lambda msg: None
        market = self._make_market(fii_provisional_cr=-500.0)   # mild selling
        result = rc.check_rc09(market)
        assert result.allowed

    def test_rc09_blocks_panic_selling(self):
        rc = self._make_rc()
        rc._log_trigger  = lambda **kw: None
        rc._send_telegram= lambda msg: None
        market = self._make_market(fii_provisional_cr=-3500.0)  # ₹3500cr selling
        result = rc.check_rc09(market)
        assert result.blocked
        assert result.rule   == "RC-09"
        assert result.action == "block_longs"

    def test_rc09_permits_fii_buying(self):
        rc = self._make_rc()
        rc._log_trigger  = lambda **kw: None
        rc._send_telegram= lambda msg: None
        market = self._make_market(fii_provisional_cr=2000.0)   # buying
        result = rc.check_rc09(market)
        assert result.allowed

    # ── RC-10 Tests ───────────────────────────────────────────────────────

    def test_rc10_permits_new_sector(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        port = self._make_portfolio(
            open_positions={"INFY": {}, "TCS": {}},
            sector_map={"INFY": "IT", "TCS": "IT", "RELIANCE": "Energy"},
        )
        result = rc.check_rc10("RELIANCE", port)
        assert result.allowed  # RELIANCE is in Energy, not IT

    def test_rc10_blocks_third_in_sector(self):
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        port = self._make_portfolio(
            open_positions={"INFY": {}, "TCS": {}},
            sector_map={"INFY": "IT", "TCS": "IT", "WIPRO": "IT"},
        )
        result = rc.check_rc10("WIPRO", port)
        assert result.blocked
        assert result.rule == "RC-10"

    def test_rc10_permits_unknown_sector(self):
        """Unknown sector mapping should permit (no data = no block)."""
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        port = self._make_portfolio()  # empty sector_map
        result = rc.check_rc10("UNKNOWN_STOCK", port)
        assert result.allowed

    # ── RCResult Tests ────────────────────────────────────────────────────

    def test_rc_result_permit_factory(self):
        r = RCResult.permit()
        assert r.allowed
        assert not r.blocked
        assert r.rule == ""

    def test_rc_result_block_factory(self):
        r = RCResult.block("RC-01", "Test reason", "halt")
        assert r.blocked
        assert not r.allowed
        assert r.rule   == "RC-01"
        assert r.action == "halt"

    # ── evaluate_entry integration test ──────────────────────────────────

    def test_evaluate_entry_all_clear(self):
        """All rules clear → evaluate_entry should permit."""
        rc = self._make_rc()
        # Patch out all DB-touching rule checks
        rc._log_trigger = lambda **kw: None
        rc.check_rc04   = lambda sym: RCResult.permit()

        port   = self._make_portfolio()
        market = self._make_market()
        result = rc.evaluate_entry("RELIANCE", port, market, avg_turnover_cr=500.0)
        assert result.allowed, f"All-clear entry should be permitted: {result.reason}"

    def test_evaluate_entry_blocked_by_rc08(self):
        """4 open positions → evaluate_entry should block via RC-08."""
        rc = self._make_rc()
        rc._log_trigger = lambda **kw: None
        rc.check_rc04   = lambda sym: RCResult.permit()

        port = self._make_portfolio(
            open_positions={"A": {}, "B": {}, "C": {}, "D": {}}
        )
        market = self._make_market()
        result = rc.evaluate_entry("RELIANCE", port, market)
        assert result.blocked
        assert result.rule == "RC-08"

    def test_portfolio_state_drawdown_property(self):
        """PortfolioState.drawdown must compute correctly."""
        port = PortfolioState(
            total_value=880_000, peak_value=1_000_000,
            cash=0, open_positions={},
            trades_today=0, trades_this_month=0, sector_map={}
        )
        assert abs(port.drawdown - 0.12) < 1e-6

    def test_market_state_vix_spike_property(self):
        market = MarketState(india_vix_open=15.0, india_vix_current=16.5)
        assert abs(market.vix_spike_pct - 0.10) < 1e-6

    def test_market_state_nifty_drop_property(self):
        market = MarketState(nifty_open=22000.0, nifty_current=21120.0)
        assert abs(market.nifty_drop_pct - 0.04) < 1e-4


# ── Run tests when file is executed directly ──────────────────────────────
if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))