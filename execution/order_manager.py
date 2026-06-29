"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Order & Position Manager                        ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : execution/order_manager.py                             ║
║         Phase   : 4 — Paper Trading & Live Monitoring                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import threading
import psycopg2

from dataclasses  import dataclass, field
from datetime     import datetime
from enum         import Enum, auto
from typing       import Optional, Dict, List, Tuple, Any
from loguru       import logger
from dotenv       import load_dotenv

from execution.upstox_executor import (
    OrderRequest, OrderResult, OrderSide,
    OrderType, OrderStatus,
)
from environment.risk_constitution import (
    RiskConstitution, PortfolioState, MarketState, RCResult,
)
from environment.position_sizer import (
    PositionSizer, SizingInput, build_position_sizer,
)

load_dotenv()

DB_URL           = os.getenv("TIMESCALE_URL", "postgresql://godseye_user:godseye_pass@localhost:5433/godseye")
MAX_POSITIONS    = 4
MAX_TRADES_PER_MONTH = 15
MAX_HOLD_DAYS_SWING  = 15
INITIAL_CAPITAL  = float(os.getenv("INITIAL_CAPITAL", "1000000"))
TRAIL_ACTIVATE_PCT   = 0.020
TRAIL_DISTANCE_PCT   = 0.008


class PositionState(Enum):
    OPEN             = auto()
    CLOSED_TP        = auto()
    CLOSED_SL        = auto()
    CLOSED_TRAIL     = auto()
    CLOSED_RC        = auto()
    CLOSED_MANUAL    = auto()
    CLOSED_EOD       = auto()
    CLOSED_MAX_HOLD  = auto()


class SignalMode(Enum):
    SWING    = "swing"
    INTRADAY = "intraday"


@dataclass
class TradeSignal:
    symbol           : str
    side             : OrderSide
    mode             : SignalMode
    confidence_score : float
    entry_price      : float
    tp_price         : float
    sl_price         : float
    atr_pct          : float    = 0.02
    vol_regime       : str      = "normal"
    mds_score        : int      = 0
    timestamp        : datetime = field(default_factory=datetime.now)
    signal_id        : str      = ""

    def __post_init__(self):
        if not self.signal_id:
            import uuid
            self.signal_id = f"SIG-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ManagedPosition:
    symbol         : str
    signal_id      : str
    order_id       : str
    mode           : SignalMode
    entry_price    : float
    entry_time     : datetime
    quantity       : int
    position_value : float
    tp_price       : float
    sl_price       : float
    trail_active   : bool           = False
    trail_peak     : float          = 0.0
    trail_stop     : float          = 0.0
    state          : PositionState  = PositionState.OPEN
    hold_days      : int            = 0
    current_price  : float          = 0.0
    unrealised_pnl : float          = 0.0
    unrealised_pct : float          = 0.0
    exit_price     : float          = 0.0
    exit_time      : Optional[datetime] = None
    realised_pnl   : float          = 0.0

    def update_price(self, price: float):
        self.current_price  = price
        self.unrealised_pnl = (price - self.entry_price) * self.quantity
        self.unrealised_pct = (price - self.entry_price) / self.entry_price

    def update_trailing_stop(self, price: float) -> bool:
        gain = (price - self.entry_price) / self.entry_price
        if gain >= TRAIL_ACTIVATE_PCT and not self.trail_active:
            self.trail_active = True
            self.trail_peak   = price
            self.trail_stop   = price * (1 - TRAIL_DISTANCE_PCT)
        if self.trail_active:
            if price > self.trail_peak:
                self.trail_peak = price
                self.trail_stop = price * (1 - TRAIL_DISTANCE_PCT)
            if price <= self.trail_stop:
                return True
        return False

    def should_exit(self, price: float) -> Tuple[bool, PositionState]:
        if price >= self.tp_price:
            return True, PositionState.CLOSED_TP
        if price <= self.sl_price:
            return True, PositionState.CLOSED_SL
        if self.update_trailing_stop(price):
            return True, PositionState.CLOSED_TRAIL
        if self.mode == SignalMode.SWING and self.hold_days >= MAX_HOLD_DAYS_SWING:
            return True, PositionState.CLOSED_MAX_HOLD
        return False, PositionState.OPEN

    @property
    def is_open(self) -> bool:
        return self.state == PositionState.OPEN

    @property
    def sl_pct(self) -> float:
        return (self.entry_price - self.sl_price) / self.entry_price

    @property
    def risk_contribution(self) -> float:
        return self.sl_pct * (self.position_value / INITIAL_CAPITAL)


@dataclass
class PortfolioSnapshot:
    total_value       : float
    cash              : float
    peak_value        : float
    drawdown          : float
    open_positions    : Dict[str, ManagedPosition]
    n_positions       : int
    trades_today      : int
    trades_this_month : int
    total_trades      : int
    unrealised_pnl    : float
    realised_pnl_today: float
    portfolio_heat    : float
    timestamp         : datetime = field(default_factory=datetime.now)


class OrderManager:
    """
    Manages the full trade lifecycle for G.O.D.S E.Y.E.
    Receives signals, validates via RC, sizes via Kelly,
    places orders via BrokerRouter, tracks positions, and exits.
    """

    def __init__(
        self,
        broker          : Any,
        initial_capital : float = INITIAL_CAPITAL,
        paper_mode      : bool  = True,
        persist_to_db   : bool  = True,
        rc              : Optional[RiskConstitution] = None,
        sizer           : Optional[PositionSizer]    = None,
    ):
        self.broker          = broker
        self.initial_capital = initial_capital
        self.paper_mode      = paper_mode
        self.persist_to_db   = persist_to_db
        self.rc              = rc    or RiskConstitution()
        self.sizer           = sizer or build_position_sizer()

        self._positions          : Dict[str, ManagedPosition] = {}
        self._cash               = initial_capital
        self._peak_value         = initial_capital
        self._lock               = threading.RLock()
        self._trades_today       = 0
        self._trades_this_month  = 0
        self._total_trades       = 0
        self._realised_pnl_today = 0.0
        self._sector_map         : Dict[str, str]   = {}
        self._last_prices        : Dict[str, float] = {}
        self._running            = False
        self._db_conn            = None

        logger.info(
            f"OrderManager initialized | "
            f"capital=₹{initial_capital:,.0f} | paper={paper_mode}"
        )

    def start(self):
        if self._running:
            return
        self._running = True
        if self.persist_to_db:
            self._ensure_db_tables()
        logger.info("OrderManager started.")

    def stop(self):
        self._running = False
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass
        logger.info("OrderManager stopped.")

    def set_sector_map(self, sector_map: Dict[str, str]):
        self._sector_map = sector_map

    def process_signal(
        self,
        signal       : TradeSignal,
        market_state : MarketState,
        corr_matrix  = None,
        symbol_idx   : Optional[Dict[str, int]] = None,
    ) -> Tuple[bool, str]:
        with self._lock:
            portfolio = self._build_portfolio_state()

            rc_result = self.rc.evaluate_entry(
                symbol          = signal.symbol,
                portfolio       = portfolio,
                market          = market_state,
                corr_matrix     = corr_matrix,
                symbol_idx      = symbol_idx,
                avg_turnover_cr = 100.0,
            )
            if rc_result.blocked:
                logger.info(f"Signal blocked by {rc_result.rule}: {rc_result.reason}")
                return False, f"{rc_result.rule}: {rc_result.reason}"

            sizing = self.sizer.compute(SizingInput(
                symbol           = signal.symbol,
                entry_price      = signal.entry_price,
                tp_price         = signal.tp_price,
                sl_price         = signal.sl_price,
                confidence_score = signal.confidence_score,
                portfolio_value  = portfolio.total_value,
                available_cash   = portfolio.cash,
                current_drawdown = portfolio.drawdown,
                open_risk_pct    = self._portfolio_heat(),
                mds_score        = signal.mds_score,
                vol_regime       = signal.vol_regime,
                atr_pct          = signal.atr_pct,
            ))
            if sizing.is_zero:
                logger.info(f"Signal sized to zero: {sizing.reason}")
                return False, f"Sizing: {sizing.reason}"

            order_req = OrderRequest(
                symbol     = signal.symbol,
                side       = signal.side,
                quantity   = sizing.quantity,
                order_type = OrderType.MARKET,
                price      = signal.entry_price,
                tag        = signal.signal_id,
            )
            result = self.broker.place_order(order_req)
            if not result.is_filled:
                logger.warning(f"Order not filled for {signal.symbol}: {result.error_message}")
                return False, f"Order rejected: {result.error_message}"

            fill_price = result.fill_price or signal.entry_price
            position   = ManagedPosition(
                symbol         = signal.symbol,
                signal_id      = signal.signal_id,
                order_id       = result.order_id,
                mode           = signal.mode,
                entry_price    = fill_price,
                entry_time     = datetime.now(),
                quantity       = result.fill_quantity,
                position_value = fill_price * result.fill_quantity,
                tp_price       = signal.tp_price,
                sl_price       = signal.sl_price,
                trail_peak     = fill_price,
                trail_stop     = signal.sl_price,
                current_price  = fill_price,
            )

            self._positions[signal.symbol] = position
            self._cash              -= position.position_value
            self._trades_today      += 1
            self._trades_this_month += 1
            self._total_trades      += 1

            logger.success(
                f"OPENED {signal.symbol} | "
                f"qty={position.quantity} @ ₹{fill_price:.2f} | "
                f"TP=₹{signal.tp_price:.2f} SL=₹{signal.sl_price:.2f}"
            )

            if self.persist_to_db:
                self._persist_trade_open(position, signal)

            return True, f"Opened {signal.symbol} qty={position.quantity}"

    def on_price_update(self, prices: Dict[str, float]):
        with self._lock:
            self._last_prices.update(prices)
            to_close: List[Tuple[str, PositionState]] = []

            for symbol, pos in self._positions.items():
                if not pos.is_open:
                    continue
                price = prices.get(symbol)
                if price is None or price <= 0:
                    continue

                pos.update_price(price)

                rc_result = self.rc.check_rc02(
                    symbol        = symbol,
                    position      = {"entry_price": pos.entry_price, "quantity": pos.quantity},
                    current_price = price,
                    portfolio     = self._build_portfolio_state(),
                )
                if rc_result.blocked and rc_result.action == "force_close":
                    to_close.append((symbol, PositionState.CLOSED_RC))
                    continue

                should_exit, reason = pos.should_exit(price)
                if should_exit:
                    to_close.append((symbol, reason))

            for symbol, reason in to_close:
                self._close_position(symbol, reason, prices.get(symbol, 0))

    def close_position_manual(self, symbol: str) -> bool:
        with self._lock:
            if symbol not in self._positions:
                logger.warning(f"Manual close: {symbol} not in positions.")
                return False
            price = self._last_prices.get(symbol, self._positions[symbol].entry_price)
            self._close_position(symbol, PositionState.CLOSED_MANUAL, price)
            return True

    def close_all_positions(self, reason: str = "manual"):
        with self._lock:
            symbols = list(self._positions.keys())
            for sym in symbols:
                price = self._last_prices.get(sym, self._positions[sym].entry_price)
                self._close_position(sym, PositionState.CLOSED_MANUAL, price)
            logger.info(f"All positions closed. Reason: {reason}")

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        with self._lock:
            pv        = self._compute_portfolio_value(self._last_prices)
            unrealised= sum(p.unrealised_pnl for p in self._positions.values())
            drawdown  = (self._peak_value - pv) / self._peak_value if self._peak_value > 0 else 0
            return PortfolioSnapshot(
                total_value       = pv,
                cash              = self._cash,
                peak_value        = self._peak_value,
                drawdown          = max(0.0, drawdown),
                open_positions    = dict(self._positions),
                n_positions       = len(self._positions),
                trades_today      = self._trades_today,
                trades_this_month = self._trades_this_month,
                total_trades      = self._total_trades,
                unrealised_pnl    = unrealised,
                realised_pnl_today= self._realised_pnl_today,
                portfolio_heat    = self._portfolio_heat(),
            )

    def get_open_positions(self) -> Dict[str, ManagedPosition]:
        with self._lock:
            return {k: v for k, v in self._positions.items() if v.is_open}

    def get_position(self, symbol: str) -> Optional[ManagedPosition]:
        with self._lock:
            return self._positions.get(symbol)

    def is_position_open(self, symbol: str) -> bool:
        with self._lock:
            pos = self._positions.get(symbol)
            return pos is not None and pos.is_open

    def increment_hold_days(self):
        with self._lock:
            for pos in self._positions.values():
                if pos.is_open and pos.mode == SignalMode.SWING:
                    pos.hold_days += 1

    def reset_daily_counters(self):
        with self._lock:
            self._trades_today       = 0
            self._realised_pnl_today = 0.0
            logger.info("Daily counters reset.")

    def reset_monthly_counters(self):
        with self._lock:
            self._trades_this_month = 0
            logger.info("Monthly trade counter reset.")

    def _close_position(self, symbol: str, reason: PositionState, exit_price: float):
        pos = self._positions.get(symbol)
        if pos is None or not pos.is_open:
            return

        exit_req = OrderRequest(
            symbol     = symbol,
            side       = OrderSide.SELL,
            quantity   = pos.quantity,
            order_type = OrderType.MARKET,
            price      = exit_price,
            tag        = f"EXIT-{pos.signal_id}",
        )
        result      = self.broker.place_order(exit_req)
        actual_exit = result.fill_price if result.is_filled else exit_price

        gross_proceeds   = actual_exit * pos.quantity
        realised_pnl     = gross_proceeds - pos.position_value

        pos.state        = reason
        pos.exit_price   = actual_exit
        pos.exit_time    = datetime.now()
        pos.realised_pnl = realised_pnl

        self._cash               += gross_proceeds
        self._realised_pnl_today += realised_pnl

        pv = self._compute_portfolio_value(self._last_prices)
        if pv > self._peak_value:
            self._peak_value = pv

        logger.success(
            f"CLOSED {symbol} | exit=₹{actual_exit:.2f} | "
            f"pnl=₹{realised_pnl:+,.2f} ({pos.unrealised_pct:+.2%}) | "
            f"reason={reason.name}"
        )
        if self.persist_to_db:
            self._persist_trade_close(pos)

    def _compute_portfolio_value(self, prices: Dict[str, float]) -> float:
        mtm = sum(
            pos.quantity * prices.get(sym, pos.entry_price)
            for sym, pos in self._positions.items()
            if pos.is_open
        )
        return self._cash + mtm

    def _portfolio_heat(self) -> float:
        total_value = self._cash + sum(
            pos.position_value for pos in self._positions.values() if pos.is_open
        )
        if total_value <= 0:
            return 0.0
        heat = sum(
            pos.sl_pct * (pos.position_value / total_value)
            for pos in self._positions.values() if pos.is_open
        )
        return min(heat, 1.0)

    def _build_portfolio_state(self) -> PortfolioState:
        pv       = self._compute_portfolio_value(self._last_prices)
        open_pos = {
            sym: {"entry_price": pos.entry_price, "quantity": pos.quantity, "sl_price": pos.sl_price}
            for sym, pos in self._positions.items() if pos.is_open
        }
        return PortfolioState(
            total_value       = pv,
            peak_value        = self._peak_value,
            cash              = self._cash,
            open_positions    = open_pos,
            trades_today      = self._trades_today,
            trades_this_month = self._trades_this_month,
            sector_map        = self._sector_map,
        )

    def _get_db_conn(self):
        if self._db_conn is None or self._db_conn.closed:
            self._db_conn = psycopg2.connect(DB_URL)
        return self._db_conn

    def _ensure_db_tables(self):
        try:
            conn = self._get_db_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_log (
                        id               SERIAL PRIMARY KEY,
                        signal_id        VARCHAR(64),
                        order_id         VARCHAR(64),
                        symbol           VARCHAR(20) NOT NULL,
                        mode             VARCHAR(10),
                        side             VARCHAR(5),
                        entry_price      NUMERIC(12,4),
                        exit_price       NUMERIC(12,4),
                        quantity         INTEGER,
                        position_value   NUMERIC(14,2),
                        tp_price         NUMERIC(12,4),
                        sl_price         NUMERIC(12,4),
                        entry_time       TIMESTAMP,
                        exit_time        TIMESTAMP,
                        hold_days        INTEGER     DEFAULT 0,
                        realised_pnl     NUMERIC(14,2),
                        exit_reason      VARCHAR(30),
                        confidence_score NUMERIC(5,4),
                        paper_mode       BOOLEAN     DEFAULT TRUE,
                        created_at       TIMESTAMP   DEFAULT NOW()
                    );
                """)
            conn.commit()
            logger.info("trade_log table ready.")
        except Exception as e:
            logger.warning(f"DB table setup failed (non-critical): {e}")

    def _persist_trade_open(self, pos: ManagedPosition, signal: TradeSignal):
        try:
            conn = self._get_db_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trade_log (
                        signal_id, order_id, symbol, mode, side,
                        entry_price, quantity, position_value,
                        tp_price, sl_price, entry_time,
                        confidence_score, paper_mode
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    pos.signal_id, pos.order_id, pos.symbol,
                    pos.mode.value, "BUY",
                    pos.entry_price, pos.quantity, pos.position_value,
                    pos.tp_price, pos.sl_price, pos.entry_time,
                    signal.confidence_score, self.paper_mode,
                ))
            conn.commit()
        except Exception as e:
            logger.warning(f"persist_trade_open failed: {e}")

    def _persist_trade_close(self, pos: ManagedPosition):
        try:
            conn = self._get_db_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trade_log
                    SET exit_price   = %s,
                        exit_time    = %s,
                        hold_days    = %s,
                        realised_pnl = %s,
                        exit_reason  = %s
                    WHERE signal_id  = %s AND exit_time IS NULL
                """, (
                    pos.exit_price, pos.exit_time,
                    pos.hold_days, pos.realised_pnl,
                    pos.state.name, pos.signal_id,
                ))
            conn.commit()
        except Exception as e:
            logger.warning(f"persist_trade_close failed: {e}")

    def __repr__(self) -> str:
        return (
            f"OrderManager(positions={len(self._positions)}, "
            f"cash=₹{self._cash:,.0f}, "
            f"trades={self._total_trades}, "
            f"paper={self.paper_mode})"
        )


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest execution/order_manager.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestOrderManager:

    class _MockBroker:
        def __init__(self, healthy=True):
            self._healthy = healthy
            self._orders  = []
        def place_order(self, req):
            status = OrderStatus.PAPER_FILL if self._healthy else OrderStatus.REJECTED
            r = OrderResult(
                order_id=f"MOCK-{len(self._orders)+1:04d}",
                symbol=req.symbol, side=req.side, quantity=req.quantity,
                status=status, fill_price=req.price or 100.0,
                fill_quantity=req.quantity if self._healthy else 0,
                broker="mock", paper_mode=True,
            )
            self._orders.append(r)
            return r
        def cancel_order(self, oid): return True
        def get_positions(self): return []
        def get_funds(self): return {"available_cash": 1_000_000.0}

    class _MockRC:
        def evaluate_entry(self, **kwargs): return RCResult.permit()
        def check_rc02(self, **kwargs): return RCResult.permit()

    class _BlockingRC:
        def evaluate_entry(self, **kwargs): return RCResult.block("RC-08", "Test block", "block")
        def check_rc02(self, **kwargs): return RCResult.permit()

    def _make_om(self, broker=None, rc=None, healthy=True):
        om = OrderManager(
            broker=broker or self._MockBroker(healthy=healthy),
            initial_capital=1_000_000.0, paper_mode=True,
            persist_to_db=False, rc=rc or self._MockRC(),
        )
        om.start()
        return om

    def _sig(self, **kw) -> TradeSignal:
        d = dict(
            symbol="RELIANCE", side=OrderSide.BUY, mode=SignalMode.SWING,
            confidence_score=0.75, entry_price=100.0,
            tp_price=104.0, sl_price=98.5,
            atr_pct=0.018, vol_regime="normal", mds_score=0, signal_id="TEST-001",
        )
        d.update(kw)
        return TradeSignal(**d)

    def _mkt(self): return MarketState(nifty_open=22000, nifty_current=22000,
        india_vix_open=15, india_vix_current=15, fii_provisional_cr=0, mds_score=0)

    # Init
    def test_initial_cash(self):
        assert self._make_om()._cash == 1_000_000.0
    def test_initial_no_positions(self):
        assert len(self._make_om().get_open_positions()) == 0
    def test_repr(self):
        assert "OrderManager" in repr(self._make_om())

    # Signal processing
    def test_process_signal_success(self):
        ok, r = self._make_om().process_signal(self._sig(), self._mkt())
        assert ok, r
    def test_process_signal_opens_position(self):
        om = self._make_om()
        om.process_signal(self._sig(), self._mkt())
        assert om.is_position_open("RELIANCE")
    def test_process_signal_reduces_cash(self):
        om = self._make_om()
        c = om._cash
        om.process_signal(self._sig(), self._mkt())
        assert om._cash < c
    def test_rc_blocked(self):
        ok, r = self._make_om(rc=self._BlockingRC()).process_signal(self._sig(), self._mkt())
        assert not ok and "RC-08" in r
    def test_broker_rejected(self):
        ok, _ = self._make_om(healthy=False).process_signal(self._sig(), self._mkt())
        assert not ok
    def test_increments_trade_count(self):
        om = self._make_om()
        om.process_signal(self._sig(), self._mkt())
        assert om._total_trades == 1
    def test_two_symbols(self):
        om = self._make_om()
        om.process_signal(self._sig(symbol="REL", signal_id="S1"), self._mkt())
        om.process_signal(self._sig(symbol="TCS", signal_id="S2"), self._mkt())
        assert len(om.get_open_positions()) == 2

    # Price updates & exits
    def test_price_update_pnl(self):
        om = self._make_om()
        om.process_signal(self._sig(), self._mkt())
        om.on_price_update({"RELIANCE": 102.0})
        assert om.get_position("RELIANCE").current_price == 102.0
    def test_tp_hit_closes(self):
        om = self._make_om()
        om.process_signal(self._sig(tp_price=104.0), self._mkt())
        om.on_price_update({"RELIANCE": 105.0})
        assert om.get_position("RELIANCE").state == PositionState.CLOSED_TP
    def test_sl_hit_closes(self):
        om = self._make_om()
        om.process_signal(self._sig(sl_price=98.5), self._mkt())
        om.on_price_update({"RELIANCE": 97.0})
        assert om.get_position("RELIANCE").state == PositionState.CLOSED_SL
    def test_trailing_activates(self):
        om = self._make_om()
        om.process_signal(self._sig(), self._mkt())
        om.on_price_update({"RELIANCE": 103.0})
        assert om.get_position("RELIANCE").trail_active
    def test_trailing_hits(self):
        om = self._make_om()
        om.process_signal(self._sig(entry_price=100.0, tp_price=110.0, sl_price=95.0), self._mkt())
        om.on_price_update({"RELIANCE": 103.0})
        pos = om.get_position("RELIANCE")
        assert pos.trail_active
        om.on_price_update({"RELIANCE": pos.trail_stop - 0.01})
        assert pos.state == PositionState.CLOSED_TRAIL
    def test_unknown_symbol_no_crash(self):
        om = self._make_om()
        om.on_price_update({"UNKNOWN": 500.0})

    # Manual close
    def test_manual_close(self):
        om = self._make_om()
        om.process_signal(self._sig(), self._mkt())
        om._last_prices["RELIANCE"] = 101.0
        assert om.close_position_manual("RELIANCE")
        assert om.get_position("RELIANCE").state == PositionState.CLOSED_MANUAL
    def test_manual_close_unknown(self):
        assert not self._make_om().close_position_manual("NONEXISTENT")
    def test_close_all(self):
        om = self._make_om()
        om.process_signal(self._sig(symbol="REL", signal_id="S1"), self._mkt())
        om.process_signal(self._sig(symbol="TCS", signal_id="S2"), self._mkt())
        om._last_prices = {"REL": 100.0, "TCS": 100.0}
        om.close_all_positions()
        assert len(om.get_open_positions()) == 0

    # Snapshot
    def test_snapshot_type(self):
        assert isinstance(self._make_om().get_portfolio_snapshot(), PortfolioSnapshot)
    def test_snapshot_initial(self):
        s = self._make_om().get_portfolio_snapshot()
        assert s.total_value == 1_000_000.0 and s.n_positions == 0
    def test_snapshot_after_trade(self):
        om = self._make_om()
        om.process_signal(self._sig(), self._mkt())
        s = om.get_portfolio_snapshot()
        assert s.n_positions == 1 and s.total_trades == 1

    # Counters
    def test_daily_reset(self):
        om = self._make_om()
        om.process_signal(self._sig(), self._mkt())
        om.reset_daily_counters()
        assert om._trades_today == 0
    def test_monthly_reset(self):
        om = self._make_om()
        om.process_signal(self._sig(), self._mkt())
        om.reset_monthly_counters()
        assert om._trades_this_month == 0
    def test_hold_days(self):
        om = self._make_om()
        om.process_signal(self._sig(), self._mkt())
        om.increment_hold_days()
        om.increment_hold_days()
        assert om.get_position("RELIANCE").hold_days == 2

    # ManagedPosition
    def test_pos_is_open(self):
        p = ManagedPosition("T","S","O",SignalMode.SWING,100.0,datetime.now(),10,1000.0,104.0,98.5)
        assert p.is_open
    def test_pos_sl_pct(self):
        p = ManagedPosition("T","S","O",SignalMode.SWING,100.0,datetime.now(),10,1000.0,104.0,98.5)
        assert abs(p.sl_pct - 0.015) < 1e-6
    def test_pos_update_price(self):
        p = ManagedPosition("T","S","O",SignalMode.SWING,100.0,datetime.now(),10,1000.0,104.0,98.5)
        p.update_price(105.0)
        assert p.unrealised_pnl == 50.0 and abs(p.unrealised_pct - 0.05) < 1e-6
    def test_pos_exit_tp(self):
        p = ManagedPosition("T","S","O",SignalMode.SWING,100.0,datetime.now(),10,1000.0,104.0,98.5)
        ok, reason = p.should_exit(105.0)
        assert ok and reason == PositionState.CLOSED_TP
    def test_pos_exit_sl(self):
        p = ManagedPosition("T","S","O",SignalMode.SWING,100.0,datetime.now(),10,1000.0,104.0,98.5)
        ok, reason = p.should_exit(98.0)
        assert ok and reason == PositionState.CLOSED_SL
    def test_pos_no_exit_midrange(self):
        p = ManagedPosition("T","S","O",SignalMode.SWING,100.0,datetime.now(),10,1000.0,104.0,98.5)
        ok, _ = p.should_exit(101.0)
        assert not ok


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))