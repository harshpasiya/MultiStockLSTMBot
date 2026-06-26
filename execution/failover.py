"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Automatic Broker Failover                       ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : execution/failover.py                                  ║
║         Phase   : 4 — Paper Trading & Live Monitoring                   ║
║                                                                          ║
║  What this module does:                                                  ║
║    Provides a unified broker interface that automatically switches       ║
║    between Zerodha Kite (primary) and Upstox (failover) based on        ║
║    real-time health checks.                                              ║
║                                                                          ║
║    The signal engine and order manager ONLY talk to BrokerRouter.       ║
║    They never reference KiteExecutor or UpstoxExecutor directly.        ║
║    This means a broker outage is handled transparently — no code        ║
║    changes, no manual intervention, no missed trades.                   ║
║                                                                          ║
║  Failover logic:                                                         ║
║    1. Primary broker (Kite) is used for all orders by default           ║
║    2. Every HEALTH_CHECK_INTERVAL seconds, both brokers are pinged      ║
║    3. If primary fails N_FAILURES_TO_FAILOVER consecutive health        ║
║       checks, traffic switches to failover (Upstox)                     ║
║    4. Primary is retested every RECOVERY_CHECK_INTERVAL seconds         ║
║    5. When primary recovers, traffic switches back automatically         ║
║    6. All switches are logged + Telegram alert sent                      ║
║                                                                          ║
║  State machine:                                                          ║
║    PRIMARY_HEALTHY  → normal operation, all orders via Kite             ║
║    PRIMARY_DEGRADED → Kite failing, warning sent, still using Kite      ║
║    FAILOVER_ACTIVE  → Kite down, all orders via Upstox                  ║
║    BOTH_DOWN        → all order placement blocked, alerts fired         ║
║                                                                          ║
║  Thread safety:                                                          ║
║    Health checks run in a background daemon thread.                      ║
║    All broker switches use a RLock to prevent race conditions.          ║
║                                                                          ║
║  Usage:                                                                  ║
║    router = BrokerRouter()                                               ║
║    router.start()          # starts background health check thread      ║
║                                                                          ║
║    result = router.place_order(OrderRequest(...))                        ║
║    positions = router.get_positions()                                    ║
║    router.stop()           # graceful shutdown                           ║
║                                                                          ║
║  Dependencies:                                                           ║
║    execution/kite_executor.py, execution/upstox_executor.py             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import time
import threading
import requests

from dataclasses  import dataclass, field
from datetime     import datetime
from enum         import Enum, auto
from typing       import Optional, Dict, List, Callable, Any
from loguru       import logger
from dotenv       import load_dotenv

from execution.upstox_executor import (
    UpstoxExecutor, OrderRequest, OrderResult,
    OrderSide, OrderType, OrderStatus,
    PositionInfo, QuoteData,
)

load_dotenv()

# ── Telegram alerts ───────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")

# ── Failover thresholds ───────────────────────────────────────────────────
HEALTH_CHECK_INTERVAL    = 30     # seconds between health checks
RECOVERY_CHECK_INTERVAL  = 60     # seconds between primary recovery checks
N_FAILURES_TO_FAILOVER   = 3      # consecutive failures before switching
N_RECOVERIES_TO_PRIMARY  = 2      # consecutive successes before switching back
REQUEST_TIMEOUT          = 5      # seconds for health check requests


# ══════════════════════════════════════════════════════════════════════════
#  ENUMERATIONS & DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

class BrokerState(Enum):
    PRIMARY_HEALTHY  = auto()
    PRIMARY_DEGRADED = auto()
    FAILOVER_ACTIVE  = auto()
    BOTH_DOWN        = auto()


class BrokerName(str, Enum):
    KITE   = "kite"
    UPSTOX = "upstox"
    NONE   = "none"


@dataclass
class BrokerHealth:
    name              : BrokerName
    is_healthy        : bool
    consecutive_fails : int      = 0
    consecutive_ok    : int      = 0
    last_check_time   : datetime = field(default_factory=datetime.now)
    last_error        : str      = ""
    response_time_ms  : float    = 0.0


@dataclass
class FailoverEvent:
    timestamp   : datetime
    from_broker : BrokerName
    to_broker   : BrokerName
    reason      : str
    state       : BrokerState


@dataclass
class RouterStatus:
    state           : BrokerState
    active_broker   : BrokerName
    primary_health  : BrokerHealth
    failover_health : BrokerHealth
    total_orders    : int
    failover_events : List[FailoverEvent]
    uptime_seconds  : float
    paper_mode      : bool


# ══════════════════════════════════════════════════════════════════════════
#  MOCK KITE EXECUTOR STUB
# ══════════════════════════════════════════════════════════════════════════

class _KiteExecutorStub:
    """
    Interface-compatible stub for KiteExecutor.
    Used in paper trading mode and unit tests.
    Delegates all calls to UpstoxExecutor in paper mode.
    """

    def __init__(self, paper_mode: bool = True):
        self._delegate  = UpstoxExecutor(paper_mode=paper_mode)
        self.paper_mode = paper_mode

    def is_healthy(self) -> bool:
        return True

    def place_order(self, request: OrderRequest) -> OrderResult:
        result        = self._delegate.place_order(request)
        result.broker = "kite_paper"
        return result

    def cancel_order(self, order_id: str) -> bool:
        return self._delegate.cancel_order(order_id)

    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        return self._delegate.get_order_status(order_id)

    def get_positions(self) -> List[PositionInfo]:
        return self._delegate.get_positions()

    def get_quote(self, instrument: str) -> Optional[QuoteData]:
        return self._delegate.get_quote(instrument)

    def get_funds(self) -> Dict[str, float]:
        return self._delegate.get_funds()


def _load_kite_executor(paper_mode: bool = True):
    """Loads real KiteExecutor if available, falls back to stub."""
    try:
        from execution.kite_executor import KiteExecutor
        return KiteExecutor(paper_mode=paper_mode)
    except Exception as e:
        logger.warning(f"KiteExecutor unavailable ({e}) — using stub.")
        return _KiteExecutorStub(paper_mode=paper_mode)


# ══════════════════════════════════════════════════════════════════════════
#  BROKER ROUTER
# ══════════════════════════════════════════════════════════════════════════

class BrokerRouter:
    """
    Unified broker interface with automatic failover.

    signal_engine.py → BrokerRouter → KiteExecutor  (primary)
                                    → UpstoxExecutor (failover)
    """

    def __init__(
        self,
        paper_mode : bool = True,
        primary    : Any  = None,
        failover   : Any  = None,
        auto_start : bool = False,
    ):
        self.paper_mode = paper_mode
        self._primary   = primary  or _load_kite_executor(paper_mode)
        self._failover  = failover or UpstoxExecutor(paper_mode=paper_mode)

        self._state        = BrokerState.PRIMARY_HEALTHY
        self._active       = BrokerName.KITE
        self._lock         = threading.RLock()
        self._start_time   = datetime.now()
        self._total_orders = 0

        self._primary_health  = BrokerHealth(name=BrokerName.KITE,   is_healthy=True)
        self._failover_health = BrokerHealth(name=BrokerName.UPSTOX, is_healthy=True)
        self._failover_events : List[FailoverEvent] = []

        self._running   = False
        self._hc_thread : Optional[threading.Thread] = None

        self._on_failover  : Optional[Callable] = None
        self._on_both_down : Optional[Callable] = None

        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info(f"BrokerRouter initialized | mode={mode_str}")

        if auto_start:
            self.start()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Starts the background health check thread."""
        if self._running:
            logger.warning("BrokerRouter already running.")
            return
        self._running   = True
        self._hc_thread = threading.Thread(
            target = self._health_check_loop,
            name   = "BrokerHealthCheck",
            daemon = True,
        )
        self._hc_thread.start()
        logger.info(f"BrokerRouter started | interval={HEALTH_CHECK_INTERVAL}s")

    def stop(self):
        """Gracefully stops the health check thread."""
        self._running = False
        if self._hc_thread and self._hc_thread.is_alive():
            self._hc_thread.join(timeout=5.0)
        logger.info("BrokerRouter stopped.")

    def register_failover_callback(self, callback: Callable):
        """Registers callback called on every broker switch."""
        self._on_failover = callback

    def register_both_down_callback(self, callback: Callable):
        """Registers callback called when both brokers are down."""
        self._on_both_down = callback

    # ── Public broker interface ───────────────────────────────────────────

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Places order via active broker; emergency fallback if rejected."""
        with self._lock:
            if self._state == BrokerState.BOTH_DOWN:
                logger.error("BOTH brokers down — order blocked.")
                return OrderResult(
                    order_id      = "",
                    symbol        = request.symbol,
                    side          = request.side,
                    quantity      = request.quantity,
                    status        = OrderStatus.REJECTED,
                    error_message = "Both brokers unavailable",
                    broker        = "none",
                )

            self._total_orders += 1
            result = self._get_active_broker().place_order(request)

            if result.is_rejected and self._state != BrokerState.BOTH_DOWN:
                logger.warning(
                    f"Active broker ({self._active.value}) rejected order — "
                    f"attempting emergency fallback..."
                )
                fallback = self._get_inactive_broker()
                if fallback is not None:
                    result = fallback.place_order(request)
                    if result.is_filled:
                        logger.info("Emergency fallback order succeeded.")

            return result

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            return self._get_active_broker().cancel_order(order_id)

    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        with self._lock:
            return self._get_active_broker().get_order_status(order_id)

    def get_positions(self) -> List[PositionInfo]:
        with self._lock:
            return self._get_active_broker().get_positions()

    def get_quote(self, instrument: str) -> Optional[QuoteData]:
        with self._lock:
            quote = self._get_active_broker().get_quote(instrument)
            if quote is None and self._state != BrokerState.BOTH_DOWN:
                fallback = self._get_inactive_broker()
                if fallback:
                    quote = fallback.get_quote(instrument)
            return quote

    def get_funds(self) -> Dict[str, float]:
        with self._lock:
            return self._get_active_broker().get_funds()

    def get_status(self) -> RouterStatus:
        with self._lock:
            uptime = (datetime.now() - self._start_time).total_seconds()
            return RouterStatus(
                state           = self._state,
                active_broker   = self._active,
                primary_health  = self._primary_health,
                failover_health = self._failover_health,
                total_orders    = self._total_orders,
                failover_events = list(self._failover_events),
                uptime_seconds  = uptime,
                paper_mode      = self.paper_mode,
            )

    @property
    def active_broker_name(self) -> BrokerName:
        return self._active

    @property
    def is_operational(self) -> bool:
        return self._state != BrokerState.BOTH_DOWN

    # ── Health check loop ─────────────────────────────────────────────────

    def _health_check_loop(self):
        logger.info("Health check loop started.")
        while self._running:
            try:
                self._check_primary_health()
                self._check_failover_health()
                self._evaluate_state()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
            time.sleep(HEALTH_CHECK_INTERVAL)
        logger.info("Health check loop stopped.")

    def _check_primary_health(self):
        start = time.monotonic()
        try:
            healthy = self._primary.is_healthy()
            elapsed = (time.monotonic() - start) * 1000
            with self._lock:
                self._primary_health.last_check_time  = datetime.now()
                self._primary_health.response_time_ms = elapsed
                if healthy:
                    self._primary_health.consecutive_fails = 0
                    self._primary_health.consecutive_ok   += 1
                    self._primary_health.is_healthy        = True
                    self._primary_health.last_error        = ""
                else:
                    self._primary_health.consecutive_fails += 1
                    self._primary_health.consecutive_ok     = 0
                    self._primary_health.is_healthy         = False
        except Exception as e:
            with self._lock:
                self._primary_health.consecutive_fails += 1
                self._primary_health.consecutive_ok     = 0
                self._primary_health.is_healthy         = False
                self._primary_health.last_error         = str(e)

    def _check_failover_health(self):
        start = time.monotonic()
        try:
            healthy = self._failover.is_healthy()
            elapsed = (time.monotonic() - start) * 1000
            with self._lock:
                self._failover_health.last_check_time  = datetime.now()
                self._failover_health.response_time_ms = elapsed
                self._failover_health.is_healthy        = healthy
                if healthy:
                    self._failover_health.consecutive_fails = 0
                    self._failover_health.consecutive_ok   += 1
                    self._failover_health.last_error        = ""
                else:
                    self._failover_health.consecutive_fails += 1
                    self._failover_health.consecutive_ok     = 0
        except Exception as e:
            with self._lock:
                self._failover_health.is_healthy = False
                self._failover_health.last_error = str(e)

    def _evaluate_state(self):
        """State machine: evaluates health data and switches broker if needed."""
        with self._lock:
            primary_failing = (
                self._primary_health.consecutive_fails >= N_FAILURES_TO_FAILOVER
            )
            primary_ok  = self._primary_health.consecutive_ok  >= N_RECOVERIES_TO_PRIMARY
            failover_ok = self._failover_health.is_healthy

            if self._state == BrokerState.PRIMARY_HEALTHY:
                if primary_failing and failover_ok:
                    self._transition_to(BrokerState.FAILOVER_ACTIVE, BrokerName.UPSTOX,
                                        "Primary broker failed health checks")
                elif primary_failing and not failover_ok:
                    self._transition_to(BrokerState.BOTH_DOWN, BrokerName.NONE,
                                        "Both brokers failing")
                elif self._primary_health.consecutive_fails > 0:
                    self._state = BrokerState.PRIMARY_DEGRADED

            elif self._state == BrokerState.PRIMARY_DEGRADED:
                if primary_failing and failover_ok:
                    self._transition_to(BrokerState.FAILOVER_ACTIVE, BrokerName.UPSTOX,
                                        "Primary degraded beyond threshold")
                elif primary_ok:
                    self._state  = BrokerState.PRIMARY_HEALTHY
                    self._active = BrokerName.KITE

            elif self._state == BrokerState.FAILOVER_ACTIVE:
                if primary_ok and self._primary_health.is_healthy:
                    self._transition_to(BrokerState.PRIMARY_HEALTHY, BrokerName.KITE,
                                        "Primary broker recovered")
                elif not failover_ok:
                    self._transition_to(BrokerState.BOTH_DOWN, BrokerName.NONE,
                                        "Failover also failed")

            elif self._state == BrokerState.BOTH_DOWN:
                if self._primary_health.is_healthy:
                    self._transition_to(BrokerState.PRIMARY_HEALTHY, BrokerName.KITE,
                                        "Primary recovered from both-down")
                elif failover_ok:
                    self._transition_to(BrokerState.FAILOVER_ACTIVE, BrokerName.UPSTOX,
                                        "Failover recovered while primary still down")

    def _transition_to(self, new_state: BrokerState, new_broker: BrokerName, reason: str):
        """Executes a broker state transition with logging and alerts."""
        event = FailoverEvent(
            timestamp   = datetime.now(),
            from_broker = self._active,
            to_broker   = new_broker,
            reason      = reason,
            state       = new_state,
        )
        self._failover_events.append(event)
        if len(self._failover_events) > 100:
            self._failover_events = self._failover_events[-100:]

        old_state    = self._state
        self._state  = new_state
        self._active = new_broker

        log_fn = logger.critical if new_state == BrokerState.BOTH_DOWN else logger.warning
        log_fn(
            f"BROKER: {old_state.name} → {new_state.name} | "
            f"active={new_broker.value} | {reason}"
        )

        self._send_telegram_alert(event)

        if self._on_failover:
            try:
                self._on_failover(event)
            except Exception as e:
                logger.warning(f"Failover callback error: {e}")

        if new_state == BrokerState.BOTH_DOWN and self._on_both_down:
            try:
                self._on_both_down(event)
            except Exception as e:
                logger.warning(f"Both-down callback error: {e}")

    def _get_active_broker(self):
        if self._active == BrokerName.KITE:
            return self._primary
        elif self._active == BrokerName.UPSTOX:
            return self._failover
        return self._primary

    def _get_inactive_broker(self):
        if self._active == BrokerName.KITE:
            return self._failover
        elif self._active == BrokerName.UPSTOX:
            return self._primary
        return None

    def _send_telegram_alert(self, event: FailoverEvent):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
        emoji = {
            BrokerState.FAILOVER_ACTIVE : "⚠️",
            BrokerState.PRIMARY_HEALTHY : "✅",
            BrokerState.BOTH_DOWN       : "🚨",
            BrokerState.PRIMARY_DEGRADED: "🟡",
        }.get(event.state, "ℹ️")
        msg = (
            f"{emoji} G.O.D.S E.Y.E — Broker Switch\n"
            f"From: {event.from_broker.value} → To: {event.to_broker.value}\n"
            f"State: {event.state.name}\n"
            f"Reason: {event.reason}"
        )
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"Telegram alert failed: {e}")

    def __repr__(self) -> str:
        return (
            f"BrokerRouter(state={self._state.name}, "
            f"active={self._active.value}, "
            f"orders={self._total_orders}, "
            f"paper={self.paper_mode})"
        )


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest execution/failover.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestBrokerRouter:

    class _MockBroker:
        def __init__(self, name: str, healthy: bool = True):
            self.name     = name
            self._healthy = healthy
            self._orders  : List[OrderResult] = []

        def is_healthy(self) -> bool:
            return self._healthy

        def place_order(self, request: OrderRequest) -> OrderResult:
            status = OrderStatus.PAPER_FILL if self._healthy else OrderStatus.REJECTED
            result = OrderResult(
                order_id      = f"{self.name}-{len(self._orders)+1:04d}",
                symbol        = request.symbol,
                side          = request.side,
                quantity      = request.quantity,
                status        = status,
                fill_price    = 100.0,
                fill_quantity = request.quantity if self._healthy else 0,
                broker        = self.name,
                paper_mode    = True,
            )
            self._orders.append(result)
            return result

        def cancel_order(self, order_id: str) -> bool:
            return self._healthy

        def get_order_status(self, order_id: str) -> Optional[OrderResult]:
            return next((o for o in self._orders if o.order_id == order_id), None)

        def get_positions(self) -> List[PositionInfo]:
            return []

        def get_quote(self, instrument: str) -> Optional[QuoteData]:
            if not self._healthy:
                return None
            return QuoteData(
                symbol=instrument, last_price=100.0,
                open_price=99.0, high_price=101.0,
                low_price=98.0, close_price=100.0, volume=1_000_000,
            )

        def get_funds(self) -> Dict[str, float]:
            return {"available_cash": 1_000_000.0}

    def _make_router(self, primary_healthy=True, failover_healthy=True) -> BrokerRouter:
        return BrokerRouter(
            paper_mode  = True,
            primary     = self._MockBroker("kite",   healthy=primary_healthy),
            failover    = self._MockBroker("upstox", healthy=failover_healthy),
            auto_start  = False,
        )

    def _req(self) -> OrderRequest:
        return OrderRequest(
            symbol="NSE_EQ|TEST", side=OrderSide.BUY,
            quantity=10, order_type=OrderType.MARKET,
        )

    # ── Init ──────────────────────────────────────────────────────────────

    def test_initial_state(self):
        r = self._make_router()
        assert r._state  == BrokerState.PRIMARY_HEALTHY
        assert r._active == BrokerName.KITE

    def test_is_operational(self):
        assert self._make_router().is_operational is True

    def test_paper_mode(self):
        assert self._make_router().paper_mode is True

    # ── Order placement ───────────────────────────────────────────────────

    def test_order_via_primary(self):
        r = self._make_router()
        assert r.place_order(self._req()).broker == "kite"

    def test_order_count(self):
        r = self._make_router()
        for _ in range(3):
            r.place_order(self._req())
        assert r._total_orders == 3

    def test_primary_rejected_falls_back(self):
        r = self._make_router(primary_healthy=False, failover_healthy=True)
        result = r.place_order(self._req())
        assert result.broker == "upstox"
        assert result.is_filled

    def test_both_down_blocks_order(self):
        r         = self._make_router()
        r._state  = BrokerState.BOTH_DOWN
        r._active = BrokerName.NONE
        result    = r.place_order(self._req())
        assert result.status == OrderStatus.REJECTED
        assert "unavailable" in result.error_message.lower()

    # ── State machine ─────────────────────────────────────────────────────

    def test_failover_after_n_failures(self):
        r = self._make_router()
        r._primary_health.consecutive_fails = N_FAILURES_TO_FAILOVER
        r._primary_health.is_healthy        = False
        r._failover_health.is_healthy       = True
        r._evaluate_state()
        assert r._state  == BrokerState.FAILOVER_ACTIVE
        assert r._active == BrokerName.UPSTOX

    def test_recovery_returns_to_primary(self):
        r = self._make_router()
        r._state  = BrokerState.FAILOVER_ACTIVE
        r._active = BrokerName.UPSTOX
        r._primary_health.consecutive_ok = N_RECOVERIES_TO_PRIMARY
        r._primary_health.is_healthy     = True
        r._evaluate_state()
        assert r._state  == BrokerState.PRIMARY_HEALTHY
        assert r._active == BrokerName.KITE

    def test_both_down_when_failover_also_fails(self):
        r = self._make_router()
        r._primary_health.consecutive_fails = N_FAILURES_TO_FAILOVER
        r._primary_health.is_healthy        = False
        r._failover_health.is_healthy       = False
        r._evaluate_state()
        assert r._state == BrokerState.BOTH_DOWN

    def test_degraded_before_failover(self):
        r = self._make_router()
        r._primary_health.consecutive_fails = 1
        r._primary_health.is_healthy        = False
        r._evaluate_state()
        assert r._state == BrokerState.PRIMARY_DEGRADED

    def test_failover_event_recorded(self):
        r = self._make_router()
        r._send_telegram_alert = lambda e: None
        r._transition_to(BrokerState.FAILOVER_ACTIVE, BrokerName.UPSTOX, "test")
        assert len(r._failover_events) == 1
        assert r._failover_events[0].to_broker == BrokerName.UPSTOX

    def test_both_down_recovery(self):
        r = self._make_router()
        r._state  = BrokerState.BOTH_DOWN
        r._active = BrokerName.NONE
        r._primary_health.is_healthy     = True
        r._primary_health.consecutive_ok = N_RECOVERIES_TO_PRIMARY
        r._send_telegram_alert           = lambda e: None
        r._evaluate_state()
        assert r._state == BrokerState.PRIMARY_HEALTHY

    # ── Health checks ─────────────────────────────────────────────────────

    def test_primary_health_healthy(self):
        r = self._make_router(primary_healthy=True)
        r._check_primary_health()
        assert r._primary_health.is_healthy is True
        assert r._primary_health.consecutive_fails == 0

    def test_primary_health_unhealthy(self):
        r = self._make_router(primary_healthy=False)
        r._check_primary_health()
        assert r._primary_health.is_healthy is False
        assert r._primary_health.consecutive_fails == 1

    def test_fails_accumulate(self):
        r = self._make_router(primary_healthy=False)
        for _ in range(3):
            r._check_primary_health()
        assert r._primary_health.consecutive_fails == 3

    def test_recovery_resets_fails(self):
        r = self._make_router(primary_healthy=False)
        r._check_primary_health()
        r._primary._healthy = True
        r._check_primary_health()
        assert r._primary_health.consecutive_fails == 0

    # ── Callbacks ─────────────────────────────────────────────────────────

    def test_failover_callback(self):
        r = self._make_router()
        events = []
        r.register_failover_callback(lambda e: events.append(e))
        r._send_telegram_alert = lambda e: None
        r._transition_to(BrokerState.FAILOVER_ACTIVE, BrokerName.UPSTOX, "test")
        assert len(events) == 1

    def test_both_down_callback(self):
        r = self._make_router()
        alerts = []
        r.register_both_down_callback(lambda e: alerts.append(e))
        r._send_telegram_alert = lambda e: None
        r._transition_to(BrokerState.BOTH_DOWN, BrokerName.NONE, "test")
        assert len(alerts) == 1

    # ── Delegates ─────────────────────────────────────────────────────────

    def test_cancel_delegates(self):
        r      = self._make_router()
        result = r.place_order(self._req())
        assert r.cancel_order(result.order_id) is True

    def test_get_positions(self):
        assert isinstance(self._make_router().get_positions(), list)

    def test_get_funds(self):
        assert "available_cash" in self._make_router().get_funds()

    def test_get_quote_primary(self):
        q = self._make_router().get_quote("NSE_EQ|TEST")
        assert isinstance(q, QuoteData)

    def test_get_quote_fallback(self):
        r = self._make_router(primary_healthy=False)
        q = r.get_quote("NSE_EQ|TEST")
        assert q is not None

    # ── Status & repr ─────────────────────────────────────────────────────

    def test_get_status(self):
        status = self._make_router().get_status()
        assert isinstance(status, RouterStatus)
        assert status.state == BrokerState.PRIMARY_HEALTHY

    def test_active_broker_name(self):
        assert self._make_router().active_broker_name == BrokerName.KITE

    def test_repr(self):
        assert "PRIMARY_HEALTHY" in repr(self._make_router())

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def test_start_stop(self):
        r = self._make_router()
        r.start()
        time.sleep(0.05)
        r.stop()
        assert not r._running

    def test_double_start(self):
        r = self._make_router()
        r.start()
        r.start()   # should warn, not crash
        r.stop()


# ── Run when executed directly ────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))