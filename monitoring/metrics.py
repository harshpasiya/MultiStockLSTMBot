"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Prometheus Metrics Exporter                     ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : monitoring/metrics.py                                  ║
║         Phase   : 4 — Paper Trading & Live Monitoring                   ║
║                                                                          ║
║  What this module does:                                                  ║
║    Exposes all G.O.D.S E.Y.E system metrics as Prometheus-compatible    ║
║    gauges/counters on an HTTP endpoint (:8000/metrics).                 ║
║    Grafana scrapes this endpoint every 15 seconds to populate the       ║
║    live trading dashboard at http://localhost:3000.                     ║
║                                                                          ║
║  Key design decision — isolated CollectorRegistry:                      ║
║    Every GodsEyeMetrics instance creates its OWN CollectorRegistry      ║
║    instead of using prometheus_client's global registry.                ║
║    This prevents "Duplicated timeseries" errors when multiple           ║
║    instances are created (e.g. in unit tests).                          ║
║                                                                          ║
║  Metrics exported:                                                       ║
║    Portfolio  : total_value, cash, drawdown, unrealised_pnl, heat       ║
║    Trades     : total_trades, trades_today, win_rate, profit_factor     ║
║    Positions  : n_open, per-symbol P&L, hold_days                       ║
║    Signals    : generated, blocked, latency_ms                          ║
║    Broker     : active_broker, failover_count, order_success_rate       ║
║    Model      : backbone_inference_ms, feature_drift_score              ║
║    RC         : per-rule trigger counts, halt_active                    ║
║    System     : uptime, last_heartbeat, paper_mode                      ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install prometheus-client loguru                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import time
import threading

from datetime import datetime
from typing   import Optional, Dict, Any
from loguru   import logger

try:
    from prometheus_client import (
        Gauge, Counter, Histogram,
        CollectorRegistry,
        make_wsgi_app,
        start_http_server,
    )
    from wsgiref.simple_server import make_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed — metrics tracked in-memory only. "
        "Install with: pip install prometheus-client"
    )

METRICS_PORT = 8000


# ══════════════════════════════════════════════════════════════════════════
#  STUB METRICS (when prometheus_client is not installed)
# ══════════════════════════════════════════════════════════════════════════

class _Stub:
    """No-op metric stub — mirrors the prometheus_client API surface."""
    def __init__(self, *a, **kw):
        self._val = 0.0
        self._children: Dict[str, "_Stub"] = {}

    def set(self, v):     self._val = float(v)
    def inc(self, v=1):   self._val += float(v)
    def dec(self, v=1):   self._val -= float(v)
    def observe(self, v): self._val = float(v)

    def labels(self, **kw) -> "_Stub":
        key = str(kw)
        if key not in self._children:
            self._children[key] = _Stub()
        return self._children[key]

    @property
    def value(self) -> float:
        return self._val


# ══════════════════════════════════════════════════════════════════════════
#  METRICS CLASS
# ══════════════════════════════════════════════════════════════════════════

class GodsEyeMetrics:
    """
    Central metrics registry for G.O.D.S E.Y.E.

    Uses an ISOLATED CollectorRegistry per instance so that:
      - Multiple instances can coexist (e.g. in unit tests)
      - No "Duplicated timeseries" errors from prometheus_client global registry
      - The HTTP server for each instance only exposes its own metrics

    Usage:
        metrics = GodsEyeMetrics()
        metrics.start_server()           # starts :8000/metrics for Grafana

        metrics.update_portfolio(snapshot)
        metrics.record_signal(latency_ms=120, blocked=False)
        metrics.set_active_broker("kite")
    """

    def __init__(self, port: int = METRICS_PORT):
        self.port            = port
        self._server_started = False
        self._lock           = threading.Lock()
        self._start_time     = time.time()

        # Each instance owns its own isolated registry
        # This is the key fix — avoids global registry duplicate errors
        if PROMETHEUS_AVAILABLE:
            self._registry = CollectorRegistry()
            self._G  = lambda name, doc, labels=None: (
                Gauge(name, doc, labels or [], registry=self._registry)
                if labels else Gauge(name, doc, registry=self._registry)
            )
            self._C  = lambda name, doc, labels=None: (
                Counter(name, doc, labels or [], registry=self._registry)
                if labels else Counter(name, doc, registry=self._registry)
            )
            self._H  = lambda name, doc, buckets=None: (
                Histogram(name, doc, buckets=buckets, registry=self._registry)
                if buckets else Histogram(name, doc, registry=self._registry)
            )
        else:
            self._registry = None
            self._G  = lambda *a, **kw: _Stub()
            self._C  = lambda *a, **kw: _Stub()
            self._H  = lambda *a, **kw: _Stub()

        # ── Portfolio ──────────────────────────────────────────────────────
        self.portfolio_value    = self._G("godseye_portfolio_value_inr",
                                          "Total portfolio value in INR")
        self.portfolio_cash     = self._G("godseye_portfolio_cash_inr",
                                          "Available cash in INR")
        self.portfolio_drawdown = self._G("godseye_portfolio_drawdown",
                                          "Current drawdown from peak")
        self.portfolio_heat     = self._G("godseye_portfolio_heat",
                                          "Total portfolio heat fraction")
        self.unrealised_pnl     = self._G("godseye_unrealised_pnl_inr",
                                          "Total unrealised P&L in INR")
        self.realised_pnl_today = self._G("godseye_realised_pnl_today_inr",
                                          "Realised P&L today in INR")
        self.peak_value         = self._G("godseye_peak_value_inr",
                                          "All-time peak portfolio value")

        # ── Trades ─────────────────────────────────────────────────────────
        self.total_trades       = self._G("godseye_total_trades",
                                          "Total trades since start")
        self.trades_today       = self._G("godseye_trades_today",
                                          "Trades today")
        self.trades_this_month  = self._G("godseye_trades_this_month",
                                          "Trades this month")
        self.win_rate           = self._G("godseye_win_rate",
                                          "Win rate over last 50 trades")
        self.profit_factor      = self._G("godseye_profit_factor",
                                          "Profit factor over last 50 trades")

        # ── Positions ──────────────────────────────────────────────────────
        self.n_open_positions   = self._G("godseye_open_positions",
                                          "Number of open positions")
        self.position_pnl       = self._G("godseye_position_pnl_pct",
                                          "Per-symbol unrealised P&L %",
                                          labels=["symbol"])
        self.position_hold_days = self._G("godseye_position_hold_days",
                                          "Days held per position",
                                          labels=["symbol"])

        # ── Signals ────────────────────────────────────────────────────────
        self.signals_generated  = self._C("godseye_signals_generated_total",
                                          "Total signals generated")
        self.signals_blocked    = self._C("godseye_signals_blocked_total",
                                          "Total signals blocked",
                                          labels=["rule"])
        self.signal_latency_ms  = self._H("godseye_signal_latency_ms",
                                          "Signal generation latency ms",
                                          buckets=[100,250,500,1000,2000,5000])
        self.last_signal_time   = self._G("godseye_last_signal_timestamp",
                                          "Unix timestamp of last signal")

        # ── Broker ─────────────────────────────────────────────────────────
        self.active_broker_kite = self._G("godseye_broker_kite_active",
                                          "1 if Kite is active broker")
        self.failover_count     = self._C("godseye_failover_total",
                                          "Total broker failover events")
        self.order_success_rate = self._G("godseye_order_success_rate",
                                          "Fraction of orders filled (last 50)")
        self.broker_latency_ms  = self._G("godseye_broker_latency_ms",
                                          "Last order placement latency ms")

        # ── Model ──────────────────────────────────────────────────────────
        self.backbone_latency_ms= self._G("godseye_backbone_inference_ms",
                                          "Backbone inference latency ms")
        self.feature_drift_score= self._G("godseye_feature_drift_score",
                                          "KL divergence drift score")
        self.drift_alert_active = self._G("godseye_drift_alert_active",
                                          "1 if drift exceeds threshold")
        self.last_retrain_time  = self._G("godseye_last_retrain_timestamp",
                                          "Unix timestamp of last retrain")

        # ── Risk Constitution ──────────────────────────────────────────────
        self.rc_triggers        = self._C("godseye_rc_triggers_total",
                                          "RC rule triggers",
                                          labels=["rule"])
        self.rc_halt_active     = self._G("godseye_rc_halt_active",
                                          "1 if RC-01 halt is active")

        # ── System ─────────────────────────────────────────────────────────
        self.uptime_seconds     = self._G("godseye_uptime_seconds",
                                          "System uptime in seconds")
        self.last_heartbeat     = self._G("godseye_last_heartbeat_timestamp",
                                          "Unix timestamp of last heartbeat")
        self.paper_mode_active  = self._G("godseye_paper_mode_active",
                                          "1 if paper trading mode")

        # ── In-memory rolling windows ──────────────────────────────────────
        self._closed_trades  : list = []   # (pnl_pct, is_win)
        self._order_results  : list = []   # bool: filled or not
        self._hb_thread      : Optional[threading.Thread] = None

    # ══════════════════════════════════════════════════════════════════════
    #  SERVER
    # ══════════════════════════════════════════════════════════════════════

    def start_server(self):
        """
        Starts the Prometheus HTTP metrics server on self.port.
        Uses the instance's isolated registry (not global).
        Call once at application startup.
        """
        if self._server_started or not PROMETHEUS_AVAILABLE:
            if not PROMETHEUS_AVAILABLE:
                logger.warning("prometheus_client not available — server not started.")
            return

        try:
            # Use make_wsgi_app with our isolated registry
            app    = make_wsgi_app(self._registry)
            server = make_server("", self.port, app)
            t      = threading.Thread(
                target=server.serve_forever,
                daemon=True,
                name="PrometheusMetricsServer",
            )
            t.start()
            self._server_started = True
            logger.info(f"Metrics server started at :{self.port}/metrics")
            self._start_heartbeat()
        except OSError as e:
            logger.warning(f"Could not start metrics server on :{self.port}: {e}")

    def _start_heartbeat(self):
        def _loop():
            while True:
                self.uptime_seconds.set(time.time() - self._start_time)
                self.last_heartbeat.set(time.time())
                time.sleep(10)
        self._hb_thread = threading.Thread(
            target=_loop, daemon=True, name="MetricsHeartbeat"
        )
        self._hb_thread.start()

    # ══════════════════════════════════════════════════════════════════════
    #  UPDATE METHODS
    # ══════════════════════════════════════════════════════════════════════

    def update_portfolio(self, snapshot: Any):
        """
        Updates all portfolio metrics from a PortfolioSnapshot.
        Called by order_manager every bar.
        """
        self.portfolio_value.set(snapshot.total_value)
        self.portfolio_cash.set(snapshot.cash)
        self.portfolio_drawdown.set(snapshot.drawdown)
        self.portfolio_heat.set(snapshot.portfolio_heat)
        self.unrealised_pnl.set(snapshot.unrealised_pnl)
        self.realised_pnl_today.set(snapshot.realised_pnl_today)
        self.peak_value.set(snapshot.peak_value)
        self.total_trades.set(snapshot.total_trades)
        self.trades_today.set(snapshot.trades_today)
        self.trades_this_month.set(snapshot.trades_this_month)
        self.n_open_positions.set(snapshot.n_positions)

        for symbol, pos in snapshot.open_positions.items():
            try:
                if hasattr(pos, "is_open") and pos.is_open:
                    self.position_pnl.labels(symbol=symbol).set(
                        getattr(pos, "unrealised_pct", 0) * 100
                    )
                    self.position_hold_days.labels(symbol=symbol).set(
                        getattr(pos, "hold_days", 0)
                    )
            except Exception:
                pass

    def record_trade_close(self, pnl_pct: float, is_win: bool):
        """
        Records a closed trade for win rate and profit factor.
        Maintains a rolling window of last 50 trades.
        """
        with self._lock:
            self._closed_trades.append((pnl_pct, is_win))
            if len(self._closed_trades) > 50:
                self._closed_trades = self._closed_trades[-50:]
            self._recompute_trade_stats()

    def _recompute_trade_stats(self):
        """Recomputes and sets win_rate and profit_factor from closed trades."""
        n = len(self._closed_trades)
        if n == 0:
            return
        wins   = sum(1 for _, w in self._closed_trades if w)
        self.win_rate.set(wins / n)
        gains  = sum(p for p, _ in self._closed_trades if p > 0)
        losses = sum(abs(p) for p, _ in self._closed_trades if p < 0)
        self.profit_factor.set(gains / losses if losses > 0 else gains)

    def record_signal(
        self,
        latency_ms  : float,
        blocked     : bool,
        block_rule  : str = "",
    ):
        """
        Records a signal generation event.

        Args:
            latency_ms : ms from bar close to signal ready
            blocked    : True if signal was blocked
            block_rule : Which RC rule blocked it (e.g. 'RC-08')
        """
        self.signals_generated.inc()
        self.signal_latency_ms.observe(latency_ms)
        self.last_signal_time.set(time.time())
        if blocked:
            self.signals_blocked.labels(rule=block_rule or "unknown").inc()

    def record_order(self, filled: bool, latency_ms: float = 0.0):
        """
        Records an order placement result.

        Args:
            filled     : True if order was filled
            latency_ms : Placement latency in ms
        """
        with self._lock:
            self._order_results.append(filled)
            if len(self._order_results) > 50:
                self._order_results = self._order_results[-50:]
            self.order_success_rate.set(
                sum(self._order_results) / len(self._order_results)
            )
        if latency_ms > 0:
            self.broker_latency_ms.set(latency_ms)

    def set_active_broker(self, broker_name: str):
        """Sets active broker metric. broker_name: 'kite' or 'upstox'."""
        self.active_broker_kite.set(1 if broker_name == "kite" else 0)

    def record_failover(self):
        """Records a broker failover event."""
        self.failover_count.inc()

    def update_model_metrics(
        self,
        backbone_latency_ms : float,
        drift_score         : float,
        drift_threshold     : float = 0.15,
    ):
        """Updates model health metrics."""
        self.backbone_latency_ms.set(backbone_latency_ms)
        self.feature_drift_score.set(drift_score)
        self.drift_alert_active.set(1 if drift_score > drift_threshold else 0)

    def record_retrain(self):
        """Records completion of nightly retraining."""
        self.last_retrain_time.set(time.time())

    def record_rc_trigger(self, rule: str):
        """Records a Risk Constitution rule trigger."""
        self.rc_triggers.labels(rule=rule).inc()

    def set_rc_halt(self, active: bool):
        """Updates RC-01 halt status."""
        self.rc_halt_active.set(1 if active else 0)

    def set_paper_mode(self, active: bool):
        """Updates paper mode status."""
        self.paper_mode_active.set(1 if active else 0)

    def get_summary(self) -> Dict:
        """
        Returns a human-readable summary of key metrics.
        Used by health check endpoint and weekly review.
        """
        with self._lock:
            n    = len(self._closed_trades)
            wins = sum(1 for _, w in self._closed_trades if w)
            wr   = (wins / n) if n > 0 else 0.0
            n_o  = len(self._order_results)
            sr   = (sum(self._order_results) / n_o) if n_o > 0 else 1.0

        return {
            "uptime_hours"        : (time.time() - self._start_time) / 3600,
            "win_rate"            : wr,
            "order_success_rate"  : sr,
            "n_closed_trades"     : n,
            "server_started"      : self._server_started,
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

    def _get_gauge_value(self, gauge) -> float:
        """
        Safely reads a gauge value regardless of whether
        prometheus_client is installed or stub is used.
        """
        if PROMETHEUS_AVAILABLE:
            try:
                return gauge._value.get()
            except Exception:
                return 0.0
        else:
            return gauge._val


# ── Module-level singleton ─────────────────────────────────────────────────
_default_metrics: Optional[GodsEyeMetrics] = None


def get_metrics() -> GodsEyeMetrics:
    """
    Returns the module-level singleton GodsEyeMetrics instance.
    Creates it on first call.

    Example:
        from monitoring.metrics import get_metrics
        metrics = get_metrics()
        metrics.record_signal(latency_ms=150, blocked=False)
    """
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = GodsEyeMetrics()
    return _default_metrics


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest monitoring/metrics.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestGodsEyeMetrics:
    """
    Unit tests for GodsEyeMetrics.
    Each test creates its own isolated instance — no shared global state.
    Tests work whether or not prometheus_client is installed.
    """

    def _make(self) -> GodsEyeMetrics:
        """Creates a fresh isolated metrics instance per test."""
        return GodsEyeMetrics(port=0)   # port=0 means server not started

    def _make_snapshot(self):
        """Minimal mock PortfolioSnapshot."""
        class _Pos:
            is_open        = True
            unrealised_pct = 0.03
            hold_days      = 5

        class _Snap:
            total_value        = 1_050_000.0
            cash               = 700_000.0
            peak_value         = 1_100_000.0
            drawdown           = 0.045
            portfolio_heat     = 0.03
            unrealised_pnl     = 15_000.0
            realised_pnl_today = 35_000.0
            n_positions        = 2
            trades_today       = 3
            trades_this_month  = 8
            total_trades       = 42
            open_positions     = {"RELIANCE": _Pos()}

        return _Snap()

    def _val(self, gauge) -> float:
        """Reads gauge value regardless of prometheus availability."""
        if PROMETHEUS_AVAILABLE:
            try:
                return gauge._value.get()
            except Exception:
                return 0.0
        return gauge._val

    # ── Initialization ────────────────────────────────────────────────────

    def test_creates_successfully(self):
        m = self._make()
        assert isinstance(m, GodsEyeMetrics)

    def test_server_not_started_on_init(self):
        m = self._make()
        assert m._server_started is False

    def test_multiple_instances_no_duplicate_error(self):
        """Each instance uses isolated registry — no duplicate metric error."""
        m1 = self._make()
        m2 = self._make()
        m3 = self._make()
        assert all(isinstance(x, GodsEyeMetrics) for x in [m1, m2, m3])

    def test_singleton_returns_same_instance(self):
        import monitoring.metrics as mod
        mod._default_metrics = None   # reset for clean test
        m1 = mod.get_metrics()
        m2 = mod.get_metrics()
        assert m1 is m2
        mod._default_metrics = None   # cleanup

    # ── Portfolio ─────────────────────────────────────────────────────────

    def test_update_portfolio_no_crash(self):
        self._make().update_portfolio(self._make_snapshot())

    def test_portfolio_value_set(self):
        m = self._make()
        m.update_portfolio(self._make_snapshot())
        assert self._val(m.portfolio_value) == 1_050_000.0

    def test_portfolio_drawdown_set(self):
        m = self._make()
        m.update_portfolio(self._make_snapshot())
        assert abs(self._val(m.portfolio_drawdown) - 0.045) < 1e-6

    def test_positions_count_set(self):
        m = self._make()
        m.update_portfolio(self._make_snapshot())
        assert self._val(m.n_open_positions) == 2

    def test_trades_today_set(self):
        m = self._make()
        m.update_portfolio(self._make_snapshot())
        assert self._val(m.trades_today) == 3

    # ── Trade recording ───────────────────────────────────────────────────

    def test_record_trade_appends(self):
        m = self._make()
        m.record_trade_close(0.04, True)
        assert len(m._closed_trades) == 1

    def test_win_rate_all_wins(self):
        m = self._make()
        for _ in range(5):
            m.record_trade_close(0.03, True)
        assert m.get_summary()["win_rate"] == 1.0

    def test_win_rate_all_losses(self):
        m = self._make()
        for _ in range(5):
            m.record_trade_close(-0.015, False)
        assert m.get_summary()["win_rate"] == 0.0

    def test_win_rate_mixed(self):
        m = self._make()
        for _ in range(3):
            m.record_trade_close(0.04, True)
        for _ in range(2):
            m.record_trade_close(-0.015, False)
        assert abs(m.get_summary()["win_rate"] - 0.6) < 1e-6

    def test_rolling_window_capped_at_50(self):
        m = self._make()
        for _ in range(60):
            m.record_trade_close(0.02, True)
        assert len(m._closed_trades) == 50

    def test_profit_factor_wins_only(self):
        m = self._make()
        for _ in range(3):
            m.record_trade_close(0.04, True)
        assert self._val(m.profit_factor) > 0

    # ── Signals ───────────────────────────────────────────────────────────

    def test_record_signal_not_blocked(self):
        m = self._make()
        m.record_signal(latency_ms=120.0, blocked=False)

    def test_record_signal_blocked(self):
        m = self._make()
        m.record_signal(latency_ms=80.0, blocked=True, block_rule="RC-08")

    def test_last_signal_time_updated(self):
        m      = self._make()
        before = time.time()
        m.record_signal(latency_ms=100.0, blocked=False)
        assert self._val(m.last_signal_time) >= before

    def test_record_signal_no_rule_no_crash(self):
        m = self._make()
        m.record_signal(latency_ms=50.0, blocked=True)   # no block_rule

    # ── Orders ────────────────────────────────────────────────────────────

    def test_record_order_filled_success_rate_1(self):
        m = self._make()
        m.record_order(filled=True)
        assert m.get_summary()["order_success_rate"] == 1.0

    def test_record_order_mixed_success_rate(self):
        m = self._make()
        m.record_order(filled=True)
        m.record_order(filled=False)
        assert abs(m.get_summary()["order_success_rate"] - 0.5) < 1e-6

    def test_order_window_capped_at_50(self):
        m = self._make()
        for _ in range(60):
            m.record_order(filled=True)
        assert len(m._order_results) == 50

    def test_record_order_with_latency(self):
        m = self._make()
        m.record_order(filled=True, latency_ms=200.0)
        assert self._val(m.broker_latency_ms) == 200.0

    # ── Broker ────────────────────────────────────────────────────────────

    def test_set_active_broker_kite(self):
        m = self._make()
        m.set_active_broker("kite")
        assert self._val(m.active_broker_kite) == 1.0

    def test_set_active_broker_upstox(self):
        m = self._make()
        m.set_active_broker("upstox")
        assert self._val(m.active_broker_kite) == 0.0

    def test_record_failover_no_crash(self):
        self._make().record_failover()

    # ── Model ─────────────────────────────────────────────────────────────

    def test_model_metrics_no_drift(self):
        m = self._make()
        m.update_model_metrics(backbone_latency_ms=45.0, drift_score=0.05)
        assert self._val(m.drift_alert_active) == 0.0
        assert self._val(m.backbone_latency_ms) == 45.0

    def test_model_metrics_drift_alert(self):
        m = self._make()
        m.update_model_metrics(backbone_latency_ms=45.0, drift_score=0.20)
        assert self._val(m.drift_alert_active) == 1.0

    def test_record_retrain_updates_timestamp(self):
        m      = self._make()
        before = time.time()
        m.record_retrain()
        assert self._val(m.last_retrain_time) >= before

    # ── RC ────────────────────────────────────────────────────────────────

    def test_record_rc_trigger_no_crash(self):
        m = self._make()
        m.record_rc_trigger("RC-01")
        m.record_rc_trigger("RC-08")

    def test_set_rc_halt_active(self):
        m = self._make()
        m.set_rc_halt(True)
        assert self._val(m.rc_halt_active) == 1.0

    def test_set_rc_halt_inactive(self):
        m = self._make()
        m.set_rc_halt(True)
        m.set_rc_halt(False)
        assert self._val(m.rc_halt_active) == 0.0

    # ── Paper mode ────────────────────────────────────────────────────────

    def test_set_paper_mode_true(self):
        m = self._make()
        m.set_paper_mode(True)
        assert self._val(m.paper_mode_active) == 1.0

    def test_set_paper_mode_false(self):
        m = self._make()
        m.set_paper_mode(False)
        assert self._val(m.paper_mode_active) == 0.0

    # ── Summary ───────────────────────────────────────────────────────────

    def test_summary_has_required_keys(self):
        m       = self._make()
        summary = m.get_summary()
        for key in ("uptime_hours", "win_rate", "order_success_rate",
                    "n_closed_trades", "prometheus_available"):
            assert key in summary

    def test_summary_no_trades(self):
        m = self._make()
        s = m.get_summary()
        assert s["win_rate"]        == 0.0
        assert s["n_closed_trades"] == 0

    def test_summary_uptime_positive(self):
        m = self._make()
        time.sleep(0.01)
        assert m.get_summary()["uptime_hours"] > 0


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))