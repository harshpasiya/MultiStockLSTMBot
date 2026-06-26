"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Alert Dispatcher                                ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : monitoring/alerts.py                                   ║
║         Phase   : 4 — Paper Trading & Live Monitoring                   ║
║                                                                          ║
║  What this module does:                                                  ║
║    Dispatches real-time alerts to Telegram and logs for all critical    ║
║    system events. Every component imports AlertManager and calls        ║
║    send() — this module handles routing, deduplication, rate-limiting,  ║
║    and formatting.                                                       ║
║                                                                          ║
║  Alert categories:                                                       ║
║    CRITICAL  : RC-01 halt, both brokers down, >10% drawdown            ║
║    WARNING   : Broker failover, >8% drawdown, drift alert, RC triggers  ║
║    INFO      : Trade opened/closed, daily summary, retrain complete     ║
║    DEBUG     : Signal generated, order placed (logged only, no Telegram)║
║                                                                          ║
║  Rate limiting:                                                          ║
║    Same alert type is suppressed for COOLDOWN_SECONDS after first fire  ║
║    Prevents alert storms during volatile market conditions               ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install requests loguru python-dotenv                             ║
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
from typing       import Optional, Dict, List, Callable
from loguru       import logger
from dotenv       import load_dotenv

load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")
TELEGRAM_TIMEOUT = 5      # seconds
MAX_MESSAGE_LEN  = 4000   # Telegram limit is 4096 chars


# ══════════════════════════════════════════════════════════════════════════
#  ENUMERATIONS & DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

class AlertLevel(Enum):
    DEBUG    = 0
    INFO     = 1
    WARNING  = 2
    CRITICAL = 3


class AlertCategory(Enum):
    TRADE       = auto()   # trade opened/closed
    RISK        = auto()   # RC trigger, drawdown breach
    BROKER      = auto()   # failover, both down
    MODEL       = auto()   # drift, retrain
    SYSTEM      = auto()   # startup, shutdown, heartbeat
    PERFORMANCE = auto()   # daily/weekly summary


LEVEL_EMOJI = {
    AlertLevel.DEBUG   : "🔍",
    AlertLevel.INFO    : "ℹ️",
    AlertLevel.WARNING : "⚠️",
    AlertLevel.CRITICAL: "🚨",
}

CATEGORY_EMOJI = {
    AlertCategory.TRADE      : "📈",
    AlertCategory.RISK       : "🛡️",
    AlertCategory.BROKER     : "🔌",
    AlertCategory.MODEL      : "🧠",
    AlertCategory.SYSTEM     : "⚙️",
    AlertCategory.PERFORMANCE: "📊",
}

# Cooldown in seconds per alert key (prevents alert storms)
DEFAULT_COOLDOWNS: Dict[str, int] = {
    "RC-01"           : 0,       # never suppress kill-switch
    "RC-02"           : 60,
    "RC-03"           : 300,
    "RC-06"           : 120,
    "RC-09"           : 0,       # never suppress FII panic
    "broker_failover" : 60,
    "broker_both_down": 0,       # never suppress
    "drawdown_warn"   : 300,
    "drawdown_crit"   : 0,
    "drift_alert"     : 600,
    "trade_open"      : 10,
    "trade_close"     : 10,
    "daily_summary"   : 0,
    "retrain_done"    : 0,
}


@dataclass
class Alert:
    """Represents a single alert event."""
    level      : AlertLevel
    category   : AlertCategory
    title      : str
    body       : str
    key        : str           = ""      # deduplication key
    timestamp  : datetime      = field(default_factory=datetime.now)
    metadata   : Dict          = field(default_factory=dict)

    def format_telegram(self) -> str:
        """Formats alert as a Telegram message string."""
        lvl_e  = LEVEL_EMOJI.get(self.level,    "ℹ️")
        cat_e  = CATEGORY_EMOJI.get(self.category, "")
        ts_str = self.timestamp.strftime("%H:%M:%S")
        lines  = [
            f"{lvl_e} {cat_e} <b>G.O.D.S E.Y.E</b>",
            f"<b>{self.title}</b>",
            f"<i>{ts_str}</i>",
            "",
            self.body,
        ]
        if self.metadata:
            lines.append("")
            for k, v in self.metadata.items():
                lines.append(f"• {k}: {v}")
        msg = "\n".join(lines)
        return msg[:MAX_MESSAGE_LEN]

    def format_log(self) -> str:
        """Formats alert as a loguru log string."""
        return f"[{self.category.name}] {self.title} | {self.body}"


# ══════════════════════════════════════════════════════════════════════════
#  TELEGRAM SENDER
# ══════════════════════════════════════════════════════════════════════════

class TelegramSender:
    """
    Handles Telegram message delivery with retry logic.
    Falls back silently if Telegram is not configured.
    """

    MAX_RETRIES  = 3
    RETRY_DELAY  = 2.0

    def __init__(self, token: str = "", chat_id: str = ""):
        self.token   = token   or TELEGRAM_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self._enabled = bool(self.token and self.chat_id)

        if not self._enabled:
            logger.debug(
                "Telegram not configured — alerts will be logged only. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable."
            )

    @property
    def is_configured(self) -> bool:
        return self._enabled

    def send(self, message: str) -> bool:
        """
        Sends a message to Telegram.

        Args:
            message : HTML-formatted message string

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._enabled:
            return False

        url     = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id"   : self.chat_id,
            "text"      : message,
            "parse_mode": "HTML",
        }

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, timeout=TELEGRAM_TIMEOUT)
                if resp.status_code == 200:
                    return True
                elif resp.status_code == 429:
                    # Rate limited by Telegram
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                    logger.warning(f"Telegram rate limited — retry after {retry_after}s")
                    time.sleep(retry_after)
                else:
                    logger.warning(
                        f"Telegram send failed (attempt {attempt}): "
                        f"HTTP {resp.status_code}"
                    )
            except requests.RequestException as e:
                logger.warning(f"Telegram send error (attempt {attempt}): {e}")

            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY * attempt)

        return False


# ══════════════════════════════════════════════════════════════════════════
#  ALERT MANAGER
# ══════════════════════════════════════════════════════════════════════════

class AlertManager:
    """
    Central alert dispatcher for G.O.D.S E.Y.E.

    Routes alerts to Telegram and loguru based on level.
    Deduplicates via cooldown keys to prevent alert storms.
    Maintains an in-memory alert history for monitoring dashboard.

    Usage:
        alerts = AlertManager()

        # Trade opened
        alerts.trade_opened("RELIANCE", qty=10, price=2850.0, confidence=0.78)

        # Risk Constitution trigger
        alerts.rc_triggered("RC-01", drawdown=0.13)

        # Broker failover
        alerts.broker_failover(from_broker="kite", to_broker="upstox",
                               reason="Health check failed")

        # Daily summary
        alerts.daily_summary(trades=5, pnl_pct=0.023, win_rate=0.6)
    """

    def __init__(
        self,
        telegram    : Optional[TelegramSender] = None,
        min_telegram_level: AlertLevel = AlertLevel.WARNING,
        cooldowns   : Optional[Dict[str, int]] = None,
    ):
        self._telegram   = telegram or TelegramSender()
        self._min_tg_lvl = min_telegram_level
        self._cooldowns  = {**DEFAULT_COOLDOWNS, **(cooldowns or {})}
        self._last_sent  : Dict[str, float] = {}   # key → last send timestamp
        self._history    : List[Alert]      = []   # in-memory history
        self._lock       = threading.Lock()
        self._callbacks  : List[Callable]   = []   # optional extra handlers

        logger.info(
            f"AlertManager initialized | "
            f"Telegram={'configured' if self._telegram.is_configured else 'disabled'} | "
            f"min_telegram_level={min_telegram_level.name}"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  CORE SEND METHOD
    # ══════════════════════════════════════════════════════════════════════

    def send(self, alert: Alert) -> bool:
        """
        Dispatches an alert to all configured channels.

        Routing:
            All levels   → loguru (always)
            WARNING+     → Telegram (if configured and not in cooldown)
            All levels   → registered callbacks

        Args:
            alert : Alert instance to dispatch

        Returns:
            True if alert was sent to at least one channel
        """
        with self._lock:
            # ── Log always ─────────────────────────────────────────────────
            log_fn = {
                AlertLevel.DEBUG   : logger.debug,
                AlertLevel.INFO    : logger.info,
                AlertLevel.WARNING : logger.warning,
                AlertLevel.CRITICAL: logger.critical,
            }.get(alert.level, logger.info)
            log_fn(alert.format_log())

            # ── Store in history ───────────────────────────────────────────
            self._history.append(alert)
            if len(self._history) > 500:
                self._history = self._history[-500:]

            # ── Fire callbacks ─────────────────────────────────────────────
            for cb in self._callbacks:
                try:
                    cb(alert)
                except Exception as e:
                    logger.warning(f"Alert callback error: {e}")

            # ── Telegram (with cooldown check) ─────────────────────────────
            if alert.level.value < self._min_tg_lvl.value:
                return True   # logged, but below Telegram threshold

            key      = alert.key or f"{alert.category.name}_{alert.level.name}"
            cooldown = self._cooldowns.get(key, 60)
            last     = self._last_sent.get(key, 0)
            now      = time.time()

            if cooldown > 0 and (now - last) < cooldown:
                remaining = int(cooldown - (now - last))
                logger.debug(
                    f"Alert '{key}' in cooldown — suppressing Telegram "
                    f"(resumes in {remaining}s)"
                )
                return True   # logged but Telegram suppressed

            sent = self._telegram.send(alert.format_telegram())
            if sent:
                self._last_sent[key] = now

            return True

    def register_callback(self, callback: Callable[[Alert], None]):
        """
        Registers an additional alert handler.
        Called for every alert regardless of level or cooldown.

        Args:
            callback : Callable(alert: Alert) → None
        """
        self._callbacks.append(callback)

    # ══════════════════════════════════════════════════════════════════════
    #  CONVENIENCE METHODS — one per alert type
    # ══════════════════════════════════════════════════════════════════════

    def trade_opened(
        self,
        symbol    : str,
        qty       : int,
        price     : float,
        tp_price  : float,
        sl_price  : float,
        confidence: float,
        mode      : str = "swing",
    ):
        """Alert: new position opened."""
        tp_pct = (tp_price - price) / price * 100
        sl_pct = (price - sl_price) / price * 100
        self.send(Alert(
            level    = AlertLevel.INFO,
            category = AlertCategory.TRADE,
            title    = f"OPENED {symbol}",
            body     = (
                f"Mode: {mode.upper()} | Qty: {qty} @ ₹{price:.2f}\n"
                f"TP: ₹{tp_price:.2f} (+{tp_pct:.1f}%) | "
                f"SL: ₹{sl_price:.2f} (-{sl_pct:.1f}%)"
            ),
            key      = "trade_open",
            metadata = {"confidence": f"{confidence:.2f}", "mode": mode},
        ))

    def trade_closed(
        self,
        symbol    : str,
        exit_price: float,
        pnl_inr   : float,
        pnl_pct   : float,
        reason    : str,
        hold_days : int,
    ):
        """Alert: position closed."""
        emoji  = "✅" if pnl_inr >= 0 else "❌"
        self.send(Alert(
            level    = AlertLevel.INFO,
            category = AlertCategory.TRADE,
            title    = f"{emoji} CLOSED {symbol}",
            body     = (
                f"Exit: ₹{exit_price:.2f} | "
                f"P&L: ₹{pnl_inr:+,.0f} ({pnl_pct:+.2%})\n"
                f"Reason: {reason} | Held: {hold_days} days"
            ),
            key      = "trade_close",
            metadata = {"pnl_inr": f"₹{pnl_inr:+,.0f}", "reason": reason},
        ))

    def rc_triggered(self, rule: str, **details):
        """Alert: Risk Constitution rule triggered."""
        level = AlertLevel.CRITICAL if rule == "RC-01" else AlertLevel.WARNING
        detail_str = " | ".join(f"{k}: {v}" for k, v in details.items())
        self.send(Alert(
            level    = level,
            category = AlertCategory.RISK,
            title    = f"Risk Constitution {rule} Triggered",
            body     = detail_str or f"{rule} fired",
            key      = rule,
            metadata = details,
        ))

    def rc_halt_cleared(self, cleared_by: str):
        """Alert: RC-01 halt has been manually cleared."""
        self.send(Alert(
            level    = AlertLevel.WARNING,
            category = AlertCategory.RISK,
            title    = "RC-01 Halt CLEARED",
            body     = f"Trading resumed. Cleared by: {cleared_by}",
            key      = "RC-01",
        ))

    def broker_failover(
        self,
        from_broker: str,
        to_broker  : str,
        reason     : str,
    ):
        """Alert: broker switched from primary to failover."""
        self.send(Alert(
            level    = AlertLevel.WARNING,
            category = AlertCategory.BROKER,
            title    = f"Broker Failover: {from_broker} → {to_broker}",
            body     = f"Reason: {reason}",
            key      = "broker_failover",
            metadata = {"from": from_broker, "to": to_broker},
        ))

    def broker_both_down(self, reason: str):
        """Alert: both brokers are unavailable — critical."""
        self.send(Alert(
            level    = AlertLevel.CRITICAL,
            category = AlertCategory.BROKER,
            title    = "BOTH BROKERS DOWN — Orders Blocked",
            body     = f"Reason: {reason}\nManual intervention required.",
            key      = "broker_both_down",
        ))

    def broker_recovered(self, broker: str):
        """Alert: primary broker recovered and is active again."""
        self.send(Alert(
            level    = AlertLevel.INFO,
            category = AlertCategory.BROKER,
            title    = f"Broker Recovered: {broker}",
            body     = "Primary broker back online. Traffic restored.",
            key      = "broker_failover",   # resets failover cooldown
        ))

    def drawdown_warning(self, drawdown_pct: float, portfolio_value: float):
        """Alert: drawdown exceeded warning threshold (8%)."""
        self.send(Alert(
            level    = AlertLevel.WARNING,
            category = AlertCategory.RISK,
            title    = f"Drawdown Warning: {drawdown_pct:.1%}",
            body     = (
                f"Portfolio: ₹{portfolio_value:,.0f}\n"
                f"Drawdown: {drawdown_pct:.2%} | Threshold: 8%\n"
                f"Position sizes reduced to 50%."
            ),
            key      = "drawdown_warn",
            metadata = {"drawdown": f"{drawdown_pct:.2%}"},
        ))

    def drawdown_critical(self, drawdown_pct: float, portfolio_value: float):
        """Alert: drawdown approaching RC-01 kill switch (10%+)."""
        self.send(Alert(
            level    = AlertLevel.CRITICAL,
            category = AlertCategory.RISK,
            title    = f"CRITICAL Drawdown: {drawdown_pct:.1%}",
            body     = (
                f"Portfolio: ₹{portfolio_value:,.0f}\n"
                f"Drawdown: {drawdown_pct:.2%} | Kill switch at 12%\n"
                f"No new positions. Review immediately."
            ),
            key      = "drawdown_crit",
            metadata = {"drawdown": f"{drawdown_pct:.2%}"},
        ))

    def drift_alert(self, drift_score: float, threshold: float = 0.15):
        """Alert: feature distribution drift detected."""
        self.send(Alert(
            level    = AlertLevel.WARNING,
            category = AlertCategory.MODEL,
            title    = f"Feature Drift Detected: KL={drift_score:.3f}",
            body     = (
                f"KL divergence {drift_score:.3f} > threshold {threshold:.3f}\n"
                f"Model may be operating outside training distribution.\n"
                f"Consider early retraining."
            ),
            key      = "drift_alert",
            metadata = {"kl_divergence": f"{drift_score:.4f}"},
        ))

    def retrain_complete(self, val_ic: float, val_acc: float, duration_min: float):
        """Alert: nightly retraining pipeline completed."""
        self.send(Alert(
            level    = AlertLevel.INFO,
            category = AlertCategory.MODEL,
            title    = "Nightly Retrain Complete",
            body     = (
                f"Val IC: {val_ic:.4f} | Val Acc: {val_acc:.1%}\n"
                f"Duration: {duration_min:.1f} min | "
                f"Model deployed at {datetime.now().strftime('%H:%M')}"
            ),
            key      = "retrain_done",
            metadata = {"val_ic": f"{val_ic:.4f}", "val_acc": f"{val_acc:.1%}"},
        ))

    def daily_summary(
        self,
        trades        : int,
        pnl_inr       : float,
        pnl_pct       : float,
        win_rate      : float,
        drawdown      : float,
        open_positions: int,
    ):
        """Alert: end-of-day performance summary."""
        emoji = "✅" if pnl_inr >= 0 else "❌"
        self.send(Alert(
            level    = AlertLevel.INFO,
            category = AlertCategory.PERFORMANCE,
            title    = f"{emoji} Daily Summary",
            body     = (
                f"Trades: {trades} | P&L: ₹{pnl_inr:+,.0f} ({pnl_pct:+.2%})\n"
                f"Win Rate: {win_rate:.1%} | Max DD: {drawdown:.2%}\n"
                f"Open Positions: {open_positions}"
            ),
            key      = "daily_summary",
            metadata = {
                "trades"  : trades,
                "pnl"     : f"₹{pnl_inr:+,.0f}",
                "win_rate": f"{win_rate:.1%}",
            },
        ))

    def system_startup(self, paper_mode: bool, capital: float):
        """Alert: system started."""
        mode = "PAPER" if paper_mode else "LIVE"
        self.send(Alert(
            level    = AlertLevel.INFO,
            category = AlertCategory.SYSTEM,
            title    = f"G.O.D.S E.Y.E Started ({mode})",
            body     = (
                f"Mode: {mode} | Capital: ₹{capital:,.0f}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ),
            key      = "system_startup",
        ))

    def system_shutdown(self, reason: str = "manual"):
        """Alert: system stopped."""
        self.send(Alert(
            level    = AlertLevel.INFO,
            category = AlertCategory.SYSTEM,
            title    = "G.O.D.S E.Y.E Stopped",
            body     = f"Reason: {reason}",
            key      = "system_shutdown",
        ))

    def fii_panic(self, fii_selling_cr: float):
        """Alert: FII selling exceeded RC-09 threshold."""
        self.send(Alert(
            level    = AlertLevel.WARNING,
            category = AlertCategory.RISK,
            title    = f"RC-09: FII Panic Selling ₹{fii_selling_cr:.0f}cr",
            body     = (
                f"FII provisional selling: ₹{fii_selling_cr:.0f} crore\n"
                f"All long signals blocked today and tomorrow.\n"
                f"MDS forced to -3."
            ),
            key      = "RC-09",
            metadata = {"fii_selling_cr": f"₹{fii_selling_cr:.0f}cr"},
        ))

    # ══════════════════════════════════════════════════════════════════════
    #  HISTORY & STATE
    # ══════════════════════════════════════════════════════════════════════

    def get_history(
        self,
        level    : Optional[AlertLevel]    = None,
        category : Optional[AlertCategory] = None,
        last_n   : int = 50,
    ) -> List[Alert]:
        """
        Returns recent alert history, optionally filtered.

        Args:
            level    : Filter by minimum alert level
            category : Filter by alert category
            last_n   : Return last N alerts

        Returns:
            List of Alert objects, newest last
        """
        with self._lock:
            results = list(self._history)

        if level is not None:
            results = [a for a in results if a.level.value >= level.value]
        if category is not None:
            results = [a for a in results if a.category == category]

        return results[-last_n:]

    def get_cooldown_state(self) -> Dict[str, float]:
        """Returns remaining cooldown seconds per alert key."""
        now   = time.time()
        state = {}
        with self._lock:
            for key, cooldown in self._cooldowns.items():
                last      = self._last_sent.get(key, 0)
                remaining = max(0.0, cooldown - (now - last))
                if remaining > 0:
                    state[key] = remaining
        return state

    def reset_cooldown(self, key: str):
        """
        Manually resets cooldown for an alert key.
        Useful in tests and for clearing suppressed critical alerts.
        """
        with self._lock:
            self._last_sent.pop(key, None)

    def clear_history(self):
        """Clears alert history (use in tests or for memory management)."""
        with self._lock:
            self._history.clear()
            self._last_sent.clear()


# ── Module-level singleton ─────────────────────────────────────────────────
_default_alerts: Optional[AlertManager] = None


def get_alerts() -> AlertManager:
    """
    Returns the module-level AlertManager singleton.

    Example:
        from monitoring.alerts import get_alerts
        alerts = get_alerts()
        alerts.trade_opened("RELIANCE", qty=10, price=2850.0,
                            tp_price=2964.0, sl_price=2807.25,
                            confidence=0.78)
    """
    global _default_alerts
    if _default_alerts is None:
        _default_alerts = AlertManager()
    return _default_alerts


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest monitoring/alerts.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestAlertManager:
    """
    Unit tests for AlertManager and TelegramSender.
    All tests use unconfigured Telegram (no real HTTP calls).
    """

    def _make_am(self, min_level=AlertLevel.WARNING) -> AlertManager:
        """Creates AlertManager with Telegram disabled."""
        sender = TelegramSender(token="", chat_id="")
        am     = AlertManager(
            telegram           = sender,
            min_telegram_level = min_level,
            cooldowns          = {k: 0 for k in DEFAULT_COOLDOWNS},  # all cooldowns=0
        )
        return am

    def _make_alert(self, level=AlertLevel.INFO, key="test") -> Alert:
        return Alert(
            level    = level,
            category = AlertCategory.TRADE,
            title    = "Test Alert",
            body     = "Test body",
            key      = key,
        )

    # ── TelegramSender ────────────────────────────────────────────────────

    def test_telegram_not_configured(self):
        s = TelegramSender(token="", chat_id="")
        assert not s.is_configured

    def test_telegram_configured(self):
        s = TelegramSender(token="fake_token", chat_id="12345")
        assert s.is_configured

    def test_telegram_send_unconfigured_returns_false(self):
        s = TelegramSender(token="", chat_id="")
        assert s.send("test") is False

    # ── Alert formatting ──────────────────────────────────────────────────

    def test_alert_format_telegram_contains_title(self):
        a = self._make_alert()
        msg = a.format_telegram()
        assert "Test Alert" in msg

    def test_alert_format_telegram_within_length(self):
        a = Alert(
            level=AlertLevel.INFO, category=AlertCategory.TRADE,
            title="T", body="B" * 5000,
        )
        assert len(a.format_telegram()) <= MAX_MESSAGE_LEN

    def test_alert_format_log_contains_title(self):
        a = self._make_alert()
        assert "Test Alert" in a.format_log()

    def test_alert_format_log_contains_category(self):
        a = self._make_alert()
        assert "TRADE" in a.format_log()

    def test_alert_metadata_in_telegram(self):
        a = Alert(
            level=AlertLevel.INFO, category=AlertCategory.TRADE,
            title="T", body="B", metadata={"key1": "val1"},
        )
        assert "key1" in a.format_telegram()
        assert "val1" in a.format_telegram()

    # ── AlertManager core ─────────────────────────────────────────────────

    def test_send_stores_in_history(self):
        am = self._make_am()
        am.send(self._make_alert())
        assert len(am.get_history()) == 1

    def test_send_multiple_stored(self):
        am = self._make_am()
        for _ in range(5):
            am.send(self._make_alert())
        assert len(am.get_history()) == 5

    def test_history_capped_at_500(self):
        am = self._make_am()
        for i in range(510):
            am.send(self._make_alert(key=f"k{i}"))
        assert len(am._history) == 500

    def test_send_returns_true(self):
        am = self._make_am()
        assert am.send(self._make_alert()) is True

    # ── Cooldown ──────────────────────────────────────────────────────────

    def test_cooldown_suppresses_telegram(self):
        """With non-zero cooldown, second send should not reach Telegram."""
        sent_calls = []
        am = AlertManager(
            telegram=TelegramSender(token="", chat_id=""),
            cooldowns={"test_key": 300},   # 5 min cooldown
        )
        am.send(Alert(level=AlertLevel.WARNING, category=AlertCategory.RISK,
                      title="T", body="B", key="test_key"))
        am.send(Alert(level=AlertLevel.WARNING, category=AlertCategory.RISK,
                      title="T", body="B", key="test_key"))
        # Both should be in history (logged always)
        assert len(am.get_history()) == 2

    def test_cooldown_zero_always_sends(self):
        am = self._make_am()   # all cooldowns = 0
        for _ in range(3):
            am.send(Alert(level=AlertLevel.WARNING, category=AlertCategory.RISK,
                          title="T", body="B", key="RC-01"))
        assert len(am.get_history()) == 3

    def test_reset_cooldown(self):
        am = AlertManager(
            telegram=TelegramSender(token="", chat_id=""),
            cooldowns={"test_key": 300},
        )
        am._last_sent["test_key"] = time.time()
        state = am.get_cooldown_state()
        assert "test_key" in state and state["test_key"] > 0
        am.reset_cooldown("test_key")
        assert "test_key" not in am.get_cooldown_state()

    # ── Callbacks ─────────────────────────────────────────────────────────

    def test_callback_called_on_send(self):
        am      = self._make_am()
        received = []
        am.register_callback(lambda a: received.append(a))
        am.send(self._make_alert())
        assert len(received) == 1
        assert isinstance(received[0], Alert)

    def test_callback_level_correct(self):
        am      = self._make_am()
        received = []
        am.register_callback(lambda a: received.append(a.level))
        am.send(self._make_alert(level=AlertLevel.CRITICAL))
        assert received[0] == AlertLevel.CRITICAL

    def test_callback_error_doesnt_crash(self):
        am = self._make_am()
        am.register_callback(lambda a: 1/0)   # raises ZeroDivisionError
        am.send(self._make_alert())   # should not crash

    def test_multiple_callbacks(self):
        am   = self._make_am()
        c1, c2 = [], []
        am.register_callback(lambda a: c1.append(a))
        am.register_callback(lambda a: c2.append(a))
        am.send(self._make_alert())
        assert len(c1) == 1 and len(c2) == 1

    # ── History filtering ─────────────────────────────────────────────────

    def test_history_filter_by_level(self):
        am = self._make_am(min_level=AlertLevel.DEBUG)
        am.send(self._make_alert(level=AlertLevel.INFO))
        am.send(self._make_alert(level=AlertLevel.WARNING))
        am.send(self._make_alert(level=AlertLevel.CRITICAL))
        warnings_plus = am.get_history(level=AlertLevel.WARNING)
        assert all(a.level.value >= AlertLevel.WARNING.value for a in warnings_plus)

    def test_history_filter_by_category(self):
        am = self._make_am()
        am.send(Alert(level=AlertLevel.INFO, category=AlertCategory.TRADE,
                      title="T", body="B"))
        am.send(Alert(level=AlertLevel.INFO, category=AlertCategory.BROKER,
                      title="B", body="B"))
        trades = am.get_history(category=AlertCategory.TRADE)
        assert all(a.category == AlertCategory.TRADE for a in trades)

    def test_history_last_n(self):
        am = self._make_am()
        for i in range(20):
            am.send(self._make_alert(key=f"k{i}"))
        assert len(am.get_history(last_n=5)) == 5

    def test_clear_history(self):
        am = self._make_am()
        am.send(self._make_alert())
        am.clear_history()
        assert len(am.get_history()) == 0

    # ── Convenience methods ───────────────────────────────────────────────

    def test_trade_opened_no_crash(self):
        am = self._make_am()
        am.trade_opened("RELIANCE", qty=10, price=2850.0,
                        tp_price=2964.0, sl_price=2807.25, confidence=0.78)
        assert len(am.get_history()) == 1

    def test_trade_closed_positive_pnl(self):
        am = self._make_am()
        am.trade_closed("RELIANCE", exit_price=2964.0,
                        pnl_inr=1140.0, pnl_pct=0.04,
                        reason="CLOSED_TP", hold_days=5)
        h = am.get_history()
        assert len(h) == 1
        assert "CLOSED_TP" in h[0].body

    def test_trade_closed_negative_pnl(self):
        am = self._make_am()
        am.trade_closed("RELIANCE", exit_price=2807.0,
                        pnl_inr=-428.0, pnl_pct=-0.015,
                        reason="CLOSED_SL", hold_days=2)
        h = am.get_history()
        assert "❌" in h[0].title

    def test_rc_triggered_rc01_is_critical(self):
        am = self._make_am(min_level=AlertLevel.DEBUG)
        am.rc_triggered("RC-01", drawdown="13%")
        h = am.get_history()
        assert h[0].level == AlertLevel.CRITICAL

    def test_rc_triggered_rc08_is_warning(self):
        am = self._make_am(min_level=AlertLevel.DEBUG)
        am.rc_triggered("RC-08", positions=4)
        h = am.get_history()
        assert h[0].level == AlertLevel.WARNING

    def test_broker_failover_no_crash(self):
        am = self._make_am()
        am.broker_failover("kite", "upstox", "Health check failed 3x")
        assert len(am.get_history()) == 1

    def test_broker_both_down_is_critical(self):
        am = self._make_am(min_level=AlertLevel.DEBUG)
        am.broker_both_down("Both APIs unreachable")
        assert am.get_history()[-1].level == AlertLevel.CRITICAL

    def test_drawdown_warning_no_crash(self):
        am = self._make_am()
        am.drawdown_warning(0.085, 915_000.0)
        assert len(am.get_history()) == 1

    def test_drawdown_critical_is_critical(self):
        am = self._make_am(min_level=AlertLevel.DEBUG)
        am.drawdown_critical(0.11, 890_000.0)
        assert am.get_history()[-1].level == AlertLevel.CRITICAL

    def test_drift_alert_no_crash(self):
        am = self._make_am()
        am.drift_alert(drift_score=0.18)
        assert len(am.get_history()) == 1

    def test_retrain_complete_no_crash(self):
        am = self._make_am()
        am.retrain_complete(val_ic=0.065, val_acc=0.57, duration_min=45.0)

    def test_daily_summary_no_crash(self):
        am = self._make_am()
        am.daily_summary(
            trades=5, pnl_inr=12_000, pnl_pct=0.012,
            win_rate=0.6, drawdown=0.02, open_positions=2,
        )

    def test_system_startup_no_crash(self):
        am = self._make_am()
        am.system_startup(paper_mode=True, capital=1_000_000.0)

    def test_system_shutdown_no_crash(self):
        am = self._make_am()
        am.system_shutdown(reason="manual")

    def test_fii_panic_no_crash(self):
        am = self._make_am()
        am.fii_panic(fii_selling_cr=3500.0)

    # ── Singleton ─────────────────────────────────────────────────────────

    def test_singleton_same_instance(self):
        import monitoring.alerts as mod
        mod._default_alerts = None
        a1 = mod.get_alerts()
        a2 = mod.get_alerts()
        assert a1 is a2
        mod._default_alerts = None


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))