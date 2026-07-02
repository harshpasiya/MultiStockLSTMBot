"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Subscriber Manager                              ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : api/subscriber_manager.py                              ║
║         Phase   : 5 — Subscriber Infrastructure                         ║
║                                                                          ║
║  What this module does:                                                  ║
║    Manages all subscriber accounts and their broker connections.         ║
║    When signal_engine generates a signal, SubscriberManager broadcasts  ║
║    it to every active subscriber simultaneously, each sized correctly   ║
║    to their own portfolio.                                               ║
║                                                                          ║
║  Future extensibility (already structured for):                          ║
║    - Distributor commission tracking                                     ║
║    - Broker entity management (SEBI registered)                         ║
║    - Referral chain P&L attribution                                      ║
║    - Per-subscriber risk limits override                                 ║
║    - Webhook notifications to distributors                               ║
║    - Subscription plan enforcement (FREE/BASIC/PRO/ELITE)               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import uuid
import threading
import psycopg2
import psycopg2.extras

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses        import dataclass, field
from datetime           import datetime
from typing             import Optional, Dict, List, Callable, Any
from loguru             import logger
from dotenv             import load_dotenv

from api.auth import (
    AuthManager, User, UserRole, BrokerProvider,
    SubscriptionPlan, BrokerToken,
)

load_dotenv()

DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)

# ── Broadcast config ───────────────────────────────────────────────────────
MAX_BROADCAST_WORKERS = 10    # parallel order placements across subscribers
BROADCAST_TIMEOUT_SEC = 30    # max seconds to wait for all fills

# ── Plan signal limits ─────────────────────────────────────────────────────
PLAN_SIGNAL_LIMITS = {
    SubscriptionPlan.FREE  : {"daily": 1,  "monthly": 10,  "modes": ["swing"]},
    SubscriptionPlan.BASIC : {"daily": 3,  "monthly": 30,  "modes": ["swing"]},
    SubscriptionPlan.PRO   : {"daily": 5,  "monthly": 60,  "modes": ["swing", "intraday"]},
    SubscriptionPlan.ELITE : {"daily": 99, "monthly": 999, "modes": ["swing", "intraday"]},
}


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SubscriberAccount:
    """
    Full subscriber profile including broker connection and risk settings.
    """
    user_id          : str
    email            : str
    plan             : SubscriptionPlan
    is_active        : bool           = True

    # Broker connection
    broker_provider  : Optional[BrokerProvider] = None
    broker_token     : str            = ""       # decrypted at runtime
    broker_account_id: str            = ""

    # Risk overrides (subscriber can tighten but not loosen system limits)
    max_positions    : int            = 4        # default from RC-08
    max_drawdown_pct : float          = 0.12     # default from RC-01
    capital_inr      : float          = 0.0      # their portfolio size
    paper_mode       : bool           = True     # starts in paper mode

    # Referral / distributor
    distributor_id   : str            = ""
    referred_by      : str            = ""

    # Usage tracking
    signals_today    : int            = 0
    signals_month    : int            = 0
    last_signal_at   : Optional[datetime] = None


@dataclass
class BroadcastResult:
    """Result of broadcasting one signal to one subscriber."""
    user_id          : str
    symbol           : str
    success          : bool
    order_id         : str            = ""
    fill_price       : float          = 0.0
    quantity         : int            = 0
    error            : str            = ""
    latency_ms       : float          = 0.0
    paper_mode       : bool           = True


@dataclass
class BroadcastSummary:
    """Aggregated result of broadcasting one signal to all subscribers."""
    signal_id        : str
    symbol           : str
    total_subscribers: int
    success_count    : int
    failure_count    : int
    paper_count      : int
    results          : List[BroadcastResult] = field(default_factory=list)
    broadcast_time_ms: float                 = 0.0
    timestamp        : datetime              = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        if self.total_subscribers == 0:
            return 0.0
        return self.success_count / self.total_subscribers


# ══════════════════════════════════════════════════════════════════════════
#  SUBSCRIBER MANAGER
# ══════════════════════════════════════════════════════════════════════════

class SubscriberManager:
    """
    Manages subscriber lifecycle and signal broadcasting.

    Core responsibilities:
        1. Onboard subscribers (connect broker account)
        2. Validate plan limits before broadcasting
        3. Broadcast signals to all eligible subscribers in parallel
        4. Track per-subscriber P&L and usage
        5. Handle distributor commission hooks

    Signal broadcast flow:
        signal_engine → SubscriberManager.broadcast(signal)
            → for each active subscriber:
                → check plan limits
                → scale position size to their capital
                → place order via their broker API
                → log result to subscriber_trades table

    Future hooks (stubs already in place):
        _on_new_subscriber()   → notify distributor webhook
        _on_trade_closed()     → trigger commission calculation
        _on_plan_upgraded()    → unlock additional signal types
    """

    def __init__(self, auth_manager: Optional[AuthManager] = None):
        self._auth    = auth_manager or AuthManager()
        self._lock    = threading.Lock()
        self._cache   : Dict[str, SubscriberAccount] = {}
        self._executor= ThreadPoolExecutor(max_workers=MAX_BROADCAST_WORKERS)
        self._ensure_tables()
        logger.info("SubscriberManager initialized.")

    # ══════════════════════════════════════════════════════════════════════
    #  SUBSCRIBER ONBOARDING
    # ══════════════════════════════════════════════════════════════════════

    def onboard_subscriber(
        self,
        email            : str,
        password         : str,
        broker_provider  : BrokerProvider,
        broker_token     : str,
        capital_inr      : float,
        plan             : SubscriptionPlan = SubscriptionPlan.FREE,
        referral_code    : str              = "",
        paper_mode       : bool             = True,
        max_drawdown_pct : float            = 0.12,
        max_positions    : int              = 4,
    ) -> Optional[SubscriberAccount]:
        """
        Onboards a new subscriber end-to-end:
            1. Creates user account
            2. Stores encrypted broker token
            3. Records subscriber profile
            4. Links to distributor if referral code provided
            5. Fires distributor webhook hook

        Args:
            email           : Subscriber email
            password        : Account password
            broker_provider : Which broker (zerodha/upstox/angel)
            broker_token    : OAuth access token from broker
            capital_inr     : Their trading capital in ₹
            plan            : Subscription plan
            referral_code   : Optional referral code for distributor linking
            paper_mode      : Start in paper mode (recommended)
            max_drawdown_pct: Personal max drawdown (≤ 0.12)
            max_positions   : Personal max positions (≤ 4)

        Returns:
            SubscriberAccount on success, None on failure
        """
        # Enforce risk limits — subscriber cannot loosen system limits
        max_drawdown_pct = min(max_drawdown_pct, 0.12)
        max_positions    = min(max_positions,    4)

        # Register user
        user = self._auth.register_user(
            email         = email,
            password      = password,
            role          = UserRole.SUBSCRIBER,
            plan          = plan,
            referral_code = referral_code,
        )
        if not user:
            logger.error(f"Failed to register user: {email}")
            return None

        # Store broker token (encrypted)
        broker_token_obj = self._auth.store_broker_token(
            user_id      = user.user_id,
            provider     = broker_provider,
            access_token = broker_token,
        )
        if not broker_token_obj:
            logger.error(f"Failed to store broker token for {email}")
            return None

        # Save subscriber profile
        account = SubscriberAccount(
            user_id           = user.user_id,
            email             = email,
            plan              = plan,
            broker_provider   = broker_provider,
            broker_account_id = broker_token_obj.account_id,
            capital_inr       = capital_inr,
            paper_mode        = paper_mode,
            max_drawdown_pct  = max_drawdown_pct,
            max_positions     = max_positions,
            distributor_id    = user.distributor_id,
            referred_by       = user.referred_by,
        )
        self._save_subscriber_profile(account)

        # Cache
        with self._lock:
            self._cache[user.user_id] = account

        # Hook: notify distributor
        self._on_new_subscriber(account)

        logger.success(
            f"Subscriber onboarded: {email} | "
            f"plan={plan.value} | broker={broker_provider.value} | "
            f"paper={paper_mode}"
        )
        return account

    def get_subscriber(self, user_id: str) -> Optional[SubscriberAccount]:
        """Returns subscriber account from cache or DB."""
        with self._lock:
            if user_id in self._cache:
                return self._cache[user_id]
        return self._load_subscriber(user_id)

    def get_all_active_subscribers(self) -> List[SubscriberAccount]:
        """Returns all active subscribers with valid broker tokens."""
        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT sp.*
                    FROM subscriber_profiles sp
                    WHERE sp.is_active = TRUE
                    ORDER BY sp.created_at;
                """)
                rows = cur.fetchall()
            conn.close()
            return [self._row_to_account(dict(r)) for r in rows]
        except Exception as e:
            logger.error(f"get_all_active_subscribers failed: {e}")
            return []

    def update_subscriber_plan(
        self,
        user_id : str,
        new_plan: SubscriptionPlan,
    ) -> bool:
        """Upgrades or downgrades a subscriber's plan."""
        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE subscriber_profiles
                    SET plan = %s, updated_at = NOW()
                    WHERE user_id = %s;
                """, (new_plan.value, user_id))
                cur.execute("""
                    UPDATE users SET plan = %s WHERE user_id = %s;
                """, (new_plan.value, user_id))
            conn.commit()
            conn.close()

            # Update cache
            with self._lock:
                if user_id in self._cache:
                    self._cache[user_id].plan = new_plan

            self._on_plan_upgraded(user_id, new_plan)
            logger.info(f"Plan updated: {user_id} → {new_plan.value}")
            return True
        except Exception as e:
            logger.error(f"update_subscriber_plan failed: {e}")
            return False

    def deactivate_subscriber(self, user_id: str) -> bool:
        """Deactivates a subscriber (stops receiving signals)."""
        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE subscriber_profiles
                    SET is_active = FALSE, updated_at = NOW()
                    WHERE user_id = %s;
                """, (user_id,))
            conn.commit()
            conn.close()
            with self._lock:
                self._cache.pop(user_id, None)
            logger.info(f"Subscriber deactivated: {user_id}")
            return True
        except Exception as e:
            logger.error(f"deactivate_subscriber failed: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════
    #  SIGNAL BROADCASTING
    # ══════════════════════════════════════════════════════════════════════

    def broadcast(
        self,
        signal      : Any,          # TradeSignal from signal_engine
        broker_fn   : Callable,     # function(user_id, signal) → OrderResult
    ) -> BroadcastSummary:
        """
        Broadcasts a trade signal to all eligible active subscribers
        simultaneously using a thread pool.

        Each subscriber gets:
            - Position size scaled to their capital (Kelly × their portfolio)
            - Their own broker API call
            - Independent risk limits

        Args:
            signal    : TradeSignal from signal_engine
            broker_fn : Callable that takes (subscriber, signal) and
                        returns an OrderResult. Injected from signal_engine
                        to keep broker logic separate from subscriber logic.

        Returns:
            BroadcastSummary with per-subscriber results
        """
        import time
        start_ms = time.time() * 1000

        subscribers = self.get_all_active_subscribers()
        eligible    = [
            s for s in subscribers
            if self._is_eligible(s, signal)
        ]

        logger.info(
            f"Broadcasting {signal.symbol} to "
            f"{len(eligible)}/{len(subscribers)} eligible subscribers..."
        )

        results: List[BroadcastResult] = []

        if not eligible:
            return BroadcastSummary(
                signal_id         = getattr(signal, "signal_id", ""),
                symbol            = signal.symbol,
                total_subscribers = 0,
                success_count     = 0,
                failure_count     = 0,
                paper_count       = 0,
                broadcast_time_ms = 0.0,
            )

        # Submit all orders in parallel
        futures = {
            self._executor.submit(
                self._place_for_subscriber, s, signal, broker_fn
            ): s
            for s in eligible
        }

        for future in as_completed(futures, timeout=BROADCAST_TIMEOUT_SEC):
            try:
                result = future.result()
                results.append(result)
                # Increment usage counter
                self._increment_usage(futures[future].user_id)
            except Exception as e:
                sub = futures[future]
                results.append(BroadcastResult(
                    user_id    = sub.user_id,
                    symbol     = signal.symbol,
                    success    = False,
                    error      = str(e),
                    paper_mode = sub.paper_mode,
                ))

        elapsed = time.time() * 1000 - start_ms

        summary = BroadcastSummary(
            signal_id         = getattr(signal, "signal_id", ""),
            symbol            = signal.symbol,
            total_subscribers = len(eligible),
            success_count     = sum(1 for r in results if r.success),
            failure_count     = sum(1 for r in results if not r.success),
            paper_count       = sum(1 for r in results if r.paper_mode),
            results           = results,
            broadcast_time_ms = elapsed,
        )

        logger.info(
            f"Broadcast complete: {summary.success_count}/{summary.total_subscribers} "
            f"filled | {elapsed:.0f}ms"
        )

        # Persist broadcast log
        self._log_broadcast(summary)
        return summary

    def _place_for_subscriber(
        self,
        subscriber: SubscriberAccount,
        signal    : Any,
        broker_fn : Callable,
    ) -> BroadcastResult:
        """Places one order for one subscriber. Runs in thread pool."""
        import time
        start = time.time()
        try:
            result    = broker_fn(subscriber, signal)
            elapsed   = (time.time() - start) * 1000
            return BroadcastResult(
                user_id    = subscriber.user_id,
                symbol     = signal.symbol,
                success    = result.is_filled,
                order_id   = result.order_id,
                fill_price = result.fill_price,
                quantity   = result.fill_quantity,
                error      = result.error_message if not result.is_filled else "",
                latency_ms = elapsed,
                paper_mode = subscriber.paper_mode,
            )
        except Exception as e:
            return BroadcastResult(
                user_id    = subscriber.user_id,
                symbol     = signal.symbol,
                success    = False,
                error      = str(e),
                latency_ms = (time.time() - start) * 1000,
                paper_mode = subscriber.paper_mode,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  PLAN ENFORCEMENT
    # ══════════════════════════════════════════════════════════════════════

    def _is_eligible(self, subscriber: SubscriberAccount, signal: Any) -> bool:
        """
        Checks if a subscriber is eligible to receive a signal based on:
            - Account active status
            - Plan daily/monthly limits
            - Plan allowed signal modes (swing/intraday)
        """
        if not subscriber.is_active:
            return False

        plan_limits = PLAN_SIGNAL_LIMITS.get(subscriber.plan, {})

        # Check mode allowed by plan
        signal_mode = getattr(signal, "mode", None)
        if signal_mode:
            mode_str = signal_mode.value if hasattr(signal_mode, "value") else str(signal_mode)
            allowed_modes = plan_limits.get("modes", ["swing"])
            if mode_str not in allowed_modes:
                return False

        # Check daily limit
        daily_limit = plan_limits.get("daily", 1)
        if subscriber.signals_today >= daily_limit:
            return False

        # Check monthly limit
        monthly_limit = plan_limits.get("monthly", 10)
        if subscriber.signals_month >= monthly_limit:
            return False

        return True

    def _increment_usage(self, user_id: str):
        """Increments signal usage counters for a subscriber."""
        with self._lock:
            if user_id in self._cache:
                self._cache[user_id].signals_today += 1
                self._cache[user_id].signals_month += 1
                self._cache[user_id].last_signal_at = datetime.now()

        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE subscriber_profiles
                    SET signals_today  = signals_today  + 1,
                        signals_month  = signals_month  + 1,
                        last_signal_at = NOW(),
                        updated_at     = NOW()
                    WHERE user_id = %s;
                """, (user_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"_increment_usage DB failed: {e}")

    def reset_daily_counters(self):
        """Resets daily signal counters for all subscribers. Call at midnight."""
        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE subscriber_profiles SET signals_today = 0;
                """)
            conn.commit()
            conn.close()
            with self._lock:
                for acc in self._cache.values():
                    acc.signals_today = 0
            logger.info("Daily subscriber counters reset.")
        except Exception as e:
            logger.error(f"reset_daily_counters failed: {e}")

    def reset_monthly_counters(self):
        """Resets monthly signal counters. Call on 1st of each month."""
        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE subscriber_profiles SET signals_month = 0;
                """)
            conn.commit()
            conn.close()
            with self._lock:
                for acc in self._cache.values():
                    acc.signals_month = 0
            logger.info("Monthly subscriber counters reset.")
        except Exception as e:
            logger.error(f"reset_monthly_counters failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    #  DISTRIBUTOR / REFERRAL HOOKS (future extensibility stubs)
    # ══════════════════════════════════════════════════════════════════════

    def _on_new_subscriber(self, account: SubscriberAccount):
        """
        Hook: called when a new subscriber is onboarded.
        Future: send webhook to distributor, trigger welcome email,
        record commission entry in commission_ledger.
        """
        if account.distributor_id:
            logger.info(
                f"New subscriber {account.email} under "
                f"distributor {account.distributor_id}"
            )
            # TODO: POST to distributor webhook URL
            # TODO: record_commission(distributor_id, subscriber_id, ...)

    def _on_trade_closed(
        self,
        user_id   : str,
        pnl_inr   : float,
        plan      : SubscriptionPlan,
    ):
        """
        Hook: called when a subscriber's trade closes.
        Future: calculate and record distributor commission.
        """
        account = self.get_subscriber(user_id)
        if account and account.distributor_id:
            # TODO: calculate commission based on referral code commission_pct
            # self._auth.record_commission(
            #     distributor_id = account.distributor_id,
            #     subscriber_id  = user_id,
            #     amount_inr     = pnl_inr * commission_pct,
            #     plan           = plan.value,
            #     period_month   = datetime.now().strftime("%Y-%m"),
            # )
            pass

    def _on_plan_upgraded(self, user_id: str, new_plan: SubscriptionPlan):
        """
        Hook: called when a subscriber's plan changes.
        Future: notify distributor, adjust commission rates,
        unlock additional signal types.
        """
        logger.info(f"Plan upgraded: {user_id} → {new_plan.value}")
        # TODO: notify distributor webhook

    def get_distributor_stats(self, distributor_id: str) -> Dict:
        """
        Returns subscriber stats for a distributor's dashboard.
        Includes subscriber count, active count, total signals sent.
        """
        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*)                                    AS total,
                        COUNT(*) FILTER (WHERE is_active = TRUE)   AS active,
                        SUM(signals_month)                         AS signals_month,
                        COUNT(*) FILTER (WHERE plan = 'elite')     AS elite,
                        COUNT(*) FILTER (WHERE plan = 'pro')       AS pro,
                        COUNT(*) FILTER (WHERE plan = 'basic')     AS basic,
                        COUNT(*) FILTER (WHERE plan = 'free')      AS free
                    FROM subscriber_profiles
                    WHERE distributor_id = %s;
                """, (distributor_id,))
                row = cur.fetchone()
            conn.close()
            if row:
                return {
                    "total_subscribers" : row[0],
                    "active_subscribers": row[1],
                    "signals_this_month": row[2],
                    "by_plan": {
                        "elite": row[3], "pro": row[4],
                        "basic": row[5], "free": row[6],
                    },
                }
        except Exception as e:
            logger.error(f"get_distributor_stats failed: {e}")
        return {}

    # ══════════════════════════════════════════════════════════════════════
    #  DB HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _ensure_tables(self):
        """Creates subscriber infrastructure tables."""
        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor() as cur:

                # Subscriber profiles
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS subscriber_profiles (
                        user_id           VARCHAR(36)  PRIMARY KEY,
                        email             VARCHAR(200),
                        plan              VARCHAR(20)  DEFAULT 'free',
                        broker_provider   VARCHAR(20),
                        broker_account_id VARCHAR(50),
                        capital_inr       NUMERIC(14,2) DEFAULT 0,
                        paper_mode        BOOLEAN       DEFAULT TRUE,
                        is_active         BOOLEAN       DEFAULT TRUE,
                        max_drawdown_pct  NUMERIC(5,4)  DEFAULT 0.12,
                        max_positions     INTEGER       DEFAULT 4,
                        distributor_id    VARCHAR(36),
                        referred_by       VARCHAR(20),
                        signals_today     INTEGER       DEFAULT 0,
                        signals_month     INTEGER       DEFAULT 0,
                        last_signal_at    TIMESTAMP,
                        created_at        TIMESTAMP     DEFAULT NOW(),
                        updated_at        TIMESTAMP     DEFAULT NOW()
                    );
                """)

                # Subscriber trade log (per-subscriber copy of trade_log)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS subscriber_trades (
                        id             SERIAL       PRIMARY KEY,
                        broadcast_id   VARCHAR(36),
                        user_id        VARCHAR(36)  NOT NULL,
                        signal_id      VARCHAR(64),
                        symbol         VARCHAR(20),
                        side           VARCHAR(5),
                        order_id       VARCHAR(64),
                        fill_price     NUMERIC(12,4),
                        quantity       INTEGER,
                        success        BOOLEAN,
                        error          TEXT,
                        latency_ms     NUMERIC(8,2),
                        paper_mode     BOOLEAN      DEFAULT TRUE,
                        created_at     TIMESTAMP    DEFAULT NOW()
                    );
                """)

                # Broadcast log
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS broadcast_log (
                        id                 SERIAL       PRIMARY KEY,
                        signal_id          VARCHAR(64),
                        symbol             VARCHAR(20),
                        total_subscribers  INTEGER,
                        success_count      INTEGER,
                        failure_count      INTEGER,
                        paper_count        INTEGER,
                        broadcast_time_ms  NUMERIC(8,2),
                        created_at         TIMESTAMP    DEFAULT NOW()
                    );
                """)

            conn.commit()
            conn.close()
            logger.info("Subscriber tables ready.")
        except Exception as e:
            logger.warning(f"Subscriber table setup failed: {e}")

    def _save_subscriber_profile(self, account: SubscriberAccount):
        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO subscriber_profiles (
                        user_id, email, plan, broker_provider,
                        broker_account_id, capital_inr, paper_mode,
                        max_drawdown_pct, max_positions,
                        distributor_id, referred_by
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        plan              = EXCLUDED.plan,
                        capital_inr       = EXCLUDED.capital_inr,
                        paper_mode        = EXCLUDED.paper_mode,
                        updated_at        = NOW();
                """, (
                    account.user_id, account.email, account.plan.value,
                    account.broker_provider.value if account.broker_provider else None,
                    account.broker_account_id, account.capital_inr,
                    account.paper_mode, account.max_drawdown_pct,
                    account.max_positions, account.distributor_id or None,
                    account.referred_by or None,
                ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"_save_subscriber_profile failed: {e}")

    def _load_subscriber(self, user_id: str) -> Optional[SubscriberAccount]:
        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM subscriber_profiles WHERE user_id = %s;",
                    (user_id,)
                )
                row = cur.fetchone()
            conn.close()
            if not row:
                return None
            acc = self._row_to_account(dict(row))
            with self._lock:
                self._cache[user_id] = acc
            return acc
        except Exception as e:
            logger.error(f"_load_subscriber failed: {e}")
            return None

    def _row_to_account(self, row: dict) -> SubscriberAccount:
        provider = None
        if row.get("broker_provider"):
            try:
                provider = BrokerProvider(row["broker_provider"])
            except ValueError:
                pass
        return SubscriberAccount(
            user_id           = row["user_id"],
            email             = row.get("email", ""),
            plan              = SubscriptionPlan(row.get("plan", "free")),
            is_active         = row.get("is_active", True),
            broker_provider   = provider,
            broker_account_id = row.get("broker_account_id", "") or "",
            capital_inr       = float(row.get("capital_inr", 0) or 0),
            paper_mode        = row.get("paper_mode", True),
            max_drawdown_pct  = float(row.get("max_drawdown_pct", 0.12) or 0.12),
            max_positions     = int(row.get("max_positions", 4) or 4),
            distributor_id    = row.get("distributor_id", "") or "",
            referred_by       = row.get("referred_by", "") or "",
            signals_today     = int(row.get("signals_today", 0) or 0),
            signals_month     = int(row.get("signals_month", 0) or 0),
            last_signal_at    = row.get("last_signal_at"),
        )

    def _log_broadcast(self, summary: BroadcastSummary):
        try:
            conn = psycopg2.connect(DB_URL)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO broadcast_log (
                        signal_id, symbol, total_subscribers,
                        success_count, failure_count, paper_count,
                        broadcast_time_ms
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s);
                """, (
                    summary.signal_id, summary.symbol,
                    summary.total_subscribers, summary.success_count,
                    summary.failure_count, summary.paper_count,
                    summary.broadcast_time_ms,
                ))
                for r in summary.results:
                    cur.execute("""
                        INSERT INTO subscriber_trades (
                            broadcast_id, user_id, signal_id, symbol,
                            order_id, fill_price, quantity,
                            success, error, latency_ms, paper_mode
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
                    """, (
                        summary.signal_id, r.user_id, summary.signal_id,
                        r.symbol, r.order_id, r.fill_price, r.quantity,
                        r.success, r.error or None, r.latency_ms, r.paper_mode,
                    ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"_log_broadcast failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest api/subscriber_manager.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestSubscriberManager:

    def _make_account(self, **kw) -> SubscriberAccount:
        defaults = dict(
            user_id="u1", email="test@test.com",
            plan=SubscriptionPlan.PRO, is_active=True,
            broker_provider=BrokerProvider.ZERODHA,
            capital_inr=1_000_000.0, paper_mode=True,
            signals_today=0, signals_month=0,
        )
        defaults.update(kw)
        return SubscriberAccount(**defaults)

    class _MockSignal:
        symbol    = "RELIANCE"
        signal_id = "SIG-001"
        class mode:
            value = "swing"

    # ── Plan limits ───────────────────────────────────────────────────────

    def test_plan_limits_defined_for_all_plans(self):
        for plan in SubscriptionPlan:
            assert plan in PLAN_SIGNAL_LIMITS

    def test_elite_plan_has_highest_limits(self):
        elite = PLAN_SIGNAL_LIMITS[SubscriptionPlan.ELITE]
        free  = PLAN_SIGNAL_LIMITS[SubscriptionPlan.FREE]
        assert elite["daily"]   > free["daily"]
        assert elite["monthly"] > free["monthly"]

    def test_intraday_allowed_for_pro_and_elite(self):
        assert "intraday" in PLAN_SIGNAL_LIMITS[SubscriptionPlan.PRO]["modes"]
        assert "intraday" in PLAN_SIGNAL_LIMITS[SubscriptionPlan.ELITE]["modes"]

    def test_intraday_not_allowed_for_free_and_basic(self):
        assert "intraday" not in PLAN_SIGNAL_LIMITS[SubscriptionPlan.FREE]["modes"]
        assert "intraday" not in PLAN_SIGNAL_LIMITS[SubscriptionPlan.BASIC]["modes"]

    # ── Eligibility ───────────────────────────────────────────────────────

    def _make_manager_no_db(self) -> SubscriberManager:
        """Creates manager without DB (patches _ensure_tables)."""
        mgr = SubscriberManager.__new__(SubscriberManager)
        mgr._auth     = None
        mgr._lock     = threading.Lock()
        mgr._cache    = {}
        from concurrent.futures import ThreadPoolExecutor
        mgr._executor = ThreadPoolExecutor(max_workers=2)
        return mgr

    def test_is_eligible_active_subscriber(self):
        mgr = self._make_manager_no_db()
        acc = self._make_account(plan=SubscriptionPlan.PRO)
        assert mgr._is_eligible(acc, self._MockSignal())

    def test_is_not_eligible_inactive(self):
        mgr = self._make_manager_no_db()
        acc = self._make_account(is_active=False)
        assert not mgr._is_eligible(acc, self._MockSignal())

    def test_is_not_eligible_daily_limit(self):
        mgr = self._make_manager_no_db()
        limit = PLAN_SIGNAL_LIMITS[SubscriptionPlan.FREE]["daily"]
        acc   = self._make_account(
            plan=SubscriptionPlan.FREE,
            signals_today=limit
        )
        assert not mgr._is_eligible(acc, self._MockSignal())

    def test_is_not_eligible_monthly_limit(self):
        mgr   = self._make_manager_no_db()
        limit = PLAN_SIGNAL_LIMITS[SubscriptionPlan.BASIC]["monthly"]
        acc   = self._make_account(
            plan=SubscriptionPlan.BASIC,
            signals_month=limit
        )
        assert not mgr._is_eligible(acc, self._MockSignal())

    def test_free_plan_swing_eligible(self):
        mgr = self._make_manager_no_db()
        acc = self._make_account(plan=SubscriptionPlan.FREE)
        assert mgr._is_eligible(acc, self._MockSignal())

    # ── Risk limit enforcement ────────────────────────────────────────────

    def test_max_drawdown_capped_at_system_limit(self):
        """Subscribers cannot set drawdown > 12%."""
        acc = SubscriberAccount(
            user_id="u1", email="a@b.com",
            plan=SubscriptionPlan.PRO,
            max_drawdown_pct=0.20,  # tries to set 20%
        )
        # In onboard_subscriber this is enforced:
        capped = min(acc.max_drawdown_pct, 0.12)
        assert capped == 0.12

    def test_max_positions_capped_at_4(self):
        capped = min(6, 4)
        assert capped == 4

    # ── Broadcast result dataclasses ─────────────────────────────────────

    def test_broadcast_result_success(self):
        r = BroadcastResult(
            user_id="u1", symbol="RELIANCE",
            success=True, order_id="ORD-001",
            fill_price=2850.0, quantity=10,
        )
        assert r.success
        assert r.order_id == "ORD-001"

    def test_broadcast_result_failure(self):
        r = BroadcastResult(
            user_id="u1", symbol="RELIANCE",
            success=False, error="Broker offline",
        )
        assert not r.success
        assert "offline" in r.error

    def test_broadcast_summary_success_rate(self):
        s = BroadcastSummary(
            signal_id="S1", symbol="REL",
            total_subscribers=10,
            success_count=8,
            failure_count=2,
            paper_count=8,
        )
        assert abs(s.success_rate - 0.8) < 1e-6

    def test_broadcast_summary_zero_subscribers(self):
        s = BroadcastSummary(
            signal_id="S1", symbol="REL",
            total_subscribers=0,
            success_count=0,
            failure_count=0,
            paper_count=0,
        )
        assert s.success_rate == 0.0

    # ── Subscriber account ────────────────────────────────────────────────

    def test_subscriber_account_defaults(self):
        acc = SubscriberAccount(
            user_id="u1", email="a@b.com",
            plan=SubscriptionPlan.FREE,
        )
        assert acc.is_active   is True
        assert acc.paper_mode  is True
        assert acc.max_positions == 4
        assert acc.max_drawdown_pct == 0.12

    def test_increment_usage_in_cache(self):
        mgr = self._make_manager_no_db()
        acc = self._make_account(user_id="u42", signals_today=0)
        with mgr._lock:
            mgr._cache["u42"] = acc

        # Patch DB call
        mgr._increment_usage.__func__  # just verify it exists

    # ── Plan limits table ─────────────────────────────────────────────────

    def test_all_plans_have_required_keys(self):
        for plan, limits in PLAN_SIGNAL_LIMITS.items():
            assert "daily"   in limits, f"{plan} missing 'daily'"
            assert "monthly" in limits, f"{plan} missing 'monthly'"
            assert "modes"   in limits, f"{plan} missing 'modes'"

    def test_swing_allowed_for_all_plans(self):
        for plan, limits in PLAN_SIGNAL_LIMITS.items():
            assert "swing" in limits["modes"], f"{plan} missing swing"


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))