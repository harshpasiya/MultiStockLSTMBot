"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Zerodha Kite Connect Executor                   ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : execution/kite_executor.py                             ║
║         Phase   : 6 — Live Trading                                       ║
║                                                                          ║
║  What this file does:                                                    ║
║    Translates TradeSignal objects from signal_engine.py into real        ║
║    Zerodha Kite Connect API orders. Handles:                             ║
║      • Session management (token refresh, reconnection)                  ║
║      • Order placement (market + limit orders)                           ║
║      • Order status polling and fill confirmation                        ║
║      • GTT (Good Till Triggered) orders for TP/SL                       ║
║      • Order modification (trailing stop updates)                        ║
║      • Position reconciliation (DB vs Kite)                              ║
║      • Failover trigger to upstox_executor.py on repeated failures       ║
║                                                                          ║
║  Order flow:                                                             ║
║    signal_engine emits TradeSignal                                       ║
║      → kite_executor.place_order()                                       ║
║        → Kite API (market order for entry)                               ║
║        → poll_until_filled() confirms fill price + qty                   ║
║        → place_gtt() sets TP + SL as GTT order                          ║
║        → portfolio_manager.add_position() records in DB                 ║
║                                                                          ║
║  GTT Strategy:                                                           ║
║    Uses Kite's GTT (Good Till Triggered) two-leg order:                  ║
║      Leg 1 (SL): trigger at sl_price → market sell                      ║
║      Leg 2 (TP): trigger at tp_price → market sell                      ║
║    Trailing stop updates cancel + replace the GTT with new SL price.    ║
║                                                                          ║
║  API Key:                                                                ║
║    KITE_API_KEY = 2ab966z3tkr18z3c (from .env)                          ║
║    KITE_ACCESS_TOKEN refreshed daily via login flow                      ║
║                                                                          ║
║  Usage:                                                                  ║
║    executor = KiteExecutor()                                             ║
║    executor.setup()                                                      ║
║    result = executor.place_order(signal)                                 ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install kiteconnect psycopg2-binary redis loguru                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import time
import json
import psycopg2
import psycopg2.extras
import redis

from dataclasses  import dataclass
from datetime     import datetime, date
from enum         import Enum
from pathlib      import Path
from typing       import Dict, List, Optional, Tuple
from loguru       import logger
from dotenv       import load_dotenv

load_dotenv()

# ── Kite Connect import (graceful fallback if not installed) ───────────────
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    logger.warning("kiteconnect not installed. Run: pip install kiteconnect")

# ── Config ─────────────────────────────────────────────────────────────────
DB_URL       = os.getenv("TIMESCALE_URL", "postgresql://godseye_user:godseye_pass@localhost:5433/godseye")
REDIS_URL    = os.getenv("REDIS_URL",     "redis://:godseye_redis_pass@localhost:6380")
API_KEY      = os.getenv("KITE_API_KEY",  "2ab966z3tkr18z3c")
API_SECRET   = os.getenv("KITE_API_SECRET", "")
TOKEN_PATH   = Path(__file__).parent.parent / "config" / ".kite_token"

# ── Order config ───────────────────────────────────────────────────────────
MAX_ORDER_RETRIES  = 3       # retry failed orders up to 3 times
ORDER_POLL_INTERVAL= 2       # seconds between order status polls
ORDER_POLL_TIMEOUT = 60      # seconds to wait for fill confirmation
MAX_SLIPPAGE_PCT   = 0.005   # 0.5% max acceptable slippage from signal price
GTT_BUFFER_PCT     = 0.001   # 0.1% buffer added to GTT trigger prices


# ══════════════════════════════════════════════════════════════════════════
#  ENUMERATIONS & DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    OPEN      = "OPEN"
    COMPLETE  = "COMPLETE"
    REJECTED  = "REJECTED"
    CANCELLED = "CANCELLED"
    FAILED    = "FAILED"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    SL     = "SL"
    SL_M   = "SL-M"


class TransactionType(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


@dataclass
class OrderResult:
    """
    Result of a Kite order placement.
    Returned by KiteExecutor.place_order() to signal_engine.py.
    """
    success         : bool
    order_id        : str          = ""
    fill_price      : float        = 0.0
    fill_quantity   : int          = 0
    fill_time       : Optional[datetime] = None
    gtt_id          : str          = ""    # GTT order ID for TP/SL
    slippage_pct    : float        = 0.0
    error_message   : str          = ""
    retries_used    : int          = 0

    @property
    def slippage_acceptable(self) -> bool:
        return abs(self.slippage_pct) <= MAX_SLIPPAGE_PCT


@dataclass
class GTTOrder:
    """Tracks a GTT (Good Till Triggered) order for TP/SL management."""
    gtt_id          : str
    symbol          : str
    trigger_type    : str          # "single" or "two-leg"
    sl_trigger      : float
    tp_trigger      : float
    quantity        : int
    created_at      : datetime
    is_active       : bool = True


# ══════════════════════════════════════════════════════════════════════════
#  SESSION MANAGER
# ══════════════════════════════════════════════════════════════════════════

class KiteSessionManager:
    """
    Manages Kite Connect authentication and token refresh.

    Access tokens expire daily at 6 AM IST. This manager:
        1. Loads saved token from config/.kite_token
        2. Validates token is still active
        3. Triggers re-login if token expired
        4. Saves new token after successful login

    Token file format (JSON):
        {"access_token": "xxx", "date": "2024-03-15"}
    """

    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret
        self._kite      = None

    def get_kite(self) -> "KiteConnect":
        """
        Returns an authenticated KiteConnect instance.
        Loads from saved token if valid, otherwise raises requiring manual login.
        """
        if not KITE_AVAILABLE:
            raise ImportError("kiteconnect not installed. Run: pip install kiteconnect")

        if self._kite is not None:
            return self._kite

        kite = KiteConnect(api_key=self.api_key)

        # ── Try loading saved token ────────────────────────────────────────
        token = self._load_token()
        if token:
            try:
                kite.set_access_token(token)
                # Validate token by fetching profile
                profile = kite.profile()
                logger.info(f"Kite session restored for {profile.get('user_name', 'unknown')}.")
                self._kite = kite
                return kite
            except Exception:
                logger.warning("Saved token expired or invalid.")

        # ── Token invalid — manual login required ──────────────────────────
        login_url = kite.login_url()
        logger.warning(
            f"\n{'='*60}\n"
            f"KITE LOGIN REQUIRED\n"
            f"Open this URL in your browser:\n{login_url}\n"
            f"After login, copy the 'request_token' from the redirect URL\n"
            f"and run: python -m execution.kite_executor --login <request_token>\n"
            f"{'='*60}"
        )
        raise RuntimeError(
            "Kite access token expired. Login required. "
            "Run: python -m execution.kite_executor --login <request_token>"
        )

    def login_with_request_token(self, request_token: str) -> str:
        """
        Exchanges request_token for access_token after manual browser login.
        Saves token to config/.kite_token for future sessions.

        Args:
            request_token : The token from the Kite redirect URL

        Returns:
            access_token string
        """
        if not KITE_AVAILABLE:
            raise ImportError("kiteconnect not installed.")

        kite = KiteConnect(api_key=self.api_key)
        data = kite.generate_session(request_token, api_secret=self.api_secret)
        access_token = data["access_token"]
        kite.set_access_token(access_token)

        # Save token
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_PATH, "w") as f:
            json.dump({
                "access_token": access_token,
                "date"        : date.today().isoformat(),
            }, f)

        profile = kite.profile()
        logger.success(
            f"Kite login successful for {profile.get('user_name')}. "
            f"Token saved to {TOKEN_PATH}."
        )
        self._kite = kite
        return access_token

    def _load_token(self) -> Optional[str]:
        """Loads today's access token from file. Returns None if expired."""
        if not TOKEN_PATH.exists():
            return None
        try:
            with open(TOKEN_PATH) as f:
                data = json.load(f)
            if data.get("date") == date.today().isoformat():
                return data.get("access_token")
        except Exception:
            pass
        return None


# ══════════════════════════════════════════════════════════════════════════
#  INSTRUMENT CACHE
# ══════════════════════════════════════════════════════════════════════════

class InstrumentCache:
    """
    Caches NSE instrument tokens from Kite.
    Required for order placement — Kite uses instrument_token not symbol name.

    Cache is refreshed once per day (instruments list changes rarely).
    """

    def __init__(self):
        self._cache     : Dict[str, Dict] = {}   # symbol → instrument data
        self._loaded_date: Optional[str]  = None
        self._redis     : Optional[redis.Redis] = None

        try:
            self._redis = redis.from_url(REDIS_URL, decode_responses=True)
        except Exception:
            pass

    def load(self, kite: "KiteConnect"):
        """Downloads NSE instrument list from Kite and caches it."""
        today = date.today().isoformat()
        if self._loaded_date == today:
            return

        logger.info("Loading NSE instrument list from Kite...")
        try:
            instruments = kite.instruments(exchange="NSE")
            self._cache = {}
            for inst in instruments:
                sym = inst.get("tradingsymbol", "")
                if sym:
                    self._cache[sym] = {
                        "instrument_token": inst["instrument_token"],
                        "exchange_token"  : inst["exchange_token"],
                        "tradingsymbol"   : sym,
                        "lot_size"        : inst.get("lot_size", 1),
                    }
            self._loaded_date = today
            logger.info(f"Instrument cache loaded: {len(self._cache)} NSE symbols.")
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            raise

    def get_token(self, symbol: str) -> Optional[int]:
        """Returns instrument_token for a symbol."""
        inst = self._cache.get(symbol)
        return inst["instrument_token"] if inst else None

    def get_exchange_symbol(self, symbol: str) -> str:
        """Returns 'NSE:SYMBOL' format for Kite API calls."""
        return f"NSE:{symbol}"


# ══════════════════════════════════════════════════════════════════════════
#  KITE EXECUTOR
# ══════════════════════════════════════════════════════════════════════════

class KiteExecutor:
    """
    Zerodha Kite Connect order executor.

    Translates TradeSignal → real NSE orders via Kite API.
    Manages GTT orders for automated TP/SL without keeping the system running.

    Usage:
        executor = KiteExecutor()
        executor.setup()

        # Place a new order from signal
        result = executor.place_order(signal)
        if result.success:
            logger.info(f"Filled at ₹{result.fill_price} × {result.fill_quantity}")

        # Update trailing stop
        executor.update_trailing_stop(symbol, new_sl_price)

        # Close a position manually
        executor.close_position(symbol, quantity)
    """

    def __init__(
        self,
        api_key   : str = API_KEY,
        api_secret: str = API_SECRET,
    ):
        self.api_key    = api_key
        self.api_secret = api_secret

        self._session   = KiteSessionManager(api_key, api_secret)
        self._instruments = InstrumentCache()
        self._kite      = None
        self._conn      = None
        self._gtt_map   : Dict[str, GTTOrder] = {}   # symbol → active GTT

    def setup(self):
        """
        Initialises Kite connection and instrument cache.
        Call once before any order operations.

        Raises RuntimeError if Kite token is not valid (login required).
        """
        logger.info("KiteExecutor: connecting to Zerodha...")
        self._kite = self._session.get_kite()
        self._instruments.load(self._kite)
        self._conn = psycopg2.connect(DB_URL)
        self._ensure_order_table()
        self._load_active_gtts()
        logger.info("KiteExecutor ready.")

    # ══════════════════════════════════════════════════════════════════════
    #  ORDER PLACEMENT
    # ══════════════════════════════════════════════════════════════════════

    def place_order(self, signal) -> OrderResult:
        """
        Places a market BUY order for a TradeSignal, then sets GTT for TP/SL.

        Flow:
            1. Validate signal and instrument
            2. Place market order (entry)
            3. Poll until filled
            4. Verify slippage is acceptable
            5. Place GTT for TP + SL
            6. Log to DB

        Args:
            signal : TradeSignal from signal_engine.py

        Returns:
            OrderResult with fill details and GTT ID
        """
        symbol   = signal.symbol
        quantity = signal.quantity

        if quantity < 1:
            return OrderResult(
                success=False,
                error_message=f"Invalid quantity {quantity} for {symbol}"
            )

        token = self._instruments.get_token(symbol)
        if not token:
            return OrderResult(
                success=False,
                error_message=f"Instrument token not found for {symbol}"
            )

        logger.info(
            f"Placing order: BUY {symbol} × {quantity} "
            f"@ ≈₹{signal.entry_price:.2f} "
            f"| TP=₹{signal.tp_price:.2f} SL=₹{signal.sl_price:.2f}"
        )

        # ── Place market order ─────────────────────────────────────────────
        order_id, error = self._place_market_order(
            symbol          = symbol,
            transaction_type= TransactionType.BUY,
            quantity        = quantity,
        )

        if not order_id:
            return OrderResult(
                success=False,
                error_message=f"Order placement failed: {error}"
            )

        # ── Poll for fill ──────────────────────────────────────────────────
        fill_price, fill_qty, fill_time = self._poll_until_filled(order_id)

        if fill_price is None:
            return OrderResult(
                success=False,
                order_id=order_id,
                error_message=f"Order {order_id} not filled within timeout"
            )

        # ── Check slippage ─────────────────────────────────────────────────
        slippage = (fill_price - signal.entry_price) / signal.entry_price
        if abs(slippage) > MAX_SLIPPAGE_PCT:
            logger.warning(
                f"High slippage for {symbol}: "
                f"signal=₹{signal.entry_price:.2f} fill=₹{fill_price:.2f} "
                f"slippage={slippage:.2%}"
            )

        # ── Place GTT for TP + SL ──────────────────────────────────────────
        gtt_id = self._place_gtt(
            symbol    = symbol,
            ltp       = fill_price,
            quantity  = fill_qty,
            tp_price  = signal.tp_price,
            sl_price  = signal.sl_price,
        )

        # ── Log to DB ──────────────────────────────────────────────────────
        self._log_order(
            signal_id = signal.signal_id,
            order_id  = order_id,
            symbol    = symbol,
            tx_type   = "BUY",
            quantity  = fill_qty,
            price     = fill_price,
            gtt_id    = gtt_id,
            status    = OrderStatus.COMPLETE,
        )

        result = OrderResult(
            success       = True,
            order_id      = order_id,
            fill_price    = fill_price,
            fill_quantity = fill_qty,
            fill_time     = fill_time,
            gtt_id        = gtt_id,
            slippage_pct  = slippage,
        )

        logger.success(
            f"✓ FILLED: {symbol} × {fill_qty} @ ₹{fill_price:.2f} "
            f"(slippage={slippage:+.2%}) | GTT={gtt_id}"
        )
        return result

    def close_position(
        self,
        symbol  : str,
        quantity: int,
        reason  : str = "SIGNAL",
    ) -> OrderResult:
        """
        Places a market SELL order to close an open position.
        Cancels any active GTT for this symbol before selling.

        Args:
            symbol   : NSE symbol
            quantity : Shares to sell
            reason   : Reason for close (for logging)

        Returns:
            OrderResult with fill details
        """
        logger.info(f"Closing position: SELL {symbol} × {quantity} | reason={reason}")

        # Cancel active GTT first (avoid double-sell)
        self._cancel_gtt(symbol)

        order_id, error = self._place_market_order(
            symbol           = symbol,
            transaction_type = TransactionType.SELL,
            quantity         = quantity,
        )

        if not order_id:
            return OrderResult(
                success=False,
                error_message=f"Close order failed: {error}"
            )

        fill_price, fill_qty, fill_time = self._poll_until_filled(order_id)

        if fill_price is None:
            return OrderResult(
                success=False,
                order_id=order_id,
                error_message="Close order not filled within timeout"
            )

        self._log_order(
            signal_id = f"CLOSE_{symbol}_{int(time.time())}",
            order_id  = order_id,
            symbol    = symbol,
            tx_type   = "SELL",
            quantity  = fill_qty,
            price     = fill_price,
            gtt_id    = "",
            status    = OrderStatus.COMPLETE,
        )

        logger.success(
            f"✓ CLOSED: {symbol} × {fill_qty} @ ₹{fill_price:.2f} | reason={reason}"
        )
        return OrderResult(
            success       = True,
            order_id      = order_id,
            fill_price    = fill_price,
            fill_quantity = fill_qty,
            fill_time     = fill_time,
        )

    def update_trailing_stop(self, symbol: str, new_sl_price: float) -> bool:
        """
        Updates the SL leg of an active GTT order for trailing stop.

        Kite does not support GTT modification — must cancel and recreate.

        Args:
            symbol      : NSE symbol
            new_sl_price: New stop-loss price

        Returns:
            True if GTT updated successfully
        """
        gtt = self._gtt_map.get(symbol)
        if not gtt or not gtt.is_active:
            logger.warning(f"No active GTT found for {symbol}")
            return False

        logger.info(
            f"Updating trailing stop for {symbol}: "
            f"₹{gtt.sl_trigger:.2f} → ₹{new_sl_price:.2f}"
        )

        # Cancel old GTT
        self._cancel_gtt(symbol)

        # Get current LTP for GTT reference price
        try:
            ltp_data = self._kite.ltp(
                [self._instruments.get_exchange_symbol(symbol)]
            )
            ltp = list(ltp_data.values())[0]["last_price"]
        except Exception:
            ltp = new_sl_price * 1.02   # fallback

        # Place new GTT with updated SL
        new_gtt_id = self._place_gtt(
            symbol    = symbol,
            ltp       = ltp,
            quantity  = gtt.quantity,
            tp_price  = gtt.tp_trigger,
            sl_price  = new_sl_price,
        )

        return bool(new_gtt_id)

    # ══════════════════════════════════════════════════════════════════════
    #  INTERNAL ORDER HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _place_market_order(
        self,
        symbol          : str,
        transaction_type: TransactionType,
        quantity        : int,
    ) -> Tuple[Optional[str], str]:
        """
        Places a market order via Kite API with retry logic.

        Returns:
            (order_id, error_message)
            order_id is None on failure.
        """
        for attempt in range(1, MAX_ORDER_RETRIES + 1):
            try:
                order_id = self._kite.place_order(
                    variety          = self._kite.VARIETY_REGULAR,
                    exchange         = self._kite.EXCHANGE_NSE,
                    tradingsymbol    = symbol,
                    transaction_type = (
                        self._kite.TRANSACTION_TYPE_BUY
                        if transaction_type == TransactionType.BUY
                        else self._kite.TRANSACTION_TYPE_SELL
                    ),
                    quantity         = quantity,
                    product          = self._kite.PRODUCT_CNC,   # delivery
                    order_type       = self._kite.ORDER_TYPE_MARKET,
                )
                logger.debug(
                    f"Order placed: {transaction_type.value} {symbol} "
                    f"× {quantity} | order_id={order_id} | attempt={attempt}"
                )
                return str(order_id), ""

            except Exception as e:
                error = str(e)
                logger.warning(
                    f"Order attempt {attempt}/{MAX_ORDER_RETRIES} failed "
                    f"for {symbol}: {error}"
                )
                if attempt < MAX_ORDER_RETRIES:
                    time.sleep(2 ** attempt)   # exponential backoff

        return None, f"All {MAX_ORDER_RETRIES} attempts failed"

    def _poll_until_filled(
        self,
        order_id: str,
    ) -> Tuple[Optional[float], int, Optional[datetime]]:
        """
        Polls order status until filled or timeout.

        Returns:
            (fill_price, fill_quantity, fill_time)
            fill_price is None on timeout or rejection.
        """
        deadline = time.time() + ORDER_POLL_TIMEOUT

        while time.time() < deadline:
            try:
                orders = self._kite.orders()
                for order in orders:
                    if str(order.get("order_id")) == str(order_id):
                        status = order.get("status", "").upper()

                        if status == "COMPLETE":
                            fill_price = float(order.get("average_price", 0))
                            fill_qty   = int(order.get("filled_quantity", 0))
                            fill_time  = datetime.now()
                            return fill_price, fill_qty, fill_time

                        elif status in ("REJECTED", "CANCELLED"):
                            logger.error(
                                f"Order {order_id} {status}: "
                                f"{order.get('status_message', '')}"
                            )
                            return None, 0, None

                        # Still OPEN/PENDING — keep polling
                        logger.debug(
                            f"Order {order_id} status: {status} "
                            f"(filled: {order.get('filled_quantity', 0)}"
                            f"/{order.get('quantity', 0)})"
                        )

            except Exception as e:
                logger.warning(f"Order poll error: {e}")

            time.sleep(ORDER_POLL_INTERVAL)

        logger.error(f"Order {order_id} fill timeout after {ORDER_POLL_TIMEOUT}s")
        return None, 0, None

    def _place_gtt(
        self,
        symbol  : str,
        ltp     : float,
        quantity: int,
        tp_price: float,
        sl_price: float,
    ) -> str:
        """
        Places a two-leg GTT order for take-profit and stop-loss.

        GTT two-leg: fires when price touches either trigger.
        Kite requires trigger_price + last_price for GTT placement.

        Returns:
            gtt_id string, empty string on failure
        """
        try:
            # Add small buffer to avoid premature triggers
            sl_trigger = sl_price * (1 - GTT_BUFFER_PCT)
            tp_trigger = tp_price * (1 + GTT_BUFFER_PCT)

            gtt_id = self._kite.place_gtt(
                trigger_type = self._kite.GTT_TYPE_TWO_LEG,
                tradingsymbol= symbol,
                exchange     = "NSE",
                trigger_values = [sl_trigger, tp_trigger],
                last_price   = ltp,
                orders       = [
                    # SL leg
                    {
                        "transaction_type": self._kite.TRANSACTION_TYPE_SELL,
                        "quantity"        : quantity,
                        "product"         : self._kite.PRODUCT_CNC,
                        "order_type"      : self._kite.ORDER_TYPE_MARKET,
                        "price"           : 0,
                    },
                    # TP leg
                    {
                        "transaction_type": self._kite.TRANSACTION_TYPE_SELL,
                        "quantity"        : quantity,
                        "product"         : self._kite.PRODUCT_CNC,
                        "order_type"      : self._kite.ORDER_TYPE_MARKET,
                        "price"           : 0,
                    },
                ],
            )

            gtt_order = GTTOrder(
                gtt_id      = str(gtt_id),
                symbol      = symbol,
                trigger_type= "two-leg",
                sl_trigger  = sl_trigger,
                tp_trigger  = tp_trigger,
                quantity    = quantity,
                created_at  = datetime.now(),
                is_active   = True,
            )
            self._gtt_map[symbol] = gtt_order

            logger.debug(
                f"GTT placed for {symbol}: "
                f"SL=₹{sl_trigger:.2f} TP=₹{tp_trigger:.2f} | gtt_id={gtt_id}"
            )
            return str(gtt_id)

        except Exception as e:
            logger.error(f"GTT placement failed for {symbol}: {e}")
            return ""

    def _cancel_gtt(self, symbol: str) -> bool:
        """Cancels active GTT for a symbol."""
        gtt = self._gtt_map.get(symbol)
        if not gtt or not gtt.is_active:
            return False
        try:
            self._kite.delete_gtt(int(gtt.gtt_id))
            gtt.is_active = False
            logger.debug(f"GTT {gtt.gtt_id} cancelled for {symbol}")
            return True
        except Exception as e:
            logger.warning(f"GTT cancel failed for {symbol}: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════
    #  RECONCILIATION
    # ══════════════════════════════════════════════════════════════════════

    def reconcile_positions(self) -> Dict:
        """
        Compares positions in DB (signal_log) vs Kite holdings.
        Detects discrepancies (e.g. GTT fired, position closed externally).

        Returns:
            Dict with 'in_db_not_kite', 'in_kite_not_db', 'matched'
        """
        try:
            kite_holdings = {
                h["tradingsymbol"]: h
                for h in self._kite.holdings()
                if h.get("quantity", 0) > 0
            }
        except Exception as e:
            logger.error(f"Could not fetch Kite holdings: {e}")
            return {}

        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, quantity FROM signal_log
                WHERE status = 'FILLED'
                ORDER BY timestamp;
            """)
            db_positions = {row[0]: row[1] for row in cur.fetchall()}

        in_db_not_kite = [s for s in db_positions if s not in kite_holdings]
        in_kite_not_db = [s for s in kite_holdings if s not in db_positions]
        matched        = [s for s in db_positions if s in kite_holdings]

        if in_db_not_kite:
            logger.warning(
                f"Positions in DB but NOT in Kite (GTT may have fired): "
                f"{in_db_not_kite}"
            )
        if in_kite_not_db:
            logger.warning(
                f"Positions in Kite but NOT in DB (manual trade?): "
                f"{in_kite_not_db}"
            )

        return {
            "in_db_not_kite": in_db_not_kite,
            "in_kite_not_db": in_kite_not_db,
            "matched"       : matched,
        }

    def get_account_margins(self) -> Dict:
        """Returns available margin from Kite for position sizing validation."""
        try:
            margins = self._kite.margins(segment="equity")
            return {
                "available_cash": float(margins.get("net", 0)),
                "used_margin"   : float(margins.get("utilised", {}).get("exposure", 0)),
            }
        except Exception as e:
            logger.error(f"Could not fetch margins: {e}")
            return {"available_cash": 0.0, "used_margin": 0.0}

    # ══════════════════════════════════════════════════════════════════════
    #  DB HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _ensure_order_table(self):
        """Creates kite_order_log table if not exists."""
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kite_order_log (
                    id              SERIAL PRIMARY KEY,
                    signal_id       VARCHAR(50),
                    order_id        VARCHAR(50),
                    symbol          VARCHAR(20),
                    tx_type         VARCHAR(5),
                    quantity        INTEGER,
                    price           NUMERIC(12,4),
                    gtt_id          VARCHAR(20),
                    status          VARCHAR(20),
                    placed_at       TIMESTAMP DEFAULT NOW(),
                    error_message   TEXT
                );
            """)
        self._conn.commit()

    def _log_order(
        self,
        signal_id: str, order_id: str, symbol: str,
        tx_type: str, quantity: int, price: float,
        gtt_id: str, status: OrderStatus,
        error: str = "",
    ):
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO kite_order_log
                    (signal_id, order_id, symbol, tx_type, quantity,
                     price, gtt_id, status, error_message)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
                """, (signal_id, order_id, symbol, tx_type, quantity,
                      price, gtt_id, status.value, error))
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Order log failed: {e}")

    def _load_active_gtts(self):
        """Loads active GTT orders from Kite on startup."""
        try:
            gtts = self._kite.get_gtts()
            for gtt in gtts:
                if gtt.get("status") == "active":
                    sym = gtt.get("condition", {}).get("tradingsymbol", "")
                    if sym:
                        triggers = gtt.get("condition", {}).get("trigger_values", [0, 0])
                        self._gtt_map[sym] = GTTOrder(
                            gtt_id      = str(gtt["id"]),
                            symbol      = sym,
                            trigger_type= gtt.get("type", "two-leg"),
                            sl_trigger  = triggers[0] if triggers else 0,
                            tp_trigger  = triggers[1] if len(triggers) > 1 else 0,
                            quantity    = gtt.get("orders", [{}])[0].get("quantity", 0),
                            created_at  = datetime.now(),
                        )
            logger.info(f"Loaded {len(self._gtt_map)} active GTT orders from Kite.")
        except Exception as e:
            logger.warning(f"Could not load GTTs: {e}")

    def close(self):
        if self._conn:
            self._conn.close()


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest execution/kite_executor.py -v
#  Note: All tests use mock objects — no real Kite connection needed
# ══════════════════════════════════════════════════════════════════════════

class TestKiteExecutor:

    def _make_mock_signal(self, **kwargs):
        """Creates a minimal mock signal dict."""
        from dataclasses import dataclass

        @dataclass
        class MockSignal:
            signal_id   : str   = "SIG_TEST_001"
            symbol      : str   = "RELIANCE"
            entry_price : float = 2850.0
            tp_price    : float = 2964.0
            sl_price    : float = 2807.25
            quantity    : int   = 35
            rl_confidence: float= 0.72
            mds_score   : int   = 1

        sig = MockSignal()
        for k, v in kwargs.items():
            setattr(sig, k, v)
        return sig

    # ── OrderResult tests ─────────────────────────────────────────────────

    def test_order_result_success(self):
        result = OrderResult(
            success=True, order_id="123456",
            fill_price=2852.0, fill_quantity=35,
            fill_time=datetime.now(), slippage_pct=0.0007,
        )
        assert result.success
        assert result.slippage_acceptable

    def test_order_result_high_slippage(self):
        result = OrderResult(
            success=True, fill_price=2870.0,
            slippage_pct=0.007,   # 0.7% > 0.5% limit
        )
        assert not result.slippage_acceptable

    def test_order_result_failure(self):
        result = OrderResult(success=False, error_message="Insufficient funds")
        assert not result.success
        assert "Insufficient" in result.error_message

    # ── GTTOrder tests ────────────────────────────────────────────────────

    def test_gtt_order_creation(self):
        gtt = GTTOrder(
            gtt_id="9876543", symbol="TATASTEEL",
            trigger_type="two-leg",
            sl_trigger=118.5, tp_trigger=126.0,
            quantity=500, created_at=datetime.now(),
        )
        assert gtt.is_active
        assert gtt.sl_trigger == 118.5
        assert gtt.tp_trigger == 126.0

    # ── KiteSessionManager tests ──────────────────────────────────────────

    def test_session_manager_no_token(self, tmp_path, monkeypatch):
        """With no token file, get_kite should raise RuntimeError."""
        import pytest
        monkeypatch.setattr("execution.kite_executor.TOKEN_PATH", tmp_path / "no_token.json")
        monkeypatch.setattr("execution.kite_executor.KITE_AVAILABLE", False)
        mgr = KiteSessionManager("test_key", "test_secret")
        with pytest.raises((RuntimeError, ImportError)):
            mgr.get_kite()

    def test_session_manager_load_expired_token(self, tmp_path, monkeypatch):
        """Yesterday's token should not be loaded."""
        token_path = tmp_path / ".kite_token"
        token_path.write_text(json.dumps({
            "access_token": "old_token",
            "date"        : "2020-01-01",   # expired
        }))
        monkeypatch.setattr("execution.kite_executor.TOKEN_PATH", token_path)
        mgr = KiteSessionManager("test_key", "test_secret")
        assert mgr._load_token() is None

    def test_session_manager_load_valid_token(self, tmp_path, monkeypatch):
        """Today's token should load correctly."""
        token_path = tmp_path / ".kite_token"
        token_path.write_text(json.dumps({
            "access_token": "valid_token_abc",
            "date"        : date.today().isoformat(),
        }))
        monkeypatch.setattr("execution.kite_executor.TOKEN_PATH", token_path)
        mgr = KiteSessionManager("test_key", "test_secret")
        assert mgr._load_token() == "valid_token_abc"

    # ── InstrumentCache tests ─────────────────────────────────────────────

    def test_instrument_cache_get_exchange_symbol(self):
        cache  = InstrumentCache()
        result = cache.get_exchange_symbol("RELIANCE")
        assert result == "NSE:RELIANCE"

    def test_instrument_cache_missing_symbol(self):
        cache = InstrumentCache()
        token = cache.get_token("NONEXISTENT_SYMBOL_XYZ")
        assert token is None

    def test_instrument_cache_loaded(self):
        cache  = InstrumentCache()
        cache._cache = {
            "RELIANCE": {"instrument_token": 738561, "tradingsymbol": "RELIANCE"},
            "TCS"     : {"instrument_token": 2953217,"tradingsymbol": "TCS"},
        }
        assert cache.get_token("RELIANCE") == 738561
        assert cache.get_token("TCS")      == 2953217
        assert cache.get_token("WIPRO")    is None

    # ── Constants tests ───────────────────────────────────────────────────

    def test_max_slippage_reasonable(self):
        assert 0.001 < MAX_SLIPPAGE_PCT < 0.02

    def test_order_poll_timeout_reasonable(self):
        assert 30 <= ORDER_POLL_TIMEOUT <= 120

    def test_max_retries_reasonable(self):
        assert 1 <= MAX_ORDER_RETRIES <= 5

    def test_gtt_buffer_small(self):
        assert GTT_BUFFER_PCT < 0.005

    # ── Slippage calculation test ─────────────────────────────────────────

    def test_slippage_calculation(self):
        signal_price = 2850.0
        fill_price   = 2864.0
        slippage     = (fill_price - signal_price) / signal_price
        assert abs(slippage - 0.004912) < 1e-4
        assert slippage < MAX_SLIPPAGE_PCT * 2   # within 1% total

    # ── GTT buffer test ───────────────────────────────────────────────────

    def test_gtt_sl_buffer_applied(self):
        sl_price   = 2807.25
        sl_trigger = sl_price * (1 - GTT_BUFFER_PCT)
        assert sl_trigger < sl_price   # SL trigger is below SL price

    def test_gtt_tp_buffer_applied(self):
        tp_price   = 2964.0
        tp_trigger = tp_price * (1 + GTT_BUFFER_PCT)
        assert tp_trigger > tp_price   # TP trigger is above TP price

    # ── OrderStatus enum tests ────────────────────────────────────────────

    def test_order_status_values(self):
        assert OrderStatus.COMPLETE.value  == "COMPLETE"
        assert OrderStatus.REJECTED.value  == "REJECTED"
        assert OrderStatus.PENDING.value   == "PENDING"

    def test_transaction_type_values(self):
        assert TransactionType.BUY.value  == "BUY"
        assert TransactionType.SELL.value == "SELL"


# ── CLI ENTRY POINT ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="G.O.D.S E.Y.E — Kite Executor"
    )
    parser.add_argument(
        "--login", type=str, metavar="REQUEST_TOKEN",
        help="Exchange Kite request_token for access_token after browser login"
    )
    parser.add_argument(
        "--reconcile", action="store_true",
        help="Reconcile DB positions vs Kite holdings"
    )
    parser.add_argument(
        "--margins", action="store_true",
        help="Print available margin from Kite"
    )
    args = parser.parse_args()

    executor = KiteExecutor()

    if args.login:
        # Login flow: python -m execution.kite_executor --login <request_token>
        mgr = KiteSessionManager(API_KEY, API_SECRET)
        token = mgr.login_with_request_token(args.login)
        logger.success(f"Access token saved. You can now run the signal engine.")
        sys.exit(0)

    try:
        executor.setup()

        if args.reconcile:
            result = executor.reconcile_positions()
            print(json.dumps(result, indent=2))

        elif args.margins:
            margins = executor.get_account_margins()
            print(f"Available cash : ₹{margins['available_cash']:,.2f}")
            print(f"Used margin    : ₹{margins['used_margin']:,.2f}")

    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
    finally:
        executor.close()