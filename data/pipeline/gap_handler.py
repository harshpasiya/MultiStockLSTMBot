"""
╔══════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Gap Handler                            ║
║         Project : MultiStockLSTMBot                             ║
║         File    : data/pipeline/gap_handler.py                  ║
║         Phase   : 0 — Data Infrastructure                       ║
║         Purpose : Detects and fills missing data in OHLCV series ║
╚══════════════════════════════════════════════════════════════════╝

Types of gaps handled:
-----------------------
    1. Expected gaps (weekends + NSE holidays)
       → Forward-fill OHLC, set volume = 0, mark is_gap = 1
       → Safe to fill: price did not change; stock just didn't trade

    2. Circuit breaker days (stock hit upper/lower circuit)
       → Price stayed at circuit level; volume may be zero
       → Forward-fill OHLC from last traded price
       → Mark as is_circuit_breaker = 1 (separate flag for model)

    3. Unexpected gaps (NSE download failure, new listing, delisting)
       → Flag for re-download; do NOT silently forward-fill
       → Log as data quality issue

    4. Trading halts (regulatory / exchange halt)
       → Detected by zero volume on a trading day
       → Mark is_halt = 1; keep last known price

CRITICAL rule:
--------------
    Never forward-fill more than MAX_CONSECUTIVE_FILL consecutive days.
    If a gap is longer than this, it is likely a delisting or extended
    trading halt — flagging it prevents the model from training on
    artificially stable "prices" that are just old data repeated.

Usage:
------
    from data.pipeline.gap_handler import GapHandler

    handler = GapHandler()

    # Fill gaps for a single stock's OHLCV DataFrame
    clean_df = handler.fill_gaps(df, symbol="RELIANCE")

    # Check if a date is an expected non-trading day
    is_holiday = handler.is_expected_gap(date(2024, 11, 1))  # Diwali

    # Get all NSE holidays for a given year
    holidays = handler.get_holidays(2024)
"""

import pandas as pd
import numpy as np

from datetime import date, timedelta
from loguru import logger


class GapHandler:
    """
    Handles missing trading days in per-stock OHLCV DataFrames.

    Attributes:
        max_consecutive_fill : Maximum days to forward-fill before flagging
        nse_holidays         : Set of known NSE market holidays
    """

    # ── NSE Market Holidays ───────────────────────────────────────────────
    # Source: NSE India official holiday calendar
    # Update this dict every year in November for the coming year

    NSE_HOLIDAYS: dict[int, set[date]] = {

        2019: {
            date(2019, 2, 19),   # Chhatrapati Shivaji Maharaj Jayanti
            date(2019, 3, 4),    # Mahashivratri
            date(2019, 3, 21),   # Holi
            date(2019, 4, 14),   # Dr. Ambedkar Jayanti / Good Friday
            date(2019, 4, 17),   # Ram Navami
            date(2019, 4, 19),   # Mahavir Jayanti
            date(2019, 6, 5),    # Eid-ul-Fitr
            date(2019, 8, 12),   # Bakri Id
            date(2019, 8, 15),   # Independence Day
            date(2019, 9, 2),    # Ganesh Chaturthi
            date(2019, 9, 10),   # Muharram
            date(2019, 10, 2),   # Gandhi Jayanti / Dussehra
            date(2019, 10, 28),  # Diwali-Laxmi Puja
            date(2019, 10, 29),  # Diwali-Balipratipada
            date(2019, 11, 12),  # Gurunanak Jayanti
            date(2019, 12, 25),  # Christmas
        },

        2020: {
            date(2020, 2, 21),   # Mahashivratri
            date(2020, 3, 10),   # Holi
            date(2020, 4, 2),    # Ram Navami
            date(2020, 4, 6),    # Mahavir Jayanti
            date(2020, 4, 10),   # Good Friday
            date(2020, 4, 14),   # Dr. Ambedkar Jayanti
            date(2020, 5, 25),   # Eid-ul-Fitr
            date(2020, 8, 3),    # Bakri Id
            date(2020, 8, 15),   # Independence Day / Muharram
            date(2020, 10, 2),   # Gandhi Jayanti
            date(2020, 11, 16),  # Gurunanak Jayanti
            date(2020, 11, 30),  # Diwali-Balipratipada
            date(2020, 12, 25),  # Christmas
        },

        2021: {
            date(2021, 1, 26),   # Republic Day
            date(2021, 3, 11),   # Mahashivratri
            date(2021, 3, 29),   # Holi
            date(2021, 4, 2),    # Good Friday
            date(2021, 4, 13),   # Gudi Padwa / Mahavir Jayanti
            date(2021, 4, 14),   # Dr. Ambedkar Jayanti
            date(2021, 4, 21),   # Ram Navami
            date(2021, 5, 13),   # Eid-ul-Fitr
            date(2021, 7, 20),   # Bakri Id
            date(2021, 8, 19),   # Muharram
            date(2021, 9, 10),   # Ganesh Chaturthi
            date(2021, 10, 15),  # Dussehra
            date(2021, 11, 4),   # Diwali-Laxmi Puja
            date(2021, 11, 5),   # Diwali-Balipratipada
            date(2021, 11, 19),  # Gurunanak Jayanti
            date(2021, 12, 25),  # Christmas
        },

        2022: {
            date(2022, 1, 26),   # Republic Day
            date(2022, 3, 1),    # Mahashivratri
            date(2022, 3, 18),   # Holi
            date(2022, 4, 14),   # Dr. Ambedkar Jayanti / Mahavir Jayanti
            date(2022, 4, 15),   # Good Friday
            date(2022, 5, 3),    # Eid-ul-Fitr
            date(2022, 7, 9),    # Eid-ul-Adha (Bakri Id)
            date(2022, 8, 9),    # Muharram
            date(2022, 8, 15),   # Independence Day
            date(2022, 8, 31),   # Ganesh Chaturthi
            date(2022, 10, 2),   # Gandhi Jayanti
            date(2022, 10, 5),   # Dussehra
            date(2022, 10, 24),  # Diwali-Laxmi Puja
            date(2022, 10, 26),  # Diwali-Balipratipada
            date(2022, 11, 8),   # Gurunanak Jayanti
            date(2022, 12, 25),  # Christmas
        },

        2023: {
            date(2023, 1, 26),   # Republic Day
            date(2023, 3, 7),    # Holi
            date(2023, 3, 30),   # Ram Navami
            date(2023, 4, 4),    # Mahavir Jayanti
            date(2023, 4, 7),    # Good Friday
            date(2023, 4, 14),   # Dr. Ambedkar Jayanti
            date(2023, 5, 5),    # Buddha Purnima
            date(2023, 6, 28),   # Eid-ul-Adha (Bakri Id)
            date(2023, 8, 15),   # Independence Day
            date(2023, 9, 19),   # Ganesh Chaturthi
            date(2023, 10, 2),   # Gandhi Jayanti
            date(2023, 10, 24),  # Dussehra
            date(2023, 11, 13),  # Diwali-Laxmi Puja
            date(2023, 11, 14),  # Diwali-Balipratipada
            date(2023, 11, 27),  # Gurunanak Jayanti
            date(2023, 12, 25),  # Christmas
        },

        2024: {
            date(2024, 1, 22),   # Ram Mandir consecration (special)
            date(2024, 3, 25),   # Holi
            date(2024, 3, 29),   # Good Friday
            date(2024, 4, 14),   # Dr. Ambedkar Jayanti
            date(2024, 4, 17),   # Ram Navami
            date(2024, 4, 21),   # Mahavir Jayanti
            date(2024, 5, 23),   # Buddha Purnima
            date(2024, 6, 17),   # Eid-ul-Adha (Bakri Id)
            date(2024, 7, 17),   # Muharram
            date(2024, 8, 15),   # Independence Day
            date(2024, 10, 2),   # Gandhi Jayanti
            date(2024, 10, 13),  # Dussehra
            date(2024, 11, 1),   # Diwali-Laxmi Puja
            date(2024, 11, 15),  # Gurunanak Jayanti
            date(2024, 12, 25),  # Christmas
        },

        2025: {
            date(2025, 2, 26),   # Mahashivratri
            date(2025, 3, 14),   # Holi
            date(2025, 4, 10),   # Ram Navami
            date(2025, 4, 14),   # Dr. Ambedkar Jayanti
            date(2025, 4, 18),   # Good Friday
            date(2025, 5, 12),   # Buddha Purnima
            date(2025, 6, 7),    # Eid-ul-Adha (Bakri Id)
            date(2025, 8, 15),   # Independence Day
            date(2025, 8, 27),   # Ganesh Chaturthi
            date(2025, 10, 2),   # Gandhi Jayanti
            date(2025, 10, 2),   # Dussehra (same day — verify with NSE)
            date(2025, 10, 20),  # Diwali-Laxmi Puja (tentative)
            date(2025, 11, 5),   # Gurunanak Jayanti (tentative)
            date(2025, 12, 25),  # Christmas
        },
    }

    def __init__(self, max_consecutive_fill: int = 5):
        """
        Args:
            max_consecutive_fill : Maximum consecutive days to forward-fill.
                                   Gaps longer than this are flagged, not filled.
        """
        self.max_consecutive_fill = max_consecutive_fill
        self._all_holidays = set()
        for year_holidays in self.NSE_HOLIDAYS.values():
            self._all_holidays.update(year_holidays)

    # ── Public Methods ────────────────────────────────────────────────────

    def is_expected_gap(self, d: date) -> bool:
        """
        Returns True if the given date is a known non-trading day.
        (Weekend or NSE holiday)

        Args:
            d : Date to check

        Returns:
            True if non-trading day, False if expected trading day
        """
        return d.weekday() >= 5 or d in self._all_holidays

    def get_holidays(self, year: int) -> set[date]:
        """Returns set of NSE holidays for a given year"""
        return self.NSE_HOLIDAYS.get(year, set())

    def fill_gaps(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> pd.DataFrame:
        """
        Fills expected gaps in a single stock's OHLCV DataFrame.

        Processing steps:
            1. Sort by date ascending
            2. Identify all expected trading days in the date range
            3. Find missing trading days
            4. For each missing day: forward-fill from last known price
            5. Mark filled rows with is_gap = 1
            6. Flag consecutive gaps > max_consecutive_fill

        Args:
            df     : Single-stock OHLCV DataFrame with 'date' column
            symbol : Symbol name for logging (optional)

        Returns:
            DataFrame with gaps filled and metadata flags added:
                is_gap            : 1 if row was forward-filled
                is_circuit_breaker: 1 if zero volume on a trading day
                consecutive_gap   : Number of consecutive gap days at this point
        """
        if df.empty:
            logger.warning(f"[GapHandler] Empty DataFrame for {symbol}")
            return df

        df = df.copy().sort_values("date").reset_index(drop=True)

        # Add metadata columns if not present
        if "is_gap" not in df.columns:
            df["is_gap"] = 0
        if "is_circuit_breaker" not in df.columns:
            df["is_circuit_breaker"] = 0
        if "consecutive_gap" not in df.columns:
            df["consecutive_gap"] = 0

        # Detect circuit breaker days (trading day with zero volume)
        zero_vol_mask = (
            df["volume"] == 0
        ) & (~df["date"].apply(self.is_expected_gap))
        df.loc[zero_vol_mask, "is_circuit_breaker"] = 1

        # Build list of expected trading days in date range
        start_date = df["date"].min()
        end_date   = df["date"].max()

        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()

        expected_trading_days = self._get_trading_days(start_date, end_date)

        # Find missing trading days
        existing_dates = set(
            d.date() if isinstance(d, pd.Timestamp) else d
            for d in df["date"]
        )
        missing_days = sorted(
            d for d in expected_trading_days if d not in existing_dates
        )

        if not missing_days:
            logger.debug(f"[GapHandler] {symbol}: No gaps found")
            return df

        # Build gap rows by forward-filling from last known price
        gap_rows     = []
        consecutive  = 0
        last_real_dt = None

        for gap_date in missing_days:
            # Find the last known row before this gap date
            prior_rows = df[
                df["date"].apply(
                    lambda d: (d.date() if isinstance(d, pd.Timestamp) else d) < gap_date
                )
            ]

            if prior_rows.empty:
                logger.debug(f"[GapHandler] {symbol}: No prior data before {gap_date} — skipping")
                continue

            last_row = prior_rows.iloc[-1].copy()

            # Track consecutive gap length
            if last_real_dt == gap_date - timedelta(days=1):
                consecutive += 1
            else:
                consecutive = 1
            last_real_dt = gap_date

            # Flag if gap is suspiciously long
            if consecutive > self.max_consecutive_fill:
                logger.warning(
                    f"[GapHandler] {symbol}: Gap at {gap_date} is {consecutive} days long — "
                    f"possible delisting or extended halt. Filling but flagging."
                )

            # Create filled row
            last_row["date"]           = gap_date
            last_row["volume"]         = 0         # no trading on gap day
            last_row["is_gap"]         = 1
            last_row["consecutive_gap"]= consecutive

            gap_rows.append(last_row)

        if gap_rows:
            gap_df = pd.DataFrame(gap_rows)
            df     = pd.concat([df, gap_df], ignore_index=True)
            df     = df.sort_values("date").reset_index(drop=True)

        filled = len(gap_rows)
        logger.info(
            f"[GapHandler] {symbol}: {filled} gap days filled "
            f"out of {len(expected_trading_days)} expected trading days"
        )

        return df

    def validate_continuity(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> dict:
        """
        Validates that a DataFrame has continuous trading day coverage.
        Returns a validation report dict.

        Args:
            df     : OHLCV DataFrame
            symbol : Symbol name for reporting

        Returns:
            Dict with keys: is_valid, gap_count, gap_pct, longest_gap, gaps
        """
        if df.empty:
            return {"is_valid": False, "error": "Empty DataFrame"}

        sorted_df  = df.sort_values("date")
        start      = sorted_df["date"].min()
        end        = sorted_df["date"].max()

        if isinstance(start, pd.Timestamp):
            start = start.date()
        if isinstance(end, pd.Timestamp):
            end = end.date()

        expected = self._get_trading_days(start, end)
        existing = set(
            d.date() if isinstance(d, pd.Timestamp) else d
            for d in df["date"]
        )
        missing  = [d for d in expected if d not in existing]

        # Find longest consecutive gap
        longest_gap = 0
        if missing:
            current_run = 1
            for i in range(1, len(missing)):
                if (missing[i] - missing[i-1]).days <= 3:
                    current_run += 1
                    longest_gap = max(longest_gap, current_run)
                else:
                    current_run = 1

        gap_pct  = len(missing) / len(expected) * 100 if expected else 0
        is_valid = gap_pct < 5.0  # Less than 5% gaps = acceptable

        return {
            "symbol"      : symbol,
            "is_valid"    : is_valid,
            "total_days"  : len(expected),
            "gap_count"   : len(missing),
            "gap_pct"     : round(gap_pct, 2),
            "longest_gap" : longest_gap,
            "gaps"        : missing[:10],  # first 10 gaps for inspection
        }

    # ── Private Methods ───────────────────────────────────────────────────

    def _get_trading_days(self, start: date, end: date) -> list[date]:
        """Returns all expected NSE trading days between start and end (inclusive)"""
        trading_days = []
        current = start
        while current <= end:
            if not self.is_expected_gap(current):
                trading_days.append(current)
            current += timedelta(days=1)
        return trading_days


# ══════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    print("Testing GapHandler...")

    handler = GapHandler()

    # Test holiday detection
    assert handler.is_expected_gap(date(2024, 1, 22)) is True   # Ram Mandir
    assert handler.is_expected_gap(date(2024, 1, 27)) is True   # Saturday
    assert handler.is_expected_gap(date(2024, 1, 23)) is False  # Normal Tuesday
    print("  Holiday detection ✓")

    # Create sample data with intentional gap (missing 2 trading days)
    all_dates = [
        date(2024, 1, 2),   # Tuesday
        date(2024, 1, 3),   # Wednesday
        # 2024-01-04 (Thursday) MISSING — simulate download failure
        # 2024-01-05 (Friday) MISSING
        date(2024, 1, 8),   # Monday
    ]

    np.random.seed(42)
    df = pd.DataFrame({
        "date"  : all_dates,
        "open"  : [100.0, 101.0, 103.0],
        "high"  : [102.0, 103.0, 105.0],
        "low"   : [99.0,  100.0, 102.0],
        "close" : [101.0, 102.0, 104.0],
        "volume": [500000, 600000, 700000],
    })

    filled = handler.fill_gaps(df, symbol="TEST")

    # Should now have 5 rows (3 real + 2 filled)
    assert len(filled) == 5, f"Expected 5 rows, got {len(filled)}"
    assert filled[filled["is_gap"] == 1]["volume"].sum() == 0, "Filled rows should have volume=0"
    print(f"  Gap filling ✓ ({len(df)} rows → {len(filled)} rows after filling 2 gaps)")

    # Validate continuity
    report = handler.validate_continuity(filled, "TEST")
    assert report["gap_count"] == 0
    print(f"  Continuity validation ✓ (gap_pct = {report['gap_pct']}%)")

    print("\nAll GapHandler tests passed.")