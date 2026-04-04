"""
╔══════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Feature Normalizer                     ║
║         Project : MultiStockLSTMBot                             ║
║         File    : data/pipeline/normalizer.py                   ║
║         Phase   : 0 — Data Infrastructure                       ║
║         Purpose : Rolling Z-score normalization for all features ║
║                   Used by all 6 pillars before LSTM/Transformer  ║
╚══════════════════════════════════════════════════════════════════╝

Why rolling normalization (not global):
---------------------------------------
    Global normalization causes look-ahead bias in backtesting.
    If you normalize using the full dataset's mean/std, your model
    "sees" future data during training — this inflates backtest results.

    Rolling normalization only uses data available at that point in time,
    making it identical to what you'd have in live trading.

    Window: 252 trading days (1 year) — long enough to be stable,
    short enough to adapt to regime changes (e.g. post-COVID vol regime).

Usage:
------
    from data.pipeline.normalizer import RollingNormalizer

    normalizer = RollingNormalizer(window=252)

    # Normalize price features
    df = normalizer.fit_transform(df, columns=['close', 'open', 'high', 'low'])

    # Normalize volume (log transform applied first)
    df = normalizer.normalize_volume(df)

    # Normalize a custom feature
    df = normalizer.normalize_single(df, column='rsi', clip_sigma=3.0)
"""

import numpy as np
import pandas as pd

from loguru import logger


class RollingNormalizer:
    """
    Applies rolling Z-score normalization to time series features.

    Formula:
        z = (x - rolling_mean) / (rolling_std + epsilon)
        z_clipped = clip(z, -clip_sigma, +clip_sigma)

    Args:
        window     : Rolling window in trading days (default: 252 = 1 year)
        min_periods: Minimum observations before normalizing (default: 20)
        clip_sigma : Clip Z-scores beyond this many std devs (default: 4.0)
        epsilon    : Small constant to prevent division by zero (default: 1e-8)
    """

    def __init__(
        self,
        window: int     = 252,
        min_periods: int = 20,
        clip_sigma: float = 4.0,
        epsilon: float   = 1e-8,
    ):
        self.window      = window
        self.min_periods = min_periods
        self.clip_sigma  = clip_sigma
        self.epsilon     = epsilon

    # ── Public Methods ────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: list[str],
        suffix: str = "_norm",
    ) -> pd.DataFrame:
        """
        Applies rolling Z-score normalization to specified columns.
        Adds normalized columns with suffix appended to original name.

        Args:
            df      : DataFrame — must be sorted ascending by date
            columns : Column names to normalize
            suffix  : Suffix for normalized column names (default: '_norm')

        Returns:
            DataFrame with original columns + normalized columns added

        Example:
            Input columns : ['close', 'volume']
            Output columns: ['close', 'volume', 'close_norm', 'volume_norm']
        """
        result = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"[Normalizer] Column '{col}' not in DataFrame — skipping")
                continue

            normalized = self._rolling_zscore(df[col])
            result[f"{col}{suffix}"] = normalized
            logger.debug(
                f"[Normalizer] '{col}' normalized | "
                f"range: [{normalized.min():.2f}, {normalized.max():.2f}]"
            )

        return result

    def normalize_volume(
        self,
        df: pd.DataFrame,
        col: str = "volume",
        output_col: str = "volume_norm",
    ) -> pd.DataFrame:
        """
        Normalizes volume using log transform + rolling Z-score.

        Volume is highly right-skewed — some stocks trade 100× more than
        others on the same day. Log transform makes the distribution roughly
        normal before Z-scoring, which prevents large-volume stocks from
        dominating the model's feature space.

        Formula:
            log_vol = log(1 + volume)   ← log1p handles zero safely
            z = rolling_zscore(log_vol)

        Args:
            df         : Input DataFrame
            col        : Volume column name (default: 'volume')
            output_col : Output column name (default: 'volume_norm')

        Returns:
            DataFrame with volume_norm column added
        """
        result = df.copy()

        if col not in df.columns:
            logger.warning(f"[Normalizer] Volume column '{col}' not found")
            return result

        # Handle zero and negative volumes safely
        vol = df[col].clip(lower=0)
        log_vol = np.log1p(vol)

        result[output_col] = self._rolling_zscore(log_vol)
        return result

    def normalize_single(
        self,
        df: pd.DataFrame,
        column: str,
        output_col: str = None,
        clip_sigma: float = None,
    ) -> pd.DataFrame:
        """
        Normalizes a single column with optional custom clip sigma.
        Useful for indicators with known bounded ranges (e.g. RSI 0-100).

        Args:
            df         : Input DataFrame
            column     : Column to normalize
            output_col : Output column name (default: column + '_norm')
            clip_sigma : Override clip threshold (default: self.clip_sigma)

        Returns:
            DataFrame with normalized column added
        """
        result     = df.copy()
        out_name   = output_col or f"{column}_norm"
        clip       = clip_sigma or self.clip_sigma

        if column not in df.columns:
            logger.warning(f"[Normalizer] Column '{column}' not found")
            return result

        normalized = self._rolling_zscore(df[column], clip_override=clip)
        result[out_name] = normalized
        return result

    def normalize_bounded(
        self,
        df: pd.DataFrame,
        column: str,
        lower_bound: float,
        upper_bound: float,
        output_col: str = None,
    ) -> pd.DataFrame:
        """
        Normalizes a bounded indicator to [-1, +1] range using min-max scaling.
        Better than Z-score for indicators with known fixed bounds (e.g. RSI, PCR).

        Formula:
            normalized = 2 × (x - lower) / (upper - lower) - 1

        Example:
            RSI [0, 100] → [-1, +1]
            PCR [0, 3]   → [-1, +1]

        Args:
            df          : Input DataFrame
            column      : Column to normalize
            lower_bound : Known minimum value of indicator
            upper_bound : Known maximum value of indicator
            output_col  : Output column name

        Returns:
            DataFrame with normalized column in [-1, +1] range
        """
        result   = df.copy()
        out_name = output_col or f"{column}_norm"

        if column not in df.columns:
            logger.warning(f"[Normalizer] Column '{column}' not found")
            return result

        range_size = upper_bound - lower_bound
        if range_size == 0:
            result[out_name] = 0.0
            return result

        normalized = 2 * (df[column].clip(lower_bound, upper_bound) - lower_bound) / range_size - 1
        result[out_name] = normalized
        return result

    def get_rolling_stats(
        self,
        series: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Returns rolling mean and std for a series.
        Useful for inspecting normalization statistics.

        Returns:
            Tuple of (rolling_mean, rolling_std)
        """
        rolling_mean = series.rolling(
            window=self.window, min_periods=self.min_periods
        ).mean()
        rolling_std = series.rolling(
            window=self.window, min_periods=self.min_periods
        ).std()
        return rolling_mean, rolling_std

    # ── Private Methods ───────────────────────────────────────────────────

    def _rolling_zscore(
        self,
        series: pd.Series,
        clip_override: float = None,
    ) -> pd.Series:
        """
        Core rolling Z-score computation.

        Args:
            series        : Input time series
            clip_override : Optional override for clip_sigma

        Returns:
            Normalized series clipped to ±clip_sigma
        """
        clip = clip_override or self.clip_sigma

        rolling_mean = series.rolling(
            window=self.window,
            min_periods=self.min_periods,
        ).mean()

        rolling_std = series.rolling(
            window=self.window,
            min_periods=self.min_periods,
        ).std()

        z_score = (series - rolling_mean) / (rolling_std + self.epsilon)
        return z_score.clip(-clip, clip)


# ══════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    print("Testing RollingNormalizer...")

    np.random.seed(42)
    n     = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    df = pd.DataFrame({
        "date"  : dates,
        "close" : 1000 + np.cumsum(np.random.randn(n) * 15),
        "volume": np.random.randint(500_000, 5_000_000, n),
        "rsi"   : np.random.uniform(20, 80, n),
    })

    normalizer = RollingNormalizer(window=60, min_periods=10)

    # Test fit_transform
    df = normalizer.fit_transform(df, columns=["close"])
    assert "close_norm" in df.columns
    assert df["close_norm"].dropna().between(-4, 4).all()
    print(f"  close_norm range: [{df['close_norm'].min():.3f}, {df['close_norm'].max():.3f}]  ✓")

    # Test volume normalization
    df = normalizer.normalize_volume(df)
    assert "volume_norm" in df.columns
    print(f"  volume_norm range: [{df['volume_norm'].min():.3f}, {df['volume_norm'].max():.3f}]  ✓")

    # Test bounded normalization (RSI)
    df = normalizer.normalize_bounded(df, "rsi", lower_bound=0, upper_bound=100)
    assert df["rsi_norm"].between(-1, 1).all()
    print(f"  rsi_norm range: [{df['rsi_norm'].min():.3f}, {df['rsi_norm'].max():.3f}]  ✓")

    print("\nAll normalizer tests passed.")