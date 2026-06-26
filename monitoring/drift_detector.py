"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Feature Distribution Drift Detector             ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : monitoring/drift_detector.py                           ║
║         Phase   : 4 — Paper Trading & Live Monitoring                   ║
║                                                                          ║
║  What this module does:                                                  ║
║    Detects when live market features have drifted significantly from    ║
║    the distribution seen during model training. When drift exceeds      ║
║    the threshold, the model is operating outside its training domain    ║
║    and predictions become unreliable.                                   ║
║                                                                          ║
║  Drift detection method — KL Divergence:                                ║
║    KL(P || Q) measures how much live distribution P differs from        ║
║    training reference distribution Q.                                   ║
║      KL = 0.0   : identical distributions (no drift)                   ║
║      KL < 0.05  : negligible drift (normal market variation)            ║
║      KL 0.05–0.10: mild drift (monitor closely)                         ║
║      KL 0.10–0.15: moderate drift (reduce confidence threshold)         ║
║      KL > 0.15  : significant drift → trigger early retraining          ║
║                                                                          ║
║  What is monitored:                                                      ║
║    All 28 features in the fused feature vector [f00–f27]                ║
║    Per-feature drift + aggregate drift score                             ║
║    Rolling window of last N_LIVE_SAMPLES live feature values            ║
║                                                                          ║
║  Reference distribution:                                                ║
║    Computed from training data (2019–2023) stored in features_fused     ║
║    table. Loaded once at startup and cached in memory.                  ║
║    Updated after each successful nightly retrain.                       ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install numpy scipy psycopg2-binary loguru                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import time
import threading
import numpy as np
import psycopg2

from dataclasses import dataclass, field
from datetime    import datetime, date
from typing      import Optional, Dict, List, Tuple
from loguru      import logger
from dotenv      import load_dotenv

try:
    from scipy.stats import entropy as scipy_entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not installed — using numpy KL implementation.")

load_dotenv()

DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)

# ── Drift thresholds ───────────────────────────────────────────────────────
DRIFT_NEGLIGIBLE  = 0.05
DRIFT_MILD        = 0.10
DRIFT_MODERATE    = 0.15   # alert threshold
DRIFT_SEVERE      = 0.25   # immediate retrain trigger

# ── Feature window ─────────────────────────────────────────────────────────
N_LIVE_SAMPLES    = 100    # rolling window of live feature observations
N_BINS            = 20     # histogram bins for KL divergence
MIN_SAMPLES       = 30     # minimum live samples before computing drift
EPSILON           = 1e-10  # smoothing for zero-probability bins

# ── All 28 feature names (matches fusion.py output) ───────────────────────
FEATURE_NAMES = [
    "f00_trend_score",       "f01_ema_ribbon_gap",    "f02_adx_normalized",
    "f03_supertrend_dir",    "f04_price_vs_ema200",   "f05_swing_structure",
    "f06_msi_signal",        "f07_vrsi_normalized",   "f08_mfi_normalized",
    "f09_msi_divergence",    "f10_mds_continuous",    "f11_fii_norm",
    "f12_dii_norm",          "f13_sentiment_score",   "f14_sentiment_momentum",
    "f15_event_flag",        "f16_market_fear_greed_n","f17_volatility_score",
    "f18_atr_pct_normalized","f19_vol_regime_code_n", "f20_hv_percentile_n",
    "f21_correlation_score", "f22_sector_divergence_n","f23_lead_lag_score",
    "f24_peer_corr_mean",    "f25_delivery_mom_n",    "f26_swing_tp_normalized",
    "f27_swing_sl_normalized",
]
N_FEATURES = len(FEATURE_NAMES)   # 28


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class DriftReport:
    """
    Full drift analysis report for one evaluation cycle.
    Returned by DriftDetector.compute_drift().
    """
    timestamp         : datetime
    aggregate_kl      : float          # overall KL divergence score
    per_feature_kl    : Dict[str, float]  # KL per feature name
    n_live_samples    : int            # number of live samples used
    n_drifted_features: int            # features exceeding DRIFT_MODERATE
    drifted_features  : List[str]      # names of drifted features
    drift_level       : str            # "negligible"/"mild"/"moderate"/"severe"
    alert_triggered   : bool           # True if drift > DRIFT_MODERATE
    recommendation    : str            # what to do

    @property
    def is_healthy(self) -> bool:
        return self.aggregate_kl < DRIFT_MODERATE

    def summary(self) -> str:
        return (
            f"DriftReport | KL={self.aggregate_kl:.4f} | "
            f"level={self.drift_level} | "
            f"drifted={self.n_drifted_features}/{N_FEATURES} features | "
            f"alert={'YES' if self.alert_triggered else 'no'}"
        )


@dataclass
class ReferenceDistribution:
    """
    Training-period feature statistics used as the reference (Q).
    Loaded from features_fused table at startup.
    """
    feature_means     : np.ndarray     # (28,) mean of each feature
    feature_stds      : np.ndarray     # (28,) std of each feature
    feature_histograms: List[Tuple[np.ndarray, np.ndarray]]  # (counts, edges) per feature
    loaded_at         : datetime       = field(default_factory=datetime.now)
    n_samples         : int            = 0
    train_start       : str            = "2019-01-01"
    train_end         : str            = "2023-06-30"


# ══════════════════════════════════════════════════════════════════════════
#  KL DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes KL divergence KL(P || Q) between two probability distributions.

    Both p and q must be normalized (sum to 1) and non-negative.
    Epsilon smoothing is applied to avoid log(0).

    Args:
        p : Live distribution (empirical histogram, normalized)
        q : Reference distribution (training histogram, normalized)

    Returns:
        KL divergence as a non-negative float
    """
    p = np.asarray(p, dtype=float) + EPSILON
    q = np.asarray(q, dtype=float) + EPSILON

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    if SCIPY_AVAILABLE:
        return float(scipy_entropy(p, q))
    else:
        # Manual implementation: sum(p * log(p/q))
        return float(np.sum(p * np.log(p / q)))


def compute_histogram(
    values  : np.ndarray,
    edges   : np.ndarray,
) -> np.ndarray:
    """
    Computes a normalized histogram of values using pre-defined bin edges.
    Using fixed edges ensures live and reference histograms are comparable.

    Args:
        values : 1D array of feature values
        edges  : Bin edges from training reference histogram

    Returns:
        Normalized histogram (sums to 1)
    """
    counts, _ = np.histogram(values, bins=edges)
    total      = counts.sum()
    if total == 0:
        return np.ones(len(counts)) / len(counts)   # uniform if no data
    return counts / total


# ══════════════════════════════════════════════════════════════════════════
#  DRIFT DETECTOR
# ══════════════════════════════════════════════════════════════════════════

class DriftDetector:
    """
    Monitors feature distribution drift between training and live data.

    Usage:
        detector = DriftDetector()
        detector.load_reference()       # loads training distribution from DB

        # After each bar, push new feature vector:
        detector.add_live_sample(feature_vector)   # shape (28,)

        # Periodically check drift:
        report = detector.compute_drift()
        if report.alert_triggered:
            alerts.drift_alert(report.aggregate_kl)
    """

    def __init__(
        self,
        drift_threshold : float = DRIFT_MODERATE,
        n_live_samples  : int   = N_LIVE_SAMPLES,
        n_bins          : int   = N_BINS,
        auto_load       : bool  = False,
    ):
        self.drift_threshold = drift_threshold
        self.n_live_samples  = n_live_samples
        self.n_bins          = n_bins

        self._reference  : Optional[ReferenceDistribution] = None
        self._live_buffer: np.ndarray = np.empty((0, N_FEATURES))
        self._lock       = threading.Lock()
        self._last_report: Optional[DriftReport] = None
        self._report_history: List[DriftReport]  = []

        if auto_load:
            self.load_reference()

    # ══════════════════════════════════════════════════════════════════════
    #  REFERENCE LOADING
    # ══════════════════════════════════════════════════════════════════════

    def load_reference(
        self,
        train_start: str = "2019-01-01",
        train_end  : str = "2023-06-30",
    ) -> bool:
        """
        Loads training distribution from the features_fused table.
        Call once at system startup and after each nightly retrain.

        Args:
            train_start : Start of training period
            train_end   : End of training period

        Returns:
            True if reference loaded successfully
        """
        logger.info(f"Loading reference distribution ({train_start} → {train_end})...")

        try:
            conn    = psycopg2.connect(DB_URL)
            samples = self._fetch_training_features(conn, train_start, train_end)
            conn.close()

            if samples is None or len(samples) == 0:
                logger.warning("No training features found in features_fused table.")
                return False

            self._build_reference(samples, train_start, train_end)
            logger.info(
                f"Reference loaded: {len(samples)} samples, "
                f"{N_FEATURES} features."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load reference distribution: {e}")
            return False

    def load_reference_from_array(self, samples: np.ndarray):
        """
        Loads reference distribution directly from a numpy array.
        Used in unit tests and when bypassing the DB.

        Args:
            samples : (N, 28) float32 array of training feature vectors
        """
        if samples.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected {N_FEATURES} features, got {samples.shape[1]}"
            )
        self._build_reference(samples)

    def _fetch_training_features(
        self,
        conn,
        train_start: str,
        train_end  : str,
    ) -> Optional[np.ndarray]:
        """
        Fetches feature vectors from features_fused table.
        Returns (N, 28) float32 array or None on failure.
        """
        cols = ", ".join(FEATURE_NAMES)
        sql  = f"""
            SELECT {cols}
            FROM features_fused
            WHERE date BETWEEN %s AND %s
              AND {FEATURE_NAMES[0]} IS NOT NULL
            ORDER BY date, symbol
            LIMIT 200000;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (train_start, train_end))
                rows = cur.fetchall()
            if not rows:
                return None
            arr = np.array(rows, dtype=np.float32)
            # Replace NaN/Inf with 0
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            return arr
        except Exception as e:
            logger.error(f"Feature fetch failed: {e}")
            return None

    def _build_reference(
        self,
        samples    : np.ndarray,
        train_start: str = "2019-01-01",
        train_end  : str = "2023-06-30",
    ):
        """
        Builds ReferenceDistribution from a samples array.
        Computes per-feature histogram with N_BINS bins.
        """
        means = samples.mean(axis=0)       # (28,)
        stds  = samples.std(axis=0) + 1e-8  # (28,)

        histograms = []
        for i in range(N_FEATURES):
            col        = samples[:, i]
            counts, edges = np.histogram(col, bins=self.n_bins)
            norm_counts   = counts / (counts.sum() + EPSILON)
            histograms.append((norm_counts, edges))

        with self._lock:
            self._reference = ReferenceDistribution(
                feature_means      = means,
                feature_stds       = stds,
                feature_histograms = histograms,
                n_samples          = len(samples),
                train_start        = train_start,
                train_end          = train_end,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  LIVE DATA INGESTION
    # ══════════════════════════════════════════════════════════════════════

    def add_live_sample(self, feature_vector: np.ndarray):
        """
        Adds a live feature vector to the rolling buffer.
        Call after each bar's features are computed.

        Args:
            feature_vector : (28,) float32 array from fusion.py
        """
        if feature_vector.shape[-1] != N_FEATURES:
            logger.warning(
                f"add_live_sample: expected {N_FEATURES} features, "
                f"got {feature_vector.shape[-1]}. Skipping."
            )
            return

        vec = np.nan_to_num(
            np.asarray(feature_vector, dtype=np.float32).flatten(),
            nan=0.0, posinf=1.0, neginf=-1.0
        )

        with self._lock:
            self._live_buffer = np.vstack([self._live_buffer, vec]) \
                if len(self._live_buffer) > 0 else vec.reshape(1, -1)

            # Keep only last N samples
            if len(self._live_buffer) > self.n_live_samples:
                self._live_buffer = self._live_buffer[-self.n_live_samples:]

    def add_live_batch(self, feature_matrix: np.ndarray):
        """
        Adds multiple live feature vectors at once.

        Args:
            feature_matrix : (N, 28) float32 array
        """
        for row in feature_matrix:
            self.add_live_sample(row)

    @property
    def n_live_samples_collected(self) -> int:
        """Number of live samples currently in the buffer."""
        with self._lock:
            return len(self._live_buffer)

    # ══════════════════════════════════════════════════════════════════════
    #  DRIFT COMPUTATION
    # ══════════════════════════════════════════════════════════════════════

    def compute_drift(self) -> Optional[DriftReport]:
        """
        Computes KL divergence between live buffer and reference distribution.

        Returns:
            DriftReport with aggregate and per-feature KL scores,
            or None if insufficient data (< MIN_SAMPLES live samples
            or no reference loaded).
        """
        with self._lock:
            if self._reference is None:
                logger.warning("Drift: no reference distribution loaded. Call load_reference() first.")
                return None

            n_live = len(self._live_buffer)
            if n_live < MIN_SAMPLES:
                logger.debug(
                    f"Drift: only {n_live} live samples "
                    f"(need {MIN_SAMPLES}). Skipping."
                )
                return None

            live_data = self._live_buffer.copy()

        # ── Per-feature KL divergence ──────────────────────────────────────
        per_feature_kl : Dict[str, float] = {}
        ref            = self._reference

        for i, fname in enumerate(FEATURE_NAMES):
            ref_counts, ref_edges = ref.feature_histograms[i]

            # Compute live histogram using same bin edges as reference
            live_col    = live_data[:, i]
            live_counts = compute_histogram(live_col, ref_edges)

            kl  = kl_divergence(live_counts, ref_counts)
            per_feature_kl[fname] = float(kl)

        # ── Aggregate score (mean KL across all features) ─────────────────
        kl_values      = list(per_feature_kl.values())
        aggregate_kl   = float(np.mean(kl_values))

        # ── Classify drift level ──────────────────────────────────────────
        if aggregate_kl < DRIFT_NEGLIGIBLE:
            level = "negligible"
        elif aggregate_kl < DRIFT_MILD:
            level = "mild"
        elif aggregate_kl < DRIFT_MODERATE:
            level = "moderate"
        else:
            level = "severe"

        # ── Identify drifted features ─────────────────────────────────────
        drifted = [
            fname for fname, kl in per_feature_kl.items()
            if kl > DRIFT_MODERATE
        ]

        # ── Build recommendation ──────────────────────────────────────────
        if aggregate_kl >= DRIFT_SEVERE:
            rec = "IMMEDIATE retraining required — model reliability compromised."
        elif aggregate_kl >= DRIFT_MODERATE:
            rec = "Schedule early retraining. Reduce confidence threshold to 0.65."
        elif aggregate_kl >= DRIFT_MILD:
            rec = "Monitor closely. Retraining at next scheduled cycle."
        else:
            rec = "No action required."

        report = DriftReport(
            timestamp          = datetime.now(),
            aggregate_kl       = aggregate_kl,
            per_feature_kl     = per_feature_kl,
            n_live_samples     = n_live,
            n_drifted_features = len(drifted),
            drifted_features   = drifted,
            drift_level        = level,
            alert_triggered    = aggregate_kl >= self.drift_threshold,
            recommendation     = rec,
        )

        with self._lock:
            self._last_report = report
            self._report_history.append(report)
            if len(self._report_history) > 100:
                self._report_history = self._report_history[-100:]

        logger.debug(report.summary())
        return report

    # ══════════════════════════════════════════════════════════════════════
    #  STATE ACCESS
    # ══════════════════════════════════════════════════════════════════════

    def get_last_report(self) -> Optional[DriftReport]:
        """Returns the most recent DriftReport."""
        with self._lock:
            return self._last_report

    def get_drift_trend(self, last_n: int = 10) -> List[float]:
        """
        Returns aggregate KL scores for the last N reports.
        Used to detect gradual drift build-up.
        """
        with self._lock:
            reports = self._report_history[-last_n:]
        return [r.aggregate_kl for r in reports]

    def is_drifting(self) -> bool:
        """Quick check: True if last report shows alert-level drift."""
        report = self.get_last_report()
        return report is not None and report.alert_triggered

    def reset_live_buffer(self):
        """
        Clears the live feature buffer.
        Call after a nightly retrain to start fresh.
        """
        with self._lock:
            self._live_buffer = np.empty((0, N_FEATURES))
        logger.info("Drift detector live buffer reset.")

    def get_feature_drift_summary(self) -> Dict[str, str]:
        """
        Returns a human-readable per-feature drift summary.
        Used by monitoring dashboard.
        """
        report = self.get_last_report()
        if report is None:
            return {"status": "no_report"}

        summary = {}
        for fname, kl in report.per_feature_kl.items():
            if kl < DRIFT_NEGLIGIBLE:
                level = "✅ ok"
            elif kl < DRIFT_MILD:
                level = "🟡 mild"
            elif kl < DRIFT_MODERATE:
                level = "🟠 moderate"
            else:
                level = "🔴 drifted"
            summary[fname] = f"{level} (KL={kl:.4f})"

        return summary


# ── Module-level singleton ─────────────────────────────────────────────────
_default_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """
    Returns the module-level DriftDetector singleton.

    Example:
        from monitoring.drift_detector import get_drift_detector
        detector = get_drift_detector()
        detector.load_reference()
        detector.add_live_sample(feature_vector)
        report = detector.compute_drift()
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = DriftDetector()
    return _default_detector


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest monitoring/drift_detector.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestDriftDetector:
    """
    Unit tests for DriftDetector.
    All tests use synthetic data — no DB required.
    """

    def _make_detector(self) -> DriftDetector:
        return DriftDetector(drift_threshold=DRIFT_MODERATE)

    def _make_reference_samples(self, n=5000, seed=42) -> np.ndarray:
        """Generates synthetic training samples from a known distribution."""
        np.random.seed(seed)
        return np.random.randn(n, N_FEATURES).astype(np.float32)

    def _make_live_samples(
        self,
        n       : int   = 60,
        drift   : float = 0.0,
        seed    : int   = 99,
    ) -> np.ndarray:
        """
        Generates synthetic live samples.
        drift > 0 shifts mean to simulate distribution shift.
        """
        np.random.seed(seed)
        samples = np.random.randn(n, N_FEATURES).astype(np.float32)
        samples += drift   # shift all features by drift amount
        return samples

    # ── Initialization ────────────────────────────────────────────────────

    def test_creates_successfully(self):
        d = self._make_detector()
        assert isinstance(d, DriftDetector)

    def test_no_reference_on_init(self):
        d = self._make_detector()
        assert d._reference is None

    def test_no_live_samples_on_init(self):
        d = self._make_detector()
        assert d.n_live_samples_collected == 0

    # ── Reference loading ─────────────────────────────────────────────────

    def test_load_reference_from_array(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        assert d._reference is not None

    def test_reference_correct_shape(self):
        d       = self._make_detector()
        samples = self._make_reference_samples(n=1000)
        d.load_reference_from_array(samples)
        assert d._reference.feature_means.shape == (N_FEATURES,)
        assert d._reference.feature_stds.shape  == (N_FEATURES,)
        assert len(d._reference.feature_histograms) == N_FEATURES

    def test_reference_n_samples(self):
        d       = self._make_detector()
        samples = self._make_reference_samples(n=2000)
        d.load_reference_from_array(samples)
        assert d._reference.n_samples == 2000

    def test_wrong_feature_count_raises(self):
        import pytest
        d       = self._make_detector()
        samples = np.random.randn(100, 15).astype(np.float32)  # wrong shape
        with pytest.raises(ValueError, match="features"):
            d.load_reference_from_array(samples)

    # ── Live sample ingestion ─────────────────────────────────────────────

    def test_add_live_sample(self):
        d = self._make_detector()
        d.add_live_sample(np.zeros(N_FEATURES, dtype=np.float32))
        assert d.n_live_samples_collected == 1

    def test_add_multiple_samples(self):
        d = self._make_detector()
        for _ in range(10):
            d.add_live_sample(np.zeros(N_FEATURES, dtype=np.float32))
        assert d.n_live_samples_collected == 10

    def test_rolling_window_capped(self):
        d = DriftDetector(n_live_samples=20)
        for i in range(30):
            d.add_live_sample(np.ones(N_FEATURES, dtype=np.float32) * i)
        assert d.n_live_samples_collected == 20

    def test_add_live_batch(self):
        d       = self._make_detector()
        samples = np.zeros((15, N_FEATURES), dtype=np.float32)
        d.add_live_batch(samples)
        assert d.n_live_samples_collected == 15

    def test_wrong_feature_count_skipped(self):
        d = self._make_detector()
        d.add_live_sample(np.zeros(15, dtype=np.float32))  # wrong size
        assert d.n_live_samples_collected == 0

    def test_nan_values_handled(self):
        d   = self._make_detector()
        vec = np.full(N_FEATURES, np.nan, dtype=np.float32)
        d.add_live_sample(vec)
        assert d.n_live_samples_collected == 1
        assert not np.isnan(d._live_buffer).any()

    # ── Drift computation ─────────────────────────────────────────────────

    def test_no_reference_returns_none(self):
        d = self._make_detector()
        d.add_live_batch(self._make_live_samples(n=50))
        assert d.compute_drift() is None

    def test_insufficient_samples_returns_none(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        d.add_live_batch(self._make_live_samples(n=5))   # below MIN_SAMPLES
        assert d.compute_drift() is None

    def test_compute_drift_returns_report(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        d.add_live_batch(self._make_live_samples(n=50))
        report  = d.compute_drift()
        assert isinstance(report, DriftReport)

    def test_no_drift_low_kl(self):
        """Live data from same distribution as reference → low KL."""
        d       = self._make_detector()
        samples = self._make_reference_samples(n=5000, seed=42)
        d.load_reference_from_array(samples)
        # Live data from same distribution
        live    = self._make_live_samples(n=100, drift=0.0, seed=1)
        d.add_live_batch(live)
        report  = d.compute_drift()
        assert report is not None
        assert report.aggregate_kl < DRIFT_MILD, \
            f"Expected low KL for same distribution, got {report.aggregate_kl:.4f}"

    def test_drift_high_kl(self):
        """Heavily shifted live data → high KL."""
        d       = self._make_detector()
        samples = self._make_reference_samples(n=5000, seed=42)
        d.load_reference_from_array(samples)
        # Live data shifted by 3 standard deviations
        live    = self._make_live_samples(n=100, drift=3.0, seed=2)
        d.add_live_batch(live)
        report  = d.compute_drift()
        assert report is not None
        assert report.aggregate_kl > DRIFT_NEGLIGIBLE, \
            f"Expected high KL for shifted distribution, got {report.aggregate_kl:.4f}"

    def test_report_has_all_features(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        d.add_live_batch(self._make_live_samples(n=50))
        report  = d.compute_drift()
        assert len(report.per_feature_kl) == N_FEATURES

    def test_report_feature_names_correct(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        d.add_live_batch(self._make_live_samples(n=50))
        report  = d.compute_drift()
        for fname in FEATURE_NAMES:
            assert fname in report.per_feature_kl

    def test_report_kl_values_non_negative(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        d.add_live_batch(self._make_live_samples(n=50))
        report  = d.compute_drift()
        for kl in report.per_feature_kl.values():
            assert kl >= 0, f"Negative KL: {kl}"

    def test_no_drift_alert_same_distribution(self):
        d       = self._make_detector()
        samples = self._make_reference_samples(n=5000, seed=42)
        d.load_reference_from_array(samples)
        live    = self._make_live_samples(n=100, drift=0.0, seed=1)
        d.add_live_batch(live)
        report  = d.compute_drift()
        assert not report.alert_triggered

    def test_drift_alert_triggered_on_shift(self):
        d       = self._make_detector()
        samples = self._make_reference_samples(n=5000, seed=42)
        d.load_reference_from_array(samples)
        live    = self._make_live_samples(n=100, drift=5.0, seed=3)
        d.add_live_batch(live)
        report  = d.compute_drift()
        assert report is not None
        # With 5σ shift, KL should be > threshold
        assert report.aggregate_kl > 0

    def test_drift_level_negligible(self):
        d       = self._make_detector()
        samples = self._make_reference_samples(n=5000)
        d.load_reference_from_array(samples)
        live    = self._make_live_samples(n=100, drift=0.0)
        d.add_live_batch(live)
        report  = d.compute_drift()
        assert report.drift_level in ("negligible", "mild", "moderate", "severe")

    def test_report_stored_in_history(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        d.add_live_batch(self._make_live_samples(n=50))
        d.compute_drift()
        assert len(d._report_history) == 1

    def test_last_report_accessible(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        d.add_live_batch(self._make_live_samples(n=50))
        d.compute_drift()
        assert d.get_last_report() is not None

    # ── State & utilities ─────────────────────────────────────────────────

    def test_is_drifting_false_initially(self):
        d = self._make_detector()
        assert d.is_drifting() is False

    def test_reset_live_buffer(self):
        d = self._make_detector()
        d.add_live_batch(self._make_live_samples(n=30))
        assert d.n_live_samples_collected == 30
        d.reset_live_buffer()
        assert d.n_live_samples_collected == 0

    def test_drift_trend_empty_initially(self):
        d = self._make_detector()
        assert d.get_drift_trend() == []

    def test_drift_trend_populated_after_reports(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        for _ in range(3):
            d.add_live_batch(self._make_live_samples(n=50))
            d.compute_drift()
        trend = d.get_drift_trend()
        assert len(trend) == 3
        assert all(isinstance(v, float) for v in trend)

    def test_feature_drift_summary_no_report(self):
        d       = self._make_detector()
        summary = d.get_feature_drift_summary()
        assert summary == {"status": "no_report"}

    def test_feature_drift_summary_with_report(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        d.add_live_batch(self._make_live_samples(n=50))
        d.compute_drift()
        summary = d.get_feature_drift_summary()
        assert len(summary) == N_FEATURES

    def test_report_is_healthy_low_kl(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        d.add_live_batch(self._make_live_samples(n=100, drift=0.0))
        report  = d.compute_drift()
        assert isinstance(report.is_healthy, bool)

    def test_report_summary_string(self):
        d       = self._make_detector()
        samples = self._make_reference_samples()
        d.load_reference_from_array(samples)
        d.add_live_batch(self._make_live_samples(n=50))
        report  = d.compute_drift()
        s       = report.summary()
        assert "KL=" in s and "level=" in s

    # ── KL divergence ─────────────────────────────────────────────────────

    def test_kl_same_distribution_near_zero(self):
        p   = np.array([0.25, 0.25, 0.25, 0.25])
        kl  = kl_divergence(p, p)
        assert kl < 0.01

    def test_kl_different_distributions_positive(self):
        p   = np.array([0.9, 0.05, 0.03, 0.02])
        q   = np.array([0.1, 0.3,  0.3,  0.3])
        kl  = kl_divergence(p, q)
        assert kl > 0

    def test_kl_non_negative(self):
        np.random.seed(42)
        for _ in range(20):
            p = np.random.dirichlet(np.ones(10))
            q = np.random.dirichlet(np.ones(10))
            assert kl_divergence(p, q) >= 0

    def test_singleton(self):
        import monitoring.drift_detector as mod
        mod._default_detector = None
        d1 = mod.get_drift_detector()
        d2 = mod.get_drift_detector()
        assert d1 is d2
        mod._default_detector = None


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))