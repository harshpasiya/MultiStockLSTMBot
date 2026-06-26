"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Embedding Pre-computation                       ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : training/precompute_embeddings.py                      ║
║                                                                          ║
║  Reads all 28 features from features_fused — exactly what the           ║
║  backbone saw during training. This is critical for signal quality.     ║
║                                                                          ║
║  During training, GodsEyeEnv built sequences like this:                 ║
║    For each date D:                                                      ║
║      seq = features_fused[symbol][D-60 : D]  →  shape (60, 28)         ║
║      embedding = backbone(seq)               →  shape (128,)            ║
║                                                                          ║
║  This script does exactly the same thing and stores results in DB.      ║
║                                                                          ║
║  Key fix from old version:                                               ║
║    OLD: built sequences from raw OHLCV (4 features + 24 zeros)         ║
║    NEW: reads all 28 features from features_fused                       ║
║                                                                          ║
║  Run after every evening feature pipeline:                               ║
║    python -m training.precompute_embeddings --device cuda               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import numpy as np
import psycopg2
import psycopg2.extras
import torch

from datetime  import date
from pathlib   import Path
from loguru    import logger
from dotenv    import load_dotenv
from tqdm      import tqdm

from models.backbone import GodsEyeBackbone

load_dotenv()

DB_URL        = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)
ROOT_DIR      = Path(__file__).parent.parent
PRETRAIN_CKPT = ROOT_DIR / "checkpoints" / "pretrain_best.pt"

# Must match training exactly
SEQ_LEN       = 60
EMBEDDING_DIM = 128
BATCH_SIZE    = 64
TRAIN_START   = "2019-01-01"

# All 28 feature columns — must match features_fused exactly
FEATURE_COLS = [
    "f00_trend_score",        "f01_ema_ribbon_gap",
    "f02_adx_normalized",     "f03_supertrend_dir",
    "f04_price_vs_ema200",    "f05_swing_structure",
    "f06_msi_signal",         "f07_vrsi_normalized",
    "f08_mfi_normalized",     "f09_msi_divergence",
    "f10_mds_continuous",     "f11_fii_norm",
    "f12_dii_norm",           "f13_sentiment_score",
    "f14_sentiment_momentum", "f15_event_flag",
    "f16_market_fear_greed_n","f17_volatility_score",
    "f18_atr_pct_normalized", "f19_vol_regime_code_n",
    "f20_hv_percentile_n",    "f21_correlation_score",
    "f22_sector_divergence_n","f23_lead_lag_score",
    "f24_peer_corr_mean",     "f25_delivery_mom_n",
    "f26_swing_tp_normalized","f27_swing_sl_normalized",
]
N_FEATURES = len(FEATURE_COLS)  # must be 28


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS backbone_embeddings (
                date      DATE        NOT NULL,
                symbol    VARCHAR(20) NOT NULL,
                embedding FLOAT4[]    NOT NULL,
                PRIMARY KEY (date, symbol)
            );
        """)
        try:
            cur.execute("""
                SELECT create_hypertable(
                    'backbone_embeddings', 'date',
                    if_not_exists => TRUE,
                    migrate_data  => TRUE
                );
            """)
        except Exception:
            pass
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_emb_symbol
            ON backbone_embeddings (symbol, date DESC);
        """)
    conn.commit()
    logger.info("backbone_embeddings table ready.")


def load_backbone(device: str) -> GodsEyeBackbone:
    backbone = GodsEyeBackbone()
    ckpt     = torch.load(
        PRETRAIN_CKPT, map_location=device, weights_only=False
    )
    state = ckpt.get("backbone_state", ckpt)
    backbone.load_state_dict(state, strict=False)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    logger.info(f"Backbone loaded on {device}.")
    return backbone


def get_symbols(conn) -> list:
    """All symbols present in features_fused."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT symbol FROM features_fused ORDER BY symbol;
        """)
        return [r[0] for r in cur.fetchall()]


def load_symbol_features(
    conn,
    symbol    : str,
    start_date: str,
    end_date  : str,
) -> tuple[list, np.ndarray]:
    """
    Loads all 28 features for one symbol from features_fused.
    Returns (dates_list, matrix_float32) where matrix is (N, 28).
    """
    cols = ", ".join(FEATURE_COLS)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT date, {cols}
            FROM features_fused
            WHERE symbol = %s
              AND date BETWEEN %s AND %s
            ORDER BY date ASC;
        """, (symbol, start_date, end_date))
        rows = cur.fetchall()

    if not rows:
        return [], np.empty((0, N_FEATURES), dtype=np.float32)

    dates  = [str(r[0]) for r in rows]
    matrix = np.array(
        [[float(v if v is not None else 0.0) for v in r[1:]]
         for r in rows],
        dtype=np.float32,
    )
    # Same preprocessing as GodsEyeEnv — replace NaN/Inf with 0
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    return dates, matrix


def precompute(
    device    : str = "cuda" if torch.cuda.is_available() else "cpu",
    start_date: str = TRAIN_START,
    end_date  : str = "",
):
    if not end_date:
        end_date = str(date.today())

    logger.info(
        f"Pre-computing embeddings from features_fused | "
        f"{start_date} → {end_date} | device={device}"
    )
    logger.info(f"Using all {N_FEATURES} features — exact training distribution")

    conn    = psycopg2.connect(DB_URL)
    conn.autocommit = False
    ensure_table(conn)

    backbone = load_backbone(device)
    symbols  = get_symbols(conn)

    if not symbols:
        logger.error(
            "No symbols in features_fused. Run feature pipeline first:\n"
            "  python -m features.trend --mode all --start 2026-02-01\n"
            "  python -m features.msi   --mode all --start 2026-02-01\n"
            "  python -m features.fusion --mode all --start 2026-02-01"
        )
        conn.close()
        return

    logger.info(f"{len(symbols)} symbols found in features_fused")

    total_done  = 0
    total_skip  = 0
    records     = []

    for sym in tqdm(symbols, desc="Embeddings"):
        dates, matrix = load_symbol_features(
            conn, sym, start_date, end_date
        )

        if len(dates) < SEQ_LEN:
            logger.debug(
                f"{sym}: only {len(dates)} feature rows "
                f"(need {SEQ_LEN}) — skipping"
            )
            total_skip += 1
            continue

        # Build rolling windows: each window is SEQ_LEN rows ending at index i
        # Window i → embedding labelled with dates[i]
        # Start from index SEQ_LEN-1 (first complete window)
        seqs        = []
        valid_dates = []

        for i in range(SEQ_LEN - 1, len(dates)):
            window = matrix[i - SEQ_LEN + 1 : i + 1]   # (SEQ_LEN, 28)
            seqs.append(window)
            valid_dates.append(dates[i])

        if not seqs:
            total_skip += 1
            continue

        # Batch inference on GPU
        for b in range(0, len(seqs), BATCH_SIZE):
            batch_seqs  = seqs[b : b + BATCH_SIZE]
            batch_dates = valid_dates[b : b + BATCH_SIZE]

            x = torch.tensor(
                np.stack(batch_seqs), dtype=torch.float32
            ).to(device)  # (B, SEQ_LEN, 28)

            with torch.no_grad():
                out = backbone(x)
                emb = (out[0] if isinstance(out, tuple) else out)\
                      .cpu().numpy()  # (B, 128)

            for i, d in enumerate(batch_dates):
                records.append({
                    "date"     : d,
                    "symbol"   : sym,
                    "embedding": emb[i].tolist(),
                })

        # Flush every 5000 records
        if len(records) >= 5000:
            _flush(conn, records)
            total_done += len(records)
            records     = []

    # Final flush
    if records:
        _flush(conn, records)
        total_done += len(records)

    conn.close()

    logger.success(
        f"Done. {total_done:,} embeddings stored | "
        f"{total_skip} symbols skipped (insufficient history)"
    )

    # Verify final state
    conn2 = psycopg2.connect(DB_URL)
    with conn2.cursor() as cur:
        cur.execute(
            "SELECT MIN(date), MAX(date), COUNT(*), "
            "COUNT(DISTINCT symbol) FROM backbone_embeddings;"
        )
        mn, mx, cnt, syms = cur.fetchone()
    conn2.close()
    logger.info(
        f"backbone_embeddings: {mn} → {mx} | "
        f"{cnt:,} vectors | {syms} symbols"
    )


def _flush(conn, records: list):
    sql = """
        INSERT INTO backbone_embeddings (date, symbol, embedding)
        VALUES (%(date)s, %(symbol)s, %(embedding)s)
        ON CONFLICT (date, symbol) DO UPDATE
            SET embedding = EXCLUDED.embedding;
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, records, page_size=500)
    conn.commit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Precompute backbone embeddings from features_fused"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--start",
        default=TRAIN_START,
        help="Start date YYYY-MM-DD (default: 2019-01-01)"
    )
    parser.add_argument(
        "--end",
        default="",
        help="End date YYYY-MM-DD (default: today)"
    )
    args = parser.parse_args()
    precompute(
        device     = args.device,
        start_date = args.start,
        end_date   = args.end,
    )