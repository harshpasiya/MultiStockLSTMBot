"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Paper Trading Control Dashboard                 ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : dashboard.py  (place in project root)                  ║
║                                                                          ║
║  Run: streamlit run dashboard.py                                         ║
║  URL: http://localhost:8501                                              ║
║                                                                          ║
║  Tabs:                                                                   ║
║    1. Overview    — portfolio snapshot, key metrics                      ║
║    2. Data        — load/backfill market data                            ║
║    3. Features    — run feature pipeline per pillar                      ║
║    4. Trading     — paper trading controls, open positions               ║
║    5. Report      — P&L report with custom date range                   ║
║    6. System      — infrastructure health, logs                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import subprocess
import time
import traceback
from datetime    import date, datetime, timedelta
from pathlib     import Path
from dotenv      import load_dotenv

load_dotenv()

import streamlit as st
import pandas    as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "G.O.D.S E.Y.E",
    page_icon  = "👁️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Project root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── DB config ──────────────────────────────────────────────────────────────
DB_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://godseye_user:godseye_pass@localhost:5433/godseye"
)

# ══════════════════════════════════════════════════════════════════════════
#  STYLING
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --navy:   #0D1B3E;
    --gold:   #C9A84C;
    --teal:   #0E7C7B;
    --red:    #B03A2E;
    --green:  #1A6B3C;
    --dark:   #080F1F;
    --card:   #111827;
    --border: #1E2D4A;
    --text:   #E8EDF5;
    --muted:  #6B7A99;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background:  var(--dark);
    color:       var(--text);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Top banner */
.gods-eye-banner {
    background: linear-gradient(135deg, var(--navy) 0%, #0a1628 60%, #0D1B3E 100%);
    border-bottom: 2px solid var(--gold);
    padding: 18px 32px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 16px;
}
.gods-eye-banner h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: var(--gold);
    margin: 0;
    letter-spacing: 0.15em;
}
.gods-eye-banner span {
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 0.1em;
}

/* Metric cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
}
.metric-card .positive { color: #4ADE80; }
.metric-card .negative { color: #F87171; }
.metric-card .neutral  { color: var(--gold); }
.metric-card .warning  { color: #FBBF24; }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    color: var(--gold);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Status badges */
.badge-ok      { background:#064E3B; color:#34D399; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-family:'Space Mono',monospace; }
.badge-warn    { background:#451A03; color:#FBBF24; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-family:'Space Mono',monospace; }
.badge-error   { background:#450A0A; color:#F87171; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-family:'Space Mono',monospace; }
.badge-paper   { background:#1E1B4B; color:#A5B4FC; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-family:'Space Mono',monospace; }

/* Streamlit overrides */
.stButton > button {
    background: var(--navy);
    color: var(--gold);
    border: 1px solid var(--gold);
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: var(--gold);
    color: var(--dark);
}
.stSelectbox label, .stDateInput label, .stTextInput label,
.stMultiSelect label, .stNumberInput label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    color: var(--muted);
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}
div[data-testid="stSidebar"] {
    background: var(--card);
    border-right: 1px solid var(--border);
}
.stDataFrame { background: var(--card); }
.stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  DB HELPERS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_db_conn():
    """Cached DB connection."""
    try:
        import psycopg2
        conn = psycopg2.connect(DB_URL)
        return conn
    except Exception as e:
        return None


def query_df(sql: str, params=None) -> pd.DataFrame:
    """Runs a SQL query and returns a DataFrame."""
    try:
        import psycopg2
        conn = psycopg2.connect(DB_URL)
        df   = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        st.error(f"DB Error: {e}")
        return pd.DataFrame()


def run_query(sql: str, params=None) -> bool:
    """Runs a non-SELECT SQL query."""
    try:
        import psycopg2
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"DB Error: {e}")
        return False


def db_ok() -> bool:
    """Quick DB health check."""
    try:
        import psycopg2
        conn = psycopg2.connect(DB_URL, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════
#  COMMAND RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_command(cmd: list, placeholder) -> tuple[bool, str]:
    """
    Runs a subprocess command and streams output to a Streamlit placeholder.
    Returns (success, output_text).
    """
    output_lines = []
    try:
        process = subprocess.Popen(
            cmd,
            stdout      = subprocess.PIPE,
            stderr      = subprocess.STDOUT,
            text        = True,
            cwd         = str(ROOT),
            env         = {**os.environ, "PYTHONPATH": str(ROOT)},
        )
        for line in process.stdout:
            output_lines.append(line.rstrip())
            placeholder.code("\n".join(output_lines[-30:]))

        process.wait()
        success = process.returncode == 0
        return success, "\n".join(output_lines)

    except Exception as e:
        err = f"Command failed: {e}"
        placeholder.error(err)
        return False, err


def python_cmd(module: str, *args) -> list:
    """Builds a python -m module command."""
    return [sys.executable, "-m", module] + list(args)


# ══════════════════════════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="gods-eye-banner">
    <div>👁️</div>
    <div>
        <h1>G.O.D.S E.Y.E</h1>
        <span>GENERATIVE OBSERVATION & DECISION SYSTEM FOR EQUITY INTELLIGENCE</span>
    </div>
    <div style="margin-left:auto; text-align:right">
        <span class="badge-paper">PAPER MODE</span>&nbsp;
        <span style="color:var(--muted); font-size:0.75rem; font-family:'Space Mono',monospace">
            NSE/BSE · Nifty 500
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="section-header">System Status</div>',
                unsafe_allow_html=True)

    # DB status
    db_status = db_ok()
    if db_status:
        st.markdown('<span class="badge-ok">● TimescaleDB</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-error">● TimescaleDB — OFFLINE</span>',
                    unsafe_allow_html=True)

    # Docker services
    for svc, port in [("Redis", 6380), ("Elasticsearch", 9200), ("Grafana", 3000)]:
        import socket
        try:
            s = socket.create_connection(("localhost", port), timeout=1)
            s.close()
            st.markdown(f'<span class="badge-ok">● {svc}</span>',
                        unsafe_allow_html=True)
        except Exception:
            st.markdown(f'<span class="badge-warn">● {svc} — check docker</span>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Quick Stats</div>',
                unsafe_allow_html=True)

    if db_status:
        try:
            row = query_df("""
                SELECT
                    (SELECT MAX(date) FROM daily_ohlcv)        AS latest_ohlcv,
                    (SELECT MAX(date) FROM features_fused)     AS latest_features,
                    (SELECT COUNT(*)  FROM trade_log
                     WHERE paper_mode=TRUE AND exit_time IS NULL) AS open_trades,
                    (SELECT COUNT(*)  FROM trade_log
                     WHERE paper_mode=TRUE)                    AS total_trades
            """)
            if not row.empty:
                r = row.iloc[0]
                st.metric("Latest Data",     str(r["latest_ohlcv"]) or "—")
                st.metric("Latest Features", str(r["latest_features"]) or "—")
                st.metric("Open Positions",  int(r["open_trades"] or 0))
                st.metric("Total Trades",    int(r["total_trades"] or 0))
        except Exception:
            pass

    st.markdown("---")
    if st.button("🔄 Refresh Dashboard"):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        '<p style="color:var(--muted);font-size:0.7rem;margin-top:2rem">'
        f'Last refresh: {datetime.now().strftime("%H:%M:%S")}</p>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📥 Data",
    "⚙️ Features",
    "🤖 Trading",
    "📈 Report",
    "🔧 System",
])


# ══════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-header">Portfolio Overview</div>',
                unsafe_allow_html=True)

    if not db_status:
        st.warning("Database offline. Start Docker: `docker-compose up -d`")
    else:
        # ── Key metrics ────────────────────────────────────────────────────
        try:
            metrics_df = query_df("""
                SELECT
                    COALESCE(SUM(realised_pnl), 0)                      AS total_pnl,
                    COUNT(*) FILTER (WHERE exit_time IS NOT NULL)        AS closed_trades,
                    COUNT(*) FILTER (WHERE exit_time IS NOT NULL
                                      AND realised_pnl > 0)             AS wins,
                    COUNT(*) FILTER (WHERE exit_time IS NULL)            AS open_trades,
                    COALESCE(AVG(realised_pnl)
                             FILTER (WHERE exit_time IS NOT NULL), 0)   AS avg_pnl,
                    COALESCE(MAX(realised_pnl)
                             FILTER (WHERE exit_time IS NOT NULL), 0)   AS best_trade,
                    COALESCE(MIN(realised_pnl)
                             FILTER (WHERE exit_time IS NOT NULL), 0)   AS worst_trade
                FROM trade_log
                WHERE paper_mode = TRUE;
            """)

            if not metrics_df.empty:
                r         = metrics_df.iloc[0]
                total_pnl = float(r["total_pnl"] or 0)
                closed    = int(r["closed_trades"] or 0)
                wins      = int(r["wins"] or 0)
                open_pos  = int(r["open_trades"] or 0)
                win_rate  = (wins / closed * 100) if closed > 0 else 0
                avg_pnl   = float(r["avg_pnl"] or 0)

                cols = st.columns(6)
                metrics = [
                    ("Total P&L",     f"₹{total_pnl:+,.0f}",
                     "positive" if total_pnl >= 0 else "negative"),
                    ("Closed Trades", str(closed),          "neutral"),
                    ("Win Rate",      f"{win_rate:.1f}%",
                     "positive" if win_rate >= 58 else "warning"),
                    ("Open Positions",str(open_pos),        "neutral"),
                    ("Avg P&L/Trade", f"₹{avg_pnl:+,.0f}",
                     "positive" if avg_pnl >= 0 else "negative"),
                    ("Best Trade",    f"₹{float(r['best_trade'] or 0):+,.0f}",
                     "positive"),
                ]
                for col, (label, value, cls) in zip(cols, metrics):
                    col.markdown(
                        f'<div class="metric-card">'
                        f'<div class="label">{label}</div>'
                        f'<div class="value {cls}">{value}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"Could not load metrics: {e}")

        st.markdown("---")

        col_left, col_right = st.columns([2, 1])

        # ── P&L curve ─────────────────────────────────────────────────────
        with col_left:
            st.markdown('<div class="section-header">Cumulative P&L</div>',
                        unsafe_allow_html=True)
            try:
                pnl_df = query_df("""
                    SELECT
                        exit_time::date              AS trade_date,
                        SUM(realised_pnl)            AS daily_pnl,
                        SUM(SUM(realised_pnl))
                            OVER (ORDER BY exit_time::date) AS cumulative_pnl
                    FROM trade_log
                    WHERE paper_mode = TRUE
                      AND exit_time IS NOT NULL
                    GROUP BY exit_time::date
                    ORDER BY exit_time::date;
                """)
                if not pnl_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x    = pnl_df["trade_date"],
                        y    = pnl_df["cumulative_pnl"],
                        mode = "lines+markers",
                        name = "Cumulative P&L",
                        line = dict(color="#C9A84C", width=2),
                        fill = "tozeroy",
                        fillcolor = "rgba(201,168,76,0.08)",
                    ))
                    fig.update_layout(
                        paper_bgcolor = "#111827",
                        plot_bgcolor  = "#111827",
                        font          = dict(color="#E8EDF5", family="Space Mono"),
                        xaxis         = dict(gridcolor="#1E2D4A", showgrid=True),
                        yaxis         = dict(gridcolor="#1E2D4A", showgrid=True,
                                             tickprefix="₹"),
                        margin        = dict(l=0, r=0, t=10, b=0),
                        height        = 280,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No closed trades yet. Run the signal engine to start.")
            except Exception as e:
                st.error(f"P&L chart error: {e}")

        # ── Open positions ─────────────────────────────────────────────────
        with col_right:
            st.markdown('<div class="section-header">Open Positions</div>',
                        unsafe_allow_html=True)
            try:
                pos_df = query_df("""
                    SELECT symbol, entry_price, quantity,
                           tp_price, sl_price, hold_days,
                           confidence_score
                    FROM trade_log
                    WHERE paper_mode = TRUE AND exit_time IS NULL
                    ORDER BY entry_time DESC;
                """)
                if not pos_df.empty:
                    for _, row in pos_df.iterrows():
                        tp_pct = (row["tp_price"] - row["entry_price"]) / row["entry_price"] * 100
                        sl_pct = (row["entry_price"] - row["sl_price"])  / row["entry_price"] * 100
                        st.markdown(
                            f'<div class="metric-card" style="text-align:left;margin-bottom:8px">'
                            f'<div style="display:flex;justify-content:space-between">'
                            f'<b style="color:#C9A84C">{row["symbol"]}</b>'
                            f'<span class="badge-paper">x{int(row["quantity"])}</span>'
                            f'</div>'
                            f'<div style="color:var(--muted);font-size:0.75rem;margin-top:4px">'
                            f'Entry ₹{row["entry_price"]:.2f} · Day {int(row["hold_days"])}'
                            f'</div>'
                            f'<div style="font-size:0.75rem;margin-top:4px">'
                            f'<span style="color:#4ADE80">TP +{tp_pct:.1f}%</span>'
                            f' &nbsp; '
                            f'<span style="color:#F87171">SL -{sl_pct:.1f}%</span>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No open positions")
            except Exception as e:
                st.error(f"Positions error: {e}")

        # ── Recent trades ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">Recent Closed Trades</div>',
                    unsafe_allow_html=True)
        try:
            recent_df = query_df("""
                SELECT symbol, mode, entry_price, exit_price,
                       quantity, realised_pnl, exit_reason,
                       hold_days, confidence_score,
                       entry_time::date AS entry_date,
                       exit_time::date  AS exit_date
                FROM trade_log
                WHERE paper_mode = TRUE AND exit_time IS NOT NULL
                ORDER BY exit_time DESC
                LIMIT 20;
            """)
            if not recent_df.empty:
                recent_df["P&L"] = recent_df["realised_pnl"].apply(
                    lambda x: f"₹{float(x):+,.0f}"
                )
                recent_df["conf"] = recent_df["confidence_score"].apply(
                    lambda x: f"{float(x):.2f}" if x else "—"
                )
                display_cols = ["symbol", "mode", "entry_date", "exit_date",
                                "entry_price", "exit_price", "quantity",
                                "P&L", "exit_reason", "hold_days", "conf"]
                st.dataframe(
                    recent_df[display_cols].rename(columns={
                        "entry_price": "Entry ₹",
                        "exit_price" : "Exit ₹",
                        "hold_days"  : "Days",
                        "exit_reason": "Reason",
                        "conf"       : "Conf",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No closed trades yet.")
        except Exception as e:
            st.error(f"Recent trades error: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 2 — DATA
# ══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">Market Data Management</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Download Data")

        mode = st.selectbox(
            "Mode",
            ["daily", "backfill", "single"],
            help="daily=today, backfill=date range, single=one date"
        )

        if mode == "backfill":
            start_date = st.date_input(
                "Start Date",
                value=date.today() - timedelta(days=30)
            )
        elif mode == "single":
            single_date = st.date_input("Date", value=date.today() - timedelta(days=1))

        data_types = st.multiselect(
            "Data Sources",
            ["OHLCV (Bhavcopy)", "FII/DII"],
            default=["OHLCV (Bhavcopy)", "FII/DII"]
        )

        if st.button("▶ Download Now", key="btn_download"):
            output_placeholder = st.empty()
            progress = st.progress(0)

            if "OHLCV (Bhavcopy)" in data_types:
                st.info("Downloading OHLCV from NSE...")
                cmd_args = ["--mode", mode]
                if mode == "backfill":
                    cmd_args += ["--start", str(start_date)]
                elif mode == "single":
                    cmd_args += ["--date", str(single_date)]

                success, out = run_command(
                    python_cmd("data.ingestion.nse_bhavcopy", *cmd_args),
                    output_placeholder
                )
                if success:
                    st.success("✓ OHLCV downloaded")
                else:
                    st.error("✗ OHLCV download failed")
                progress.progress(50)

            if "FII/DII" in data_types:
                st.info("Downloading FII/DII from NSE...")
                fii_mode = "final" if mode == "daily" else "backfill"
                fii_args = ["--mode", fii_mode]
                if mode == "backfill":
                    fii_args += ["--start", str(start_date)]

                success, out = run_command(
                    python_cmd("data.ingestion.fii_dii_scraper", *fii_args),
                    output_placeholder
                )
                if success:
                    st.success("✓ FII/DII downloaded")
                else:
                    st.warning("⚠ FII/DII download failed (may be holiday)")
                progress.progress(100)

    with col2:
        st.subheader("📊 Database Status")
        if db_status:
            try:
                status_df = query_df("""
                    SELECT
                        'daily_ohlcv'    AS table_name,
                        COUNT(*)         AS rows,
                        MIN(date)::text  AS from_date,
                        MAX(date)::text  AS to_date,
                        COUNT(DISTINCT symbol) AS symbols
                    FROM daily_ohlcv
                    UNION ALL
                    SELECT
                        'features_fused',
                        COUNT(*),
                        MIN(date)::text,
                        MAX(date)::text,
                        COUNT(DISTINCT symbol)
                    FROM features_fused
                    UNION ALL
                    SELECT
                        'fii_dii_flow',
                        COUNT(*),
                        MIN(date)::text,
                        MAX(date)::text,
                        1
                    FROM fii_dii_flow
                    UNION ALL
                    SELECT
                        'trade_log',
                        COUNT(*),
                        MIN(entry_time)::date::text,
                        MAX(entry_time)::date::text,
                        COUNT(DISTINCT symbol)
                    FROM trade_log;
                """)
                if not status_df.empty:
                    st.dataframe(
                        status_df.rename(columns={
                            "table_name": "Table",
                            "rows"      : "Rows",
                            "from_date" : "From",
                            "to_date"   : "To",
                            "symbols"   : "Symbols",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as e:
                st.error(f"Status query failed: {e}")

            # Missing dates checker
            st.markdown('<div class="section-header">Gap Detector</div>',
                        unsafe_allow_html=True)
            try:
                gap_df = query_df("""
                    SELECT date::text AS missing_date
                    FROM generate_series(
                        (SELECT MIN(date) FROM daily_ohlcv),
                        CURRENT_DATE - 1,
                        '1 day'::interval
                    ) AS gs(date)
                    WHERE EXTRACT(DOW FROM date) NOT IN (0,6)
                      AND date::date NOT IN
                          (SELECT DISTINCT date FROM daily_ohlcv)
                    ORDER BY date DESC
                    LIMIT 15;
                """)
                if not gap_df.empty:
                    st.warning(f"Found {len(gap_df)} missing trading days:")
                    st.dataframe(gap_df, use_container_width=True, hide_index=True)
                else:
                    st.success("✓ No data gaps found")
            except Exception as e:
                st.info(f"Gap check skipped: {e}")
        else:
            st.error("Database offline")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 3 — FEATURES
# ══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">Feature Engineering Pipeline</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("⚙️ Run Feature Pipeline")

        feat_mode = st.selectbox(
            "Mode",
            ["daily", "all"],
            help="daily=today only, all=full date range"
        )

        if feat_mode == "all":
            feat_start = st.date_input(
                "From Date",
                value=date.today() - timedelta(days=30),
                key="feat_start"
            )
        else:
            feat_start = date.today() - timedelta(days=1)

        pillars = st.multiselect(
            "Pillars to Run",
            ["trend", "msi", "fii_dii", "sentiment",
             "volatility", "correlation", "fusion"],
            default=["trend", "msi", "fii_dii",
                     "volatility", "correlation", "fusion"]
        )

        run_embeddings = st.checkbox("Also recompute embeddings", value=True)

        if st.button("▶ Run Feature Pipeline", key="btn_features"):
            output_placeholder = st.empty()
            total = len(pillars) + (1 if run_embeddings else 0)
            progress = st.progress(0)

            for i, pillar in enumerate(pillars):
                st.info(f"Running {pillar}...")
                args = ["--mode", feat_mode]
                if feat_mode == "all":
                    args += ["--start", str(feat_start)]

                success, _ = run_command(
                    python_cmd(f"features.{pillar}", *args),
                    output_placeholder
                )
                emoji = "✓" if success else "✗"
                if success:
                    st.success(f"{emoji} {pillar} complete")
                else:
                    st.error(f"{emoji} {pillar} failed")
                progress.progress((i + 1) / total)

            if run_embeddings:
                st.info("Recomputing backbone embeddings...")
                success, _ = run_command(
                    python_cmd("training.precompute_embeddings", "--incremental"),
                    output_placeholder
                )
                if success:
                    st.success("✓ Embeddings updated")
                else:
                    st.error("✗ Embeddings failed")
                progress.progress(1.0)

    with col2:
        st.subheader("📊 Feature Coverage")
        if db_status:
            try:
                cov_df = query_df("""
                    SELECT
                        'trend'       AS pillar,
                        COUNT(*)      AS rows,
                        MAX(date)::text AS latest
                    FROM features_trend
                    UNION ALL
                    SELECT 'msi', COUNT(*), MAX(date)::text
                    FROM features_msi
                    UNION ALL
                    SELECT 'fii_dii', COUNT(*), MAX(date)::text
                    FROM features_fii_dii
                    UNION ALL
                    SELECT 'sentiment', COUNT(*), MAX(date)::text
                    FROM features_sentiment
                    UNION ALL
                    SELECT 'volatility', COUNT(*), MAX(date)::text
                    FROM features_volatility
                    UNION ALL
                    SELECT 'correlation', COUNT(*), MAX(date)::text
                    FROM features_correlation
                    UNION ALL
                    SELECT 'fused', COUNT(*), MAX(date)::text
                    FROM features_fused;
                """)
                if not cov_df.empty:
                    today_str = str(date.today())
                    cov_df["status"] = cov_df["latest"].apply(
                        lambda d: "✅ Current" if d and d >= str(date.today() - timedelta(days=2))
                        else "⚠️ Stale"
                    )
                    st.dataframe(
                        cov_df.rename(columns={
                            "pillar": "Pillar",
                            "rows"  : "Rows",
                            "latest": "Latest Date",
                            "status": "Status",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as e:
                st.warning(f"Feature tables not found yet: {e}")

        # Drift check
        st.markdown('<div class="section-header">Feature Drift Check</div>',
                    unsafe_allow_html=True)
        if st.button("🔍 Run Drift Check", key="btn_drift"):
            with st.spinner("Computing KL divergence..."):
                try:
                    from monitoring.drift_detector import DriftDetector
                    import numpy as np

                    detector = DriftDetector()
                    loaded   = detector.load_reference()

                    if loaded and db_status:
                        live_df = query_df(f"""
                            SELECT f00_trend_score,f01_ema_ribbon_gap,
                                   f02_adx_normalized,f03_supertrend_dir,
                                   f04_price_vs_ema200,f05_swing_structure,
                                   f06_msi_signal,f07_vrsi_normalized,
                                   f08_mfi_normalized,f09_msi_divergence,
                                   f10_mds_continuous,f11_fii_norm,
                                   f12_dii_norm,f13_sentiment_score,
                                   f14_sentiment_momentum,f15_event_flag,
                                   f16_market_fear_greed_n,f17_volatility_score,
                                   f18_atr_pct_normalized,f19_vol_regime_code_n,
                                   f20_hv_percentile_n,f21_correlation_score,
                                   f22_sector_divergence_n,f23_lead_lag_score,
                                   f24_peer_corr_mean,f25_delivery_mom_n,
                                   f26_swing_tp_normalized,f27_swing_sl_normalized
                            FROM features_fused
                            WHERE date >= '{date.today() - timedelta(days=30)}'
                            LIMIT 500;
                        """)
                        if not live_df.empty:
                            arr = np.nan_to_num(
                                live_df.values.astype(np.float32)
                            )
                            detector.add_live_batch(arr)
                            report = detector.compute_drift()
                            if report:
                                kl = report.aggregate_kl
                                level = report.drift_level
                                color = (
                                    "green"  if kl < 0.05 else
                                    "yellow" if kl < 0.10 else
                                    "orange" if kl < 0.15 else
                                    "red"
                                )
                                st.markdown(
                                    f'<div class="metric-card">'
                                    f'<div class="label">KL Divergence Score</div>'
                                    f'<div class="value" style="color:{color}">'
                                    f'{kl:.4f}</div>'
                                    f'<div style="color:var(--muted);font-size:0.75rem">'
                                    f'{level.upper()} · {report.recommendation}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("Not enough live feature data for drift check.")
                    else:
                        st.info("Drift check requires training reference. "
                                "Run feature pipeline first.")
                except Exception as e:
                    st.error(f"Drift check failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 4 — TRADING
# ══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">Paper Trading Controls</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🤖 Signal Engine")

        st.markdown("""
        <div class="metric-card" style="text-align:left;margin-bottom:16px">
        <div class="label">Mode</div>
        <div style="margin-top:4px">
            <span class="badge-paper">PAPER TRADING</span>&nbsp;
            <span style="color:var(--muted);font-size:0.75rem">
                No real orders · Simulated fills · Full risk tracking
            </span>
        </div>
        </div>
        """, unsafe_allow_html=True)

        signal_mode = st.radio(
            "Run Mode",
            ["once", "continuous"],
            horizontal=True,
            help="once=single run, continuous=runs every bar"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶ Generate Signals", key="btn_signals", type="primary"):
                output_placeholder = st.empty()
                with st.spinner("Running signal engine..."):
                    success, out = run_command(
                        python_cmd("execution.signal_engine",
                                   "--mode", "paper",
                                   "--" + signal_mode),
                        output_placeholder
                    )
                if success:
                    st.success("✓ Signals generated. Check Overview tab.")
                    st.cache_data.clear()
                else:
                    st.error("✗ Signal engine failed. Check logs.")

        with col_b:
            if st.button("🔴 Close All Positions", key="btn_close_all"):
                st.warning("This will close all open paper positions.")
                if st.button("⚠️ Confirm Close All", key="btn_confirm_close"):
                    try:
                        success = run_query("""
                            UPDATE trade_log
                            SET exit_time    = NOW(),
                                exit_price   = entry_price,
                                exit_reason  = 'MANUAL_CLOSE',
                                realised_pnl = 0
                            WHERE paper_mode = TRUE
                              AND exit_time IS NULL;
                        """)
                        if success:
                            st.success("All positions closed.")
                            st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Close failed: {e}")

        st.markdown('<div class="section-header">Manual Signal Override</div>',
                    unsafe_allow_html=True)
        st.info("Use this to manually record a paper trade for testing.")

        with st.form("manual_trade"):
            m_symbol = st.text_input("Symbol (e.g. RELIANCE)")
            m_side   = st.selectbox("Side", ["BUY", "SELL"])
            m_price  = st.number_input("Entry Price ₹", min_value=0.01, value=100.0)
            m_qty    = st.number_input("Quantity", min_value=1, value=10)
            m_tp     = st.number_input("TP Price ₹", min_value=0.01,
                                        value=round(100.0 * 1.04, 2))
            m_sl     = st.number_input("SL Price ₹", min_value=0.01,
                                        value=round(100.0 * 0.985, 2))
            m_conf   = st.slider("Confidence Score", 0.55, 1.0, 0.75, 0.01)

            if st.form_submit_button("📝 Record Manual Trade"):
                if m_symbol and m_price > 0 and m_qty > 0:
                    import uuid
                    success = run_query("""
                        INSERT INTO trade_log (
                            signal_id, order_id, symbol, mode, side,
                            entry_price, quantity, position_value,
                            tp_price, sl_price, entry_time,
                            confidence_score, paper_mode, hold_days
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),%s,TRUE,0)
                    """, (
                        f"MANUAL-{uuid.uuid4().hex[:8]}",
                        f"MANUAL-ORDER",
                        m_symbol.upper(), "swing", m_side,
                        m_price, m_qty, m_price * m_qty,
                        m_tp, m_sl, m_conf,
                    ))
                    if success:
                        st.success(f"✓ Manual {m_side} for {m_symbol} recorded.")
                        st.cache_data.clear()
                else:
                    st.warning("Fill in all fields.")

    with col2:
        st.subheader("📋 Current Positions Detail")
        if db_status:
            try:
                pos_df = query_df("""
                    SELECT
                        symbol,
                        side,
                        entry_price,
                        quantity,
                        position_value,
                        tp_price,
                        sl_price,
                        hold_days,
                        confidence_score,
                        entry_time::date AS entry_date,
                        ROUND(((tp_price - entry_price) / entry_price * 100)::numeric, 2)
                            AS tp_pct,
                        ROUND(((entry_price - sl_price) / entry_price * 100)::numeric, 2)
                            AS sl_pct
                    FROM trade_log
                    WHERE paper_mode = TRUE AND exit_time IS NULL
                    ORDER BY entry_time DESC;
                """)
                if not pos_df.empty:
                    st.dataframe(
                        pos_df.rename(columns={
                            "entry_price"      : "Entry ₹",
                            "position_value"   : "Value ₹",
                            "tp_price"         : "TP ₹",
                            "sl_price"         : "SL ₹",
                            "hold_days"        : "Days",
                            "confidence_score" : "Conf",
                            "entry_date"       : "Date",
                            "tp_pct"           : "TP%",
                            "sl_pct"           : "SL%",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No open positions")
            except Exception as e:
                st.error(f"Positions error: {e}")

        # RC Constitution status
        st.markdown('<div class="section-header">Risk Constitution Status</div>',
                    unsafe_allow_html=True)
        try:
            rc_df = query_df("""
                SELECT rule, symbol, details,
                       triggered_at::date AS date
                FROM rc_trigger_log
                ORDER BY triggered_at DESC
                LIMIT 10;
            """)
            if not rc_df.empty:
                st.dataframe(rc_df, use_container_width=True, hide_index=True)
            else:
                st.success("✓ No RC triggers in log")
        except Exception:
            st.info("RC trigger log not yet created.")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 5 — REPORT
# ══════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-header">Performance Report</div>',
                unsafe_allow_html=True)

    # Date range selector
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        report_start = st.date_input(
            "From",
            value=date.today() - timedelta(days=42),
            key="report_start"
        )
    with col2:
        report_end = st.date_input(
            "To",
            value=date.today(),
            key="report_end"
        )
    with col3:
        preset = st.selectbox(
            "Quick Range",
            ["Custom", "Last 7 days", "Last 30 days",
             "Last 6 weeks", "All time"],
            key="preset_range"
        )
        if preset == "Last 7 days":
            report_start = date.today() - timedelta(days=7)
            report_end   = date.today()
        elif preset == "Last 30 days":
            report_start = date.today() - timedelta(days=30)
            report_end   = date.today()
        elif preset == "Last 6 weeks":
            report_start = date.today() - timedelta(weeks=6)
            report_end   = date.today()
        elif preset == "All time":
            report_start = date(2019, 1, 1)
            report_end   = date.today()

    if db_status:
        # ── Summary stats for date range ──────────────────────────────────
        try:
            rpt_df = query_df("""
                SELECT
                    COUNT(*)                                     AS trades,
                    COUNT(*) FILTER (WHERE realised_pnl > 0)    AS wins,
                    COUNT(*) FILTER (WHERE realised_pnl <= 0)   AS losses,
                    COALESCE(SUM(realised_pnl), 0)              AS total_pnl,
                    COALESCE(AVG(realised_pnl), 0)              AS avg_pnl,
                    COALESCE(MAX(realised_pnl), 0)              AS best,
                    COALESCE(MIN(realised_pnl), 0)              AS worst,
                    COALESCE(AVG(hold_days), 0)                 AS avg_hold,
                    COALESCE(
                        SUM(realised_pnl) FILTER (WHERE realised_pnl > 0) /
                        NULLIF(ABS(SUM(realised_pnl)
                               FILTER (WHERE realised_pnl < 0)), 0),
                        0
                    )                                            AS profit_factor
                FROM trade_log
                WHERE paper_mode = TRUE
                  AND exit_time IS NOT NULL
                  AND exit_time::date BETWEEN %s AND %s;
            """, (str(report_start), str(report_end)))

            if not rpt_df.empty and int(rpt_df.iloc[0]["trades"] or 0) > 0:
                r        = rpt_df.iloc[0]
                trades   = int(r["trades"])
                wins     = int(r["wins"])
                losses   = int(r["losses"])
                total    = float(r["total_pnl"])
                win_rate = wins / trades * 100 if trades > 0 else 0
                pf       = float(r["profit_factor"] or 0)

                # Metric row
                m_cols = st.columns(7)
                mdata  = [
                    ("Trades",      str(trades),           "neutral"),
                    ("Win Rate",    f"{win_rate:.1f}%",
                     "positive" if win_rate >= 58 else "warning"),
                    ("Total P&L",   f"₹{total:+,.0f}",
                     "positive" if total >= 0 else "negative"),
                    ("Avg P&L",     f"₹{float(r['avg_pnl']):+,.0f}",
                     "positive" if float(r["avg_pnl"]) >= 0 else "negative"),
                    ("Best Trade",  f"₹{float(r['best']):+,.0f}", "positive"),
                    ("Worst Trade", f"₹{float(r['worst']):+,.0f}", "negative"),
                    ("Profit Factor",f"{pf:.2f}",
                     "positive" if pf >= 2.0 else "warning"),
                ]
                for col, (label, val, cls) in zip(m_cols, mdata):
                    col.markdown(
                        f'<div class="metric-card">'
                        f'<div class="label">{label}</div>'
                        f'<div class="value {cls}">{val}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("---")
                chart_col1, chart_col2 = st.columns(2)

                # Daily P&L bar chart
                with chart_col1:
                    st.markdown('<div class="section-header">Daily P&L</div>',
                                unsafe_allow_html=True)
                    daily_df = query_df("""
                        SELECT
                            exit_time::date AS trade_date,
                            SUM(realised_pnl) AS daily_pnl
                        FROM trade_log
                        WHERE paper_mode = TRUE
                          AND exit_time IS NOT NULL
                          AND exit_time::date BETWEEN %s AND %s
                        GROUP BY exit_time::date
                        ORDER BY exit_time::date;
                    """, (str(report_start), str(report_end)))

                    if not daily_df.empty:
                        colors = [
                            "#4ADE80" if v >= 0 else "#F87171"
                            for v in daily_df["daily_pnl"]
                        ]
                        fig = go.Figure(go.Bar(
                            x             = daily_df["trade_date"],
                            y             = daily_df["daily_pnl"],
                            marker_color  = colors,
                            name          = "Daily P&L",
                        ))
                        fig.update_layout(
                            paper_bgcolor = "#111827",
                            plot_bgcolor  = "#111827",
                            font          = dict(color="#E8EDF5",
                                                 family="Space Mono"),
                            xaxis = dict(gridcolor="#1E2D4A"),
                            yaxis = dict(gridcolor="#1E2D4A", tickprefix="₹"),
                            margin= dict(l=0, r=0, t=10, b=0),
                            height= 250,
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Win/Loss by symbol
                with chart_col2:
                    st.markdown('<div class="section-header">P&L by Symbol</div>',
                                unsafe_allow_html=True)
                    sym_df = query_df("""
                        SELECT
                            symbol,
                            SUM(realised_pnl)   AS total_pnl,
                            COUNT(*)            AS trades,
                            COUNT(*) FILTER (WHERE realised_pnl > 0) AS wins
                        FROM trade_log
                        WHERE paper_mode = TRUE
                          AND exit_time IS NOT NULL
                          AND exit_time::date BETWEEN %s AND %s
                        GROUP BY symbol
                        ORDER BY total_pnl DESC;
                    """, (str(report_start), str(report_end)))

                    if not sym_df.empty:
                        colors = [
                            "#4ADE80" if v >= 0 else "#F87171"
                            for v in sym_df["total_pnl"]
                        ]
                        fig = go.Figure(go.Bar(
                            x            = sym_df["symbol"],
                            y            = sym_df["total_pnl"],
                            marker_color = colors,
                            text         = sym_df["total_pnl"].apply(
                                lambda v: f"₹{float(v):+,.0f}"
                            ),
                            textposition = "outside",
                        ))
                        fig.update_layout(
                            paper_bgcolor = "#111827",
                            plot_bgcolor  = "#111827",
                            font          = dict(color="#E8EDF5",
                                                 family="Space Mono"),
                            xaxis = dict(gridcolor="#1E2D4A"),
                            yaxis = dict(gridcolor="#1E2D4A", tickprefix="₹"),
                            margin= dict(l=0, r=0, t=30, b=0),
                            height= 250,
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Exit reason breakdown
                st.markdown('<div class="section-header">Exit Reasons</div>',
                            unsafe_allow_html=True)
                reason_df = query_df("""
                    SELECT
                        exit_reason,
                        COUNT(*)          AS count,
                        SUM(realised_pnl) AS total_pnl,
                        AVG(realised_pnl) AS avg_pnl
                    FROM trade_log
                    WHERE paper_mode = TRUE
                      AND exit_time IS NOT NULL
                      AND exit_time::date BETWEEN %s AND %s
                    GROUP BY exit_reason
                    ORDER BY count DESC;
                """, (str(report_start), str(report_end)))

                if not reason_df.empty:
                    col_r1, col_r2 = st.columns([1, 2])
                    with col_r1:
                        st.dataframe(
                            reason_df.rename(columns={
                                "exit_reason": "Reason",
                                "count"      : "Count",
                                "total_pnl"  : "Total P&L",
                                "avg_pnl"    : "Avg P&L",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
                    with col_r2:
                        fig = px.pie(
                            reason_df,
                            values = "count",
                            names  = "exit_reason",
                            color_discrete_sequence = [
                                "#C9A84C","#4ADE80","#F87171",
                                "#60A5FA","#A78BFA","#34D399"
                            ],
                        )
                        fig.update_layout(
                            paper_bgcolor = "#111827",
                            font          = dict(color="#E8EDF5",
                                                 family="Space Mono"),
                            margin        = dict(l=0, r=0, t=0, b=0),
                            height        = 200,
                            legend        = dict(font=dict(size=10)),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Full trade table
                st.markdown('<div class="section-header">All Trades</div>',
                            unsafe_allow_html=True)
                all_trades_df = query_df("""
                    SELECT
                        symbol, mode, side,
                        entry_price, exit_price, quantity,
                        ROUND(realised_pnl::numeric, 2)        AS pnl,
                        ROUND(confidence_score::numeric, 2)    AS conf,
                        exit_reason, hold_days,
                        entry_time::date AS entry_date,
                        exit_time::date  AS exit_date
                    FROM trade_log
                    WHERE paper_mode = TRUE
                      AND exit_time IS NOT NULL
                      AND exit_time::date BETWEEN %s AND %s
                    ORDER BY exit_time DESC;
                """, (str(report_start), str(report_end)))

                if not all_trades_df.empty:
                    # Export button
                    csv = all_trades_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label    = "⬇ Download CSV",
                        data     = csv,
                        file_name= f"paper_trading_{report_start}_{report_end}.csv",
                        mime     = "text/csv",
                    )
                    st.dataframe(
                        all_trades_df,
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info(f"No closed trades between "
                        f"{report_start} and {report_end}.")
        except Exception as e:
            st.error(f"Report error: {e}")
            st.code(traceback.format_exc())
    else:
        st.error("Database offline.")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 6 — SYSTEM
# ══════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown('<div class="section-header">System Health & Controls</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🐳 Infrastructure")

        # Docker status
        try:
            result = subprocess.run(
                ["docker", "ps", "--format",
                 "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    containers = []
                    for line in lines[1:]:
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            name   = parts[0]
                            status = parts[1]
                            ok     = "Up" in status
                            badge  = "badge-ok" if ok else "badge-error"
                            label  = "● " + name
                            containers.append(
                                f'<span class="{badge}">{label}</span>'
                            )
                    st.markdown(
                        " &nbsp; ".join(containers),
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("No Docker containers running. "
                               "Run: `docker-compose up -d`")
            else:
                st.error("Docker not accessible")
        except Exception as e:
            st.warning(f"Docker check failed: {e}")

        # Docker controls
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🟢 Start Docker", key="docker_start"):
                with st.spinner("Starting containers..."):
                    result = subprocess.run(
                        ["docker-compose", "up", "-d"],
                        capture_output=True, text=True, cwd=str(ROOT)
                    )
                    if result.returncode == 0:
                        st.success("✓ Docker started")
                    else:
                        st.error(f"Failed: {result.stderr[:200]}")

        with c2:
            if st.button("🔴 Stop Docker", key="docker_stop"):
                result = subprocess.run(
                    ["docker-compose", "down"],
                    capture_output=True, text=True, cwd=str(ROOT)
                )
                if result.returncode == 0:
                    st.success("✓ Docker stopped")

        # Model checkpoints
        st.markdown('<div class="section-header">Model Checkpoints</div>',
                    unsafe_allow_html=True)
        ckpt_dir = ROOT / "checkpoints"
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("*.pt"))
            if ckpts:
                ckpt_data = []
                for ckpt in sorted(ckpts):
                    stat = ckpt.stat()
                    ckpt_data.append({
                        "File"    : ckpt.name,
                        "Size MB" : round(stat.st_size / 1024 / 1024, 1),
                        "Modified": datetime.fromtimestamp(stat.st_mtime)
                                          .strftime("%Y-%m-%d %H:%M"),
                    })
                st.dataframe(
                    pd.DataFrame(ckpt_data),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No checkpoints found.")
        else:
            st.info("Checkpoints directory not found.")

    with col2:
        st.subheader("📋 Recent Logs")

        log_dir = ROOT / "logs"
        if log_dir.exists():
            log_files = sorted(log_dir.rglob("*.log"), key=lambda f: f.stat().st_mtime,
                               reverse=True)[:5]
            if log_files:
                selected_log = st.selectbox(
                    "Log File",
                    [f.name for f in log_files],
                    key="log_selector"
                )
                log_path = next(
                    (f for f in log_files if f.name == selected_log), None
                )
                if log_path and st.button("📖 Load Log", key="load_log"):
                    try:
                        lines = log_path.read_text(
                            encoding="utf-8", errors="ignore"
                        ).split("\n")
                        last_100 = "\n".join(lines[-100:])
                        st.code(last_100, language="text")
                    except Exception as e:
                        st.error(f"Could not read log: {e}")
            else:
                st.info("No log files found.")
        else:
            st.info("Logs directory not found.")

        # Nightly retrain controls
        st.markdown('<div class="section-header">Nightly Retrain (Manual)</div>',
                    unsafe_allow_html=True)
        st.info("Use this to manually trigger the nightly pipeline "
                "without waiting for the scheduled run.")

        retrain_steps = st.multiselect(
            "Steps to Run",
            ["Download Data", "Run Features",
             "Update Embeddings", "Fine-tune Model"],
            default=["Download Data", "Run Features", "Update Embeddings"],
            key="retrain_steps"
        )

        if st.button("▶ Run Selected Steps", key="btn_retrain"):
            output_placeholder = st.empty()
            progress = st.progress(0)
            n = len(retrain_steps)

            for i, step in enumerate(retrain_steps):
                if step == "Download Data":
                    st.info("Downloading data...")
                    run_command(
                        python_cmd("data.ingestion.nse_bhavcopy",
                                   "--mode", "daily"),
                        output_placeholder
                    )
                    run_command(
                        python_cmd("data.ingestion.fii_dii_scraper",
                                   "--mode", "final"),
                        output_placeholder
                    )

                elif step == "Run Features":
                    st.info("Running feature pipeline...")
                    run_command(
                        python_cmd("features.fusion", "--mode", "daily"),
                        output_placeholder
                    )

                elif step == "Update Embeddings":
                    st.info("Updating embeddings...")
                    run_command(
                        python_cmd("training.precompute_embeddings",
                                   "--incremental"),
                        output_placeholder
                    )

                elif step == "Fine-tune Model":
                    st.info("Fine-tuning backbone (2 epochs)...")
                    run_command(
                        python_cmd("training.pretrain_backbone",
                                   "--mode", "finetune",
                                   "--epochs", "2"),
                        output_placeholder
                    )

                progress.progress((i + 1) / n)

            st.success("✓ Selected steps completed.")
            st.cache_data.clear()

        # Gate criteria check
        st.markdown('<div class="section-header">Phase 4 Gate Check</div>',
                    unsafe_allow_html=True)
        if st.button("🎯 Check Gate Criteria", key="btn_gates"):
            if db_status:
                try:
                    gate_df = query_df("""
                        SELECT
                            COUNT(*) FILTER (WHERE exit_time IS NOT NULL)
                                AS closed_trades,
                            COUNT(*) FILTER (WHERE exit_time IS NOT NULL
                                              AND realised_pnl > 0)
                                AS wins,
                            COALESCE(SUM(realised_pnl)
                                     FILTER (WHERE exit_time IS NOT NULL), 0)
                                AS total_pnl,
                            MIN(entry_time)::date AS first_trade_date
                        FROM trade_log
                        WHERE paper_mode = TRUE;
                    """)
                    if not gate_df.empty:
                        r      = gate_df.iloc[0]
                        closed = int(r["closed_trades"] or 0)
                        wins   = int(r["wins"] or 0)
                        wr     = wins / closed * 100 if closed > 0 else 0
                        first  = r["first_trade_date"]
                        weeks  = ((date.today() - first).days // 7
                                  if first else 0)

                        gates = [
                            ("6-week track record",
                             f"{weeks} weeks",
                             weeks >= 6),
                            ("Win rate 45–75%",
                             f"{wr:.1f}%",
                             45 <= wr <= 75 or closed == 0),
                            ("Min 20 closed trades",
                             str(closed),
                             closed >= 20),
                        ]
                        for gate_name, val, passed in gates:
                            icon = "✅" if passed else "❌"
                            st.markdown(
                                f"{icon} **{gate_name}**: {val}"
                            )
                except Exception as e:
                    st.error(f"Gate check failed: {e}")
            else:
                st.error("Database offline.")