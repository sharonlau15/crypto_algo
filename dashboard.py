"""
dashboard.py
============
Interactive Streamlit dashboard — two independent sections:
  📊 Backtest Analysis  — historical walk-forward strategy performance
  🟢 Live Trading       — real-time Binance Testnet positions and NAV

Run with: streamlit run dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timezone
from config.settings import UNIVERSE

RESULT_DIR = Path("results")
STARTING_CAPITAL = 10_000  # must match settings.PORTFOLIO_USDT

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 10px; color: white; text-align: center;
    }
    .metric-title { font-size: 14px; opacity: 0.9; }
    .metric-value { font-size: 28px; font-weight: bold; }
    .live-badge {
        display: inline-block; background: #22c55e; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 13px; font-weight: bold;
    }
    .testnet-badge {
        display: inline-block; background: #f59e0b; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 13px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_backtest_data():
    data = {
        "metrics": {},
        "returns": None,
        "cumulative_returns": None,
        "monthly_seasonality": None,
        "regime_performance": None,
        "attribution": None,
        "pnl_summary": None,
    }
    if (f := RESULT_DIR / "strategy_metrics.json").exists():
        with open(f) as fh:
            data["metrics"] = json.load(fh)
    if (f := RESULT_DIR / "portfolio_returns.csv").exists():
        data["returns"] = pd.read_csv(f, index_col=0, parse_dates=True)
    if (f := RESULT_DIR / "cumulative_returns.csv").exists():
        data["cumulative_returns"] = pd.read_csv(f, index_col=0, parse_dates=True)
    if (f := RESULT_DIR / "monthly_seasonality.csv").exists():
        data["monthly_seasonality"] = pd.read_csv(f, index_col=0)
    if (f := RESULT_DIR / "regime_performance.csv").exists():
        data["regime_performance"] = pd.read_csv(f, index_col=0)
    if (f := RESULT_DIR / "attribution_report.csv").exists():
        data["attribution"] = pd.read_csv(f, index_col=0)
    if (f := RESULT_DIR / "pnl_summary.csv").exists():
        data["pnl_summary"] = pd.read_csv(f, index_col=0)
    return data


def _load_state_file():
    """Read live_state.json, recovering gracefully from mid-write corruption."""
    state_file = RESULT_DIR / "live_state.json"
    if not state_file.exists():
        return None
    try:
        with open(state_file) as f:
            content = f.read()
        # json.load raises JSONDecodeError("Extra data") when two JSON objects
        # are concatenated (atomic-write failure). raw_decode recovers the first.
        return json.JSONDecoder().raw_decode(content)[0]
    except Exception:
        return None


@st.cache_data(ttl=30)  # auto-refresh every 30 s while dashboard is open
def load_live_data():
    return _load_state_file()


@st.cache_data(ttl=30)
def load_strategy_monitor_data():
    state = _load_state_file()
    if state is None:
        return None
    return {
        "hypothetical":      state.get("hypothetical", {}),
        "active_strategies": state.get("active_strategies", []),
    }


@st.cache_data(ttl=30)
def fetch_binance_live_balances() -> dict | None:
    """
    Pull real token quantities and live prices directly from Binance.
    Returns None if API keys are unavailable or the request fails.

    Token quantities come from the actual account balances (exact).
    Prices come from the live ticker (real-time, not cached OHLCV).
    We deliberately ignore the Binance USDT balance — testnet accounts
    carry a large fake USDT (~$100k+) that does not reflect our capital.
    Cash is read from state["cash_usdt"] which the engine maintains
    accurately through actual trade fills.
    """
    try:
        from config.client import get_client
        client = get_client(for_trading=True)

        tickers = {t["symbol"]: float(t["price"])
                   for t in client.get_all_tickers()
                   if t["symbol"] in UNIVERSE}

        account   = client.get_account()
        positions = {sym: 0.0 for sym in UNIVERSE}
        for balance in account.get("balances", []):
            sym = balance["asset"] + "USDT"
            if sym in UNIVERSE:
                qty = float(balance["free"]) + float(balance["locked"])
                if qty > 0:
                    positions[sym] = qty

        values         = {sym: positions[sym] * tickers.get(sym, 0.0) for sym in UNIVERSE}
        position_value = sum(values.values())

        return {
            "positions":      positions,
            "prices":         tickers,
            "values":         values,
            "position_value": position_value,
        }
    except Exception:
        return None


def load_last_close_prices() -> dict:
    """Fallback: read the most recent close price from cached parquet files."""
    prices = {}
    cache_dir = Path("data/cache")
    if not cache_dir.exists():
        return prices
    for pq in cache_dir.glob("*_1d.parquet"):
        sym = pq.stem.replace("_1d", "")
        try:
            df = pd.read_parquet(pq, columns=["close"])
            prices[sym] = float(df["close"].iloc[-1])
        except Exception:
            pass
    return prices


def fmt(value, decimals=2, pct=False, currency=False):
    if pd.isna(value) or value is None:
        return "N/A"
    if currency:
        return f"${value:,.{decimals}f}"
    if pct:
        return f"{value * 100:.{decimals}f}%"
    return f"{value:.{decimals}f}"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Crypto Dashboard")
    section = st.radio(
        "Section",
        ["📊 Backtest Analysis", "🟢 Live Trading (Testnet)", "📈 Strategy Monitor"],
        index=0,
    )
    st.markdown("---")

    if section == "📊 Backtest Analysis":
        bt_data = load_backtest_data()
        strategies = list(bt_data["metrics"].keys())

        selected_strategies = st.multiselect(
            "Strategies",
            strategies,
            default=strategies[:3] if len(strategies) > 3 else strategies,
        )
        if not selected_strategies:
            selected_strategies = strategies[:1]

        view_mode = st.radio(
            "View",
            ["Overview", "Individual Analysis", "Comparison", "Risk Analysis"],
        )

    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    if section in ["🟢 Live Trading (Testnet)", "📈 Strategy Monitor"]:
        if st.button("🔄 Refresh live data"):
            st.cache_data.clear()
            st.rerun()


# ── Backtest plot helpers ──────────────────────────────────────────────────────
def plot_cumulative_returns(bt_data, strategies):
    if bt_data["cumulative_returns"] is None:
        st.warning("No cumulative returns data available")
        return
    cols = [c for c in strategies if c in bt_data["cumulative_returns"].columns]
    if not cols:
        return
    cum = bt_data["cumulative_returns"][cols]
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, strat in enumerate(cum.columns):
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[strat], mode="lines", name=strat,
            line=dict(width=2, color=colors[i % len(colors)]),
        ))
    fig.update_layout(
        title="Cumulative Returns", xaxis_title="Date",
        yaxis_title="Cumulative Return", hovermode="x unified",
        height=500, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_daily_returns_distribution(bt_data, strategies):
    if bt_data["returns"] is None:
        st.warning("No daily returns data available")
        return
    cols = [c for c in strategies if c in bt_data["returns"].columns]
    returns = bt_data["returns"][cols]
    fig = go.Figure()
    for strat in returns.columns:
        fig.add_trace(go.Histogram(x=returns[strat] * 100, name=strat, opacity=0.7, nbinsx=50))
    fig.update_layout(
        title="Distribution of Daily Returns", xaxis_title="Daily Return (%)",
        yaxis_title="Frequency", barmode="overlay", height=400, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_drawdown(bt_data, strategies):
    if bt_data["cumulative_returns"] is None:
        st.warning("No data available")
        return
    cols = [c for c in strategies if c in bt_data["cumulative_returns"].columns]
    cum = bt_data["cumulative_returns"][cols]
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, strat in enumerate(cum.columns):
        running_max = cum[strat].expanding().max()
        drawdown = (cum[strat] - running_max) / running_max
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown * 100, mode="lines", name=strat,
            fill="tozeroy", line=dict(width=1, color=colors[i % len(colors)]),
        ))
    fig.update_layout(
        title="Drawdown Over Time", xaxis_title="Date", yaxis_title="Drawdown (%)",
        hovermode="x unified", height=400, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_heatmap(bt_data, strategy):
    if bt_data["monthly_seasonality"] is None or strategy not in bt_data["monthly_seasonality"].columns:
        st.warning(f"No monthly data for {strategy}")
        return
    monthly_data = bt_data["monthly_seasonality"][strategy].dropna()
    if len(monthly_data) == 0:
        return
    n_rows = len(monthly_data) // 12
    if n_rows == 0:
        return
    trimmed = monthly_data.iloc[:n_rows * 12]
    fig = go.Figure(data=go.Heatmap(
        z=trimmed.values.reshape(n_rows, 12),
        colorscale="RdYlGn", zmid=0,
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        title=f"Monthly Returns Heatmap — {strategy}",
        xaxis_title="Month", yaxis_title="Year", height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_rolling_sharpe(bt_data, strategies, window=30):
    if bt_data["returns"] is None:
        st.warning("No daily returns data available")
        return
    cols = [c for c in strategies if c in bt_data["returns"].columns]
    returns = bt_data["returns"][cols]
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, strat in enumerate(returns.columns):
        rs = (
            returns[strat].rolling(window).mean()
            / returns[strat].rolling(window).std()
            * np.sqrt(252)
        )
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs, mode="lines", name=strat,
            line=dict(width=2, color=colors[i % len(colors)]),
        ))
    fig.update_layout(
        title=f"Rolling {window}-Day Sharpe Ratio", xaxis_title="Date",
        yaxis_title="Sharpe Ratio", hovermode="x unified",
        height=400, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BACKTEST ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if section == "📊 Backtest Analysis":
    st.title("📊 Backtest Analysis")
    st.markdown("---")

    bt_data = load_backtest_data()

    if not bt_data["metrics"]:
        st.error("No backtest results found. Run: `python main.py --mode backtest`")
        st.stop()

    metrics = bt_data["metrics"]

    # ── Overview ──────────────────────────────────────────────────────────────
    if view_mode == "Overview":
        st.header("Performance Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_sharpe = np.mean([metrics[s].get("sharpe", 0) for s in selected_strategies if s in metrics])
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
        with col2:
            avg_cagr = np.mean([metrics[s].get("cagr", 0) for s in selected_strategies if s in metrics])
            st.metric("Avg CAGR", fmt(avg_cagr, pct=True))
        with col3:
            avg_dd = np.mean([abs(metrics[s].get("max_drawdown", 0)) for s in selected_strategies if s in metrics])
            st.metric("Avg Max Drawdown", fmt(-avg_dd, pct=True))
        with col4:
            avg_wr = np.mean([metrics[s].get("win_rate", 0) for s in selected_strategies if s in metrics])
            st.metric("Avg Win Rate", fmt(avg_wr, pct=True))

        st.markdown("---")
        plot_cumulative_returns(bt_data, selected_strategies)

        st.subheader("Strategy Metrics")
        rows = []
        for strat in selected_strategies:
            if strat in metrics:
                m = metrics[strat]
                rows.append({
                    "Strategy": strat,
                    "Sharpe": fmt(m.get("sharpe")),
                    "Sortino": fmt(m.get("sortino")),
                    "CAGR": fmt(m.get("cagr"), pct=True),
                    "Max DD": fmt(m.get("max_drawdown"), pct=True),
                    "Ann Vol": fmt(m.get("annual_vol"), pct=True),
                    "Win Rate": fmt(m.get("win_rate"), pct=True),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

        if bt_data["pnl_summary"] is not None:
            st.subheader("P&L Summary")
            pnl = bt_data["pnl_summary"]
            avail = [s for s in selected_strategies if s in pnl.index]
            if avail:
                st.dataframe(pnl.loc[avail], use_container_width=True)

    # ── Individual Analysis ───────────────────────────────────────────────────
    elif view_mode == "Individual Analysis":
        st.header("Individual Strategy Analysis")
        selected_strat = st.selectbox("Select Strategy", selected_strategies)

        if selected_strat in metrics:
            m = metrics[selected_strat]
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Sharpe Ratio", f"{m.get('sharpe', 0):.2f}")
            with col2: st.metric("CAGR", fmt(m.get("cagr", 0), pct=True))
            with col3: st.metric("Max Drawdown", fmt(m.get("max_drawdown", 0), pct=True))
            with col4: st.metric("Win Rate", fmt(m.get("win_rate", 0), pct=True))

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Sortino", f"{m.get('sortino', 0):.2f}")
            with col2: st.metric("Annual Vol", fmt(m.get("annual_vol", 0), pct=True))
            with col3: st.metric("Best Day", fmt(m.get("best_day_pnl", 0), currency=True))
            with col4: st.metric("Worst Day", fmt(m.get("worst_day_pnl", 0), currency=True))

            st.markdown("---")
            col_l, col_r = st.columns(2)
            with col_l: plot_cumulative_returns(bt_data, [selected_strat])
            with col_r: plot_drawdown(bt_data, [selected_strat])

            col_l, col_r = st.columns(2)
            with col_l: plot_daily_returns_distribution(bt_data, [selected_strat])
            with col_r: plot_rolling_sharpe(bt_data, [selected_strat])

            st.subheader("Monthly Performance")
            plot_monthly_heatmap(bt_data, selected_strat)

    # ── Comparison ────────────────────────────────────────────────────────────
    elif view_mode == "Comparison":
        st.header("Strategy Comparison")
        if len(selected_strategies) < 2:
            st.warning("Select at least 2 strategies for comparison.")
        else:
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Cumulative Returns")
                plot_cumulative_returns(bt_data, selected_strategies)
            with col_r:
                st.subheader("Drawdowns")
                plot_drawdown(bt_data, selected_strategies)

            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Daily Returns Distribution")
                plot_daily_returns_distribution(bt_data, selected_strategies)
            with col_r:
                st.subheader("Rolling Sharpe")
                plot_rolling_sharpe(bt_data, selected_strategies)

            st.subheader("Metrics Comparison")
            rows = []
            for strat in selected_strategies:
                if strat in metrics:
                    m = metrics[strat]
                    rows.append({
                        "Strategy": strat,
                        "Sharpe": m.get("sharpe", 0),
                        "Sortino": m.get("sortino", 0),
                        "CAGR": m.get("cagr", 0),
                        "Max DD": m.get("max_drawdown", 0),
                        "Ann Vol": m.get("annual_vol", 0),
                        "Win Rate": m.get("win_rate", 0),
                    })
            if rows:
                comp_df = pd.DataFrame(rows).set_index("Strategy")
                st.dataframe(
                    comp_df.style
                        .highlight_max(axis=0, subset=["Sharpe", "Sortino", "CAGR", "Win Rate"])
                        .highlight_min(axis=0, subset=["Max DD", "Ann Vol"]),
                    use_container_width=True,
                )

    # ── Risk Analysis ─────────────────────────────────────────────────────────
    elif view_mode == "Risk Analysis":
        st.header("Risk Analysis")
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Drawdown Analysis")
            plot_drawdown(bt_data, selected_strategies)
        with col_r:
            st.subheader("Rolling Volatility (30d)")
            if bt_data["returns"] is not None:
                cols = [c for c in selected_strategies if c in bt_data["returns"].columns]
                returns = bt_data["returns"][cols]
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly
                for i, strat in enumerate(returns.columns):
                    rv = returns[strat].rolling(30).std() * np.sqrt(252) * 100
                    fig.add_trace(go.Scatter(
                        x=rv.index, y=rv, mode="lines", name=strat,
                        line=dict(width=2, color=colors[i % len(colors)]),
                    ))
                fig.update_layout(
                    title="30-Day Rolling Volatility (Annualised)",
                    xaxis_title="Date", yaxis_title="Volatility (%)",
                    hovermode="x unified", height=400, template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk Metrics Summary")
        rows = []
        for strat in selected_strategies:
            if strat in metrics:
                m = metrics[strat]
                rows.append({
                    "Strategy": strat,
                    "Max Drawdown": fmt(m.get("max_drawdown", 0), pct=True),
                    "Annual Vol": fmt(m.get("annual_vol", 0), pct=True),
                    "Sharpe": fmt(m.get("sharpe", 0)),
                    "Sortino": fmt(m.get("sortino", 0)),
                    "Win Rate": fmt(m.get("win_rate", 0), pct=True),
                    "Profit Factor": fmt(m.get("profit_factor", 0)),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LIVE TRADING (TESTNET)
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🟢 Live Trading (Testnet)":
    st.title("🟢 Live Trading Monitor")
    st.markdown(
        '<span class="testnet-badge">BINANCE TESTNET — Paper Money</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    live = load_live_data()

    if live is None:
        st.warning(
            "No live trading data found yet.\n\n"
            "Start the live engine with an immediate cycle:\n"
            "```\npython main.py --mode live --run-now\n```\n"
            "Then come back here — this page auto-refreshes every 30 seconds."
        )
        st.stop()

    # ── Status banner ─────────────────────────────────────────────────────────
    last_run = live.get("last_run")
    if last_run:
        last_run_ts = pd.Timestamp(last_run)
        age = pd.Timestamp.utcnow() - last_run_ts.tz_localize("UTC") if last_run_ts.tzinfo is None else pd.Timestamp.utcnow() - last_run_ts
        hours_ago = int(age.total_seconds() // 3600)
        mins_ago  = int((age.total_seconds() % 3600) // 60)
        st.success(f"Last rebalance: **{last_run_ts.strftime('%Y-%m-%d %H:%M UTC')}** ({hours_ago}h {mins_ago}m ago)")
    else:
        st.info("Engine is running — no rebalance cycle has completed yet.")

    # ── Active Strategy Banner ────────────────────────────────────────────────
    active_strats  = live.get("active_strategies", [])
    active_weights = live.get("active_strategy_weights", {})
    sig_updated_at = live.get("signals_updated_at")

    if active_strats:
        st.markdown("### 🎯 Currently Active Strategies")
        cols_strat = st.columns(len(active_strats))
        for i, name in enumerate(active_strats):
            blend_w = active_weights.get(name, 1.0 / len(active_strats))
            with cols_strat[i]:
                st.markdown(
                    f"<div style='background:#1e3a5f;border-radius:10px;padding:14px;text-align:center'>"
                    f"<div style='color:#93c5fd;font-size:13px;font-weight:bold'>ACTIVE STRATEGY</div>"
                    f"<div style='color:white;font-size:20px;font-weight:bold;margin:6px 0'>{name}</div>"
                    f"<div style='color:#22c55e;font-size:16px'>Blend: {blend_w*100:.0f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        if sig_updated_at:
            sig_ts = pd.Timestamp(sig_updated_at)
            st.caption(
                f"Signals last computed: {sig_ts.strftime('%Y-%m-%d %H:%M UTC')}  |  "
                "Strategy blend set by backtest regime/seasonality score + live 48h rolling Sharpe."
            )
    else:
        st.info("No active strategies recorded yet — appears after the first signal cycle.")

    # ── Parse state ───────────────────────────────────────────────────────────
    cash_usdt        = live.get("cash_usdt", STARTING_CAPITAL)
    nav_history      = live.get("nav_history", [])
    position_entries = live.get("position_entries", {})
    start_nav        = live.get("initial_nav", STARTING_CAPITAL)

    # ── Live Binance balances (real quantities + real-time prices) ────────────
    binance = fetch_binance_live_balances()
    if binance:
        positions      = binance["positions"]
        last_prices    = binance["prices"]
        position_value = binance["position_value"]
        # Live NAV = engine-tracked cash + positions at current market price
        current_nav    = cash_usdt + position_value
        nav_source     = "Binance live"
    else:
        # Fallback to state file + cached OHLCV if Binance is unreachable
        positions      = live.get("positions", {})
        last_prices    = load_last_close_prices()
        position_value = sum(positions.get(sym, 0) * last_prices.get(sym, 0) for sym in UNIVERSE)
        if nav_history:
            nav_df_tmp  = pd.DataFrame(nav_history)
            current_nav = pd.to_numeric(nav_df_tmp["nav"]).iloc[-1]
        else:
            current_nav = cash_usdt
        nav_source = "state file (Binance unavailable)"

    open_positions = {sym: qty for sym, qty in positions.items() if qty != 0}

    # ── NAV chart data ────────────────────────────────────────────────────────
    nav_df = None
    if nav_history:
        nav_df = pd.DataFrame(nav_history)
        nav_df["nav"] = pd.to_numeric(nav_df["nav"])

    total_pnl    = current_nav - start_nav
    total_return = total_pnl / start_nav if start_nav else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Live NAV", f"${current_nav:,.2f}", help=f"Source: {nav_source}")
    col2.metric("Cash (USDT)", f"${cash_usdt:,.2f}", help="Tracked by engine through actual fills")
    col3.metric("Open Positions", len(open_positions))
    col4.metric(
        "Total P&L",
        f"{total_return * 100:+.2f}%",
        delta=f"${total_pnl:+,.2f}",
        delta_color="normal",
    )

    st.markdown("---")

    # ── NAV History chart ─────────────────────────────────────────────────────
    st.subheader("NAV History")
    if nav_df is not None and len(nav_df) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nav_df["date"], y=nav_df["nav"],
            mode="lines+markers", name="Portfolio NAV",
            line=dict(width=2, color="#22c55e"),
            marker=dict(size=6),
        ))
        fig.add_hline(
            y=STARTING_CAPITAL, line_dash="dash", line_color="#94a3b8",
            annotation_text=f"Starting Capital ${STARTING_CAPITAL:,}",
            annotation_position="bottom right",
        )
        fig.update_layout(
            xaxis_title="Date", yaxis_title="NAV (USDT)",
            hovermode="x unified", height=400, template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("NAV history will appear here after the first rebalance cycle completes.")

    st.markdown("---")

    # ── Positions ─────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Open Positions")
        if open_positions:
            rows = []
            for sym, qty in open_positions.items():
                entry       = position_entries.get(sym, {})
                entry_price = entry.get("entry_price")
                last_price  = last_prices.get(sym)
                mkt_value   = binance["values"].get(sym) if binance else (qty * last_price if last_price else None)

                if entry_price and last_price:
                    unreal_pct = (last_price - entry_price) / entry_price * 100
                    unreal_usd = (last_price - entry_price) * qty
                else:
                    unreal_pct = None
                    unreal_usd = None

                rows.append({
                    "Symbol":         sym,
                    "Qty":            round(qty, 6),
                    "Entry Price":    f"${entry_price:,.2f}" if entry_price else "—",
                    "Live Price":     f"${last_price:,.2f}" if last_price else "—",
                    "Mkt Value":      f"${mkt_value:,.2f}" if mkt_value else "—",
                    "Unrealised P&L": f"{unreal_pct:+.2f}%" if unreal_pct is not None else "—",
                    "Unrealised $":   f"${unreal_usd:+,.2f}" if unreal_usd is not None else "—",
                    "Entry Date":     (entry.get("entry_date", "")[:10] if entry.get("entry_date") else "—"),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Symbol"), use_container_width=True)
            price_note = "Live prices from Binance." if binance else "⚠️ Binance unavailable — using cached OHLCV prices."
            st.caption(price_note)
        else:
            st.info("No open positions after the last rebalance.")

    with col_right:
        st.subheader("All Positions (Binance)")
        all_rows = []
        for sym, qty in positions.items():
            val = (binance["values"].get(sym, 0) if binance else qty * last_prices.get(sym, 0))
            all_rows.append({"Symbol": sym, "Qty": round(qty, 6), "Value $": round(val, 2)})
        all_df = pd.DataFrame(all_rows).set_index("Symbol")
        st.dataframe(all_df.style.applymap(
            lambda v: "color: #22c55e" if v > 0 else ("color: #ef4444" if v < 0 else ""),
            subset=["Qty"],
        ), use_container_width=True)

    # ── Stop / Profit tracking ────────────────────────────────────────────────
    if position_entries:
        st.markdown("---")
        st.subheader("Position Risk Tracker")
        from config.settings import STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRAILING_STOP_PCT
        risk_rows = []
        for sym, entry in position_entries.items():
            ep  = entry.get("entry_price", 0)
            lp  = last_prices.get(sym, 0)
            pk  = entry.get("peak_price", ep)
            pnl = (lp - ep) / ep * 100 if ep else None
            dd  = (pk - lp) / pk * 100 if pk else None
            risk_rows.append({
                "Symbol":         sym,
                "Entry":          f"${ep:,.2f}",
                "Peak":           f"${pk:,.2f}" if pk else "—",
                "Current":        f"${lp:,.2f}" if lp else "—",
                "P&L %":          f"{pnl:+.2f}%" if pnl is not None else "—",
                "DD from Peak %": f"{dd:.2f}%" if dd is not None else "—",
                "Stop Level":     f"${ep * (1 - STOP_LOSS_PCT):,.2f}" if ep else "—",
                "TP Level":       f"${ep * (1 + TAKE_PROFIT_PCT):,.2f}" if ep else "—",
            })
        st.dataframe(pd.DataFrame(risk_rows).set_index("Symbol"), use_container_width=True)
        st.caption(
            f"Stop Loss: {STOP_LOSS_PCT*100:.0f}% | "
            f"Take Profit: {TAKE_PROFIT_PCT*100:.0f}% | "
            f"Trailing Stop: {TRAILING_STOP_PCT*100:.0f}%"
        )

    # ── Strategy Signal Heatmap ───────────────────────────────────────────────
    latest_signals = live.get("latest_signals", {})
    if latest_signals:
        st.markdown("---")
        st.subheader("📡 Current Strategy Signals")
        st.caption(
            "Signal strength per strategy × token. "
            "Green = bullish (long), Red = bearish (short), Gray = neutral. "
            "Values are in [-1, +1]."
        )

        # Build matrix: rows=strategies, cols=tokens (strip USDT for readability)
        token_labels = [s.replace("USDT", "") for s in UNIVERSE]
        sig_matrix   = []
        row_labels   = []
        for name, sig_dict in latest_signals.items():
            row = [float(sig_dict.get(sym, 0)) for sym in UNIVERSE]
            sig_matrix.append(row)
            marker = " ★" if name in active_strats else ""
            row_labels.append(f"{name}{marker}")

        if sig_matrix:
            sig_df_plot = pd.DataFrame(sig_matrix, index=row_labels, columns=token_labels)

            heatmap_fig = go.Figure(data=go.Heatmap(
                z=sig_df_plot.values,
                x=sig_df_plot.columns.tolist(),
                y=sig_df_plot.index.tolist(),
                colorscale=[
                    [0.0,  "#ef4444"],   # -1 → red (short)
                    [0.5,  "#f1f5f9"],   # 0  → light grey (neutral)
                    [1.0,  "#22c55e"],   # +1 → green (long)
                ],
                zmin=-1, zmax=1,
                text=[[f"{v:+.2f}" for v in row] for row in sig_df_plot.values],
                texttemplate="%{text}",
                textfont=dict(size=11),
                hoverongaps=False,
            ))
            heatmap_fig.update_layout(
                height=max(300, 50 * len(row_labels) + 80),
                xaxis_title="Token",
                yaxis_title="Strategy  (★ = active)",
                margin=dict(l=180, r=20, t=20, b=40),
                template="plotly_white",
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

            # Active-only detail table
            active_detail = {
                name.rstrip(" ★"): latest_signals[name.rstrip(" ★")]
                for name in row_labels
                if "★" in name and name.rstrip(" ★") in latest_signals
            }
            if active_detail:
                st.markdown("**Active strategy signals — top positions:**")
                detail_cols = st.columns(len(active_detail))
                for i, (name, sigs) in enumerate(active_detail.items()):
                    with detail_cols[i]:
                        st.markdown(f"**{name}**")
                        sig_s = pd.Series({k.replace("USDT",""): v for k, v in sigs.items()})
                        longs  = sig_s[sig_s > 0.01].nlargest(5)
                        shorts = sig_s[sig_s < -0.01].nsmallest(5)
                        if not longs.empty:
                            st.markdown("🟢 **Long signals:**")
                            for tok, val in longs.items():
                                st.markdown(f"&nbsp;&nbsp;{tok}: `{val:+.3f}`", unsafe_allow_html=True)
                        if not shorts.empty:
                            st.markdown("🔴 **Short signals:**")
                            for tok, val in shorts.items():
                                st.markdown(f"&nbsp;&nbsp;{tok}: `{val:+.3f}`", unsafe_allow_html=True)
                        if longs.empty and shorts.empty:
                            st.caption("No signals above threshold")

    # ── Trade History ─────────────────────────────────────────────────────────
    trade_log = live.get("trade_log", [])
    st.markdown("---")
    st.subheader("Trade History")
    if trade_log:
        trades_df = pd.DataFrame(trade_log)

        trades_df["time"] = pd.to_datetime(trades_df["time"]).dt.strftime("%Y-%m-%d %H:%M UTC")
        if "price" in trades_df.columns:
            trades_df["price"] = trades_df["price"].apply(lambda x: f"${float(x):,.2f}")
        if "pnl" in trades_df.columns:
            trades_df["pnl"] = trades_df["pnl"].apply(
                lambda x: f"{float(x):+.2f}" if pd.notna(x) and str(x) not in ("", "None") else "—"
            )
        if "qty" in trades_df.columns:
            trades_df["qty"] = trades_df["qty"].apply(lambda x: f"{float(x):.6f}")

        col_map = {
            "time":     "Time (UTC)",
            "symbol":   "Symbol",
            "side":     "Side",
            "qty":      "Qty",
            "price":    "Price",
            "pnl":      "Realised P&L",
            "strategy": "Strategy",
            "reason":   "Reason",
        }
        available  = [c for c in col_map if c in trades_df.columns]
        trades_df  = trades_df[available].rename(columns=col_map).iloc[::-1].reset_index(drop=True)

        def _color_side(val):
            if val == "BUY":
                return "color: #22c55e; font-weight: bold"
            if val == "SELL":
                return "color: #ef4444; font-weight: bold"
            return ""

        def _color_pnl(val):
            try:
                v = float(str(val).replace(",", ""))
                return "color: #22c55e" if v > 0 else ("color: #ef4444" if v < 0 else "")
            except Exception:
                return ""

        style = trades_df.style
        if "Side" in trades_df.columns:
            style = style.applymap(_color_side, subset=["Side"])
        if "Realised P&L" in trades_df.columns:
            style = style.applymap(_color_pnl, subset=["Realised P&L"])

        st.dataframe(style, use_container_width=True)

        n_buys  = (trades_df.get("Side", pd.Series()) == "BUY").sum()
        n_sells = (trades_df.get("Side", pd.Series()) == "SELL").sum()
        st.caption(
            f"Total: {len(trade_log)} trades  |  "
            f"🟢 BUY: {n_buys}  |  🔴 SELL: {n_sells}"
        )
    else:
        st.info("No trades executed yet — orders will appear here after the first rebalance.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STRATEGY MONITOR (Hypothetical paper portfolios)
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📈 Strategy Monitor":
    st.title("📈 Strategy Monitor")
    st.markdown(
        "Each of the 10 strategies runs as an independent **$10,000 hypothetical paper portfolio** "
        "updated every signal cycle. Strategies selected by the seasonality/regime layer are shown "
        "with **★** — compare their live hypothetical performance against backtest predictions."
    )
    st.markdown("---")

    monitor_data = load_strategy_monitor_data()
    bt_data      = load_backtest_data()

    if monitor_data is None or not monitor_data.get("hypothetical"):
        st.warning(
            "No strategy monitor data yet.\n\n"
            "Start the live engine and wait for the first signal cycle:\n"
            "```\npython main.py --mode live --run-now\n```\n"
            "Hypothetical portfolios are initialised on the first cycle and updated every hour."
        )
        st.stop()

    hypo          = monitor_data["hypothetical"]
    active_strats = monitor_data["active_strategies"]
    bt_metrics    = bt_data.get("metrics", {})

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Performance Summary")
    rows = []
    for name, data in hypo.items():
        nav_val   = float(data.get("nav", STARTING_CAPITAL))
        ret_pct   = (nav_val - STARTING_CAPITAL) / STARTING_CAPITAL * 100
        bt_sharpe = bt_metrics.get(name, {}).get("sharpe")
        is_active = name in active_strats
        rows.append({
            "Strategy":        name,
            "Active":          "★ LIVE" if is_active else "—",
            "Hypo NAV":        f"${nav_val:,.2f}",
            "Total Return":    f"{ret_pct:+.2f}%",
            "Backtest Sharpe": f"{bt_sharpe:.2f}" if bt_sharpe is not None else "—",
            "Updates":         len(data.get("nav_history", [])),
        })
    if rows:
        summary_df = pd.DataFrame(rows).set_index("Strategy")
        st.dataframe(
            summary_df.style.applymap(
                lambda v: "color: #22c55e; font-weight: bold" if v == "★ LIVE" else "",
                subset=["Active"],
            ),
            use_container_width=True,
        )
        if active_strats:
            st.caption(
                f"★ LIVE = currently active: **{', '.join(active_strats)}**  |  "
                "Selection = backtest score (jumpoff) blended with 48h rolling Sharpe from live hypotheticals "
                "(live weight ramps 0→80% over 2 days). A better-performing strategy "
                "can take over within 2 days once its live score is high enough. "
                "Hypothetical portfolios run full long/short (50/50). "
                "Live portfolio is long-only (Binance spot cannot short without margin)."
            )

    st.markdown("---")

    # ── Live Competition Panel ────────────────────────────────────────────────
    st.subheader("🏆 Live Competition — Strategy Takeover Tracker")

    # Compute rolling 7-day Sharpe for each strategy from NAV history
    live_scores = {}
    earliest_date = None
    for name, data in hypo.items():
        hist = data.get("nav_history", [])
        if len(hist) < 3:
            continue
        try:
            navs = pd.Series(
                [h["nav"] for h in hist],
                index=pd.to_datetime([h["date"] for h in hist])
            ).sort_index()
            if earliest_date is None or navs.index[0] < earliest_date:
                earliest_date = navs.index[0]
            cutoff = navs.index[-1] - pd.Timedelta(days=7)
            recent = navs[navs.index >= cutoff]
            if len(recent) >= 3:
                rets = recent.pct_change().dropna()
                if rets.std() > 0:
                    live_scores[name] = float(rets.mean() / rets.std())
                else:
                    live_scores[name] = 0.0
        except Exception:
            pass

    if live_scores:
        days_live = 0.0
        if earliest_date is not None:
            days_live = (pd.Timestamp.utcnow() - earliest_date).total_seconds() / 86400
        live_weight_pct = min(days_live / 2, 0.80) * 100

        col_prog, col_info = st.columns([2, 3])
        with col_prog:
            st.metric(
                "Live Data Trust Level",
                f"{live_weight_pct:.0f}%",
                help="Ramps from 0% → 80% over 2 days. Backtest score is just a jumpoff "
                     "point — live rolling Sharpe quickly takes majority weight.",
            )
            st.progress(min(live_weight_pct / 80, 1.0))
            st.caption(
                f"Day {days_live:.1f} of 2 — "
                f"{'Live scores driving selection.' if live_weight_pct >= 80 else f'Building up ({live_weight_pct:.0f}% live, {100 - live_weight_pct:.0f}% backtest).'}"
            )

        with col_info:
            score_df = pd.DataFrame([
                {
                    "Strategy":        name,
                    "7-day Sharpe":    round(score, 3),
                    "Currently Live":  "★" if name in active_strats else "",
                    "Would Takeover":  "→ CANDIDATE" if (
                        score > 0 and name not in active_strats and
                        len(live_scores) > 0 and
                        score >= sorted(live_scores.values(), reverse=True)[:2][-1]
                    ) else "",
                }
                for name, score in sorted(live_scores.items(), key=lambda x: -x[1])
            ])

            def _color_score(v):
                try:
                    v = float(v)
                    if v > 0.5: return "color: #22c55e; font-weight: bold"
                    if v < 0: return "color: #ef4444"
                except Exception:
                    pass
                return ""

            st.dataframe(
                score_df.set_index("Strategy").style
                    .applymap(_color_score, subset=["7-day Sharpe"])
                    .applymap(lambda v: "color: #22c55e; font-weight: bold" if v == "★" else "", subset=["Currently Live"])
                    .applymap(lambda v: "color: #f59e0b; font-weight: bold" if "CANDIDATE" in str(v) else "", subset=["Would Takeover"]),
                use_container_width=True,
            )
            st.caption(
                "**→ CANDIDATE**: outperforming enough to displace a current active strategy "
                "once live weight is sufficient. The engine re-evaluates every signal cycle."
            )
    else:
        st.info(
            "Live competition scores appear after a few signal cycles of NAV history. "
            "Check back after 3+ updates."
        )

    st.markdown("---")

    # ── NAV History chart — all strategies ───────────────────────────────────
    st.subheader("Hypothetical NAV History — All Strategies")
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    fig = go.Figure()

    for i, (name, data) in enumerate(hypo.items()):
        hist = data.get("nav_history", [])
        if len(hist) < 2:
            continue
        hist_df = pd.DataFrame(hist)
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df["nav"]  = pd.to_numeric(hist_df["nav"])
        is_active       = name in active_strats
        fig.add_trace(go.Scatter(
            x=hist_df["date"], y=hist_df["nav"],
            mode="lines", name=name,
            line=dict(
                width=3 if is_active else 1.5,
                color=colors[i % len(colors)],
                dash="solid" if is_active else "dot",
            ),
        ))

    fig.add_hline(
        y=STARTING_CAPITAL, line_dash="dash", line_color="#94a3b8",
        annotation_text=f"Starting Capital ${STARTING_CAPITAL:,}",
        annotation_position="bottom right",
    )
    if len(fig.data) > 0:
        fig.update_layout(
            xaxis_title="Date", yaxis_title="NAV (USDT)",
            hovermode="x unified", height=520, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Active strategies (★) shown with solid, thicker lines. Others are dotted.")
    else:
        st.info(
            "NAV history chart appears after the second signal cycle — "
            "the first cycle establishes the baseline price, the second cycle computes returns."
        )

    st.markdown("---")

    # ── Return bar chart comparison ───────────────────────────────────────────
    st.subheader("Total Return Comparison")
    ret_rows = [
        {
            "Strategy": name,
            "Return %": round((float(d.get("nav", STARTING_CAPITAL)) - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2),
            "Active":   name in active_strats,
        }
        for name, d in hypo.items()
    ]
    if ret_rows:
        ret_df = pd.DataFrame(ret_rows).sort_values("Return %", ascending=False)
        bar_colors = [
            "#22c55e" if row["Active"] else ("#ef4444" if row["Return %"] < 0 else "#60a5fa")
            for _, row in ret_df.iterrows()
        ]
        bar_fig = go.Figure(go.Bar(
            x=ret_df["Strategy"], y=ret_df["Return %"],
            marker_color=bar_colors,
            text=[f"{v:+.2f}%" for v in ret_df["Return %"]],
            textposition="outside",
        ))
        bar_fig.add_hline(y=0, line_color="#94a3b8")
        bar_fig.update_layout(
            xaxis_title="Strategy", yaxis_title="Total Return (%)",
            height=380, template="plotly_white",
            xaxis_tickangle=-30,
        )
        st.plotly_chart(bar_fig, use_container_width=True)
        st.caption("Green = currently active strategy. Blue = positive return. Red = negative return.")

    st.markdown("---")

    # ── Per-strategy positions + trade history ────────────────────────────────
    st.subheader("Current Would-Be Positions & Trade History by Strategy")

    for name in hypo.keys():
        data       = hypo[name]
        weights    = data.get("weights", {})
        nav_val    = float(data.get("nav", STARTING_CAPITAL))
        ret_pct    = (nav_val - STARTING_CAPITAL) / STARTING_CAPITAL * 100
        marker     = " ★" if name in active_strats else ""
        ret_color  = "#22c55e" if ret_pct >= 0 else "#ef4444"
        trade_hist = data.get("trade_history", [])

        with st.expander(
            f"{name}{marker}  —  ${nav_val:,.0f}  ({ret_pct:+.1f}%)  |  {len(trade_hist)} trades",
            expanded=(name in active_strats),
        ):
            col_w, col_t = st.columns([1, 2])

            with col_w:
                st.markdown("**Current positions**")
                all_pos = {
                    sym.replace("USDT", ""): float(w)
                    for sym, w in weights.items()
                    if abs(float(w)) > 0.001
                }
                if all_pos:
                    w_df = (
                        pd.DataFrame(list(all_pos.items()), columns=["Token", "Weight"])
                        .sort_values("Weight", ascending=False)
                        .set_index("Token")
                    )
                    w_df["Side"]   = w_df["Weight"].apply(lambda x: "LONG" if x > 0 else "SHORT")
                    w_df["Weight"] = w_df["Weight"].apply(lambda x: f"{abs(x)*100:.1f}%")
                    st.dataframe(w_df, use_container_width=True)
                else:
                    st.caption("No positions (holding cash)")

            with col_t:
                # Compute total realized P&L for the header
                total_rpnl = sum(
                    float(t.get("pnl", 0)) for t in trade_hist if float(t.get("pnl", 0)) != 0
                )
                pnl_label  = f"  |  Realized P&L: ${total_rpnl:+,.2f}" if total_rpnl != 0 else ""
                st.markdown(f"**Trade history** (most recent first){pnl_label}")
                if trade_hist:
                    th_df = pd.DataFrame(trade_hist)
                    th_df["time"]        = pd.to_datetime(th_df["time"]).dt.strftime("%m-%d %H:%M")
                    th_df["price"]       = th_df["price"].apply(lambda x: f"${float(x):,.2f}")
                    th_df["hypo_value"]  = th_df["hypo_value"].apply(lambda x: f"${float(x):,.0f}")
                    th_df["weight_from"] = th_df["weight_from"].apply(lambda x: f"{x:.0f}%")
                    th_df["weight_to"]   = th_df["weight_to"].apply(lambda x: f"{x:.0f}%")
                    th_df["symbol"]      = th_df["symbol"].str.replace("USDT", "")
                    if "pnl" in th_df.columns:
                        th_df["pnl"] = th_df["pnl"].apply(
                            lambda x: f"${float(x):+,.2f}" if float(x) != 0 else "—"
                        )
                    th_df = th_df.rename(columns={
                        "time":        "Time",
                        "symbol":      "Token",
                        "side":        "Side",
                        "weight_from": "From",
                        "weight_to":   "To",
                        "price":       "Price",
                        "hypo_value":  "Value",
                        "pnl":         "Realized P&L",
                    })
                    col_order = ["Time", "Token", "Side", "From", "To", "Price", "Value", "Realized P&L"]
                    th_df = th_df[[c for c in col_order if c in th_df.columns]]

                    def _color_pnl_hypo(val):
                        try:
                            v = float(str(val).replace("$", "").replace(",", ""))
                            if v > 0: return "color: #22c55e"
                            if v < 0: return "color: #ef4444"
                        except Exception:
                            pass
                        return ""

                    style = th_df.iloc[::-1].reset_index(drop=True).style
                    if "Side" in th_df.columns:
                        style = style.applymap(
                            lambda v: "color: #22c55e; font-weight:bold" if v == "BUY"
                                      else ("color: #ef4444; font-weight:bold" if v == "SELL" else ""),
                            subset=["Side"],
                        )
                    if "Realized P&L" in th_df.columns:
                        style = style.applymap(_color_pnl_hypo, subset=["Realized P&L"])
                    st.dataframe(style, use_container_width=True)
                else:
                    st.caption("No trades yet — appears after second signal cycle.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#888;font-size:12px;'>"
    f"Crypto Strategy Dashboard — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True,
)
