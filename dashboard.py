"""
dashboard.py
============
Interactive Streamlit dashboard — two independent sections:
  📊 Backtest Analysis  — historical walk-forward strategy performance
  🟢 Live Trading       — real-time Binance Testnet positions and NAV

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime, timezone

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


@st.cache_data(ttl=30)  # auto-refresh every 30 s while dashboard is open
def load_live_data():
    state_file = RESULT_DIR / "live_state.json"
    if not state_file.exists():
        return None
    with open(state_file) as f:
        return json.load(f)


@st.cache_data(ttl=30)
def load_strategy_monitor_data():
    state_file = RESULT_DIR / "live_state.json"
    if not state_file.exists():
        return None
    with open(state_file) as f:
        state = json.load(f)
    return {
        "hypothetical":      state.get("hypothetical", {}),
        "active_strategies": state.get("active_strategies", []),
    }


def load_last_close_prices() -> dict:
    """Read the most recent close price for each symbol from cached parquet files."""
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

    # ── Parse state ───────────────────────────────────────────────────────────
    positions        = live.get("positions", {})
    cash_usdt        = live.get("cash_usdt", STARTING_CAPITAL)
    nav_history      = live.get("nav_history", [])
    position_entries = live.get("position_entries", {})

    open_positions = {sym: qty for sym, qty in positions.items() if qty != 0}

    # ── NAV summary ───────────────────────────────────────────────────────────
    if nav_history:
        nav_df = pd.DataFrame(nav_history)
        nav_df["nav"] = pd.to_numeric(nav_df["nav"])
        current_nav = nav_df["nav"].iloc[-1]
        start_nav   = nav_df["nav"].iloc[0]
    else:
        current_nav = cash_usdt
        start_nav   = STARTING_CAPITAL
        nav_df      = None

    total_pnl    = current_nav - start_nav
    total_return = total_pnl / start_nav if start_nav else 0

    # Pull last close prices from cache for unrealised P&L
    last_prices = load_last_close_prices()

    # Compute position market values
    position_value = sum(
        qty * last_prices.get(sym, 0)
        for sym, qty in open_positions.items()
    )
    live_nav = cash_usdt + position_value  # best estimate from cached prices

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current NAV", f"${current_nav:,.2f}", help="From last rebalance log")
    col2.metric("Cash (USDT)", f"${cash_usdt:,.2f}")
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

                if entry_price and last_price:
                    unreal_pct = (last_price - entry_price) / entry_price * 100
                    unreal_usd = (last_price - entry_price) * qty
                else:
                    unreal_pct = None
                    unreal_usd = None

                rows.append({
                    "Symbol":        sym,
                    "Qty":           round(qty, 6),
                    "Entry Price":   f"${entry_price:,.2f}" if entry_price else "—",
                    "Last Price":    f"${last_price:,.2f}" if last_price else "—",
                    "Unrealised P&L": f"{unreal_pct:+.2f}%" if unreal_pct is not None else "—",
                    "Unrealised $":  f"${unreal_usd:+,.2f}" if unreal_usd is not None else "—",
                    "Entry Date":    (entry.get("entry_date", "")[:10] if entry.get("entry_date") else "—"),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Symbol"), use_container_width=True)
            st.caption("Last Price sourced from cached daily OHLCV — not real-time.")
        else:
            st.info("No open positions after the last rebalance.")

    with col_right:
        st.subheader("All Positions")
        all_rows = [{"Symbol": sym, "Qty": round(qty, 6)} for sym, qty in positions.items()]
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

    # ── Trade History ─────────────────────────────────────────────────────────
    trade_log = live.get("trade_log", [])
    st.markdown("---")
    st.subheader("Trade History")
    if trade_log:
        trades_df = pd.DataFrame(trade_log)

        # Format columns
        trades_df["time"] = pd.to_datetime(trades_df["time"]).dt.strftime("%Y-%m-%d %H:%M UTC")
        if "price" in trades_df.columns:
            trades_df["price"] = trades_df["price"].apply(lambda x: f"${float(x):,.2f}")
        if "pnl" in trades_df.columns:
            trades_df["pnl"] = trades_df["pnl"].apply(
                lambda x: f"{float(x):+.2f}" if pd.notna(x) and str(x) != "" else "—"
            )
        if "qty" in trades_df.columns:
            trades_df["qty"] = trades_df["qty"].apply(lambda x: f"{float(x):.6f}")

        # Pick and rename columns
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
        available = [c for c in col_map if c in trades_df.columns]
        trades_df = trades_df[available].rename(columns=col_map)

        # Most recent first
        st.dataframe(trades_df.iloc[::-1].reset_index(drop=True), use_container_width=True)
        st.caption(f"Total trades executed: {len(trade_log)}")
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
                f"★ LIVE = currently blended by the seasonality/regime layer: "
                f"**{', '.join(active_strats)}**"
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

    # ── Per-strategy current weights ──────────────────────────────────────────
    st.subheader("Current Would-Be Positions by Strategy")

    strat_names  = list(hypo.keys())
    cols_per_row = 2

    for row_start in range(0, len(strat_names), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, name in enumerate(strat_names[row_start:row_start + cols_per_row]):
            data    = hypo[name]
            weights = data.get("weights", {})
            nav_val = float(data.get("nav", STARTING_CAPITAL))
            ret_pct = (nav_val - STARTING_CAPITAL) / STARTING_CAPITAL * 100
            marker  = " ★" if name in active_strats else ""

            with cols[col_idx]:
                ret_color = "#22c55e" if ret_pct >= 0 else "#ef4444"
                st.markdown(
                    f"**{name}{marker}** &nbsp; "
                    f"`${nav_val:,.0f}` "
                    f"<span style='color:{ret_color}'>({ret_pct:+.1f}%)</span>",
                    unsafe_allow_html=True,
                )
                pos_weights = {
                    sym.replace("USDT", ""): w
                    for sym, w in weights.items()
                    if float(w) > 0.001
                }
                if pos_weights:
                    w_df = (
                        pd.DataFrame(list(pos_weights.items()), columns=["Token", "Weight"])
                        .sort_values("Weight", ascending=False)
                        .set_index("Token")
                    )
                    w_df["Weight"] = w_df["Weight"].apply(lambda x: f"{float(x)*100:.1f}%")
                    st.dataframe(w_df, use_container_width=True)
                else:
                    st.caption("No long positions (holding cash)")

    # ── Hypothetical Trade History ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hypothetical Trade History")

    all_trades = []
    for name, data in hypo.items():
        for t in data.get("trade_history", []):
            all_trades.append({**t, "strategy": name})

    if all_trades:
        # Strategy filter
        all_strat_names = list(hypo.keys())
        selected = st.multiselect(
            "Filter by strategy",
            all_strat_names,
            default=all_strat_names,
            key="hypo_trade_filter",
        )

        filtered = [t for t in all_trades if t["strategy"] in selected]

        if filtered:
            th_df = pd.DataFrame(filtered)
            th_df["time"] = pd.to_datetime(th_df["time"]).dt.strftime("%Y-%m-%d %H:%M UTC")
            th_df["price"] = th_df["price"].apply(lambda x: f"${float(x):,.2f}")
            th_df["hypo_value"] = th_df["hypo_value"].apply(lambda x: f"${float(x):,.0f}")
            th_df["weight_from"] = th_df["weight_from"].apply(lambda x: f"{x:.1f}%")
            th_df["weight_to"]   = th_df["weight_to"].apply(lambda x: f"{x:.1f}%")
            th_df["symbol"] = th_df["symbol"].str.replace("USDT", "")

            th_df = th_df.rename(columns={
                "time":        "Time (UTC)",
                "strategy":    "Strategy",
                "symbol":      "Token",
                "side":        "Side",
                "weight_from": "From %",
                "weight_to":   "To %",
                "price":       "Price",
                "hypo_value":  "Hypo Value",
            })

            col_order = ["Time (UTC)", "Strategy", "Token", "Side", "From %", "To %", "Price", "Hypo Value"]
            th_df = th_df[[c for c in col_order if c in th_df.columns]]

            # Most recent first
            st.dataframe(th_df.iloc[::-1].reset_index(drop=True), use_container_width=True)
            st.caption(
                f"Showing {len(filtered)} hypothetical trades across {len(selected)} strategies. "
                "A trade is logged when a strategy's weight in a token shifts by more than 1%."
            )
        else:
            st.info("No trades match the selected strategies.")
    else:
        st.info("Hypothetical trades will appear here after the second signal cycle.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#888;font-size:12px;'>"
    f"Crypto Strategy Dashboard — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True,
)
