"""
dashboard/app.py — Dash trading dashboard for the crypto algo system.

Run with:  python dashboard/app.py
Accessible at http://0.0.0.0:8050
"""

import sys
from pathlib import Path

# Ensure the project root (crypto_algo/) is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dash_table, dcc, html

from config.settings import (
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT,
    UNIVERSE,
)
from dashboard.data import (
    get_all_hyp_strategies,
    get_connection_status,
    get_current_prices,
    get_engine_controls,
    get_hypothetical_nav,
    get_hypothetical_trades,
    get_live_state,
    get_nav_history,
    get_portfolio_returns,
    get_strategy_metrics,
    get_strategy_summary,
    get_trade_log,
    save_engine_controls,
)

# ── Dark chart theme ──────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
    font={"color": "white"},
    xaxis={"gridcolor": "#2a2a4a", "color": "white"},
    yaxis={"gridcolor": "#2a2a4a", "color": "white"},
    margin={"l": 50, "r": 20, "t": 40, "b": 40},
)

TABLE_STYLE_HEADER = {"backgroundColor": "#1a1a2e", "color": "white", "fontWeight": "bold"}
TABLE_STYLE_DATA   = {"backgroundColor": "#16213e", "color": "white"}
CARD_STYLE         = {"border": "1px solid #444"}
CARD_DANGER_STYLE  = {"border": "2px solid red"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _badge(text: str, color: str) -> dbc.Badge:
    return dbc.Badge(text, color=color, className="fs-6 ms-2")


def _stat_card(title: str, value_id: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.85rem"}),
            html.H4("—", id=value_id, className="mb-0"),
        ]),
        className="mb-3",
        style=CARD_STYLE,
    )


def _empty_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, **CHART_LAYOUT)
    return fig


# ── App initialisation ────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
)
app.title = "Crypto Algo Dashboard"


# ── Layout helpers (tabs) ─────────────────────────────────────────────────────

def _tab_live_overview() -> dbc.Container:
    positions_columns = [
        {"name": c, "id": c}
        for c in ["Symbol", "Side", "Qty", "Entry $", "Current $", "P&L $", "P&L %", "SL $", "TP $"]
    ]
    trades_columns = [
        {"name": c, "id": c}
        for c in ["executed_at", "symbol", "side", "qty", "price", "reason", "strategy"]
    ]
    return dbc.Container([
        # ── Stat cards ────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("NAV ($)", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H4("—", id="nav-value", className="mb-0"),
            ]), className="mb-3", style=CARD_STYLE), width=2),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Cash (USDT)", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H4("—", id="cash-value", className="mb-0"),
            ]), className="mb-3", style=CARD_STYLE), width=2),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Total P&L ($)", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H4("—", id="pnl-value", className="mb-0", style={"color": "white"}),
            ]), className="mb-3", style=CARD_STYLE), width=2),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Open Positions", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H4("—", id="positions-count", className="mb-0"),
            ]), className="mb-3", style=CARD_STYLE), width=2),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Engine Status", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.Div(id="engine-status-badge"),
            ]), className="mb-3", style=CARD_STYLE), width=2),
        ], className="mt-3"),

        # ── Active strategy blend ─────────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.H6("Active Strategy Blend", className="text-muted mb-2"),
                    html.Div(id="active-strategies-panel"),
                ]), style=CARD_STYLE),
                width=12, className="mb-3",
            ),
        ]),

        # ── NAV chart + Positions table ───────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H6("NAV History (48h)", className="text-muted mb-2"),
                    dcc.Graph(id="nav-chart", figure=_empty_fig("NAV History"), style={"height": "320px"}),
                ]), style=CARD_STYLE),
            ], width=8),

            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H6("Current Positions", className="text-muted mb-2"),
                    dash_table.DataTable(
                        id="positions-table",
                        columns=positions_columns,
                        data=[],
                        style_header=TABLE_STYLE_HEADER,
                        style_data=TABLE_STYLE_DATA,
                        style_cell={"textAlign": "right", "padding": "4px 8px", "fontSize": "0.8rem"},
                        style_cell_conditional=[{"if": {"column_id": "Symbol"}, "textAlign": "left"}],
                        style_data_conditional=[
                            {
                                "if": {"filter_query": '{P&L $} > 0'},
                                "color": "#28a745",
                            },
                            {
                                "if": {"filter_query": '{P&L $} < 0'},
                                "color": "#dc3545",
                            },
                        ],
                        page_size=12,
                    ),
                ]), style=CARD_STYLE),
            ], width=4),
        ], className="mb-3"),

        # ── Recent trades ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H6("Recent Trades (last 20)", className="text-muted mb-2"),
                    dash_table.DataTable(
                        id="trades-table",
                        columns=trades_columns,
                        data=[],
                        style_header=TABLE_STYLE_HEADER,
                        style_data=TABLE_STYLE_DATA,
                        style_cell={"textAlign": "right", "padding": "4px 8px", "fontSize": "0.8rem"},
                        style_cell_conditional=[
                            {"if": {"column_id": "symbol"},   "textAlign": "left"},
                            {"if": {"column_id": "reason"},   "textAlign": "left"},
                            {"if": {"column_id": "strategy"}, "textAlign": "left"},
                        ],
                        style_data_conditional=[
                            {"if": {"filter_query": '{side} = "BUY"'},  "color": "#28a745"},
                            {"if": {"filter_query": '{side} = "SELL"'}, "color": "#dc3545"},
                        ],
                        page_size=20,
                    ),
                ]), style=CARD_STYLE),
            ], width=12),
        ]),
    ], fluid=True)


def _tab_risk_controls() -> dbc.Container:
    kill_switch_card = dbc.Card([
        dbc.CardHeader(html.H5("Kill Switch", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Span("Status: "),
                    html.Span(id="kill-switch-badge"),
                ], width=12, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Mode", className="text-muted"),
                    dbc.RadioItems(
                        id="kill-switch-mode",
                        options=[
                            {"label": "Halt New Orders",      "value": "halt"},
                            {"label": "Close All Positions",  "value": "close_all"},
                        ],
                        value="halt",
                        inline=True,
                    ),
                ], width=12, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Note", className="text-muted"),
                    dbc.Input(id="kill-switch-note", type="text", placeholder="Optional note…"),
                ], width=12, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Button("⚡ ACTIVATE",   id="kill-activate-btn",   color="danger",  className="me-2"),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button("✓ DEACTIVATE", id="kill-deactivate-btn", color="success"),
                    width="auto",
                ),
            ], className="mb-3"),
            html.Div(id="kill-switch-status-msg", className="text-info"),
        ]),
    ], id="kill-switch-card", className="mb-3", style=CARD_STYLE)

    override_inputs = []
    for sym in UNIVERSE:
        override_inputs.append(
            html.Tr([
                html.Td(sym, style={"color": "white", "paddingRight": "12px"}),
                html.Td(id=f"current-weight-{sym}", style={"color": "#aaa", "paddingRight": "12px"}),
                html.Td(
                    dbc.Input(
                        id=f"override-input-{sym}",
                        type="number",
                        min=0,
                        placeholder="—",
                        style={"width": "100px", "backgroundColor": "#1a1a2e", "color": "white"},
                    )
                ),
            ])
        )

    override_card = dbc.Card([
        dbc.CardHeader(html.H5("Position Override", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Span("Status: "),
                    html.Span(id="override-badge"),
                ], width=12, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Mode", className="text-muted"),
                    dbc.RadioItems(
                        id="override-mode",
                        options=[
                            {"label": "Target Weights (%)",  "value": "weights"},
                            {"label": "Direct Quantities",   "value": "quantities"},
                        ],
                        value="weights",
                        inline=True,
                    ),
                ], width=12, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Table(
                        [
                            html.Thead(html.Tr([
                                html.Th("Symbol",         style={"color": "#aaa", "paddingRight": "12px"}),
                                html.Th("Current Weight", style={"color": "#aaa", "paddingRight": "12px"}),
                                html.Th("Manual Input",   style={"color": "#aaa"}),
                            ])),
                            html.Tbody(override_inputs),
                        ],
                        style={"borderCollapse": "separate", "borderSpacing": "0 6px"},
                    ),
                ], width=12, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Button("APPLY OVERRIDE",   id="override-apply-btn",   color="warning",   className="me-2"),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button("RELEASE OVERRIDE", id="override-release-btn", color="secondary"),
                    width="auto",
                ),
            ], className="mb-3"),
            html.Div(id="override-status-msg", className="text-info"),
        ]),
    ], className="mb-3", style=CARD_STYLE)

    connection_card = dbc.Card([
        dbc.CardHeader(html.H5("Connection Status", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Span("Binance: ", className="text-muted"),
                    html.Span(id="binance-status-text"),
                    html.Br(),
                    html.Small(id="binance-status-detail", className="text-muted"),
                ], width=6),
                dbc.Col([
                    html.Span("Database: ", className="text-muted"),
                    html.Span(id="db-status-text"),
                    html.Br(),
                    html.Small(id="db-status-detail", className="text-muted"),
                ], width=6),
            ]),
        ]),
    ], className="mb-3", style=CARD_STYLE)

    return dbc.Container([
        dbc.Row([dbc.Col(connection_card, width=12)]),
        dbc.Row([dbc.Col(kill_switch_card, width=12)]),
        dbc.Row([dbc.Col(override_card,    width=12)]),
    ], fluid=True)


def _tab_strategy_performance() -> dbc.Container:
    summary_cols = [
        {"name": c, "id": c} for c in [
            "strategy", "live_nav", "live_return_pct", "live_trades",
            "sharpe", "sortino", "max_drawdown", "win_rate",
        ]
    ]
    return dbc.Container([
        # ── Comparative NAV chart (all strategies) ────────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.H6("All Strategies — Hypothetical NAV", className="text-muted mb-2"),
                    dcc.Graph(id="comparative-nav-chart", figure=_empty_fig(), style={"height": "360px"}),
                ]), style=CARD_STYLE),
                width=12, className="mb-3 mt-3",
            ),
        ]),

        # ── Strategy summary table ────────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.H6("Strategy Summary", className="text-muted mb-2"),
                    dash_table.DataTable(
                        id="strategy-summary-table",
                        columns=[
                            {"name": "Strategy",        "id": "strategy"},
                            {"name": "Live NAV ($)",    "id": "live_nav"},
                            {"name": "Live Return %",   "id": "live_return_pct"},
                            {"name": "Live Trades",     "id": "live_trades"},
                            {"name": "Sharpe",          "id": "sharpe"},
                            {"name": "Sortino",         "id": "sortino"},
                            {"name": "Max DD %",        "id": "max_drawdown"},
                            {"name": "Win Rate % (BT)", "id": "win_rate_pct"},
                        ],
                        data=[],
                        style_header=TABLE_STYLE_HEADER,
                        style_data=TABLE_STYLE_DATA,
                        style_cell={"textAlign": "right", "padding": "4px 8px", "fontSize": "0.8rem"},
                        style_cell_conditional=[{"if": {"column_id": "strategy"}, "textAlign": "left"}],
                        style_data_conditional=[
                            {"if": {"filter_query": "{live_return_pct} > 0"}, "color": "#28a745"},
                            {"if": {"filter_query": "{live_return_pct} < 0"}, "color": "#dc3545"},
                        ],
                        sort_action="native",
                        page_size=15,
                    ),
                ]), style=CARD_STYLE),
                width=12, className="mb-3",
            ),
        ]),

        # ── Per-strategy detail ───────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Label("Strategy Detail", className="text-muted me-2"),
                dcc.Dropdown(
                    id="strategy-dropdown",
                    options=[],
                    value=None,
                    clearable=False,
                    style={"backgroundColor": "#1a1a2e", "color": "black", "minWidth": "260px"},
                ),
            ], width=4, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.H6("Hypothetical NAV", className="text-muted mb-2"),
                    dcc.Graph(id="strategy-nav-chart", figure=_empty_fig(), style={"height": "300px"}),
                ]), style=CARD_STYLE),
                width=12, className="mb-3",
            ),
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.H6("Hypothetical Trades (last 50)", className="text-muted mb-2"),
                    dash_table.DataTable(
                        id="strategy-trades-table",
                        columns=[{"name": c, "id": c} for c in ["executed_at", "symbol", "side", "qty", "price"]],
                        data=[],
                        style_header=TABLE_STYLE_HEADER,
                        style_data=TABLE_STYLE_DATA,
                        style_cell={"textAlign": "right", "padding": "4px 8px", "fontSize": "0.8rem"},
                        style_cell_conditional=[{"if": {"column_id": "symbol"}, "textAlign": "left"}],
                        style_data_conditional=[
                            {"if": {"filter_query": '{side} = "BUY"'},  "color": "#28a745"},
                            {"if": {"filter_query": '{side} = "SELL"'}, "color": "#dc3545"},
                        ],
                        page_size=20,
                    ),
                ]), style=CARD_STYLE),
                width=12,
            ),
        ]),
    ], fluid=True)


def _tab_backtest_results() -> dbc.Container:
    metrics_cols = [
        "strategy", "sharpe", "sortino", "calmar", "cagr",
        "max_drawdown", "total_return", "win_rate", "profit_factor",
    ]
    return dbc.Container([
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.H6("Strategy Metrics", className="text-muted mb-2"),
                    dash_table.DataTable(
                        id="metrics-table",
                        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in metrics_cols],
                        data=[],
                        style_header=TABLE_STYLE_HEADER,
                        style_data=TABLE_STYLE_DATA,
                        style_cell={"textAlign": "right", "padding": "4px 8px", "fontSize": "0.8rem"},
                        style_cell_conditional=[{"if": {"column_id": "strategy"}, "textAlign": "left"}],
                        sort_action="native",
                        page_size=20,
                    ),
                ]), style=CARD_STYLE),
                width=12, className="mb-3 mt-3",
            ),
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.H6("Cumulative Returns by Strategy", className="text-muted mb-2"),
                    dcc.Graph(id="backtest-returns-chart", figure=_empty_fig(), style={"height": "380px"}),
                ]), style=CARD_STYLE),
                width=12,
            ),
        ]),
    ], fluid=True)


# ── Root layout ───────────────────────────────────────────────────────────────

app.layout = dbc.Container([
    # Global 30-second refresh interval
    dcc.Interval(id="refresh-interval", interval=30_000, n_intervals=0),

    # Header
    dbc.Row([
        dbc.Col(
            html.H3("⚡ Crypto Algo — Live Trading Dashboard", className="mb-0"),
            width="auto",
        ),
        dbc.Col(
            html.Small(id="last-updated-text", className="text-muted align-self-end"),
            width="auto",
            className="ms-auto",
        ),
    ], className="py-3 border-bottom border-secondary align-items-center"),

    # Kill switch alert banner (hidden unless active)
    dbc.Alert(
        "⚠️  KILL SWITCH ACTIVE — Engine is halted or closing positions.",
        id="kill-switch-banner",
        color="danger",
        is_open=False,
        dismissable=False,
        className="mt-2 mb-0",
    ),

    # Tabs
    dbc.Tabs([
        dbc.Tab(_tab_live_overview(),          label="Live Overview",        tab_id="tab-live"),
        dbc.Tab(_tab_risk_controls(),          label="Risk Controls",        tab_id="tab-risk"),
        dbc.Tab(_tab_strategy_performance(),   label="Strategy Performance", tab_id="tab-strategy"),
        dbc.Tab(_tab_backtest_results(),       label="Backtest Results",     tab_id="tab-backtest"),
    ], id="main-tabs", active_tab="tab-live", className="mt-2"),

], fluid=True, style={"backgroundColor": "#0d0d1a", "minHeight": "100vh"})


# ── Callback 1 — Live overview auto-refresh ───────────────────────────────────

@app.callback(
    Output("nav-value",               "children"),
    Output("cash-value",              "children"),
    Output("pnl-value",               "children"),
    Output("pnl-value",               "style"),
    Output("positions-count",         "children"),
    Output("engine-status-badge",     "children"),
    Output("nav-chart",               "figure"),
    Output("positions-table",         "data"),
    Output("positions-table",         "columns"),
    Output("trades-table",            "data"),
    Output("trades-table",            "columns"),
    Output("last-updated-text",       "children"),
    Output("kill-switch-banner",      "is_open"),
    Output("kill-switch-badge",       "children"),
    Output("kill-switch-badge",       "style"),
    Output("override-badge",          "children"),
    Output("override-badge",          "style"),
    Output("kill-switch-card",        "style"),
    Output("active-strategies-panel", "children"),
    # Current weights displayed next to override inputs
    *[Output(f"current-weight-{sym}", "children") for sym in UNIVERSE],
    Input("refresh-interval", "n_intervals"),
)
def update_live_tab(n_intervals):
    state    = get_live_state()
    nav_hist = get_nav_history(hours=48)
    trades   = get_trade_log(limit=20)
    controls = get_engine_controls()
    prices   = get_current_prices()
    metrics  = get_strategy_metrics()

    positions        = state.get("positions", {})
    cash_usdt        = state.get("cash_usdt", 0.0)
    initial_nav      = state.get("initial_nav", 0.0)
    current_weights  = state.get("current_weights", {})
    position_entries = state.get("position_entries", {})

    # Current NAV = positions market value + cash
    pos_value = sum(
        abs(qty) * prices.get(sym, 0.0)
        for sym, qty in positions.items()
    )
    current_nav = pos_value + cash_usdt
    pnl         = current_nav - initial_nav if initial_nav else 0.0
    pnl_color   = {"color": "#28a745"} if pnl >= 0 else {"color": "#dc3545"}

    open_pos_count = sum(1 for qty in positions.values() if qty != 0)
    kill_active    = controls.get("kill_switch", False)
    engine_badge   = dbc.Badge("HALTED", color="danger") if kill_active else dbc.Badge("LIVE", color="success")

    # NAV chart
    if not nav_hist.empty:
        fig_nav = go.Figure(go.Scatter(
            x=nav_hist["recorded_at"],
            y=nav_hist["nav"],
            mode="lines",
            fill="tozeroy",
            line={"color": "#00b4d8", "width": 1.5},
            fillcolor="rgba(0,180,216,0.15)",
            name="NAV",
        ))
        fig_nav.update_layout(title="NAV (48h)", showlegend=False, **CHART_LAYOUT)
    else:
        fig_nav = _empty_fig("NAV (48h)")

    # Positions table
    pos_cols = [
        {"name": c, "id": c}
        for c in ["Symbol", "Side", "Qty", "Entry $", "Current $", "P&L $", "P&L %", "SL $", "TP $"]
    ]
    pos_rows = []
    for sym, qty in positions.items():
        if qty == 0:
            continue
        side         = "LONG" if qty > 0 else "SHORT"
        entry_info   = position_entries.get(sym, {})
        entry_price  = entry_info.get("entry_price", 0.0)
        current_px   = prices.get(sym, 0.0)
        pnl_per_unit = (current_px - entry_price) * (1 if qty > 0 else -1)
        pnl_dollar   = pnl_per_unit * abs(qty)
        pnl_pct      = (pnl_per_unit / entry_price * 100) if entry_price else 0.0
        sl_price     = entry_price * (1 - STOP_LOSS_PCT)   if qty > 0 else entry_price * (1 + STOP_LOSS_PCT)
        tp_price     = entry_price * (1 + TAKE_PROFIT_PCT) if qty > 0 else entry_price * (1 - TAKE_PROFIT_PCT)
        pos_rows.append({
            "Symbol":    sym,
            "Side":      side,
            "Qty":       round(abs(qty), 6),
            "Entry $":   round(entry_price, 4),
            "Current $": round(current_px, 4),
            "P&L $":     round(pnl_dollar, 2),
            "P&L %":     round(pnl_pct, 2),
            "SL $":      round(sl_price, 4),
            "TP $":      round(tp_price, 4),
        })

    # Trades table
    if not trades.empty:
        trades_data = trades.to_dict("records")
        trades_cols = [{"name": c, "id": c} for c in trades.columns]
    else:
        trades_data = []
        trades_cols = [
            {"name": c, "id": c}
            for c in ["executed_at", "symbol", "side", "qty", "price", "reason", "strategy"]
        ]

    now_str = f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Kill switch badge
    ks_text  = "ACTIVE"   if kill_active else "INACTIVE"
    ks_color = {"color": "#dc3545", "fontWeight": "bold"} if kill_active else {"color": "#28a745", "fontWeight": "bold"}
    ks_card_style = CARD_DANGER_STYLE if kill_active else CARD_STYLE

    # Override badge
    ov_active = controls.get("override_active", False)
    ov_text   = "ACTIVE"   if ov_active else "INACTIVE"
    ov_color  = {"color": "#ffc107", "fontWeight": "bold"} if ov_active else {"color": "#6c757d", "fontWeight": "bold"}

    # Active strategies panel
    active_strats  = state.get("active_strategies", [])
    active_weights = state.get("active_strategy_weights", {})
    win_rates      = {}
    if not metrics.empty and "strategy" in metrics.columns and "win_rate" in metrics.columns:
        win_rates = dict(zip(metrics["strategy"], metrics["win_rate"]))

    if active_strats:
        import pandas as _pd
        rows = []
        for strat in active_strats:
            blend = active_weights.get(strat, 0) * 100
            wr    = win_rates.get(strat)
            wr_str  = f"{wr*100:.1f}%" if wr is not None and not _pd.isna(wr) else "—"
            wr_color = "#28a745" if (wr or 0) > 0.5 else "#ffc107"
            rows.append(html.Tr([
                html.Td(strat.replace("_", " ").title(),
                        style={"color": "white", "paddingRight": "24px"}),
                html.Td(f"{blend:.1f}%",
                        style={"color": "#00b4d8", "textAlign": "right", "paddingRight": "24px"}),
                html.Td(wr_str,
                        style={"color": wr_color, "textAlign": "right"}),
            ]))
        active_panel = html.Table([
            html.Thead(html.Tr([
                html.Th("Strategy",  style={"color": "#aaa", "paddingRight": "24px"}),
                html.Th("Blend",     style={"color": "#aaa", "textAlign": "right", "paddingRight": "24px"}),
                html.Th("Win Rate (BT)", style={"color": "#aaa", "textAlign": "right"}),
            ])),
            html.Tbody(rows),
        ])
    else:
        active_panel = html.Span("No active strategies yet", className="text-muted")

    # Current weights for override table
    weight_displays = []
    for sym in UNIVERSE:
        w = current_weights.get(sym, 0.0)
        weight_displays.append(f"{w*100:.1f}%")

    return (
        f"${current_nav:,.2f}",
        f"${cash_usdt:,.2f}",
        f"${pnl:+,.2f}",
        pnl_color,
        str(open_pos_count),
        engine_badge,
        fig_nav,
        pos_rows,
        pos_cols,
        trades_data,
        trades_cols,
        now_str,
        kill_active,
        ks_text,
        ks_color,
        ov_text,
        ov_color,
        ks_card_style,
        active_panel,
        *weight_displays,
    )


# ── Callback 2 — Activate kill switch ────────────────────────────────────────

@app.callback(
    Output("kill-switch-status-msg", "children"),
    Input("kill-activate-btn",  "n_clicks"),
    State("kill-switch-mode",   "value"),
    State("kill-switch-note",   "value"),
    prevent_initial_call=True,
)
def activate_kill_switch(n_clicks, mode_value, note_value):
    if not n_clicks:
        return ""
    try:
        note = note_value or ""
        if mode_value == "close_all":
            save_engine_controls(
                kill_switch=True,
                kill_mode="halt",
                close_all_triggered=True,
                note=note,
            )
            return "Kill switch ACTIVATED — Close All triggered."
        else:
            save_engine_controls(
                kill_switch=True,
                kill_mode="halt",
                close_all_triggered=False,
                note=note,
            )
            return "Kill switch ACTIVATED — Halt mode."
    except Exception as e:
        return f"Error: {e}"


# ── Callback 3 — Deactivate kill switch ──────────────────────────────────────

@app.callback(
    Output("kill-switch-status-msg", "children", allow_duplicate=True),
    Input("kill-deactivate-btn", "n_clicks"),
    prevent_initial_call=True,
)
def deactivate_kill_switch(n_clicks):
    if not n_clicks:
        return ""
    try:
        save_engine_controls(kill_switch=False, close_all_triggered=False)
        return "Kill switch DEACTIVATED — engine resumed."
    except Exception as e:
        return f"Error: {e}"


# ── Callback 4 — Apply position override ─────────────────────────────────────

@app.callback(
    Output("override-status-msg", "children"),
    Input("override-apply-btn", "n_clicks"),
    State("override-mode", "value"),
    *[State(f"override-input-{sym}", "value") for sym in UNIVERSE],
    prevent_initial_call=True,
)
def apply_override(n_clicks, mode_value, *input_values):
    if not n_clicks:
        return ""
    try:
        if mode_value == "weights":
            manual_weights = {
                sym: float(val) / 100.0
                for sym, val in zip(UNIVERSE, input_values)
                if val is not None
            }
            save_engine_controls(
                override_active=True,
                override_mode="weights",
                manual_weights=manual_weights,
                manual_quantities={},
            )
            return f"Override APPLIED — weights mode ({len(manual_weights)} symbols)."
        else:
            manual_quantities = {
                sym: float(val)
                for sym, val in zip(UNIVERSE, input_values)
                if val is not None and float(val) > 0
            }
            save_engine_controls(
                override_active=True,
                override_mode="quantities",
                manual_weights={},
                manual_quantities=manual_quantities,
            )
            return f"Override APPLIED — quantities mode ({len(manual_quantities)} symbols)."
    except Exception as e:
        return f"Error: {e}"


# ── Callback 5 — Release position override ───────────────────────────────────

@app.callback(
    Output("override-status-msg", "children", allow_duplicate=True),
    Input("override-release-btn", "n_clicks"),
    prevent_initial_call=True,
)
def release_override(n_clicks):
    if not n_clicks:
        return ""
    try:
        save_engine_controls(override_active=False)
        return "Override RELEASED — engine using optimizer weights."
    except Exception as e:
        return f"Error: {e}"


# ── Callback 6 — Strategy performance tab ────────────────────────────────────

@app.callback(
    Output("strategy-nav-chart",    "figure"),
    Output("strategy-trades-table", "data"),
    Output("strategy-trades-table", "columns"),
    Input("strategy-dropdown", "value"),
    prevent_initial_call=True,
)
def update_strategy_tab(strategy_value):
    empty_cols = [{"name": c, "id": c} for c in ["executed_at", "symbol", "side", "qty", "price"]]
    if not strategy_value:
        return _empty_fig("Select a strategy"), [], empty_cols

    nav_df    = get_hypothetical_nav(strategies=[strategy_value])
    trades_df = get_hypothetical_trades(strategy=strategy_value, limit=50)

    if not nav_df.empty:
        fig = go.Figure(go.Scatter(
            x=nav_df["recorded_at"],
            y=nav_df["nav"],
            mode="lines",
            fill="tozeroy",
            line={"color": "#f4a261", "width": 1.5},
            fillcolor="rgba(244,162,97,0.15)",
            name=strategy_value,
        ))
        fig.update_layout(title=f"Hypothetical NAV — {strategy_value}", showlegend=False, **CHART_LAYOUT)
    else:
        fig = _empty_fig(f"No data for {strategy_value}")

    if not trades_df.empty:
        trades_data = trades_df.to_dict("records")
        trades_cols = [{"name": c, "id": c} for c in trades_df.columns]
    else:
        trades_data = []
        trades_cols = empty_cols

    return fig, trades_data, trades_cols


# ── Callback 7 — Populate strategy dropdown ───────────────────────────────────

@app.callback(
    Output("strategy-dropdown", "options"),
    Output("strategy-dropdown", "value"),
    Input("refresh-interval", "n_intervals"),
)
def populate_strategy_dropdown(n_intervals):
    strategies = get_all_hyp_strategies()
    options    = [{"label": s, "value": s} for s in strategies]
    value      = strategies[0] if strategies else None
    return options, value


# ── Callback 8 — Backtest results tab ────────────────────────────────────────

@app.callback(
    Output("metrics-table",         "data"),
    Output("metrics-table",         "columns"),
    Output("backtest-returns-chart","figure"),
    Input("refresh-interval", "n_intervals"),
)
def update_backtest_tab(n_intervals):
    metrics_df = get_strategy_metrics()
    returns_df = get_portfolio_returns()

    display_cols = [
        "strategy", "sharpe", "sortino", "calmar", "cagr",
        "max_drawdown", "total_return", "win_rate", "profit_factor",
    ]

    if not metrics_df.empty:
        pct_cols = {"win_rate", "max_drawdown", "total_return", "cagr"}
        for col in display_cols[1:]:
            if col not in metrics_df.columns:
                continue
            if col in pct_cols:
                metrics_df[col] = (metrics_df[col] * 100).round(2)
            else:
                metrics_df[col] = metrics_df[col].round(3)
        available_cols = [c for c in display_cols if c in metrics_df.columns]
        col_labels = {
            "strategy": "Strategy", "sharpe": "Sharpe", "sortino": "Sortino",
            "calmar": "Calmar", "cagr": "CAGR %", "max_drawdown": "Max DD %",
            "total_return": "Total Return %", "win_rate": "Win Rate %",
            "profit_factor": "Profit Factor",
        }
        metrics_data = metrics_df[available_cols].to_dict("records")
        metrics_cols = [{"name": col_labels.get(c, c), "id": c} for c in available_cols]
    else:
        metrics_data = []
        metrics_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in display_cols]

    # Cumulative returns chart
    fig = go.Figure()
    if not returns_df.empty:
        # Expect columns: date (or index) + one column per strategy
        date_col = None
        for candidate in ["date", "Date", "recorded_at", "index"]:
            if candidate in returns_df.columns:
                date_col = candidate
                break
        if date_col is None and returns_df.index.name:
            returns_df = returns_df.reset_index()
            date_col = returns_df.columns[0]
        elif date_col is None:
            returns_df = returns_df.reset_index()
            date_col = returns_df.columns[0]

        strategy_cols = [c for c in returns_df.columns if c != date_col]
        colours = [
            "#00b4d8", "#f4a261", "#2ec4b6", "#e71d36", "#ff9f1c",
            "#6a0572", "#48cae4", "#d62828", "#023e8a", "#80b918",
            "#f72585", "#4361ee",
        ]
        for i, col in enumerate(strategy_cols):
            cumret = (1 + returns_df[col].fillna(0)).cumprod()
            fig.add_trace(go.Scatter(
                x=returns_df[date_col],
                y=cumret,
                mode="lines",
                name=col,
                line={"color": colours[i % len(colours)], "width": 1.5},
            ))
        fig.update_layout(title="Cumulative Returns", **CHART_LAYOUT)
    else:
        fig = _empty_fig("Cumulative Returns (no data)")

    return metrics_data, metrics_cols, fig


# ── Callback 9 — Connection status ───────────────────────────────────────────

@app.callback(
    Output("binance-status-text",  "children"),
    Output("binance-status-text",  "style"),
    Output("binance-status-detail","children"),
    Output("db-status-text",       "children"),
    Output("db-status-text",       "style"),
    Output("db-status-detail",     "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_connection_status(n_intervals):
    s = get_connection_status()
    ok_style  = {"color": "#28a745", "fontWeight": "bold"}
    err_style = {"color": "#dc3545", "fontWeight": "bold"}

    b = s["binance"]
    d = s["db"]
    return (
        "CONNECTED"    if b["ok"] else "DISCONNECTED",
        ok_style       if b["ok"] else err_style,
        b["detail"],
        "CONNECTED"    if d["ok"] else "DISCONNECTED",
        ok_style       if d["ok"] else err_style,
        d["detail"],
    )


# ── Callback 10 — Strategy performance overview ───────────────────────────────

STRAT_COLOURS = [
    "#00b4d8", "#f4a261", "#2ec4b6", "#e71d36", "#ff9f1c",
    "#6a0572", "#48cae4", "#d62828", "#023e8a", "#80b918",
    "#f72585", "#4361ee",
]

@app.callback(
    Output("comparative-nav-chart",  "figure"),
    Output("strategy-summary-table", "data"),
    Input("refresh-interval", "n_intervals"),
)
def update_strategy_overview(n_intervals):
    nav_df   = get_hypothetical_nav()
    summary  = get_strategy_summary()

    # Comparative NAV chart
    fig = go.Figure()
    if not nav_df.empty:
        strategies = nav_df["strategy"].unique()
        for i, strat in enumerate(strategies):
            s = nav_df[nav_df["strategy"] == strat]
            fig.add_trace(go.Scatter(
                x=s["recorded_at"],
                y=s["nav"],
                mode="lines",
                name=strat.replace("_", " ").title(),
                line={"color": STRAT_COLOURS[i % len(STRAT_COLOURS)], "width": 1.5},
            ))
    fig.update_layout(
        title="Hypothetical NAV — All Strategies",
        legend={"orientation": "h", "y": -0.15, "font": {"size": 10}},
        **CHART_LAYOUT,
    )

    # Summary table
    summary_data = summary.to_dict("records") if not summary.empty else []

    return fig, summary_data


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
