"""
dashboard/data.py — Data access helpers for the Dash trading dashboard.

All public functions return empty DataFrames / dicts on failure so the UI
degrades gracefully when the DB is unreachable or tables are empty.
"""

import sys
from pathlib import Path

# Ensure the project root (crypto_algo/) is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from contextlib import contextmanager

import pandas as pd
from loguru import logger

from db.connection import get_conn, put_conn
from db.controls import load_controls, save_controls
from config.settings import UNIVERSE, RESULT_DIR


# ── DB context manager ────────────────────────────────────────────────────────

@contextmanager
def _db():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        put_conn(conn)


# ── Binance price fetch ───────────────────────────────────────────────────────

def fetch_current_prices() -> dict:
    """
    Fetch the latest mark prices for all symbols in UNIVERSE from Binance.

    Inlined here (not imported from live_engine) to avoid pulling in
    APScheduler and the full live-engine module at import time.
    """
    from config.client import get_client
    try:
        client = get_client(for_trading=False)
        return {
            t["symbol"]: float(t["price"])
            for t in client.futures_symbol_ticker()
            if t["symbol"] in UNIVERSE
        }
    except Exception:
        return {}


# ── Live state ────────────────────────────────────────────────────────────────

def get_live_state() -> dict:
    """
    Return the current live trading state as a flat dict.

    Keys: positions (dict), cash_usdt (float), initial_nav (float),
          current_weights (dict), last_run (str | None),
          position_entries (dict keyed by symbol).
    """
    default = {
        "positions":               {},
        "cash_usdt":               0.0,
        "initial_nav":             0.0,
        "current_weights":         {},
        "last_run":                None,
        "position_entries":        {},
        "active_strategies":       [],
        "active_strategy_weights": {},
    }
    try:
        with _db() as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT positions, cash_usdt, initial_nav, current_weights, last_run,
                       active_strategies, active_strategy_weights
                FROM live_state
                WHERE id = 1
            """)
            row = cur.fetchone()
            if row is None:
                return default

            positions, cash_usdt, initial_nav, current_weights, last_run, \
                active_strategies, active_strategy_weights = row

            state = {
                "positions":               positions if isinstance(positions, dict) else {},
                "cash_usdt":               float(cash_usdt) if cash_usdt is not None else 0.0,
                "initial_nav":             float(initial_nav) if initial_nav is not None else 0.0,
                "current_weights":         current_weights if isinstance(current_weights, dict) else {},
                "last_run":                str(last_run) if last_run is not None else None,
                "position_entries":        {},
                "active_strategies":       active_strategies if isinstance(active_strategies, list) else [],
                "active_strategy_weights": active_strategy_weights if isinstance(active_strategy_weights, dict) else {},
            }

            cur.execute("""
                SELECT symbol, entry_price, entry_date, peak_price
                FROM position_entries
            """)
            for sym, ep, ed, pp in cur.fetchall():
                state["position_entries"][sym] = {
                    "entry_price": float(ep),
                    "entry_date":  str(ed),
                    "peak_price":  float(pp) if pp is not None else float(ep),
                }

            return state

    except Exception as e:
        logger.error(f"get_live_state failed: {e}")
        return default


# ── NAV history ───────────────────────────────────────────────────────────────

def get_nav_history(hours: int = 48) -> pd.DataFrame:
    """
    Return NAV history for the last `hours` hours.
    Columns: recorded_at, nav, event.
    """
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT recorded_at, nav, event
                FROM nav_history
                WHERE recorded_at >= NOW() - INTERVAL '%s hours'
                ORDER BY recorded_at ASC
            """, (hours,))
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame(columns=["recorded_at", "nav", "event"])
            df = pd.DataFrame(rows, columns=["recorded_at", "nav", "event"])
            df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
            return df
    except Exception as e:
        logger.error(f"get_nav_history failed: {e}")
        return pd.DataFrame(columns=["recorded_at", "nav", "event"])


# ── Trade log ─────────────────────────────────────────────────────────────────

def get_trade_log(limit: int = 50) -> pd.DataFrame:
    """
    Return the `limit` most recent trade log entries.
    Columns: executed_at, symbol, side, qty, price, reason.
    """
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT executed_at, symbol, side, qty, price, reason
                FROM trade_log
                ORDER BY executed_at DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame(columns=["executed_at", "symbol", "side", "qty", "price", "reason"])
            df = pd.DataFrame(rows, columns=["executed_at", "symbol", "side", "qty", "price", "reason"])
            df["qty"]   = pd.to_numeric(df["qty"],   errors="coerce")
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            return df
    except Exception as e:
        logger.error(f"get_trade_log failed: {e}")
        return pd.DataFrame(columns=["executed_at", "symbol", "side", "qty", "price", "reason"])


# ── Strategy metrics ──────────────────────────────────────────────────────────

def get_strategy_metrics() -> pd.DataFrame:
    """
    Return all rows from strategy_metrics.
    Columns: strategy, sharpe, sortino, calmar, cagr, max_drawdown,
             total_return, win_rate, profit_factor, recorded_at.
    """
    cols = [
        "strategy", "sharpe", "sortino", "calmar", "cagr",
        "max_drawdown", "total_return", "win_rate", "profit_factor", "recorded_at",
    ]
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT strategy, sharpe, sortino, calmar, cagr,
                       max_drawdown, total_return, win_rate, profit_factor, recorded_at
                FROM strategy_metrics
                ORDER BY sharpe DESC NULLS LAST
            """)
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame(columns=cols)
            df = pd.DataFrame(rows, columns=cols)
            for c in ["sharpe", "sortino", "calmar", "cagr", "max_drawdown",
                      "total_return", "win_rate", "profit_factor"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
    except Exception as e:
        logger.error(f"get_strategy_metrics failed: {e}")
        return pd.DataFrame(columns=cols)


# ── Hypothetical strategies ───────────────────────────────────────────────────

def get_all_hyp_strategies() -> list:
    """Return a sorted list of distinct strategy names from hypothetical_nav."""
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT strategy FROM hypothetical_nav ORDER BY strategy")
            return [r[0] for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"get_all_hyp_strategies failed: {e}")
        return []


def get_hypothetical_nav(strategies: list | None = None) -> pd.DataFrame:
    """
    Return hypothetical NAV history.
    Columns: recorded_at, nav, strategy.
    Optionally filtered to `strategies` list.
    """
    cols = ["recorded_at", "nav", "strategy"]
    try:
        with _db() as conn:
            cur = conn.cursor()
            if strategies:
                placeholders = ",".join(["%s"] * len(strategies))
                cur.execute(
                    f"""
                    SELECT recorded_at, nav, strategy
                    FROM hypothetical_nav
                    WHERE strategy IN ({placeholders})
                    ORDER BY recorded_at ASC
                    """,
                    strategies,
                )
            else:
                cur.execute("""
                    SELECT recorded_at, nav, strategy
                    FROM hypothetical_nav
                    ORDER BY recorded_at ASC
                """)
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame(columns=cols)
            df = pd.DataFrame(rows, columns=cols)
            df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
            return df
    except Exception as e:
        logger.error(f"get_hypothetical_nav failed: {e}")
        return pd.DataFrame(columns=cols)


def get_hypothetical_trades(strategy: str, limit: int = 50) -> pd.DataFrame:
    """
    Return the `limit` most recent hypothetical trades for `strategy`.
    Columns: executed_at, symbol, side, qty, price.
    """
    cols = ["executed_at", "symbol", "side", "qty", "price"]
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT executed_at, symbol, side, qty, price
                FROM hypothetical_trades
                WHERE strategy = %s
                ORDER BY executed_at DESC
                LIMIT %s
            """, (strategy, limit))
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame(columns=cols)
            df = pd.DataFrame(rows, columns=cols)
            df["qty"]   = pd.to_numeric(df["qty"],   errors="coerce")
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            return df
    except Exception as e:
        logger.error(f"get_hypothetical_trades failed: {e}")
        return pd.DataFrame(columns=cols)


# ── Strategy summary (live hypothetical + backtest metrics) ──────────────────

def get_strategy_summary() -> pd.DataFrame:
    """
    Merge live hypothetical performance with backtest metrics.

    Columns: strategy, live_nav, live_return_pct, live_win_rate,
             sharpe, sortino, max_drawdown, win_rate (backtest).
    """
    from config.settings import PORTFOLIO_USDT
    cols = ["strategy", "live_nav", "live_return_pct", "live_trades",
            "sharpe", "sortino", "max_drawdown", "win_rate_pct"]
    try:
        with _db() as conn:
            cur = conn.cursor()

            # Latest NAV per strategy — DISTINCT ON is faster and correct
            cur.execute("""
                SELECT DISTINCT ON (strategy) strategy, nav
                FROM hypothetical_nav
                ORDER BY strategy, recorded_at DESC
            """)
            nav_rows = {r[0]: float(r[1]) for r in cur.fetchall()}

            # Trade counts per strategy
            cur.execute("""
                SELECT strategy, COUNT(*) AS total_trades
                FROM hypothetical_trades
                GROUP BY strategy
            """)
            trade_counts = {r[0]: int(r[1]) for r in cur.fetchall()}

            # Backtest metrics — only show strategies that have live hypothetical data
            cur.execute("""
                SELECT strategy, sharpe, sortino, max_drawdown, win_rate
                FROM strategy_metrics
            """)
            bt_rows = {r[0]: r[1:] for r in cur.fetchall()}

            # Only include strategies that have live hypothetical NAV data
            strategies = sorted(nav_rows.keys())
            records = []
            for strat in strategies:
                nav = nav_rows[strat]
                ret_pct = (nav - PORTFOLIO_USDT) / PORTFOLIO_USDT * 100
                bt = bt_rows.get(strat, (None, None, None, None))
                wr_raw = bt[3]
                records.append({
                    "strategy":        strat,
                    "live_nav":        round(nav, 4),
                    "live_return_pct": round(ret_pct, 4),
                    "live_trades":     trade_counts.get(strat, 0),
                    "sharpe":          round(float(bt[0]), 3) if bt[0] is not None else None,
                    "sortino":         round(float(bt[1]), 3) if bt[1] is not None else None,
                    "max_drawdown":    round(float(bt[2]) * 100, 2) if bt[2] is not None else None,
                    "win_rate_pct":    round(float(wr_raw) * 100, 1) if wr_raw is not None else None,
                })

            return pd.DataFrame(records, columns=cols)

    except Exception as e:
        logger.error(f"get_strategy_summary failed: {e}")
        return pd.DataFrame(columns=cols)


# ── Connection health ─────────────────────────────────────────────────────────

def get_connection_status() -> dict:
    """
    Returns DB and Binance connectivity status.
    Each value is a dict with keys: ok (bool), detail (str).
    """
    status = {"db": {"ok": False, "detail": ""}, "binance": {"ok": False, "detail": ""}}

    # DB check
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT updated_at FROM engine_controls WHERE id = 1")
            row = cur.fetchone()
            last = str(row[0]) if row and row[0] else "never"
            status["db"] = {"ok": True, "detail": f"Connected — last control update: {last}"}
    except Exception as e:
        status["db"] = {"ok": False, "detail": str(e)[:120]}

    # Binance check — ping real endpoint for connectivity, demo client for auth
    # (PAPER_TRADING=True: engine uses demo/testnet; real_client has no secret)
    try:
        from config.client import get_client, real_client, demo_client
        real_client.ping()   # unauthenticated — confirms network reachability
        # Auth check via demo client (testnet) — same creds the engine uses
        balances = demo_client.futures_account_balance()
        usdt = next((float(b["balance"]) for b in balances if b["asset"] == "USDT"), 0.0)
        status["binance"] = {"ok": True, "detail": f"Testnet connected — USDT balance: ${usdt:,.2f}"}
    except Exception as e:
        status["binance"] = {"ok": False, "detail": str(e)[:120]}

    return status


# ── Engine controls ───────────────────────────────────────────────────────────

def get_engine_controls() -> dict:
    """Delegate to db.controls.load_controls()."""
    return load_controls()


def save_engine_controls(**kwargs) -> None:
    """Delegate to db.controls.save_controls() with arbitrary kwargs."""
    save_controls(**kwargs)


# ── Current prices ────────────────────────────────────────────────────────────

def get_current_prices() -> dict:
    """Fetch live mark prices from Binance for all UNIVERSE symbols."""
    return fetch_current_prices()


# ── Portfolio returns CSV ─────────────────────────────────────────────────────

def get_portfolio_returns() -> pd.DataFrame:
    """
    Load portfolio_returns.csv from RESULT_DIR if it exists.
    Returns an empty DataFrame if the file is missing or unreadable.
    """
    csv_path = Path(RESULT_DIR) / "portfolio_returns.csv"
    try:
        if not csv_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.error(f"get_portfolio_returns failed: {e}")
        return pd.DataFrame()
