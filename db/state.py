"""
db/state.py — PostgreSQL-backed replacement for live_state.json (crypto_algo).

load_state() returns a dict with the same shape the rest of the engine expects.
save_state() upserts core fields and appends any new nav/trade rows.

Offset tracking: _nav_db_count and _trade_db_count are stored inside the state
dict itself (prefixed _) so each job invocation tracks its own baseline
independently — no shared mutable globals between APScheduler threads.
"""

import json
from contextlib import contextmanager
from loguru import logger

from db.connection import get_conn, put_conn
from config.settings import UNIVERSE, PORTFOLIO_USDT


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


def load_state() -> dict | None:
    """
    Load state from PostgreSQL. Returns None when the DB has no live_state row
    yet (first ever run) so live_engine.py can fall back to Binance bootstrap.
    """
    try:
        with _db() as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT positions, cash_usdt, initial_nav, current_weights, last_run,
                       active_strategies, active_strategy_weights
                FROM live_state WHERE id = 1
            """)
            row = cur.fetchone()
            if row is None:
                return None  # signal to caller: no persisted state yet

            positions, cash_usdt, initial_nav, current_weights, last_run, \
                active_strategies, active_strategy_weights = row
            state: dict = {
                "positions":               positions or {sym: 0.0 for sym in UNIVERSE},
                "cash_usdt":               float(cash_usdt or 0),
                "initial_nav":             float(initial_nav) if initial_nav is not None else None,
                "current_weights":         current_weights or {},
                "last_run":                str(last_run) if last_run else None,
                "active_strategies":       active_strategies if isinstance(active_strategies, list) else [],
                "active_strategy_weights": active_strategy_weights if isinstance(active_strategy_weights, dict) else {},
                "position_entries": {},
                "nav_history":      [],
                "trade_log":        [],
                "hypothetical":     {},
                "latest_signals":   {},
                "signals_updated_at": None,
            }

            # Position entries
            cur.execute("""
                SELECT symbol, entry_price, entry_date, peak_price
                FROM position_entries
            """)
            for sym, ep, ed, pp in cur.fetchall():
                state["position_entries"][sym] = {
                    "entry_price": float(ep),
                    "entry_date":  str(ed),
                    "peak_price":  float(pp),
                }

            # Recent nav_history (last 2880 — 2 days at 1-min cadence)
            cur.execute("""
                SELECT recorded_at, nav, event
                FROM nav_history
                ORDER BY recorded_at DESC LIMIT 2880
            """)
            rows = cur.fetchall()
            state["nav_history"] = [
                {"date": str(r[0]), "nav": float(r[1]), "event": r[2]}
                for r in reversed(rows)
            ]

            # Recent trade_log (last 500)
            cur.execute("""
                SELECT executed_at, symbol, side, qty, price, reason, order_id
                FROM trade_log
                ORDER BY executed_at DESC LIMIT 500
            """)
            rows = cur.fetchall()
            state["trade_log"] = [
                {
                    "time":     str(r[0]),
                    "symbol":   r[1],
                    "side":     r[2],
                    "qty":      float(r[3]),
                    "price":    float(r[4]),
                    "reason":   r[5],
                    "order_id": r[6],
                }
                for r in reversed(rows)
            ]

            # Watermarks for incremental saves
            state["_nav_db_count"]   = len(state["nav_history"])
            state["_trade_db_count"] = len(state["trade_log"])

            # Hypothetical portfolios — current nav + recent history
            cur.execute("SELECT DISTINCT strategy FROM hypothetical_nav")
            strategies = [r[0] for r in cur.fetchall()]

            hyp_counts: dict = {}
            for strat in strategies:
                cur.execute("""
                    SELECT nav FROM hypothetical_nav
                    WHERE strategy = %s ORDER BY recorded_at DESC LIMIT 1
                """, (strat,))
                nav_row = cur.fetchone()

                cur.execute("""
                    SELECT recorded_at, nav FROM hypothetical_nav
                    WHERE strategy = %s
                    ORDER BY recorded_at DESC LIMIT 2880
                """, (strat,))
                nav_hist = [
                    {"date": str(r[0]), "nav": float(r[1])}
                    for r in reversed(cur.fetchall())
                ]

                cur.execute("""
                    SELECT executed_at, symbol, side, qty, price
                    FROM hypothetical_trades
                    WHERE strategy = %s
                    ORDER BY executed_at DESC LIMIT 500
                """, (strat,))
                trade_hist = [
                    {
                        "time":   str(r[0]),
                        "symbol": r[1],
                        "side":   r[2],
                        "qty":    float(r[3]),
                        "price":  float(r[4]),
                    }
                    for r in reversed(cur.fetchall())
                ]

                state["hypothetical"][strat] = {
                    "nav":           float(nav_row[0]) if nav_row else PORTFOLIO_USDT,
                    "nav_history":   nav_hist,
                    "trade_history": trade_hist,
                    "weights":       {},
                    "last_prices":   {},
                    "entry_prices":  {},
                }
                hyp_counts[strat] = {
                    "nav":    len(nav_hist),
                    "trades": len(trade_hist),
                }

            state["_hyp_counts"] = hyp_counts

        return state

    except Exception as e:
        logger.error(f"DB load_state failed: {e}")
        return None


def save_state(state: dict):
    """Persist live trading state to PostgreSQL."""
    try:
        with _db() as conn:
            cur = conn.cursor()

            # Upsert core live_state row
            cur.execute("""
                INSERT INTO live_state
                    (id, positions, cash_usdt, initial_nav, current_weights, last_run,
                     active_strategies, active_strategy_weights)
                VALUES (1, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    positions               = EXCLUDED.positions,
                    cash_usdt               = EXCLUDED.cash_usdt,
                    initial_nav             = EXCLUDED.initial_nav,
                    current_weights         = EXCLUDED.current_weights,
                    last_run                = EXCLUDED.last_run,
                    active_strategies       = EXCLUDED.active_strategies,
                    active_strategy_weights = EXCLUDED.active_strategy_weights
            """, (
                json.dumps(state.get("positions", {})),
                state.get("cash_usdt", 0),
                state.get("initial_nav"),
                json.dumps(state.get("current_weights", {})),
                state.get("last_run"),
                json.dumps(state.get("active_strategies", [])),
                json.dumps(state.get("active_strategy_weights", {})),
            ))

            # Sync position_entries — delete and reinsert (small table, always current)
            cur.execute("DELETE FROM position_entries")
            for sym, entry in state.get("position_entries", {}).items():
                cur.execute("""
                    INSERT INTO position_entries (symbol, entry_price, entry_date, peak_price)
                    VALUES (%s, %s, %s, %s)
                """, (
                    sym,
                    entry["entry_price"],
                    entry["entry_date"],
                    entry["peak_price"],
                ))

            # Append only new nav_history entries (beyond the load-time watermark)
            nav_offset = state.get("_nav_db_count", 0)
            for row in state.get("nav_history", [])[nav_offset:]:
                cur.execute("""
                    INSERT INTO nav_history (recorded_at, nav, event)
                    VALUES (%s, %s, %s)
                """, (row.get("date"), row.get("nav"), row.get("event")))

            # Append only new trade_log entries
            trade_offset = state.get("_trade_db_count", 0)
            for row in state.get("trade_log", [])[trade_offset:]:
                cur.execute("""
                    INSERT INTO trade_log
                        (executed_at, symbol, side, qty, price, reason, order_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    row.get("time"),
                    row.get("symbol"),
                    row.get("side"),
                    row.get("qty"),
                    row.get("price"),
                    row.get("reason"),
                    row.get("order_id"),
                ))

            # Hypothetical portfolios — append only NEW nav and trade entries per strategy.
            # CRITICAL: update _hyp_counts after each write so subsequent save_state()
            # calls don't re-insert from offset 0 and duplicate all rows.
            hyp_counts = state.setdefault("_hyp_counts", {})
            for name, hyp in state.get("hypothetical", {}).items():
                nav_list     = hyp.get("nav_history", [])
                trade_list   = hyp.get("trade_history", [])
                nav_offset   = hyp_counts.get(name, {}).get("nav", 0)
                trade_offset = hyp_counts.get(name, {}).get("trades", 0)

                for row in nav_list[nav_offset:]:
                    cur.execute("""
                        INSERT INTO hypothetical_nav (strategy, recorded_at, nav)
                        VALUES (%s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (name, row.get("date"), row.get("nav")))

                for row in trade_list[trade_offset:]:
                    price = row.get("price") or 0.0
                    qty   = row.get("qty")
                    if qty is None:
                        hypo_val = row.get("hypo_value") or 0.0
                        qty = round(hypo_val / price, 6) if price > 0 else 0.0
                    cur.execute("""
                        INSERT INTO hypothetical_trades
                            (strategy, executed_at, symbol, side, qty, price)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        name,
                        row.get("time"),
                        row.get("symbol"),
                        row.get("side"),
                        qty,
                        price,
                    ))

                # Advance watermarks so next call only writes truly new rows
                hyp_counts[name] = {
                    "nav":    len(nav_list),
                    "trades": len(trade_list),
                }

    except Exception as e:
        logger.error(f"DB save_state failed: {e}")
        raise


def load_nav_history_for_report() -> list[dict]:
    """Used by _generate_final_reports() to get full nav history from DB."""
    with _db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT recorded_at, nav, event FROM nav_history ORDER BY recorded_at")
        return [{"date": str(r[0]), "nav": float(r[1]), "event": r[2]} for r in cur.fetchall()]


def load_trade_log_for_report() -> list[dict]:
    """Used by _generate_final_reports() to get full trade log from DB."""
    with _db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT executed_at, symbol, side, qty, price, reason, order_id
            FROM trade_log ORDER BY executed_at
        """)
        return [
            {
                "time":     str(r[0]),
                "symbol":   r[1],
                "side":     r[2],
                "qty":      float(r[3]),
                "price":    float(r[4]),
                "reason":   r[5],
                "order_id": r[6],
            }
            for r in cur.fetchall()
        ]
