"""
execution/live_engine.py
========================
Real-time continuous trading engine.

Two concurrent loops
--------------------
 FAST  (every PRICE_MONITOR_SECS, default 60s)
   → Fetch current prices for all positions
   → Check stop loss / trailing stop / take profit
   → Execute immediate exit orders if triggered

 SIGNAL (every SIGNAL_RECOMPUTE_MINS, default 60min)
   → Fetch fresh OHLCV data (no cache)
   → Recompute all strategy signals on latest data
   → Compute new target portfolio weights
   → Compare to last-known weights (stored in state)
   → Execute rebalance ONLY if total weight delta > REBALANCE_THRESHOLD
   → Log reason and skip if signal unchanged

This means the engine trades whenever the signal shifts — not on a fixed clock.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler

from config.client import get_client
from config.settings import (
    UNIVERSE, PORTFOLIO_USDT, MIN_ORDER_USDT,
    PAPER_TRADING, LOG_DIR, RESULT_DIR,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT, USE_TRAILING_STOP,
    SIGNAL_RECOMPUTE_MINS, PRICE_MONITOR_SECS, REBALANCE_THRESHOLD,
    MAX_LIVE_POSITIONS, MAX_POSITION_SIZE,
)
from data.ingestion import get_universe_ohlcv, build_close_matrix, build_return_matrix
from data.ingestion import get_fear_greed_index, get_universe_funding_rates
from seasonality.analyzer import blend_signals
from utils.logger import setup_logger


STATE_FILE      = RESULT_DIR / "live_state.json"
_last_heartbeat = 0.0   # tracks when the monitor last printed a full portfolio snapshot


# ── State management ───────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "positions":        {sym: 0.0 for sym in UNIVERSE},
        "cash_usdt":        PORTFOLIO_USDT,
        "last_run":         None,
        "nav_history":      [],
        "current_weights":  {},
        "position_entries": {},
        "trade_log":        [],
        "hypothetical":     {},
        "active_strategies": [],
    }


_NAV_HISTORY_CAP    = 2880   # keep last 2 days at 1-min cadence
_TRADE_HISTORY_CAP  = 500    # per-strategy hypothetical trade log


def save_state(state: dict):
    # Cap unbounded lists to prevent the state file growing indefinitely
    if len(state.get("nav_history", [])) > _NAV_HISTORY_CAP:
        state["nav_history"] = state["nav_history"][-_NAV_HISTORY_CAP:]

    for hyp in state.get("hypothetical", {}).values():
        if len(hyp.get("nav_history", [])) > _NAV_HISTORY_CAP:
            hyp["nav_history"] = hyp["nav_history"][-_NAV_HISTORY_CAP:]
        if len(hyp.get("trade_history", [])) > _TRADE_HISTORY_CAP:
            hyp["trade_history"] = hyp["trade_history"][-_TRADE_HISTORY_CAP:]

    # Atomic write: write to .tmp then rename so a mid-write crash never
    # produces a corrupted file (os.replace is atomic on POSIX).
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_FILE)


# ── Portfolio NAV ──────────────────────────────────────────────────────────────
def compute_nav(state: dict, current_prices: dict = None) -> float:
    if current_prices is None:
        client = get_client(for_trading=False)
        current_prices = {}
        for sym in UNIVERSE:
            try:
                current_prices[sym] = float(client.get_symbol_ticker(symbol=sym)["price"])
            except Exception:
                pass

    nav = state["cash_usdt"]
    for sym, qty in state["positions"].items():
        if qty != 0 and sym in current_prices:
            nav += qty * current_prices[sym]
    return nav


# ── Fetch current prices (all universe) ───────────────────────────────────────
def fetch_current_prices() -> dict:
    client = get_client(for_trading=False)
    prices = {}
    for sym in UNIVERSE:
        try:
            prices[sym] = float(client.get_symbol_ticker(symbol=sym)["price"])
        except Exception as e:
            logger.warning(f"Price fetch failed {sym}: {e}")
    return prices


# ── Stop loss / Take profit / Trailing stop check ─────────────────────────────
def check_position_exits(state: dict, current_prices: dict) -> dict:
    """Return {symbol: exit_price} for positions that breached a risk limit."""
    exits = {}
    if "position_entries" not in state:
        state["position_entries"] = {}

    for sym, qty in state["positions"].items():
        if qty == 0 or sym not in state["position_entries"]:
            continue

        entry        = state["position_entries"][sym]
        entry_price  = entry["entry_price"]
        current      = current_prices.get(sym)
        if current is None:
            continue

        pct = (current - entry_price) / entry_price

        if pct <= -STOP_LOSS_PCT:
            exits[sym] = current
            logger.warning(
                f"[STOP LOSS] {sym} {pct*100:.2f}% "
                f"(entry={entry_price:.2f} now={current:.2f})"
            )
            continue

        if pct >= TAKE_PROFIT_PCT:
            exits[sym] = current
            logger.success(
                f"[TAKE PROFIT] {sym} +{pct*100:.2f}% "
                f"(entry={entry_price:.2f} now={current:.2f})"
            )
            continue

        if USE_TRAILING_STOP and TRAILING_STOP_PCT > 0:
            peak = entry.get("peak_price", entry_price)
            if current > peak:
                entry["peak_price"] = current
                peak = current
            dd = (peak - current) / peak
            if dd >= TRAILING_STOP_PCT:
                exits[sym] = current
                logger.warning(
                    f"[TRAILING STOP] {sym} -{dd*100:.2f}% from peak "
                    f"(peak={peak:.2f} now={current:.2f})"
                )

    return exits


# ── Execute exit orders only (no rebalance) ───────────────────────────────────
def execute_exits_only(forced_exits: dict, state: dict) -> dict:
    """Close specific positions immediately. Used by the price monitor."""
    client = get_client(for_trading=True)

    for sym, exit_price in forced_exits.items():
        qty = state["positions"].get(sym, 0)
        if qty == 0:
            continue
        try:
            if PAPER_TRADING:
                order = client.create_order(
                    symbol=sym,
                    side="SELL" if qty > 0 else "BUY",
                    type="MARKET",
                    quantity=abs(qty),
                )
                proceeds = qty * exit_price
                state["positions"][sym] = 0.0
                state["cash_usdt"] += proceeds
                state["position_entries"].pop(sym, None)

                entry_price = state.get("position_entries", {}).get(sym, {}).get("entry_price", exit_price)
                pos_pnl     = (exit_price - entry_price) * qty
                pos_pct     = (exit_price - entry_price) / entry_price * 100 if entry_price else 0

                state.setdefault("trade_log", []).append({
                    "time":      str(datetime.now(timezone.utc)),
                    "symbol":    sym,
                    "side":      "SELL",
                    "qty":       abs(qty),
                    "price":     exit_price,
                    "pnl":       round(pos_pnl, 4),
                    "reason":    "stop/tp",
                    "strategy":  ", ".join(state.get("active_strategies", [])),
                    "order_id":  order.get("orderId"),
                })
                logger.warning(
                    f"📝 [EXIT ORDER] SELL {abs(qty):.6f} {sym} @ ${exit_price:,.2f} "
                    f"| P&L: {pos_pnl:+.2f} ({pos_pct:+.2f}%) "
                    f"| orderId={order.get('orderId')}"
                )
        except Exception as e:
            logger.error(f"Exit order failed {sym}: {e}")

    return state


# ── Full rebalance (signal-driven) ────────────────────────────────────────────
def execute_rebalance(target_weights: pd.Series, state: dict, nav: float,
                      current_prices: dict) -> dict:
    """Place orders to shift current weights towards target weights."""
    client = get_client(for_trading=True)

    # Reconcile internal cash tracking with actual testnet USDT balance.
    # Fees (0.1% per trade) and lot-size rounding cause drift over time.
    try:
        account_info = client.get_account()
        actual_usdt  = float(next(
            (a["free"] for a in account_info.get("balances", []) if a["asset"] == "USDT"),
            state["cash_usdt"],
        ))
        if abs(actual_usdt - state["cash_usdt"]) > 1.0:
            logger.info(
                f"Cash reconciliation: internal=${state['cash_usdt']:,.2f} "
                f"→ actual testnet=${actual_usdt:,.2f} "
                f"(drift=${actual_usdt - state['cash_usdt']:+.2f}, likely fees)"
            )
            state["cash_usdt"] = actual_usdt
    except Exception as e:
        logger.warning(f"Cash reconciliation skipped: {e}")

    # Leave a small buffer so fees on the last BUY don't push over the limit
    available_cash = state["cash_usdt"] * 0.999

    current_values = {
        sym: state["positions"].get(sym, 0) * current_prices.get(sym, 0)
        for sym in UNIVERSE
    }
    current_w = {sym: v / nav for sym, v in current_values.items()}

    for sym in UNIVERSE:
        target_w  = float(target_weights.get(sym, 0.0))
        current_w_ = current_w.get(sym, 0.0)
        delta_w    = target_w - current_w_
        delta_usdt = delta_w * nav

        if abs(delta_usdt) < MIN_ORDER_USDT:
            continue

        price = current_prices.get(sym)
        if not price:
            continue

        try:
            info = client.get_symbol_info(sym)
            if info is None:
                logger.warning(f"Symbol info not found for {sym} — skipping (possibly delisted or renamed)")
                continue
            step = next(
                float(f["stepSize"])
                for f in info["filters"]
                if f["filterType"] == "LOT_SIZE"
            )
            qty  = abs(delta_usdt) / price
            qty  = round(qty // step * step, 8)
            if qty <= 0:
                continue

            side = "BUY" if delta_usdt > 0 else "SELL"

            # Cap BUY size to available cash (fees + rounding can exceed balance)
            if side == "BUY":
                delta_usdt = min(delta_usdt, available_cash)
                if delta_usdt < MIN_ORDER_USDT:
                    logger.info(f"  Skipping {sym}: insufficient cash (${available_cash:.2f} available)")
                    continue
                qty = round((delta_usdt / price) // step * step, 8)
                if qty <= 0:
                    continue

            if PAPER_TRADING:
                order = client.create_order(
                    symbol=sym, side=side, type="MARKET", quantity=qty,
                )
                cost = qty * price
                logger.success(
                    f"📝 [TESTNET ORDER] {side} {qty:.6f} {sym} @ ${price:,.2f} "
                    f"| Δw={delta_w*100:+.1f}% | {'Cost' if side=='BUY' else 'Proceeds'}=${cost:,.2f} "
                    f"| orderId={order.get('orderId')}"
                )

                state.setdefault("position_entries", {})
                if side == "BUY":
                    is_new = state["positions"].get(sym, 0) == 0
                    if is_new:
                        state["position_entries"][sym] = {
                            "entry_price": price,
                            "entry_date":  str(pd.Timestamp.now(tz="UTC")),
                            "peak_price":  price,
                        }
                        sl_price = price * (1 - STOP_LOSS_PCT)
                        tp_price = price * (1 + TAKE_PROFIT_PCT)
                        logger.info(
                            f"   ↳ Entry=${price:,.2f} | "
                            f"Stop Loss=${sl_price:,.2f} (-{STOP_LOSS_PCT*100:.0f}%) | "
                            f"Take Profit=${tp_price:,.2f} (+{TAKE_PROFIT_PCT*100:.0f}%)"
                        )
                    state["positions"][sym]  = state["positions"].get(sym, 0) + qty
                    state["cash_usdt"]      -= cost
                    available_cash          -= cost  # track within this rebalance loop
                else:
                    entry_price = state.get("position_entries", {}).get(sym, {}).get("entry_price", price)
                    sell_pnl    = (price - entry_price) * qty
                    sell_pct    = (price - entry_price) / entry_price * 100 if entry_price else 0
                    logger.info(f"   ↳ Realised P&L: {sell_pnl:+.2f} ({sell_pct:+.2f}%)")
                    new_qty = state["positions"].get(sym, 0) - qty
                    if new_qty <= 0:
                        state["position_entries"].pop(sym, None)
                    state["positions"][sym] = max(new_qty, 0)
                    state["cash_usdt"]     += qty * price

                trade_pnl = sell_pnl if side == "SELL" else 0.0
                state.setdefault("trade_log", []).append({
                    "time":      str(datetime.now(timezone.utc)),
                    "symbol":    sym,
                    "side":      side,
                    "qty":       qty,
                    "price":     price,
                    "pnl":       round(trade_pnl, 4),
                    "reason":    "signal_rebalance",
                    "strategy":  ", ".join(state.get("active_strategies", [])),
                    "order_id":  order.get("orderId"),
                })

        except Exception as e:
            logger.error(f"Order failed {sym}: {e}")

    return state


# ── Signal weight computation ──────────────────────────────────────────────────
def _signal_to_weights(signal_series: pd.Series, long_short: bool = False) -> pd.Series:
    """
    Convert raw signals to position weights with iterative MAX_POSITION_SIZE capping.

    long_short=False  →  long-only (used for live spot trading, spot testnet
                          cannot execute short sells without margin)
    long_short=True   →  50% gross long / 50% gross short, total |w| ≤ 100%
                          (used for hypothetical paper portfolios)
    """
    def _cap_and_scale(sigs: pd.Series, budget: float) -> pd.Series:
        w = sigs / sigs.sum() * budget
        for _ in range(20):
            over = w > MAX_POSITION_SIZE
            if not over.any():
                break
            w[over] = MAX_POSITION_SIZE
            leftover = budget - w[over].sum()
            uncapped = ~over
            if uncapped.any() and w[uncapped].sum() > 0:
                w[uncapped] = w[uncapped] / w[uncapped].sum() * leftover
            else:
                break
        return w.clip(upper=MAX_POSITION_SIZE)

    weights  = pd.Series(0.0, index=UNIVERSE)
    pos_sigs = signal_series[signal_series > 0].nlargest(MAX_LIVE_POSITIONS)
    neg_sigs = signal_series[signal_series < 0].nsmallest(MAX_LIVE_POSITIONS).abs()

    has_longs  = not pos_sigs.empty
    has_shorts = long_short and not neg_sigs.empty

    # Split 50/50 when both sides present; one side gets 100% if alone
    long_budget  = 0.5 if (has_longs and has_shorts) else (1.0 if has_longs else 0.0)
    short_budget = 0.5 if (has_longs and has_shorts) else (1.0 if has_shorts else 0.0)

    if has_longs:
        weights[pos_sigs.index] = _cap_and_scale(pos_sigs, long_budget)
    if has_shorts:
        weights[neg_sigs.index] = -_cap_and_scale(neg_sigs, short_budget)

    return weights.reindex(UNIVERSE, fill_value=0)


def compute_target_weights(
    seasonality_analyzer,
    signals_dict: dict,
    close: pd.DataFrame,
    hypotheticals: dict = None,
) -> tuple:
    """Return (target_weights, active_selection) for the LIVE portfolio (long-only, spot-safe)."""
    selection = seasonality_analyzer.select_strategy(
        current_date=close.index[-1], top_n=2,
        hypotheticals=hypotheticals,
    )
    logger.info(f"Active strategy blend: {selection}")

    blended      = blend_signals(signals_dict, selection, close)
    current_sigs = blended.iloc[-1].reindex(UNIVERSE, fill_value=0)

    if current_sigs[current_sigs > 0].empty:
        logger.warning("No positive signals from active strategies — holding cash")
        return pd.Series(0.0, index=UNIVERSE), selection

    # long_short=False: spot testnet cannot execute actual short sells
    return _signal_to_weights(current_sigs, long_short=False), selection


# ── Hypothetical paper portfolios (all 10 strategies simultaneously) ───────────
def update_hypotheticals(signals_dict: dict, current_prices: dict, state: dict):
    """
    Track all 10 strategies as independent $10k paper portfolios.
    Each strategy computes its own signal-proportional weights independently.
    Period return = sum(prev_weight_i * price_return_i) applied to prev NAV.
    """
    hypo = state.setdefault("hypothetical", {})

    for name, sig_df in signals_dict.items():
        if sig_df is None or sig_df.empty:
            continue
        try:
            latest_sig  = sig_df.iloc[-1].reindex(UNIVERSE, fill_value=0)
            new_weights = _signal_to_weights(latest_sig, long_short=True)

            ts_now = str(pd.Timestamp.now(tz="UTC"))

            if name not in hypo:
                # Log initial positions as entry trades (P&L = 0 at entry)
                initial_entry_prices = {}
                initial_trades = []
                for sym in UNIVERSE:
                    w = float(new_weights.get(sym, 0))
                    p = float(current_prices.get(sym, 0))
                    if abs(w) >= 0.01:
                        initial_entry_prices[sym] = p
                        initial_trades.append({
                            "time":        ts_now,
                            "symbol":      sym,
                            "side":        "BUY" if w > 0 else "SELL",
                            "weight_from": 0.0,
                            "weight_to":   round(w * 100, 1),
                            "price":       p,
                            "hypo_value":  round(PORTFOLIO_USDT * abs(w), 2),
                            "pnl":         0.0,
                        })
                hypo[name] = {
                    "nav":           PORTFOLIO_USDT,
                    "weights":       new_weights.to_dict(),
                    "last_prices":   {s: current_prices.get(s, 0) for s in UNIVERSE},
                    "nav_history":   [{"date": ts_now, "nav": PORTFOLIO_USDT}],
                    "trade_history": initial_trades,
                    "entry_prices":  initial_entry_prices,
                }
                logger.info(
                    f"  📊 [HYPO INIT] {name} — starting at ${PORTFOLIO_USDT:,.0f} "
                    f"| {len(initial_trades)} initial positions"
                )
                continue

            # Period return = held weights × fractional price moves since last cycle
            # Negative weights (shorts) are included — w*(p1-p0)/p0 is negative when
            # short and price rises, positive when short and price falls (correct).
            prev_weights = pd.Series(hypo[name].get("weights", {})).reindex(UNIVERSE, fill_value=0)
            prev_prices  = hypo[name].get("last_prices", {})
            prev_nav     = float(hypo[name]["nav"])

            period_return = 0.0
            for sym in UNIVERSE:
                w  = float(prev_weights.get(sym, 0))
                p0 = float(prev_prices.get(sym, 0))
                p1 = float(current_prices.get(sym, 0))
                if abs(w) > 0 and p0 > 0 and p1 > 0:
                    period_return += w * (p1 - p0) / p0

            # If trade_history is missing (state file from before this fix), backfill
            # the current weights as initial entries so history is never empty
            trade_history = hypo[name].setdefault("trade_history", [])
            if not trade_history:
                for sym in UNIVERSE:
                    pw = float(prev_weights.get(sym, 0))
                    if abs(pw) >= 0.01:
                        trade_history.append({
                            "time":        ts_now,
                            "symbol":      sym,
                            "side":        "BUY" if pw > 0 else "SELL",
                            "weight_from": 0.0,
                            "weight_to":   round(pw * 100, 1),
                            "price":       float(current_prices.get(sym, 0)),
                            "hypo_value":  round(prev_nav * abs(pw), 2),
                        })

            new_nav = round(prev_nav * (1 + period_return), 2)
            ret_pct = (new_nav - PORTFOLIO_USDT) / PORTFOLIO_USDT * 100

            # Log hypothetical trades — any weight shift > 1% counts as a trade
            entry_prices = hypo[name].setdefault("entry_prices", {})
            for sym in UNIVERSE:
                prev_w  = float(prev_weights.get(sym, 0))
                new_w   = float(new_weights.get(sym, 0))
                delta_w = new_w - prev_w
                if abs(delta_w) < 0.01:
                    continue
                price      = float(current_prices.get(sym, 0))
                hypo_value = round(prev_nav * abs(delta_w), 2)

                # Realized P&L: only when reducing / closing a position
                ep  = entry_prices.get(sym, price)
                if delta_w < 0 and abs(prev_w) > 0 and price > 0:
                    sold_usdt = prev_nav * min(abs(delta_w), abs(prev_w))
                    sold_qty  = sold_usdt / price
                    trade_pnl = round((price - ep) * sold_qty, 4)
                else:
                    trade_pnl = 0.0

                # Update entry price: weighted average on additions, clear on full exit
                if delta_w > 0 and price > 0:
                    prev_qty = prev_nav * abs(prev_w) / price if abs(prev_w) > 0 else 0
                    add_qty  = prev_nav * abs(delta_w) / price
                    total    = prev_qty + add_qty
                    entry_prices[sym] = (ep * prev_qty + price * add_qty) / total if total > 0 else price
                elif abs(new_w) < 0.001:
                    entry_prices.pop(sym, None)

                trade_history.append({
                    "time":        ts_now,
                    "symbol":      sym,
                    "side":        "BUY" if delta_w > 0 else "SELL",
                    "weight_from": round(prev_w * 100, 1),
                    "weight_to":   round(new_w * 100, 1),
                    "price":       price,
                    "hypo_value":  hypo_value,
                    "pnl":         trade_pnl,
                })

            hypo[name]["nav"]         = new_nav
            hypo[name]["weights"]     = new_weights.to_dict()
            hypo[name]["last_prices"] = {s: current_prices.get(s, 0) for s in UNIVERSE}
            hypo[name]["nav_history"].append({
                "date": str(pd.Timestamp.now(tz="UTC")),
                "nav":  new_nav,
            })

            logger.info(
                f"  📊 [HYPO] {name:<28} NAV=${new_nav:,.2f}  ({ret_pct:+.2f}% vs start)"
            )
        except Exception as e:
            logger.error(f"Hypothetical update failed for {name}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# JOB 1 — PRICE MONITOR  (runs every PRICE_MONITOR_SECS seconds)
# ══════════════════════════════════════════════════════════════════════════════
def price_monitor_job():
    """
    Fast loop: poll current prices and immediately exit any position
    that breaches its stop loss, take profit, or trailing stop.
    Does NOT rebalance — just closes the breached position.
    """
    state = load_state()

    open_positions = {sym: qty for sym, qty in state["positions"].items() if qty != 0}
    if not open_positions:
        return  # nothing to monitor

    prices = fetch_current_prices()
    forced_exits = check_position_exits(state, prices)

    if forced_exits:
        logger.warning(
            f"[MONITOR] Risk limits hit for: {list(forced_exits.keys())} — closing now"
        )
        state = execute_exits_only(forced_exits, state)
        nav   = compute_nav(state, prices)
        state["nav_history"].append({
            "date": str(pd.Timestamp.now(tz="UTC")),
            "nav":  round(nav, 2),
            "event": "stop_tp_exit",
        })
        save_state(state)

    # Portfolio heartbeat every 30 seconds
    global _last_heartbeat
    if time.time() - _last_heartbeat >= 30:
        _last_heartbeat = time.time()
        nav        = compute_nav(state, prices)
        nav_hist   = state.get("nav_history", [])
        start_nav  = nav_hist[0]["nav"] if nav_hist else PORTFOLIO_USDT
        pnl        = nav - start_nav
        pnl_pct    = pnl / start_nav * 100 if start_nav else 0
        ts         = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        n_open     = len(open_positions)
        logger.info(
            f"📈 [{ts}] NAV=${nav:,.2f} | Cash=${state['cash_usdt']:,.2f} "
            f"| Positions={n_open} | P&L={pnl:+.2f} ({pnl_pct:+.2f}%)"
        )
        entries = state.get("position_entries", {})
        for sym, qty in open_positions.items():
            ep = entries.get(sym, {}).get("entry_price", 0)
            cp = prices.get(sym, 0)
            if ep and cp:
                pos_pnl = (cp - ep) * qty
                pos_pct = (cp - ep) / ep * 100
                sl      = ep * (1 - STOP_LOSS_PCT)
                tp      = ep * (1 + TAKE_PROFIT_PCT)
                logger.info(
                    f"   {sym:<12} {qty:.6f} | entry=${ep:,.2f} | now=${cp:,.2f} "
                    f"| P&L={pos_pnl:+.2f} ({pos_pct:+.2f}%) "
                    f"| SL=${sl:,.2f} | TP=${tp:,.2f}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# JOB 2 — SIGNAL RECOMPUTE + CONDITIONAL REBALANCE
#          (runs every SIGNAL_RECOMPUTE_MINS minutes)
# ══════════════════════════════════════════════════════════════════════════════
def signal_rebalance_job(strategies: list, seasonality_analyzer, signals_dict: dict):
    """
    Signal loop: recompute all strategy signals on the latest market data.
    Only places orders when the new target weights differ from the current
    weights by more than REBALANCE_THRESHOLD (sum of |Δweight|).

    This is what makes the engine signal-driven rather than clock-driven.
    """
    logger.info("=" * 55)
    logger.info(f"Signal recompute — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    state = load_state()

    # 1. Fresh data (bypass cache so we always have the latest bar)
    logger.info("Fetching latest market data...")
    try:
        universe_data = get_universe_ohlcv(use_cache=False)
        close         = build_close_matrix(universe_data)
        returns       = build_return_matrix(close)
        funding       = get_universe_funding_rates()
        fg            = get_fear_greed_index()
    except Exception as e:
        logger.error(f"Data fetch failed — skipping this cycle: {e}")
        return

    # 2. Recompute all strategy signals
    logger.info(f"Recomputing signals for {len(strategies)} strategies...")
    high_df = pd.DataFrame({s: universe_data[s]["high"] for s in universe_data})
    low_df  = pd.DataFrame({s: universe_data[s]["low"]  for s in universe_data})
    for strategy in strategies:
        try:
            signals = strategy.run(
                close=close, returns=returns,
                high=high_df, low=low_df,
                funding_rates=funding, fear_greed=fg,
            )
            signals_dict[strategy.name] = signals
        except Exception as e:
            logger.error(f"Signal failed for {strategy.name}: {e}")

    # Print per-strategy signal snapshot (latest bar)
    logger.info("─" * 55)
    logger.info("STRATEGY SIGNALS — latest bar")
    for name, sig_df in signals_dict.items():
        try:
            if sig_df is None or sig_df.empty:
                continue
            latest    = sig_df.iloc[-1].reindex(UNIVERSE, fill_value=0)
            bulls     = latest[latest > 0.01].nlargest(4)
            bears     = latest[latest < -0.01].nsmallest(3)
            bull_str  = "  ".join(f"{s.replace('USDT','')}={v:+.2f}" for s, v in bulls.items())
            bear_str  = "  ".join(f"{s.replace('USDT','')}={v:+.2f}" for s, v in bears.items())
            signal_line = bull_str or "(no long signals)"
            if bear_str:
                signal_line += f"  |  Short: {bear_str}"
            logger.info(f"  🎯 {name:<24} {signal_line}")
        except Exception:
            pass
    logger.info("─" * 55)

    # 3. Snapshot latest signals to state (dashboard reads these)
    signals_snapshot = {}
    for name, sig_df in signals_dict.items():
        if sig_df is not None and not sig_df.empty:
            latest = sig_df.iloc[-1].reindex(UNIVERSE, fill_value=0)
            signals_snapshot[name] = {k: round(float(v), 4) for k, v in latest.items()}
    state["latest_signals"] = signals_snapshot
    state["signals_updated_at"] = str(datetime.now(timezone.utc))

    # 4. Fetch current prices (used for hypotheticals + potential rebalance)
    prices = fetch_current_prices()

    # 5. Update hypothetical paper portfolios for all 10 strategies
    logger.info("─" * 55)
    logger.info("Updating hypothetical paper portfolios...")
    update_hypotheticals(signals_dict, prices, state)

    # 6. New target weights for LIVE portfolio (live hypothetical perf feeds selection)
    try:
        new_weights, active_sel = compute_target_weights(
            seasonality_analyzer=seasonality_analyzer,
            signals_dict=signals_dict,
            close=close,
            hypotheticals=state.get("hypothetical", {}),
        )
        state["active_strategies"]      = [name for name, _ in active_sel]
        state["active_strategy_weights"] = {name: round(w, 4) for name, w in active_sel}
    except Exception as e:
        logger.error(f"Weight computation failed: {e}")
        save_state(state)  # persist hypotheticals even on error
        return

    # 6. Compare ACTUAL live positions (not saved target) vs new target.
    #    Using saved target caused a permanent HOLD when initial execution was partial:
    #    saved-target == new-target → delta = 0 even though actual positions differ.
    nav_actual = compute_nav(state, prices)
    if nav_actual > 0:
        prev_weights = pd.Series({
            sym: (state["positions"].get(sym, 0) * prices.get(sym, 0)) / nav_actual
            for sym in UNIVERSE
        }).reindex(UNIVERSE, fill_value=0)
    else:
        prev_weights = pd.Series(state.get("current_weights", {})).reindex(UNIVERSE, fill_value=0)

    new_aligned  = new_weights.reindex(UNIVERSE, fill_value=0)
    weight_delta = (new_aligned - prev_weights).abs().sum()

    # Print weight delta table
    logger.info(f"{'Symbol':<12} {'Actual':>8} {'Target':>8} {'Delta':>8}  {'Action'}")
    logger.info("─" * 52)
    for sym in UNIVERSE:
        pw = float(prev_weights.get(sym, 0))
        nw = float(new_aligned.get(sym, 0))
        dw = nw - pw
        if abs(dw) < 0.001 and pw == 0 and nw == 0:
            continue
        action = "▲ BUY" if dw > 0.01 else ("▼ SELL" if dw < -0.01 else "· hold")
        logger.info(f"{sym:<12} {pw*100:>7.1f}% {nw*100:>7.1f}% {dw*100:>+7.1f}%  {action}")
    logger.info("─" * 52)
    logger.info(
        f"Total weight delta: {weight_delta:.4f} "
        f"(threshold={REBALANCE_THRESHOLD}) "
        f"→ {'⚡ REBALANCE' if weight_delta >= REBALANCE_THRESHOLD else '✋ HOLD'}"
    )

    if weight_delta < REBALANCE_THRESHOLD:
        logger.info("Signal unchanged — holding current positions, no orders placed.")
        save_state(state)  # persist hypotheticals
        return

    # 7. Signal shifted — rebalance
    logger.info(f"Signal shift detected (Δ={weight_delta:.4f}) — placing orders...")

    nav = compute_nav(state, prices)
    logger.info(f"Portfolio NAV: ${nav:,.2f} USDT")

    state = execute_rebalance(new_weights, state, nav, prices)

    # 8. Persist
    state["current_weights"] = new_aligned.to_dict()
    state["last_run"]        = str(datetime.now(timezone.utc))
    state["nav_history"].append({
        "date":  str(pd.Timestamp.now(tz="UTC")),
        "nav":   round(nav, 2),
        "event": "signal_rebalance",
    })
    save_state(state)
    logger.success(f"Rebalance complete. NAV=${nav:,.2f}")


# ── Graceful shutdown reporting ────────────────────────────────────────────────
def _generate_final_reports():
    try:
        if not STATE_FILE.exists():
            logger.warning("No live state file to report on.")
            return

        with open(STATE_FILE) as f:
            state = json.load(f)

        logger.info("\n" + "=" * 55)
        logger.info("FINAL REPORT — Live Trading Session")
        logger.info("=" * 55)
        logger.info(f"Cash remaining: ${state['cash_usdt']:,.2f} USDT")

        open_pos = {sym: qty for sym, qty in state["positions"].items() if qty != 0}
        logger.info(f"Open positions ({len(open_pos)}):")
        for sym, qty in open_pos.items():
            logger.info(f"  {sym}: {qty:.6f}")

        trade_log = state.get("trade_log", [])
        logger.info(f"Total trades executed: {len(trade_log)}")

        nav_history = state.get("nav_history", [])
        if nav_history:
            nav_df    = pd.DataFrame(nav_history)
            nav_df["nav"] = pd.to_numeric(nav_df["nav"])
            start_nav = nav_df["nav"].iloc[0]
            end_nav   = nav_df["nav"].iloc[-1]
            ret_pct   = (end_nav - start_nav) / start_nav * 100
            logger.info(f"Start NAV: ${start_nav:,.2f}  End NAV: ${end_nav:,.2f}  Return: {ret_pct:+.2f}%")

            nav_export = RESULT_DIR / "nav_history_export.csv"
            nav_df.to_csv(nav_export, index=False)
            logger.success(f"NAV history → {nav_export}")

        if trade_log:
            trades_df = pd.DataFrame(trade_log)
            trades_export = RESULT_DIR / "trade_log.csv"
            trades_df.to_csv(trades_export, index=False)
            logger.success(f"Trade log   → {trades_export}")

        logger.info("=" * 55)

    except Exception as e:
        logger.error(f"Report generation error: {e}")


# ── Scheduler setup ────────────────────────────────────────────────────────────
def start_scheduler(strategies: list, seasonality_analyzer, signals_dict: dict,
                    run_now: bool = False):
    """
    Start two concurrent APScheduler jobs:
      • price_monitor_job  — every PRICE_MONITOR_SECS seconds (stop/TP)
      • signal_rebalance_job — every SIGNAL_RECOMPUTE_MINS minutes (signal-driven trades)

    Uses BackgroundScheduler so both loops run independently without blocking
    each other. Main thread stays alive via a KeyboardInterrupt-safe sleep loop.
    """
    setup_logger()

    scheduler = BackgroundScheduler(timezone="UTC")

    # Job 1: Fast price / risk monitor
    scheduler.add_job(
        func=price_monitor_job,
        trigger="interval",
        seconds=PRICE_MONITOR_SECS,
        id="price_monitor",
        name=f"Price & stop monitor (every {PRICE_MONITOR_SECS}s)",
        max_instances=1,
        coalesce=True,
    )

    # Job 2: Signal recompute + conditional rebalance
    scheduler.add_job(
        func=signal_rebalance_job,
        trigger="interval",
        minutes=SIGNAL_RECOMPUTE_MINS,
        args=[strategies, seasonality_analyzer, signals_dict],
        id="signal_rebalance",
        name=f"Signal recompute & rebalance (every {SIGNAL_RECOMPUTE_MINS}min)",
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()

    logger.info("=" * 55)
    logger.info("🚀  REAL-TIME ALGO TRADING ENGINE — BINANCE TESTNET")
    logger.info("=" * 55)
    logger.info(f"  📊 REAL DATA  |  🎮 TESTNET EXECUTION (paper money)")
    logger.info(f"  Signal recompute : every {SIGNAL_RECOMPUTE_MINS} min  → places orders only if signal shifts")
    logger.info(f"  Stop/TP monitor  : every {PRICE_MONITOR_SECS}s   → exits positions immediately on breach")
    logger.info(f"  Rebalance trigger: total |Δweight| > {REBALANCE_THRESHOLD:.0%}")
    logger.info(f"  Risk management  : SL={STOP_LOSS_PCT*100:.0f}%  TP={TAKE_PROFIT_PCT*100:.0f}%  TrailStop={TRAILING_STOP_PCT*100:.0f}%")
    logger.info(f"  Universe         : {len(UNIVERSE)} tokens — {', '.join(s.replace('USDT','') for s in UNIVERSE)}")
    logger.info("  Press Ctrl+C to stop and generate final report")
    logger.info("=" * 55)

    if run_now:
        logger.info("run_now=True — firing initial signal rebalance immediately...")
        signal_rebalance_job(strategies, seasonality_analyzer, signals_dict)

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("\nGraceful shutdown initiated...")
        scheduler.shutdown(wait=False)
        _generate_final_reports()
        logger.success("Engine stopped.")
