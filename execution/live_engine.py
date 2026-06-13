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
    LIVE_BOOK_STRATEGIES, LIVE_REBALANCE_FREQ_DAYS,
    MAX_LIVE_POSITIONS, MAX_POSITION_SIZE, FUTURES_LEVERAGE,
    TRANSACTION_COST_BP, SLIPPAGE_BP,
)
from data.ingestion import get_universe_ohlcv, build_close_matrix, build_return_matrix
from data.ingestion import get_fear_greed_index, get_universe_funding_rates
from seasonality.analyzer import blend_signals
from utils.logger import setup_logger
from db.state import (
    load_state as _db_load_state,
    save_state,
    load_nav_history_for_report,
    load_trade_log_for_report,
)
from db.controls import load_controls, acknowledge_close_all

_last_heartbeat = 0.0   # tracks when the monitor last printed a full portfolio snapshot


# ── State management ───────────────────────────────────────────────────────────
def _bootstrap_from_binance() -> dict:
    """
    Called on fresh start (no state file). Reads actual Binance FUTURES account
    state so the engine picks up existing positions and real wallet balance.

    Quantities are signed (positive = long, negative = short).
    NAV = futures wallet balance + total unrealized PnL.
    """
    client  = get_client(for_trading=True)
    tickers = {t["symbol"]: float(t["price"])
               for t in client.futures_symbol_ticker()
               if t["symbol"] in UNIVERSE}

    account = client.futures_account()

    # totalWalletBalance is the account-level USDT-equivalent sum of all assets
    # (USDT + USDC + BTC collateral etc.) — always prefer this over per-asset lookup
    cash = float(account.get("totalWalletBalance", 0.0))

    # Open futures positions (signed quantities) + their Binance entry prices
    positions        = {sym: 0.0 for sym in UNIVERSE}
    position_entries = {}
    for p in account.get("positions", []):
        sym = p["symbol"]
        if sym in UNIVERSE:
            qty = float(p["positionAmt"])
            if qty != 0:
                positions[sym] = qty
                ep = float(p.get("entryPrice", 0))
                if ep > 0:
                    position_entries[sym] = {
                        "entry_price": ep,
                        "entry_date":  str(pd.Timestamp.now(tz="UTC")),
                        "peak_price":  tickers.get(sym, ep),
                    }

    unrealized_pnl = float(account.get("totalUnrealizedProfit", 0))
    nav = round(cash + unrealized_pnl, 2)

    logger.info(
        f"Bootstrap from Binance Futures: wallet=${cash:,.2f} | "
        f"unrealized PnL=${unrealized_pnl:,.2f} | NAV=${nav:,.2f}"
    )

    return {
        "positions":        positions,
        "cash_usdt":        cash,
        "initial_nav":      nav,
        "last_run":         None,
        "nav_history":      [],
        "current_weights":  {},
        "position_entries": position_entries,
        "trade_log":        [],
        "hypothetical":     {},
        "active_strategies": [],
    }


def load_state() -> dict:
    # Try DB first
    state = _db_load_state()
    if state is not None:
        return state

    # No DB row yet — bootstrap from Binance (real positions + real USDT balance)
    try:
        return _bootstrap_from_binance()
    except Exception as e:
        logger.warning(f"Binance bootstrap failed ({e}) — starting with empty state")

    # Last-resort fallback (Binance unreachable at startup)
    return {
        "positions":        {sym: 0.0 for sym in UNIVERSE},
        "cash_usdt":        0.0,
        "initial_nav":      None,
        "last_run":         None,
        "nav_history":      [],
        "current_weights":  {},
        "position_entries": {},
        "trade_log":        [],
        "hypothetical":     {},
        "active_strategies": [],
        "_nav_db_count":    0,
        "_trade_db_count":  0,
        "_hyp_counts":      {},
    }


def reconcile_with_binance(state: dict, prices: dict) -> dict:
    """
    Sync futures positions and USDT wallet balance from the actual Binance account.

    What comes from Binance (authoritative):
      - positionAmt    — signed qty per symbol (positive=long, negative=short)
      - walletBalance  — USDT including all realized PnL and fees
      - entryPrice     — seeded for any positions we didn't open this session
      - initial_nav    — stamped ONCE on the first successful reading

    What stays internal:
      - position_entries["entry_price"] — used for per-position unrealized P&L
    """
    try:
        client  = get_client(for_trading=True)
        account = client.futures_account()

        # totalWalletBalance = USDT-equivalent sum of all collateral (USDT+USDC+BTC...)
        wallet_balance = float(account.get("totalWalletBalance", state.get("cash_usdt", 0.0)))

        # ── Signed futures positions + seed entry prices for unknown positions ─
        new_positions = {sym: 0.0 for sym in UNIVERSE}
        for p in account.get("positions", []):
            sym = p["symbol"]
            if sym in UNIVERSE:
                qty = float(p["positionAmt"])
                if qty != 0:
                    new_positions[sym] = qty
                    if sym not in state.get("position_entries", {}):
                        ep = float(p.get("entryPrice", 0))
                        if ep > 0:
                            state.setdefault("position_entries", {})[sym] = {
                                "entry_price": ep,
                                "entry_date":  str(pd.Timestamp.now(tz="UTC")),
                                "peak_price":  prices.get(sym, ep),
                            }

        # Clear entry data for positions that are now closed
        for sym in list(state.get("position_entries", {}).keys()):
            if new_positions.get(sym, 0) == 0:
                state["position_entries"].pop(sym, None)

        drift = wallet_balance - state.get("cash_usdt", wallet_balance)
        if abs(drift) > 0.10:
            logger.info(
                f"Wallet reconciliation: internal=${state.get('cash_usdt', 0):,.2f} "
                f"→ Binance=${wallet_balance:,.2f}  (Δ={drift:+.2f})"
            )

        state["positions"] = new_positions
        state["cash_usdt"] = wallet_balance

        # ── Stamp initial_nav once from the first real Binance reading ─────────
        if state.get("initial_nav") is None:
            unrealized  = float(account.get("totalUnrealizedProfit", 0))
            live_nav    = round(wallet_balance + unrealized, 2)
            if live_nav > 0:
                state["initial_nav"] = live_nav
                logger.info(f"Initial NAV anchored from Binance Futures: ${live_nav:,.2f}")

    except Exception as e:
        logger.warning(f"Binance futures reconciliation skipped ({e}) — using internal state")

    return state


_NAV_HISTORY_CAP   = 2880   # keep last 2 days at 1-min cadence in-memory
_TRADE_HISTORY_CAP = 500    # per-strategy hypothetical trade log in-memory


# ── Portfolio NAV ──────────────────────────────────────────────────────────────
def compute_nav(state: dict, current_prices: dict = None) -> float:
    """
    Futures NAV = wallet balance + unrealized PnL.

    unrealized PnL = Σ qty_i × (current_price_i - entry_price_i)
    Works for both longs (qty > 0) and shorts (qty < 0).
    """
    if current_prices is None:
        current_prices = fetch_current_prices()

    wallet   = state["cash_usdt"]
    entries  = state.get("position_entries", {})
    unrealized = 0.0
    for sym, qty in state["positions"].items():
        if qty != 0 and sym in current_prices:
            ep = entries.get(sym, {}).get("entry_price", current_prices[sym])
            unrealized += qty * (current_prices[sym] - ep)
    return wallet + unrealized


# ── Fetch current prices (futures) ────────────────────────────────────────────
def fetch_current_prices() -> dict:
    client = get_client(for_trading=False)
    try:
        return {t["symbol"]: float(t["price"])
                for t in client.futures_symbol_ticker()
                if t["symbol"] in UNIVERSE}
    except Exception as e:
        logger.warning(f"Bulk futures price fetch failed ({e}) — falling back to per-symbol")
    prices = {}
    for sym in UNIVERSE:
        try:
            prices[sym] = float(client.futures_symbol_ticker(symbol=sym)["price"])
        except Exception as ex:
            logger.warning(f"Price fetch failed {sym}: {ex}")
    return prices


# ── Stop loss / Take profit / Trailing stop check ─────────────────────────────
def check_position_exits(state: dict, current_prices: dict) -> dict:
    """
    Return {symbol: exit_price} for positions that breached a risk limit.

    pct is signed by direction so the same thresholds apply to longs and shorts:
      long  (qty > 0): profit when price rises  → pct = +(current-entry)/entry
      short (qty < 0): profit when price falls  → pct = -(current-entry)/entry
    """
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

        direction = 1 if qty > 0 else -1
        pct = direction * (current - entry_price) / entry_price  # signed P&L %

        if pct <= -STOP_LOSS_PCT:
            exits[sym] = current
            logger.warning(
                f"[STOP LOSS] {sym} {pct*100:.2f}% "
                f"(entry={entry_price:.2f} now={current:.2f} "
                f"{'long' if qty > 0 else 'short'})"
            )
            continue

        if pct >= TAKE_PROFIT_PCT:
            exits[sym] = current
            logger.success(
                f"[TAKE PROFIT] {sym} +{pct*100:.2f}% "
                f"(entry={entry_price:.2f} now={current:.2f})"
            )
            continue

        # Trailing stop: track peak-price for longs, trough-price for shorts
        if USE_TRAILING_STOP and TRAILING_STOP_PCT > 0:
            if qty > 0:
                peak = entry.get("peak_price", entry_price)
                if current > peak:
                    entry["peak_price"] = current
                    peak = current
                dd = (peak - current) / peak
            else:
                trough = entry.get("peak_price", entry_price)
                if current < trough:
                    entry["peak_price"] = current
                    trough = current
                dd = (current - trough) / trough if trough > 0 else 0
            if dd >= TRAILING_STOP_PCT:
                exits[sym] = current
                logger.warning(
                    f"[TRAILING STOP] {sym} -{dd*100:.2f}% from "
                    f"{'peak' if qty > 0 else 'trough'} "
                    f"(ref={entry.get('peak_price', entry_price):.2f} now={current:.2f})"
                )

    return exits


# ── Execute exit orders only (no rebalance) ───────────────────────────────────
def execute_exits_only(forced_exits: dict, state: dict) -> dict:
    """Close specific futures positions immediately. Used by the price monitor."""
    client = get_client(for_trading=True)

    for sym, exit_price in forced_exits.items():
        qty = state["positions"].get(sym, 0)
        if qty == 0:
            continue
        # Close long → SELL; close short → BUY; reduceOnly ensures no reversal
        side    = "SELL" if qty > 0 else "BUY"
        abs_qty = abs(qty)
        try:
            if PAPER_TRADING:
                order = client.futures_create_order(
                    symbol=sym,
                    side=side,
                    type="MARKET",
                    quantity=abs_qty,
                    reduceOnly=True,
                )
                entry_price = state.get("position_entries", {}).get(sym, {}).get("entry_price", exit_price)
                pos_pnl     = qty * (exit_price - entry_price)   # signed correctly for long/short
                direction   = 1 if qty > 0 else -1
                pos_pct     = direction * (exit_price - entry_price) / entry_price * 100 if entry_price else 0

                state["positions"][sym] = 0.0
                state["position_entries"].pop(sym, None)
                state["cash_usdt"] += pos_pnl  # approx; reconcile will correct from Binance

                state.setdefault("trade_log", []).append({
                    "time":      str(datetime.now(timezone.utc)),
                    "symbol":    sym,
                    "side":      side,
                    "qty":       abs_qty,
                    "price":     exit_price,
                    "pnl":       round(pos_pnl, 4),
                    "reason":    "stop/tp",
                    "strategy":  ", ".join(state.get("active_strategies", [])),
                    "order_id":  order.get("orderId"),
                })
                logger.warning(
                    f"📝 [FUTURES EXIT] {side} {abs_qty:.6f} {sym} @ ${exit_price:,.2f} "
                    f"| P&L: {pos_pnl:+.2f} ({pos_pct:+.2f}%) "
                    f"| orderId={order.get('orderId')}"
                )
        except Exception as e:
            logger.error(f"Exit order failed {sym}: {e}")

    return state


# ── Full rebalance (signal-driven, futures) ────────────────────────────────────
def execute_rebalance(target_weights: pd.Series, state: dict, nav: float,
                      current_prices: dict) -> dict:
    """
    Place futures orders to shift signed position weights towards target weights.

    target_weights[sym] > 0  → long exposure
    target_weights[sym] < 0  → short exposure
    target_weights[sym] == 0 → flat (close any open position)

    current_w[sym] = qty * price / nav  (signed; negative for shorts)
    """
    if nav <= 0:
        logger.critical(
            f"NAV is {nav:.2f} — cannot size orders. "
            "Fund your Binance Futures wallet and restart. Stopping now."
        )
        os._exit(1)

    client = get_client(for_trading=True)

    # Fetch futures LOT_SIZE filters once for the whole rebalance
    try:
        exchange_info = {
            s["symbol"]: s
            for s in client.futures_exchange_info()["symbols"]
            if s["symbol"] in UNIVERSE
        }
    except Exception as e:
        logger.error(f"Could not fetch futures exchange info: {e}")
        return state

    for sym in UNIVERSE:
        target_w   = float(target_weights.get(sym, 0.0))
        qty_cur    = float(state["positions"].get(sym, 0.0))  # signed
        price      = current_prices.get(sym)
        if not price:
            continue

        # Signed current weight (negative = short)
        current_w  = (qty_cur * price) / nav
        delta_w    = target_w - current_w
        delta_usdt = delta_w * nav

        if abs(delta_usdt) < MIN_ORDER_USDT:
            continue

        sym_info = exchange_info.get(sym)
        if sym_info is None:
            logger.warning(f"Futures symbol info not found for {sym} — skipping")
            continue
        step = next(
            (float(f["stepSize"]) for f in sym_info["filters"] if f["filterType"] == "LOT_SIZE"),
            0.001,
        )

        qty  = abs(delta_usdt) / price
        qty  = round(qty // step * step, 8)
        if qty <= 0:
            continue

        side = "BUY" if delta_w > 0 else "SELL"

        try:
            if PAPER_TRADING:
                order = client.futures_create_order(
                    symbol=sym, side=side, type="MARKET", quantity=qty,
                )
                notional = qty * price
                logger.success(
                    f"📝 [FUTURES DEMO] {side} {qty:.6f} {sym} @ ${price:,.2f} "
                    f"| Δw={delta_w*100:+.1f}% | Notional=${notional:,.2f} "
                    f"| orderId={order.get('orderId')}"
                )

                state.setdefault("position_entries", {})
                new_qty  = round(qty_cur + (qty if side == "BUY" else -qty), 8)
                trade_pnl = 0.0

                # Determine if this is opening/adding vs reducing/closing
                is_opening = (qty_cur == 0) or (qty_cur > 0 and side == "BUY") or (qty_cur < 0 and side == "SELL")

                if is_opening:
                    if sym in state["position_entries"]:
                        # Weighted average entry price when adding to a position
                        old_ep  = state["position_entries"][sym]["entry_price"]
                        old_qty = abs(qty_cur)
                        new_ep  = (old_ep * old_qty + price * qty) / (old_qty + qty)
                        state["position_entries"][sym]["entry_price"] = new_ep
                    else:
                        sl_price = price * (1 - STOP_LOSS_PCT) if side == "BUY" else price * (1 + STOP_LOSS_PCT)
                        tp_price = price * (1 + TAKE_PROFIT_PCT) if side == "BUY" else price * (1 - TAKE_PROFIT_PCT)
                        state["position_entries"][sym] = {
                            "entry_price": price,
                            "entry_date":  str(pd.Timestamp.now(tz="UTC")),
                            "peak_price":  price,
                        }
                        logger.info(
                            f"   ↳ Entry=${price:,.2f} | "
                            f"Stop Loss=${sl_price:,.2f} | "
                            f"Take Profit=${tp_price:,.2f}"
                        )
                else:
                    # Reducing or flipping — compute realized PnL for the closed portion
                    ep = state["position_entries"].get(sym, {}).get("entry_price", price)
                    direction = 1 if qty_cur > 0 else -1
                    trade_pnl = direction * qty * (price - ep)
                    pnl_pct   = direction * (price - ep) / ep * 100 if ep else 0
                    logger.info(f"   ↳ Realised P&L: {trade_pnl:+.2f} ({pnl_pct:+.2f}%)")
                    state["cash_usdt"] += trade_pnl  # approx; reconcile will sync

                    # Clear entry data when fully closed or flipped
                    if abs(new_qty) < step:
                        state["position_entries"].pop(sym, None)
                        new_qty = 0.0

                state["positions"][sym] = new_qty
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
            logger.error(f"Futures order failed {sym}: {e}")

    return state


# ── Signal weight computation ──────────────────────────────────────────────────
def _signal_to_weights(signal_series: pd.Series, long_short: bool = False) -> pd.Series:
    """
    Convert raw signals to position weights with iterative MAX_POSITION_SIZE capping.

    long_short=False  →  long-only (used for live spot trading, spot demo
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
    """Return (target_weights, active_selection) for the LIVE futures portfolio.
    Live book: LIVE_BOOK_STRATEGIES only, inverse-vol (equal-risk) weighted.
    """
    vols = {}
    if hypotheticals:
        for name in LIVE_BOOK_STRATEGIES:
            hyp = hypotheticals.get(name, {})
            nav_hist = hyp.get("nav_history", [])
            if len(nav_hist) >= 12:
                try:
                    dates = pd.to_datetime([h["date"] for h in nav_hist])
                    navs  = pd.Series([h["nav"] for h in nav_hist], index=dates).sort_index()
                    daily = navs.resample("1D").last().dropna()
                    rets  = daily.pct_change().dropna()
                    if len(rets) >= 10:
                        vols[name] = float(rets.std())
                except Exception:
                    pass

    if len(vols) == len(LIVE_BOOK_STRATEGIES):
        inv_vols = {n: 1.0 / max(v, 1e-6) for n, v in vols.items()}
        total    = sum(inv_vols.values())
        weights  = {n: inv_vols[n] / total for n in LIVE_BOOK_STRATEGIES}
    else:
        weights = {n: 1.0 / len(LIVE_BOOK_STRATEGIES) for n in LIVE_BOOK_STRATEGIES}

    selection = [(n, round(weights[n], 4)) for n in LIVE_BOOK_STRATEGIES]
    logger.info(f"Live book (equal-risk weighted): {selection}")

    live_sigs = {k: v for k, v in signals_dict.items() if k in LIVE_BOOK_STRATEGIES}
    blended   = blend_signals(live_sigs, selection, close)
    current_sigs = blended.iloc[-1].reindex(UNIVERSE, fill_value=0)

    if current_sigs.abs().max() == 0:
        logger.warning("No signals from live strategies — holding current positions")
        return pd.Series(0.0, index=UNIVERSE), selection

    return _signal_to_weights(current_sigs, long_short=True), selection


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
                        _bp = float(current_prices.get(sym, 0))
                        trade_history.append({
                            "time":        ts_now,
                            "symbol":      sym,
                            "side":        "BUY" if pw > 0 else "SELL",
                            "qty":         round(prev_nav * abs(pw) / _bp, 6) if _bp > 0 else 0.0,
                            "weight_from": 0.0,
                            "weight_to":   round(pw * 100, 1),
                            "price":       _bp,
                            "hypo_value":  round(prev_nav * abs(pw), 2),
                        })

            # Deduct same transaction costs as live trading so paper Sharpe
            # is net-of-costs and the selector is not biased toward high-turnover strategies.
            _turnover = sum(
                abs(float(new_weights.get(s, 0)) - float(prev_weights.get(s, 0)))
                for s in UNIVERSE
            )
            period_return -= (TRANSACTION_COST_BP + SLIPPAGE_BP) / 10_000 * _turnover

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
                    "qty":         round(hypo_value / price, 6) if price > 0 else 0.0,
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

    # ── Engine controls (kill switch / close-all / override) ──────────────────
    try:
        controls = load_controls()
    except Exception as e:
        logger.warning(f"Could not read engine_controls ({e}) — proceeding normally")
        controls = {}

    if controls.get("close_all_triggered"):
        logger.warning("[CONTROL] CLOSE ALL triggered from dashboard — flattening all positions")
        state = load_state()
        prices = fetch_current_prices()
        exits = {
            sym: prices[sym]
            for sym, qty in state["positions"].items()
            if qty != 0 and sym in prices
        }
        if exits:
            state = execute_exits_only(exits, state)
            nav = compute_nav(state, prices)
            state["nav_history"].append({
                "date":  str(pd.Timestamp.now(tz="UTC")),
                "nav":   round(nav, 2),
                "event": "close_all",
            })
            save_state(state)
        acknowledge_close_all()
        logger.warning("[CONTROL] All positions closed. Kill switch remains ACTIVE — no further orders.")
        return

    if controls.get("kill_switch"):
        logger.warning("[CONTROL] Kill switch ACTIVE — signal rebalance blocked this cycle")
        return
    # ── End engine controls ───────────────────────────────────────────────────

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

    # Funding diagnostic — carry and exhaustion_fade return flat if this is empty/stale.
    if funding is None or funding.empty:
        logger.warning("FUNDING DATA EMPTY — carry and exhaustion_fade will return flat signals")
    else:
        missing_cols = [s for s in UNIVERSE if s not in funding.columns]
        last_ts      = funding.index[-1]
        last_utc     = last_ts if last_ts.tzinfo else pd.Timestamp(last_ts, tz="UTC")
        age_hours    = (pd.Timestamp.now(tz="UTC") - last_utc).total_seconds() / 3600
        logger.info(
            f"Funding data: shape={funding.shape} | "
            f"last={last_ts} ({age_hours:.1f}h ago)"
            + (f" | MISSING SYMBOLS: {missing_cols}" if missing_cols else "")
        )

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

    # 4b. Reconcile positions and cash from Binance before doing anything with NAV.
    #     This keeps state["positions"] and state["cash_usdt"] in sync with reality
    #     and stamps initial_nav from the real account balance if not yet set.
    state = reconcile_with_binance(state, prices)

    # 5. Update hypothetical paper portfolios for all 10 strategies
    logger.info("─" * 55)
    logger.info("Updating hypothetical paper portfolios...")
    update_hypotheticals(signals_dict, prices, state)

    # 6. New target weights for LIVE portfolio
    try:
        if controls.get("override_active"):
            # Manual override from dashboard — skip signal/optimizer entirely
            if controls.get("override_mode") == "quantities":
                nav_for_override = compute_nav(state, prices)
                manual_qty = controls.get("manual_quantities", {})
                new_weights = pd.Series({
                    sym: (float(manual_qty.get(sym, 0)) * prices.get(sym, 0)) / nav_for_override
                    if nav_for_override > 0 else 0.0
                    for sym in UNIVERSE
                }).reindex(UNIVERSE, fill_value=0.0)
                logger.info(f"[OVERRIDE] Using manual quantities: { {k: v for k, v in manual_qty.items() if v} }")
            else:
                manual_w = controls.get("manual_weights", {})
                new_weights = pd.Series(manual_w).reindex(UNIVERSE, fill_value=0.0)
                logger.info(f"[OVERRIDE] Using manual weights: { {k: round(v*100,1) for k, v in manual_w.items() if v} }%")
            state["active_strategies"]       = ["MANUAL_OVERRIDE"]
            state["active_strategy_weights"] = {"MANUAL_OVERRIDE": 1.0}
        else:
            new_weights, active_sel = compute_target_weights(
                seasonality_analyzer=seasonality_analyzer,
                signals_dict=signals_dict,
                close=close,
                hypotheticals=state.get("hypothetical", {}),
            )
            state["active_strategies"]       = [name for name, _ in active_sel]
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

    if not controls.get("override_active"):
        last_rebal_ts = state.get("last_rebalance_date")
        if last_rebal_ts is not None:
            try:
                lt = pd.Timestamp(last_rebal_ts)
                lt = lt if lt.tzinfo else lt.tz_localize("UTC")
                days_since = (pd.Timestamp.now(tz="UTC") - lt).days
                if days_since < LIVE_REBALANCE_FREQ_DAYS:
                    logger.info(
                        f"Weekly cadence: {days_since}d since last rebalance "
                        f"(gate={LIVE_REBALANCE_FREQ_DAYS}d) — skipping orders this cycle"
                    )
                    save_state(state)
                    return
            except Exception as e:
                logger.warning(f"Weekly cadence check error ({e}) — proceeding normally")

    if weight_delta < REBALANCE_THRESHOLD:
        logger.info("Signal unchanged — holding current positions, no orders placed.")
        save_state(state)  # persist hypotheticals
        return

    # 7. Signal shifted — rebalance
    logger.info(f"Signal shift detected (Δ={weight_delta:.4f}) — placing orders...")

    nav = compute_nav(state, prices)
    logger.info(f"Portfolio NAV: ${nav:,.2f} USDT")

    if nav <= 0:
        logger.critical(
            "NAV is zero — Binance Futures wallet appears unfunded. "
            "Fund your futures account and restart the engine. Stopping now."
        )
        os._exit(1)

    state = execute_rebalance(new_weights, state, nav, prices)

    # 8. Persist
    state["current_weights"]      = new_aligned.to_dict()
    state["last_run"]             = str(datetime.now(timezone.utc))
    state["last_rebalance_date"]  = str(pd.Timestamp.now(tz="UTC"))
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
        state = _db_load_state()
        if state is None:
            logger.warning("No live state in DB to report on.")
            return

        logger.info("\n" + "=" * 55)
        logger.info("FINAL REPORT — Live Trading Session")
        logger.info("=" * 55)
        logger.info(f"Cash remaining: ${state['cash_usdt']:,.2f} USDT")

        open_pos = {sym: qty for sym, qty in state["positions"].items() if qty != 0}
        logger.info(f"Open positions ({len(open_pos)}):")
        for sym, qty in open_pos.items():
            logger.info(f"  {sym}: {qty:.6f}")

        trade_log = load_trade_log_for_report()
        logger.info(f"Total trades executed: {len(trade_log)}")

        nav_history = load_nav_history_for_report()
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


# ── Futures leverage setup ─────────────────────────────────────────────────────
def setup_futures_leverage():
    """Set leverage for every universe symbol at engine startup."""
    client = get_client(for_trading=True)
    for sym in UNIVERSE:
        try:
            client.futures_change_leverage(symbol=sym, leverage=FUTURES_LEVERAGE)
            logger.info(f"  Leverage set: {sym} → {FUTURES_LEVERAGE}x")
        except Exception as e:
            logger.warning(f"  Could not set leverage for {sym}: {e}")


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

    logger.info("Setting futures leverage for all universe symbols...")
    setup_futures_leverage()

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
    logger.info("🚀  REAL-TIME ALGO TRADING ENGINE — BINANCE FUTURES DEMO")
    logger.info("=" * 55)
    logger.info(f"  📊 REAL DATA  |  🎮 FUTURES DEMO EXECUTION (virtual money, real prices)")
    logger.info(f"  Leverage         : {FUTURES_LEVERAGE}x on all positions")
    logger.info(f"  Mode             : Long/Short enabled (real shorts via perpetual futures)")
    logger.info(f"  Live book        : {', '.join(LIVE_BOOK_STRATEGIES)} (equal-risk weighted)")
    logger.info(f"  Rebalance cadence: weekly (every {LIVE_REBALANCE_FREQ_DAYS}d)")
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
