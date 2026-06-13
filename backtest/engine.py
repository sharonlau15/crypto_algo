"""
backtest/engine.py
==================
Walk-forward backtester with strict T+1/T+2 execution.

Execution rule (from project spec)
-----------------------------------
  Signal computed from data up to close of day T
  → Portfolio rebalanced at close of T+1
  → Returns measured from T+1 to T+2

This means:
  - signals.shift(1) → rebalance lag
  - returns.shift(-1) on the weight series → forward return
  All implemented transparently via the signal shift in base.py
  and the return calculation here.

Output
------
  BacktestResult dataclass containing:
    - portfolio_returns  : pd.Series
    - weights_history    : pd.DataFrame
    - signal_history     : pd.DataFrame
    - metrics            : dict (Sharpe, Sortino, max_dd, CAGR, etc.)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger

from config.settings import (
    TRANSACTION_COST_BP, SLIPPAGE_BP,
    RISK_LOOKBACK_DAYS, MIN_ANNUALIZED_VOL,
    MAX_WEIGHT_SUM, LONG_SHORT, MAX_POSITION_SIZE,
    BACKTEST_TEST_START, REBALANCE_THRESHOLD, RESULT_DIR,
)


# ── Result container ───────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    strategy_name:     str
    portfolio_returns: pd.Series          # Net daily P&L series (post-cost)
    weights_history:   pd.DataFrame       # w at each rebalance
    signal_history:    pd.DataFrame       # Raw signals
    gross_returns:     pd.Series          = field(default_factory=pd.Series)
    annual_turnover:   float              = 0.0   # avg daily |Δw| × 365
    rebalance_count:   int                = 0
    metrics:           dict = field(default_factory=dict)   # Full-period net metrics
    gross_metrics:     dict = field(default_factory=dict)   # Full-period gross metrics
    in_sample_metrics: dict = field(default_factory=dict)   # In-sample net only
    oos_metrics:       dict = field(default_factory=dict)   # Out-of-sample net only
    gross_oos_metrics: dict = field(default_factory=dict)   # Out-of-sample gross only

    def __post_init__(self):
        self.metrics = compute_metrics(self.portfolio_returns, self.strategy_name)
        if len(self.gross_returns) > 0:
            self.gross_metrics = compute_metrics(
                self.gross_returns, f"{self.strategy_name}[gross]"
            )


# ── Core metrics ───────────────────────────────────────────────────────────────
def compute_metrics(returns: pd.Series, name: str = "", initial_capital: float = 10_000) -> dict:
    """
    Compute standard performance metrics on a daily return series.
    All annualized assuming 365 trading days (crypto never closes).
    """
    r = returns.dropna()
    if len(r) < 30:
        return {"error": "insufficient data"}

    ann    = 365
    mu     = r.mean() * ann
    vol    = r.std()  * np.sqrt(ann)
    sharpe = mu / vol if vol > 0 else np.nan

    downside = r[r < 0].std() * np.sqrt(ann)
    sortino  = mu / downside if downside > 0 else np.nan

    cum     = (1 + r).cumprod()
    rolling_max = cum.cummax()
    dd      = (cum - rolling_max) / rolling_max
    max_dd  = dd.min()

    calmar  = mu / abs(max_dd) if max_dd != 0 else np.nan
    total_r = cum.iloc[-1] - 1 if len(cum) > 0 else np.nan
    n_years = len(r) / ann
    cagr    = (1 + total_r) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    win_rate = (r > 0).mean()
    avg_win  = r[r > 0].mean() if (r > 0).any() else 0
    avg_loss = r[r < 0].mean() if (r < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

    # P&L in absolute terms
    pnl_series = r * initial_capital  # Daily P&L
    total_pnl = pnl_series.sum()
    avg_daily_pnl = pnl_series.mean()
    best_day = pnl_series.max()
    worst_day = pnl_series.min()
    final_capital = initial_capital * cum.iloc[-1] if len(cum) > 0 else initial_capital

    return {
        "strategy":      name,
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "calmar":        round(calmar, 3),
        "cagr":          round(cagr, 4),
        "annual_vol":    round(vol, 4),
        "max_drawdown":  round(max_dd, 4),
        "total_return":  round(total_r, 4),
        "win_rate":      round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "n_days":        len(r),
        # P&L metrics
        "initial_capital":  round(initial_capital, 2),
        "final_capital":    round(final_capital, 2),
        "total_pnl":        round(total_pnl, 2),
        "avg_daily_pnl":    round(avg_daily_pnl, 2),
        "best_day_pnl":     round(best_day, 2),
        "worst_day_pnl":    round(worst_day, 2),
        "pnl_ratio":        round(final_capital / initial_capital, 4) if initial_capital > 0 else 0,
    }


# ── Transaction cost model ─────────────────────────────────────────────────────
def apply_transaction_costs(
    returns:    pd.Series,
    weights:    pd.DataFrame,
    cost_bp:    int = TRANSACTION_COST_BP,
    slip_bp:    int = SLIPPAGE_BP,
) -> pd.Series:
    """
    Deduct round-trip costs proportional to turnover.
    Turnover = sum of |Δweight| per day.
    Total cost per day = turnover * (cost_bp + slip_bp) / 10_000
    """
    total_bp  = (cost_bp + slip_bp) / 10_000
    turnover  = weights.diff().abs().sum(axis=1)
    cost      = turnover * total_bp
    return returns - cost.reindex(returns.index, fill_value=0)


# ── Main backtester ────────────────────────────────────────────────────────────
class WalkForwardBacktester:
    """
    Runs a strategy over historical data with walk-forward portfolio
    construction and proper T+1/T+2 execution timing.

    Parameters
    ----------
    signals      : pd.DataFrame — pre-computed signal matrix (already shifted)
    close        : pd.DataFrame — close price matrix
    returns      : pd.DataFrame — log return matrix
    optimizer    : callable     — takes (signals_row, cov_matrix) → weights dict
    rebal_freq   : str          — pandas offset alias e.g. "1D", "1W"
    """

    def __init__(
        self,
        signals:    pd.DataFrame,
        close:      pd.DataFrame,
        returns:    pd.DataFrame,
        optimizer,
        rebal_freq: str = "1D",
    ):
        self.signals    = signals
        self.close      = close
        self.returns    = returns
        self.optimizer  = optimizer
        self.rebal_freq = rebal_freq

    def run(self, strategy_name: str = "unnamed",
            freeze_between_bands: bool = False) -> BacktestResult:
        logger.info(
            f"Backtesting: {strategy_name}"
            + (" [band-freeze / inv-vol]" if freeze_between_bands else "")
        )

        weights_history = pd.DataFrame(
            0.0, index=self.close.index, columns=self.close.columns
        )

        if self.rebal_freq == "1D":
            rebal_dates = self.close.index[RISK_LOOKBACK_DAYS:]
        else:
            rebal_dates = self.close.resample(self.rebal_freq).last().index

        current_weights  = pd.Series(0.0, index=self.close.columns)
        rebalance_count  = 0
        n_frozen         = 0   # days: direction unchanged, weights held exactly
        n_band_cross     = 0   # days: direction changed for ≥1 token
        n_resize_no_band = 0   # (optimizer mode only) rebalance with no direction change

        # ── Band-freeze mode (momentum / mean_reversion) ──────────────────────
        # Weights computed once per band-state change using inverse-vol allocation.
        # Between band events they are frozen unconditionally — no covariance drift.
        if freeze_between_bands:
            prev_state = None

            for dt in rebal_dates:
                if dt not in self.signals.index:
                    continue

                signal_row = self.signals.loc[dt]
                new_state  = np.sign(signal_row).fillna(0).reindex(
                    self.close.columns, fill_value=0
                )

                # No direction change for any token → freeze
                if prev_state is not None and (new_state - prev_state).abs().sum() < 0.5:
                    weights_history.loc[dt] = current_weights
                    n_frozen += 1
                    continue

                n_band_cross += 1
                prev_state = new_state.copy()

                # Compute per-symbol annualised vol from rolling window
                hist_end   = self.close.index.get_loc(dt)
                hist_start = max(0, hist_end - RISK_LOOKBACK_DAYS)
                vols = (
                    self.returns.iloc[hist_start:hist_end]
                    .std() * np.sqrt(365)
                ).clip(lower=MIN_ANNUALIZED_VOL)

                longs  = list(self.close.columns[new_state > 0])
                shorts = list(self.close.columns[new_state < 0])
                new_w  = pd.Series(0.0, index=self.close.columns)

                if longs:
                    iv = (1.0 / vols[longs]).replace([np.inf, -np.inf], 0)
                    if iv.sum() > 0:
                        lb = 0.5 if shorts else 1.0
                        new_w[longs] = lb * iv / iv.sum()

                if shorts:
                    iv = (1.0 / vols[shorts]).replace([np.inf, -np.inf], 0)
                    if iv.sum() > 0:
                        sb = 0.5 if longs else 1.0
                        new_w[shorts] = -sb * iv / iv.sum()

                new_w = new_w.clip(-MAX_POSITION_SIZE, MAX_POSITION_SIZE)

                if (new_w - current_weights).abs().sum() >= REBALANCE_THRESHOLD:
                    current_weights = new_w
                    rebalance_count += 1

                weights_history.loc[dt] = current_weights

            total = n_frozen + n_band_cross
            logger.info(
                f"  {strategy_name} trigger breakdown — "
                f"band_crossings: {n_band_cross} ({n_band_cross/max(total,1):.1%}) | "
                f"frozen: {n_frozen} ({n_frozen/max(total,1):.1%}) | "
                f"resize_no_band: 0 (impossible in band-freeze mode)"
            )

        # ── Optimizer mode (all other strategies) ─────────────────────────────
        else:
            prev_signal = None
            prev_state  = None

            for dt in rebal_dates:
                if dt not in self.signals.index:
                    continue

                signal_row = self.signals.loc[dt]
                new_state  = np.sign(signal_row).fillna(0).reindex(
                    self.close.columns, fill_value=0
                )

                # Stage 1 — covariance-drift guard.
                if prev_signal is not None and (signal_row - prev_signal).abs().sum() < 1e-6:
                    weights_history.loc[dt] = current_weights
                    n_frozen += 1
                    prev_state = new_state.copy()
                    continue

                state_changed = (
                    prev_state is None
                    or (new_state - prev_state).abs().sum() >= 0.5
                )
                prev_signal = signal_row.copy()
                prev_state  = new_state.copy()

                hist_end   = self.close.index.get_loc(dt)
                hist_start = max(0, hist_end - RISK_LOOKBACK_DAYS)
                cov_matrix = (
                    self.returns.iloc[hist_start:hist_end]
                    .cov() * 365
                )

                try:
                    new_weights = self.optimizer(
                        signals    = signal_row,
                        cov        = cov_matrix,
                        long_short = LONG_SHORT,
                    )
                    tentative = pd.Series(new_weights).reindex(
                        self.close.columns, fill_value=0
                    )

                    # Stage 2 — weight-delta gate.
                    if (tentative - current_weights).abs().sum() >= REBALANCE_THRESHOLD:
                        current_weights = tentative
                        rebalance_count += 1
                        if state_changed:
                            n_band_cross += 1
                        else:
                            n_resize_no_band += 1

                except Exception as e:
                    logger.warning(
                        f"Optimizer failed on {dt}: {e} — holding previous weights"
                    )

                weights_history.loc[dt] = current_weights

            total_rebal = n_band_cross + n_resize_no_band
            logger.info(
                f"  {strategy_name} trigger breakdown — "
                f"band_crossings: {n_band_cross} ({n_band_cross/max(total_rebal,1):.1%} of rebalances) | "
                f"resize_no_band: {n_resize_no_band} ({n_resize_no_band/max(total_rebal,1):.1%} of rebalances) | "
                f"frozen: {n_frozen}"
            )

        # Forward-fill warmup period and any non-1D gaps.
        weights_history = weights_history.replace(0, np.nan).ffill().fillna(0)

        # ── Gross returns (before any transaction costs) ───────────────────────
        fwd_returns   = self.returns.shift(-1)
        gross_returns = (weights_history * fwd_returns).sum(axis=1)
        gross_returns = gross_returns.iloc[RISK_LOOKBACK_DAYS:-1]  # trim lookahead tail

        w_trimmed = weights_history.loc[gross_returns.index]

        # ── Net returns (after transaction costs) ──────────────────────────────
        # Costs are zero on hold days because w_trimmed.diff() = 0 there.
        net_returns = apply_transaction_costs(gross_returns, w_trimmed)

        # ── Turnover stats ─────────────────────────────────────────────────────
        turnover        = w_trimmed.diff().abs().sum(axis=1)
        annual_turnover = float(turnover.mean() * 365)
        cost_bp_rate    = (TRANSACTION_COST_BP + SLIPPAGE_BP) / 10_000
        logger.info(
            f"  {strategy_name}: {rebalance_count} rebalances | "
            f"annual turnover {annual_turnover:.1%} | "
            f"annual cost drag {annual_turnover * cost_bp_rate:.3%}"
        )

        # ── Persist weights for gross-Sharpe verification ──────────────────────
        try:
            w_trimmed.to_parquet(RESULT_DIR / f"weights_{strategy_name}.parquet")
        except Exception as e:
            logger.warning(f"Could not save weights for {strategy_name}: {e}")

        # ── Train / test split ─────────────────────────────────────────────────
        test_cutoff = pd.Timestamp(BACKTEST_TEST_START, tz="UTC")
        net_is      = net_returns[net_returns.index < test_cutoff]
        net_oos     = net_returns[net_returns.index >= test_cutoff]
        gross_oos   = gross_returns[gross_returns.index >= test_cutoff]

        is_metrics      = compute_metrics(net_is,    f"{strategy_name}[in-sample]")
        oos_metrics     = compute_metrics(net_oos,   f"{strategy_name}[out-of-sample]")
        gross_oos_met   = compute_metrics(gross_oos, f"{strategy_name}[gross-oos]")

        if "error" not in oos_metrics:
            logger.info(
                f"  In-sample  Sharpe={is_metrics.get('sharpe')} "
                f"CAGR={is_metrics.get('cagr')}"
            )
            logger.info(
                f"  Out-sample Sharpe={oos_metrics.get('sharpe')} "
                f"CAGR={oos_metrics.get('cagr')} | "
                f"Gross OOS Sharpe={gross_oos_met.get('sharpe')}"
            )
        else:
            logger.info(f"  Out-of-sample period too short to evaluate yet")

        return BacktestResult(
            strategy_name     = strategy_name,
            portfolio_returns = net_returns,
            gross_returns     = gross_returns,
            weights_history   = weights_history,
            signal_history    = self.signals,
            annual_turnover   = annual_turnover,
            rebalance_count   = rebalance_count,
            in_sample_metrics = is_metrics,
            oos_metrics       = oos_metrics,
            gross_oos_metrics = gross_oos_met,
        )


def run_all_backtests(
    strategies:   list,
    signals_dict: dict,     # name → signal DataFrame
    close:        pd.DataFrame,
    returns:      pd.DataFrame,
    optimizer,
) -> dict[str, BacktestResult]:
    """
    Run backtests for all strategies and return results dict.
    """
    results = {}
    for strategy in strategies:
        name = strategy.name
        if name not in signals_dict:
            logger.warning(f"No signals for {name} — skipping")
            continue
        bt = WalkForwardBacktester(
            signals   = signals_dict[name],
            close     = close,
            returns   = returns,
            optimizer = optimizer,
        )
        results[name] = bt.run(
            strategy_name        = name,
            freeze_between_bands = getattr(strategy, "freeze_between_bands", False),
        )
        r = results[name]
        oos = r.oos_metrics
        logger.success(
            f"{name}: Sharpe={r.metrics.get('sharpe')} | CAGR={r.metrics.get('cagr')}"
            + (f" | OOS Sharpe={oos.get('sharpe')} OOS CAGR={oos.get('cagr')}"
               if "error" not in oos else " | OOS: insufficient data")
        )
    return results