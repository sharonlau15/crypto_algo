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
    MAX_WEIGHT_SUM, LONG_SHORT,
)


# ── Result container ───────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    strategy_name:    str
    portfolio_returns: pd.Series          # Daily P&L
    weights_history:  pd.DataFrame        # w at each rebalance
    signal_history:   pd.DataFrame        # Raw signals
    metrics:          dict = field(default_factory=dict)

    def __post_init__(self):
        self.metrics = compute_metrics(self.portfolio_returns, self.strategy_name)


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

    def run(self, strategy_name: str = "unnamed") -> BacktestResult:
        logger.info(f"Backtesting: {strategy_name}")

        weights_history = pd.DataFrame(
            0.0, index=self.close.index, columns=self.close.columns
        )

        # Determine rebalance dates
        if self.rebal_freq == "1D":
            rebal_dates = self.close.index[RISK_LOOKBACK_DAYS:]
        else:
            rebal_dates = self.close.resample(self.rebal_freq).last().index

        current_weights = pd.Series(0.0, index=self.close.columns)

        for dt in rebal_dates:
            if dt not in self.signals.index:
                continue

            # Covariance from 1-year lookback
            hist_end = self.close.index.get_loc(dt)
            hist_start = max(0, hist_end - RISK_LOOKBACK_DAYS)
            cov_matrix = (
                self.returns.iloc[hist_start:hist_end]
                .cov() * 365
            )

            signal_row = self.signals.loc[dt]

            try:
                new_weights = self.optimizer(
                    signals   = signal_row,
                    cov       = cov_matrix,
                    long_short = LONG_SHORT,
                )
                current_weights = pd.Series(new_weights).reindex(
                    self.close.columns, fill_value=0
                )
            except Exception as e:
                logger.warning(f"Optimizer failed on {dt}: {e} — holding previous weights")

            # T+1 implementation: weight set at dt is used from dt+1 forward
            # (the signal shift in base.py handles this; weights_history records
            # the target weight decided on dt)
            weights_history.loc[dt] = current_weights

        # Forward-fill weights between rebalance dates
        weights_history = weights_history.replace(0, np.nan).ffill().fillna(0)

        # Portfolio return = sum(w_i * r_i) for each day
        # Use T+1 forward returns: weights at T earn returns at T+1
        fwd_returns  = self.returns.shift(-1)
        port_returns = (weights_history * fwd_returns).sum(axis=1)
        port_returns = port_returns.iloc[RISK_LOOKBACK_DAYS:-1]  # trim lookahead tail

        # Apply transaction costs
        port_returns = apply_transaction_costs(
            port_returns,
            weights_history.loc[port_returns.index],
        )

        return BacktestResult(
            strategy_name    = strategy_name,
            portfolio_returns = port_returns,
            weights_history  = weights_history,
            signal_history   = self.signals,
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
        results[name] = bt.run(strategy_name=name)
        logger.success(
            f"{name}: Sharpe={results[name].metrics.get('sharpe')} "
            f"| CAGR={results[name].metrics.get('cagr')}"
        )
    return results