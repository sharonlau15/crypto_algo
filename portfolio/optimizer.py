"""
portfolio/optimizer.py
======================
Max-Sharpe portfolio optimizer subject to project constraints:
  - Annualized volatility ≥ 3%  (MIN_ANNUALIZED_VOL)
  - Sum of |weights| ≤ 100%     (MAX_WEIGHT_SUM)
  - Long-only or long-short     (LONG_SHORT)
  - Risk calc: 1-year daily returns covariance (RISK_LOOKBACK_DAYS)

Uses scipy.optimize for transparency (no black-box library).
Fallback: equal-weight if optimization fails (never leaves flat).
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger

from config.settings import (
    MIN_ANNUALIZED_VOL,
    MAX_WEIGHT_SUM,
    MAX_POSITION_SIZE,
    LONG_SHORT,
)


def _portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    """Return (annualized_return, annualized_vol, sharpe) for weight vector w."""
    port_ret = float(w @ mu)
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(max(port_var, 0))
    sharpe   = port_ret / port_vol if port_vol > 1e-8 else 0.0
    return port_ret, port_vol, sharpe


def max_sharpe_optimize(
    signals:    pd.Series,       # Signal vector: [-1, +1] per symbol
    cov:        pd.DataFrame,    # Annualized covariance matrix
    long_short: bool = LONG_SHORT,
    min_vol:    float = MIN_ANNUALIZED_VOL,
    max_w_sum:  float = MAX_WEIGHT_SUM,
    max_pos:    float = MAX_POSITION_SIZE,
) -> dict:
    """
    Maximize Sharpe ratio subject to risk and weight constraints.

    Strategy signals act as an expected return proxy:
      μ_i = signal_i  (directional conviction)

    Returns
    -------
    dict: symbol → optimal weight
    """
    symbols = signals.index.tolist()
    n       = len(symbols)
    sig     = signals.fillna(0).values.astype(float)

    # Use signal as expected return proxy
    mu  = sig.copy()
    cov_arr = cov.reindex(index=symbols, columns=symbols).fillna(0).values

    # Add small regularization to avoid singular covariance
    cov_arr += np.eye(n) * 1e-6

    # ── Bounds ────────────────────────────────────────────────────────────────
    if long_short:
        bounds = [(-max_pos, max_pos)] * n
    else:
        bounds = [(0.0, max_pos)] * n

    # ── Objective: negative Sharpe ─────────────────────────────────────────────
    def neg_sharpe(w):
        _, vol, sharpe = _portfolio_stats(w, mu, cov_arr)
        return -sharpe

    def neg_sharpe_grad(w):
        port_ret = float(w @ mu)
        port_var = float(w @ cov_arr @ w)
        port_vol = np.sqrt(max(port_var, 0))
        if port_vol < 1e-8:
            return np.zeros(n)
        d_vol = (cov_arr @ w) / port_vol
        d_ret = mu
        sharpe = port_ret / port_vol
        return -(d_ret / port_vol - sharpe * d_vol / port_vol)

    # ── Constraints ────────────────────────────────────────────────────────────
    constraints = [
        # |w|_1 ≤ max_w_sum
        {
            "type": "ineq",
            "fun": lambda w: max_w_sum - np.abs(w).sum(),
        },
        # Annualized vol ≥ min_vol  (project spec: 3%)
        # Clamp to 0 before sqrt: w@cov@w can go slightly negative due to
        # floating-point precision in near-singular covariance matrices.
        {
            "type": "ineq",
            "fun": lambda w: float(np.sqrt(max(0.0, float(w @ cov_arr @ w)))) - min_vol,
        },
    ]

    # Initial guess: signal-proportional, normalized
    w0 = sig / (np.abs(sig).sum() + 1e-8) * max_w_sum
    w0 = np.clip(w0, *(bounds[0]))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
        result = minimize(
            fun         = neg_sharpe,
            x0          = w0,
            jac         = neg_sharpe_grad,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"maxiter": 1000, "ftol": 1e-9},
        )

    if not result.success:
        logger.debug(f"Optimizer: {result.message} — falling back to equal-weight")
        return _equal_weight_fallback(signals, long_short, max_w_sum, max_pos)

    weights = pd.Series(result.x, index=symbols)

    # Zero out negligible weights (< 0.1%)
    weights[weights.abs() < 0.001] = 0.0

    # Enforce min vol constraint: if still below, scale up
    port_vol = np.sqrt(max(0.0, float(weights.values @ cov_arr @ weights.values)))
    if port_vol < min_vol and port_vol > 0:
        scale = min_vol / port_vol
        weights = (weights * scale).clip(-max_pos, max_pos)

    return weights.to_dict()


def _equal_weight_fallback(
    signals:    pd.Series,
    long_short: bool,
    max_w_sum:  float,
    max_pos:    float,
) -> dict:
    """Simple fallback: equal-weight long tokens with positive signal."""
    pos_signals = signals[signals > 0]
    neg_signals = signals[signals < 0]

    weights = pd.Series(0.0, index=signals.index)

    if not pos_signals.empty:
        w = min(max_w_sum / len(pos_signals), max_pos)
        weights[pos_signals.index] = w

    if long_short and not neg_signals.empty:
        w = min(max_w_sum / len(neg_signals), max_pos)
        weights[neg_signals.index] = -w

    return weights.to_dict()


def apply_vol_target_overlay(
    weights:    pd.DataFrame,
    returns:    pd.DataFrame,
    target_vol: float = 0.15,
    band:       float = 0.20,
    vol_window: int   = 63,
    max_scale:  float = 2.0,
) -> pd.DataFrame:
    """
    Band-triggered portfolio vol-targeting overlay.

    Scales the ENTIRE weight vector by a single scalar so that realized
    annualized portfolio vol stays near `target_vol`.  The scalar only
    updates when realized vol drifts more than ±`band` fraction away from
    target (banding prevents the high turnover of continuous rescaling).

    Replaces MIN_ANNUALIZED_VOL enforcement — do not stack.

    Parameters
    ----------
    weights    : (T × N) weights DataFrame — same shape as weights_history
    returns    : (T × N) simple/log returns — same index as weights
    target_vol : annualized vol target (default 15%)
    band       : ±fraction corridor; only resize when realized_vol is
                 outside [target*(1-band), target*(1+band)]
    vol_window : trailing days used to estimate realized vol
    max_scale  : hard cap on the multiplicative scaling factor (2 = 2×)

    Returns
    -------
    pd.DataFrame — overlaid weights, same shape and index as `weights`
    """
    from config.settings import MAX_POSITION_SIZE

    common = weights.index.intersection(returns.index)
    w = weights.reindex(common).fillna(0.0)
    r = returns.reindex(common).fillna(0.0)

    # Realized portfolio returns using the ORIGINAL (un-scaled) weights
    port_ret = (w * r).sum(axis=1)

    # Rolling annualized vol
    realized_vol = port_ret.rolling(vol_window).std() * np.sqrt(365)

    low_trigger  = target_vol * (1 - band)   # default 12%
    high_trigger = target_vol * (1 + band)   # default 18%

    # Stateful band logic: update scale only when outside corridor
    scales = np.ones(len(common))
    current_scale = 1.0

    for i in range(len(common)):
        rv = realized_vol.iloc[i]
        if pd.isna(rv) or rv <= 0.0:
            scales[i] = current_scale
            continue
        if rv < low_trigger or rv > high_trigger:
            current_scale = min(target_vol / rv, max_scale)
        scales[i] = current_scale

    scale_series = pd.Series(scales, index=common)

    scaled = w.mul(scale_series, axis=0).clip(-MAX_POSITION_SIZE, MAX_POSITION_SIZE)

    # Restore full original index (rows not in `common` stay zero)
    return scaled.reindex(weights.index).fillna(0.0)


def compute_portfolio_vol(weights: dict, cov: pd.DataFrame) -> float:
    """Compute annualized portfolio volatility from weights and covariance."""
    w = pd.Series(weights).reindex(cov.index, fill_value=0).values
    return float(np.sqrt(max(0.0, float(w @ cov.values @ w))))


def compute_portfolio_sharpe(
    weights: dict,
    signals: pd.Series,
    cov:     pd.DataFrame,
) -> float:
    """Compute expected Sharpe from weights, signal-as-mu, and covariance."""
    w   = pd.Series(weights).reindex(signals.index, fill_value=0).values
    mu  = signals.reindex(cov.index, fill_value=0).values
    ret, vol, sharpe = _portfolio_stats(w, mu, cov.values)
    return sharpe