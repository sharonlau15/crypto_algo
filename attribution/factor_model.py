"""
attribution/factor_model.py
============================
Performance attribution via OLS factor regression.

Factors used
------------
  BTC_ret   — Bitcoin returns (crypto systematic risk)
  ETH_ret   — Ethereum returns (smart contract factor)
  vol_factor — realized vol change (risk appetite)
  mom_factor — cross-sectional momentum factor (constructed)
  funding    — average funding rate (carry factor)

For each strategy's return series, we run:
  r_t = α + Σ β_k * F_k,t + ε_t

Output
------
  - Factor loadings (betas)
  - Alpha (annualized)
  - R² (how much variance explained by factors)
  - Hedging analysis: what happens to Sharpe if BTC beta = 0
  - Residual (idiosyncratic) return series
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from loguru import logger

from config.settings import UNIVERSE


# ── Factor construction ────────────────────────────────────────────────────────
def build_factor_matrix(
    close:   pd.DataFrame,
    returns: pd.DataFrame,
    funding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construct a daily factor return matrix.

    Factors
    -------
    btc_ret    : BTC daily log return
    eth_ret    : ETH daily log return
    vol_factor : 10-day realized vol change (risk aversion proxy)
    mom_factor : equal-weight portfolio of top-5 minus bottom-5 tokens (XS momentum)
    carry      : average daily funding rate across universe (optional)

    Returns
    -------
    pd.DataFrame: index=date, columns=factor_names
    """
    factors = {}

    # BTC and ETH market factors
    for sym in ["BTCUSDT", "ETHUSDT"]:
        if sym in returns.columns:
            factors[sym.replace("USDT", "").lower() + "_ret"] = returns[sym]

    # Realized vol change factor
    if "BTCUSDT" in returns.columns:
        rv  = returns["BTCUSDT"].rolling(10).std()
        factors["vol_factor"] = rv.pct_change()

    # Cross-sectional momentum factor (constructed)
    if returns.shape[1] >= 10:
        cum_5d = returns.rolling(5).sum()
        ranked = cum_5d.rank(axis=1, ascending=True)
        n = returns.shape[1]
        long_mask  = ranked >= n - 4
        short_mask = ranked <= 5
        long_ret   = returns[long_mask].mean(axis=1)
        short_ret  = returns[short_mask].mean(axis=1)
        factors["mom_factor"] = long_ret - short_ret

    # Carry factor (optional)
    if funding is not None and not funding.empty:
        factors["carry_factor"] = funding.mean(axis=1)

    factor_df = pd.DataFrame(factors).sort_index()
    factor_df = factor_df.dropna(how="all")
    return factor_df


# ── OLS factor regression ──────────────────────────────────────────────────────
def run_factor_regression(
    portfolio_returns: pd.Series,
    factor_matrix:     pd.DataFrame,
    annualize:         int = 365,
) -> dict:
    """
    OLS regression of portfolio returns on factor returns.

    Returns
    -------
    dict with:
      alpha         : annualized intercept
      betas         : dict of factor → beta coefficient
      t_stats       : dict of factor → t-statistic
      p_values      : dict of factor → p-value
      r_squared     : float
      residuals     : pd.Series (idiosyncratic returns)
      information_ratio : alpha / residual_vol (annualized)
    """
    # Align on common dates
    r   = portfolio_returns.dropna()
    F   = factor_matrix.reindex(r.index).dropna()
    r   = r.reindex(F.index)

    if len(r) < 30:
        logger.warning("Insufficient data for factor regression")
        return {}

    X = sm.add_constant(F)
    model = sm.OLS(r, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    alpha_daily = model.params.get("const", 0)
    alpha_ann   = alpha_daily * annualize

    betas   = {col: model.params[col]   for col in F.columns if col in model.params}
    tstats  = {col: model.tvalues[col]  for col in F.columns if col in model.tvalues}
    pvals   = {col: model.pvalues[col]  for col in F.columns if col in model.pvalues}

    resid    = model.resid
    resid_vol = resid.std() * np.sqrt(annualize)
    ir        = alpha_ann / resid_vol if resid_vol > 0 else np.nan

    return {
        "alpha":              round(alpha_ann, 4),
        "betas":              {k: round(v, 4) for k, v in betas.items()},
        "t_stats":            {k: round(v, 3) for k, v in tstats.items()},
        "p_values":           {k: round(v, 4) for k, v in pvals.items()},
        "r_squared":          round(model.rsquared, 4),
        "adj_r_squared":      round(model.rsquared_adj, 4),
        "information_ratio":  round(ir, 3),
        "residuals":          resid,
        "n_obs":              len(r),
    }


# ── Hedging analysis ───────────────────────────────────────────────────────────
def analyze_hedging_impact(
    portfolio_returns:  pd.Series,
    factor_matrix:      pd.DataFrame,
    hedge_factor:       str = "btc_ret",
    annualize:          int = 365,
) -> dict:
    """
    Show what happens to portfolio Sharpe if we hedge out a specific beta.

    Methodology:
      hedged_returns = portfolio_returns - beta_k * factor_k

    Returns
    -------
    dict with pre-hedge and post-hedge metrics.
    """
    reg = run_factor_regression(portfolio_returns, factor_matrix, annualize)
    if not reg or hedge_factor not in reg.get("betas", {}):
        logger.warning(f"Cannot hedge: {hedge_factor} not in regression output")
        return {}

    beta_k = reg["betas"][hedge_factor]
    factor_k = factor_matrix[hedge_factor].reindex(portfolio_returns.index).fillna(0)

    hedged = portfolio_returns - beta_k * factor_k

    def sharpe(r):
        ann_r = r.mean() * annualize
        ann_v = r.std()  * np.sqrt(annualize)
        return ann_r / ann_v if ann_v > 0 else 0

    def max_dd(r):
        cum = (1 + r).cumprod()
        return ((cum - cum.cummax()) / cum.cummax()).min()

    pre  = portfolio_returns.dropna()
    post = hedged.dropna()

    return {
        "hedge_factor":           hedge_factor,
        "hedge_beta":             round(beta_k, 4),
        "pre_hedge_sharpe":       round(sharpe(pre), 3),
        "post_hedge_sharpe":      round(sharpe(post), 3),
        "pre_hedge_max_drawdown": round(max_dd(pre), 4),
        "post_hedge_max_drawdown":round(max_dd(post), 4),
        "pre_hedge_vol":          round(pre.std() * np.sqrt(annualize), 4),
        "post_hedge_vol":         round(post.std() * np.sqrt(annualize), 4),
        "hedged_returns":         hedged,
    }


# ── Attribution summary ────────────────────────────────────────────────────────
def full_attribution_report(
    backtest_results: dict,
    factor_matrix:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Run factor regression for all strategies and return a summary table.

    Returns
    -------
    pd.DataFrame: index=strategy_name, columns=alpha + betas + r2 + IR
    """
    rows = []
    for name, result in backtest_results.items():
        reg = run_factor_regression(result.portfolio_returns, factor_matrix)
        if not reg:
            continue

        row = {"strategy": name, "alpha": reg["alpha"], "r2": reg["r_squared"],
               "IR": reg["information_ratio"]}
        row.update({f"beta_{k}": v for k, v in reg["betas"].items()})
        rows.append(row)

    df = pd.DataFrame(rows).set_index("strategy")
    return df