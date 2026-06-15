"""
utils/reporting.py
==================
Console summary tables and result persistence.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from config.settings import BACKTEST_TEST_START


def _gross_net_table(
    backtest_results: dict,
    overlay_results:  dict | None = None,
) -> pd.DataFrame:
    """
    Build gross OOS Sharpe / net OOS Sharpe / annual turnover table.
    Includes overlay variants (e.g. momentum_vt) when overlay_results is supplied.
    """
    all_results = dict(backtest_results)
    if overlay_results:
        all_results.update(overlay_results)

    rows = []
    for name, r in all_results.items():
        gross_sh = r.gross_oos_metrics.get("sharpe", np.nan) if r.gross_oos_metrics else np.nan
        net_sh   = r.oos_metrics.get("sharpe",       np.nan) if r.oos_metrics       else np.nan
        drag     = round(gross_sh - net_sh, 3) if not (np.isnan(gross_sh) or np.isnan(net_sh)) else np.nan
        rows.append({
            "strategy":             name,
            "gross_sharpe_oos":     gross_sh,
            "net_sharpe_oos":       net_sh,
            "cost_drag":            drag,
            "annual_turnover_pct":  round(r.annual_turnover * 100, 1),
            "rebalance_days":       r.rebalance_count,
        })
    df = pd.DataFrame(rows).set_index("strategy").sort_values("net_sharpe_oos", ascending=False)
    return df


def print_summary_table(
    metrics_summary:  dict,
    attr_report:      pd.DataFrame,
    regime_df:        pd.DataFrame,
    hedge_analysis:   dict,
    backtest_results: dict | None = None,
):
    """Print a formatted performance summary to console."""

    print("\n" + "=" * 80)
    print("  STRATEGY PERFORMANCE SUMMARY")
    print("=" * 80)

    rows = []
    for name, m in metrics_summary.items():
        if "error" in m:
            continue
        rows.append({
            "Strategy":    name,
            "Sharpe":      m.get("sharpe"),
            "Sortino":     m.get("sortino"),
            "CAGR":        f"{m.get('cagr', 0)*100:.1f}%",
            "Max DD":      f"{m.get('max_drawdown', 0)*100:.1f}%",
            "Ann Vol":     f"{m.get('annual_vol', 0)*100:.1f}%",
            "Win Rate":    f"{m.get('win_rate', 0)*100:.1f}%",
        })

    if rows:
        df = pd.DataFrame(rows).set_index("Strategy")
        df = df.sort_values("Sharpe", ascending=False)
        print(df.to_string())

    # P&L Summary
    print("\n" + "=" * 80)
    print("  PROFIT & LOSS SUMMARY")
    print("=" * 80)
    pnl_rows = []
    for name, m in metrics_summary.items():
        if "error" not in m:
            pnl_rows.append({
                "Strategy":          name,
                "Initial Capital":   f"${m.get('initial_capital', 0):,.2f}",
                "Final Capital":     f"${m.get('final_capital', 0):,.2f}",
                "Total P&L":         f"${m.get('total_pnl', 0):,.2f}",
                "Avg Daily P&L":     f"${m.get('avg_daily_pnl', 0):,.2f}",
                "Best Day":          f"${m.get('best_day_pnl', 0):,.2f}",
                "Worst Day":         f"${m.get('worst_day_pnl', 0):,.2f}",
                "Return Multiple":   f"{m.get('pnl_ratio', 1):.2f}x",
            })
    
    if pnl_rows:
        pnl_df = pd.DataFrame(pnl_rows).set_index("Strategy")
        print(pnl_df.to_string())

    if backtest_results is not None:
        gn = _gross_net_table(backtest_results)
        print("\n" + "=" * 80)
        print(f"  GROSS vs NET SHARPE (OOS: {BACKTEST_TEST_START} → today) + ANNUAL TURNOVER")
        print("=" * 80)
        print(gn.to_string())

    print("\n" + "=" * 80)
    print("  FACTOR ATTRIBUTION (alpha, betas, R²)")
    print("=" * 80)
    if not attr_report.empty:
        print(attr_report.round(3).to_string())

    print("\n" + "=" * 80)
    print("  REGIME-CONDITIONAL PERFORMANCE (Sharpe)")
    print("=" * 80)
    if not regime_df.empty:
        print(regime_df.round(3).to_string())

    print("\n" + "=" * 80)
    print(f"  HEDGING ANALYSIS — Factor: {hedge_analysis.get('hedge_factor', 'N/A')}")
    print("=" * 80)
    for k, v in hedge_analysis.items():
        if k not in ("hedged_returns",):
            print(f"  {k:<30} {v}")

    print("=" * 80 + "\n")


def save_results(
    backtest_results: dict,
    monthly_df:       pd.DataFrame,
    regime_df:        pd.DataFrame,
    attr_report:      pd.DataFrame,
    season_report:    pd.DataFrame,
    output_dir:       Path,
    overlay_results:  dict | None = None,
):
    """Persist all results as CSV and JSON for dashboard consumption."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge overlay variants for complete persistence
    all_results = dict(backtest_results)
    if overlay_results:
        all_results.update(overlay_results)

    # Strategy metrics as JSON — full period + in-sample + out-of-sample split
    metrics = {
        name: {
            "full":          r.metrics,
            "in_sample":     r.in_sample_metrics,
            "out_of_sample": r.oos_metrics,
        }
        for name, r in all_results.items()
    }
    with open(output_dir / "strategy_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Portfolio return series (backtest_results only — overlay uses same returns)
    returns_df = pd.DataFrame({
        name: r.portfolio_returns
        for name, r in backtest_results.items()
    })
    returns_df.to_csv(output_dir / "portfolio_returns.csv")

    # Seasonality
    monthly_df.to_csv(output_dir / "monthly_seasonality.csv")
    regime_df.to_csv(output_dir / "regime_performance.csv")
    season_report.to_csv(output_dir / "seasonality_report.csv")

    # Attribution
    if not attr_report.empty:
        attr_report.to_csv(output_dir / "attribution_report.csv")

    # P&L Metrics (backtest_results only — overlay variants share IS capital)
    pnl_metrics = {
        name: {
            "initial_capital": r.metrics.get("initial_capital"),
            "final_capital":   r.metrics.get("final_capital"),
            "total_pnl":       r.metrics.get("total_pnl"),
            "avg_daily_pnl":   r.metrics.get("avg_daily_pnl"),
            "best_day_pnl":    r.metrics.get("best_day_pnl"),
            "worst_day_pnl":   r.metrics.get("worst_day_pnl"),
            "pnl_ratio":       r.metrics.get("pnl_ratio"),
        }
        for name, r in backtest_results.items()
    }
    pnl_df = pd.DataFrame(pnl_metrics).T
    pnl_df.to_csv(output_dir / "pnl_summary.csv")

    # Cumulative returns
    cum_returns = (1 + returns_df).cumprod()
    cum_returns.to_csv(output_dir / "cumulative_returns.csv")

    # Gross vs net Sharpe table — includes overlay variants
    gn = _gross_net_table(backtest_results, overlay_results)
    gn.to_csv(output_dir / "gross_net_sharpe.csv")

    logger.success(f"Results saved to {output_dir}")

    # Persist all metrics (full-period + OOS) to PostgreSQL
    try:
        from db.connection import get_conn, put_conn
        conn = get_conn()
        try:
            cur = conn.cursor()

            # Idempotent migration: add OOS columns if not present
            cur.execute("""
                ALTER TABLE strategy_metrics
                    ADD COLUMN IF NOT EXISTS gross_sharpe_oos    NUMERIC,
                    ADD COLUMN IF NOT EXISTS net_sharpe_oos      NUMERIC,
                    ADD COLUMN IF NOT EXISTS cost_drag           NUMERIC,
                    ADD COLUMN IF NOT EXISTS annual_turnover_pct NUMERIC,
                    ADD COLUMN IF NOT EXISTS rebalance_days      INTEGER
            """)
            conn.commit()

            for name, r in all_results.items():
                m        = r.metrics or {}
                gross_sh = r.gross_oos_metrics.get("sharpe") if r.gross_oos_metrics else None
                net_sh   = r.oos_metrics.get("sharpe")       if r.oos_metrics       else None
                drag     = (round(gross_sh - net_sh, 6)
                            if gross_sh is not None and net_sh is not None else None)
                turnover = round(r.annual_turnover * 100, 2)

                cur.execute("""
                    INSERT INTO strategy_metrics
                        (strategy, sharpe, sortino, calmar, cagr, max_drawdown,
                         total_return, win_rate, profit_factor,
                         gross_sharpe_oos, net_sharpe_oos, cost_drag,
                         annual_turnover_pct, rebalance_days,
                         recorded_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                    ON CONFLICT (strategy) DO UPDATE SET
                        sharpe               = EXCLUDED.sharpe,
                        sortino              = EXCLUDED.sortino,
                        calmar               = EXCLUDED.calmar,
                        cagr                 = EXCLUDED.cagr,
                        max_drawdown         = EXCLUDED.max_drawdown,
                        total_return         = EXCLUDED.total_return,
                        win_rate             = EXCLUDED.win_rate,
                        profit_factor        = EXCLUDED.profit_factor,
                        gross_sharpe_oos     = EXCLUDED.gross_sharpe_oos,
                        net_sharpe_oos       = EXCLUDED.net_sharpe_oos,
                        cost_drag            = EXCLUDED.cost_drag,
                        annual_turnover_pct  = EXCLUDED.annual_turnover_pct,
                        rebalance_days       = EXCLUDED.rebalance_days,
                        recorded_at          = EXCLUDED.recorded_at
                """, (
                    name,
                    m.get("sharpe"),       m.get("sortino"),
                    m.get("calmar"),       m.get("cagr"),
                    m.get("max_drawdown"), m.get("total_return"),
                    m.get("win_rate"),     m.get("profit_factor"),
                    gross_sh, net_sh, drag, turnover, r.rebalance_count,
                ))
            conn.commit()
            logger.success("Strategy metrics (full + OOS) written to PostgreSQL")
        finally:
            put_conn(conn)
    except Exception as e:
        logger.warning(f"DB write for strategy_metrics skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH REPORTING — overlay comparison, correlation matrix, eligibility gate
# ══════════════════════════════════════════════════════════════════════════════

def print_overlay_comparison(
    base_results:    dict,
    overlay_results: dict,
):
    """
    Side-by-side gross/net OOS Sharpe table for strategies WITH and WITHOUT
    the vol-targeting overlay.  `overlay_results` keys are "{name}_vt".
    """
    logger.info("=" * 90)
    logger.info("  VOL-TARGETING OVERLAY — EFFECT ON OOS SHARPE")
    logger.info(f"  (OOS: {BACKTEST_TEST_START} → today | target_vol=15% | band=±20%)")
    logger.info("=" * 90)

    rows = []
    overlay_base_names = {k.replace("_vt", "") for k in overlay_results}

    for name in sorted(overlay_base_names):
        if name not in base_results:
            continue
        base = base_results[name]
        vt   = overlay_results.get(f"{name}_vt")

        def _g(r):
            return (
                r.gross_oos_metrics.get("sharpe", np.nan) if r and r.gross_oos_metrics else np.nan
            )
        def _n(r):
            return r.oos_metrics.get("sharpe", np.nan) if r and r.oos_metrics else np.nan
        def _t(r):
            return round(r.annual_turnover * 100, 1) if r else np.nan

        rows.append({
            "strategy":          name,
            "gross_oos (base)":  _g(base),
            "net_oos  (base)":   _n(base),
            "gross_oos (+VT)":   _g(vt),
            "net_oos  (+VT)":    _n(vt),
            "turnover% (base)":  _t(base),
            "turnover% (+VT)":   _t(vt),
        })

    if rows:
        df = pd.DataFrame(rows).set_index("strategy")
        for line in df.round(3).to_string().splitlines():
            logger.info(line)
    logger.info("=" * 90)


def print_oos_correlation_matrix(
    backtest_results: dict,
    flag_threshold:   float = 0.5,
):
    """
    Compute pairwise return correlations over the OOS period.
    Flags any pair with |ρ| > flag_threshold.

    Correlation is computed on NET daily return series, filtered to the
    OOS period so it does not reflect in-sample fit.
    """
    test_cutoff = pd.Timestamp(BACKTEST_TEST_START)

    oos_returns = {}
    for name, r in backtest_results.items():
        ret = r.portfolio_returns.dropna()
        idx = ret.index
        if idx.tz is not None:
            test_ts = pd.Timestamp(BACKTEST_TEST_START, tz="UTC")
        else:
            test_ts = test_cutoff
        oos = ret[idx >= test_ts]
        if len(oos) >= 30:
            oos_returns[name] = oos

    if len(oos_returns) < 2:
        logger.info("[OOS correlation] Insufficient data — skipping.")
        return

    corr = pd.DataFrame(oos_returns).dropna(how="all").corr()
    names = sorted(corr.columns)
    corr  = corr.loc[names, names]

    logger.info("=" * 90)
    logger.info(f"  OOS RETURN CORRELATION MATRIX  (post {BACKTEST_TEST_START})")
    logger.info("=" * 90)
    for line in corr.round(2).to_string().splitlines():
        logger.info(line)

    # Flag high-correlation pairs
    flags = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho = corr.iloc[i, j]
            if abs(rho) > flag_threshold:
                flags.append(
                    f"  ⚠  |ρ({names[i]}, {names[j]})| = {rho:.2f} > {flag_threshold}"
                )
    if flags:
        logger.info(f"  High-correlation pairs (|ρ| > {flag_threshold}):")
        for f in flags:
            logger.info(f)
    else:
        logger.info(f"  No pairs with |ρ| > {flag_threshold}.")
    logger.info("=" * 90)


def print_eligibility_gate(
    backtest_results:       dict,
    live_book:              list | None = None,
    gross_sharpe_threshold: float = 0.5,
    net_sharpe_threshold:   float = 0.0,
    max_turnover_pct:       float = 1000.0,
):
    """
    Apply the promotion eligibility gate to every strategy NOT already in
    the live book.  Reports PASS/FAIL with the three contributing metrics.

    Gate: gross OOS Sharpe > threshold  AND  net OOS Sharpe > 0
          AND  annual turnover < 1000%.

    Does NOT promote anything — caller decides.
    """
    if live_book is None:
        from config.settings import LIVE_BOOK_STRATEGIES
        live_book = LIVE_BOOK_STRATEGIES

    logger.info("=" * 90)
    logger.info("  ELIGIBILITY GATE  (research → live promotion criteria)")
    logger.info(
        f"  Criteria: gross_OOS_Sharpe > {gross_sharpe_threshold}  "
        f"AND  net_OOS_Sharpe > {net_sharpe_threshold}  "
        f"AND  turnover < {max_turnover_pct:.0f}%"
    )
    logger.info("=" * 90)

    rows = []
    for name, r in backtest_results.items():
        if name in live_book:
            continue  # already live — not evaluated here

        gross_sh = (r.gross_oos_metrics.get("sharpe", np.nan)
                    if r.gross_oos_metrics else np.nan)
        net_sh   = r.oos_metrics.get("sharpe", np.nan) if r.oos_metrics else np.nan
        turnover = round(r.annual_turnover * 100, 1)

        g_pass = not np.isnan(gross_sh) and gross_sh > gross_sharpe_threshold
        n_pass = not np.isnan(net_sh)   and net_sh   > net_sharpe_threshold
        t_pass = not np.isnan(turnover) and turnover  < max_turnover_pct
        overall = "PASS" if (g_pass and n_pass and t_pass) else "FAIL"

        rows.append({
            "strategy":           name,
            "gross_OOS_Sharpe":   round(gross_sh, 3) if not np.isnan(gross_sh) else "N/A",
            "net_OOS_Sharpe":     round(net_sh, 3)   if not np.isnan(net_sh)   else "N/A",
            "annual_turnover%":   turnover,
            "gross_ok":           "Y" if g_pass else "N",
            "net_ok":             "Y" if n_pass else "N",
            "turnover_ok":        "Y" if t_pass else "N",
            "gate":               overall,
        })

    if rows:
        df = pd.DataFrame(rows).set_index("strategy")
        for line in df.to_string().splitlines():
            logger.info(line)
    else:
        logger.info("  All strategies already in live book — nothing to evaluate.")
    logger.info("=" * 90)