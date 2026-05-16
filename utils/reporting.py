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


def print_summary_table(
    metrics_summary: dict,
    attr_report:     pd.DataFrame,
    regime_df:       pd.DataFrame,
    hedge_analysis:  dict,
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
):
    """Persist all results as CSV and JSON for dashboard consumption."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strategy metrics as JSON — full period + in-sample + out-of-sample split
    metrics = {
        name: {
            "full":         r.metrics,
            "in_sample":    r.in_sample_metrics,
            "out_of_sample": r.oos_metrics,
        }
        for name, r in backtest_results.items()
    }
    with open(output_dir / "strategy_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Portfolio return series
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

    # P&L Metrics
    pnl_metrics = {
        name: {
            "initial_capital": r.metrics.get("initial_capital"),
            "final_capital": r.metrics.get("final_capital"),
            "total_pnl": r.metrics.get("total_pnl"),
            "avg_daily_pnl": r.metrics.get("avg_daily_pnl"),
            "best_day_pnl": r.metrics.get("best_day_pnl"),
            "worst_day_pnl": r.metrics.get("worst_day_pnl"),
            "pnl_ratio": r.metrics.get("pnl_ratio"),
        }
        for name, r in backtest_results.items()
    }
    pnl_df = pd.DataFrame(pnl_metrics).T
    pnl_df.to_csv(output_dir / "pnl_summary.csv")

    # Cumulative returns
    cum_returns = (1 + returns_df).cumprod()
    cum_returns.to_csv(output_dir / "cumulative_returns.csv")

    logger.success(f"Results saved to {output_dir}")

    # Also persist strategy metrics to PostgreSQL
    try:
        from db.connection import get_conn, put_conn
        conn = get_conn()
        try:
            cur = conn.cursor()
            for name, r in backtest_results.items():
                m = r.metrics
                cur.execute("""
                    INSERT INTO strategy_metrics
                        (strategy, sharpe, sortino, calmar, cagr, max_drawdown,
                         total_return, win_rate, profit_factor, recorded_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (strategy) DO UPDATE SET
                        sharpe        = EXCLUDED.sharpe,
                        sortino       = EXCLUDED.sortino,
                        calmar        = EXCLUDED.calmar,
                        cagr          = EXCLUDED.cagr,
                        max_drawdown  = EXCLUDED.max_drawdown,
                        total_return  = EXCLUDED.total_return,
                        win_rate      = EXCLUDED.win_rate,
                        profit_factor = EXCLUDED.profit_factor,
                        recorded_at   = EXCLUDED.recorded_at
                """, (
                    name,
                    m.get("sharpe"),
                    m.get("sortino"),
                    m.get("calmar"),
                    m.get("cagr"),
                    m.get("max_drawdown"),
                    m.get("total_return"),
                    m.get("win_rate"),
                    m.get("profit_factor"),
                ))
            conn.commit()
            logger.success("Strategy metrics written to PostgreSQL")
        finally:
            put_conn(conn)
    except Exception as e:
        logger.warning(f"DB write for strategy_metrics skipped: {e}")