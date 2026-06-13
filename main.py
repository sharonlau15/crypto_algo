"""
main.py
=======
Master entrypoint for the QF623 crypto algo trading system.

Usage
-----
  # Full backtest + analysis (no live trading)
  python main.py --mode backtest

  # Run live trading on Binance Testnet (zero-touch, blocking)
  python main.py --mode live

  # Backtest then immediately go live
  python main.py --mode full

  # Print latest strategy performance summary
  python main.py --mode report
"""

import argparse
import sys
import os
from pathlib import Path

# Resolve project root relative to this file so the script works on any machine
project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd
from loguru import logger

from config.settings import UNIVERSE, BACKTEST_START, RESULT_DIR
from config.client import check_connectivity
from data.ingestion import (
    get_universe_ohlcv, build_close_matrix, build_return_matrix,
    get_fear_greed_index, get_universe_funding_rates,
)
from strategies.alpha import get_all_strategies
from backtest.engine import run_all_backtests, compute_metrics
from portfolio.optimizer import max_sharpe_optimize
from seasonality.analyzer import SeasonalityAnalyzer, blend_signals
from attribution.factor_model import (
    build_factor_matrix, full_attribution_report, analyze_hedging_impact
)
from execution.live_engine import start_scheduler
from utils.logger import setup_logger
from utils.reporting import print_summary_table, save_results



def run_backtest_pipeline(use_cache: bool = True):
    """
    Full offline backtest pipeline.
    Returns all artifacts needed for live trading.
    """
    logger.info("=" * 60)
    logger.info("QF623 Crypto Algo — Backtest Pipeline")
    logger.info("=" * 60)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    logger.info("Step 1/6: Loading universe data")
    universe_data = get_universe_ohlcv(use_cache=use_cache)
    close         = build_close_matrix(universe_data)
    returns       = build_return_matrix(close)

    high_df = pd.DataFrame({s: universe_data[s]["high"] for s in universe_data})
    low_df  = pd.DataFrame({s: universe_data[s]["low"]  for s in universe_data})

    logger.info(f"Universe: {list(universe_data.keys())}")
    logger.info(f"Date range: {close.index[0].date()} → {close.index[-1].date()}")
    logger.info(f"Total bars: {len(close)}")

    # ── 2. Extra data (funding, sentiment) ────────────────────────────────────
    logger.info("Step 2/6: Loading funding rates and sentiment data")
    funding = get_universe_funding_rates()
    fg      = get_fear_greed_index()

    # ── 3. Generate all signals ────────────────────────────────────────────────
    logger.info("Step 3/6: Generating signals for all 10 strategies")
    strategies   = get_all_strategies()
    signals_dict = {}

    for strategy in strategies:
        logger.info(f"  → {strategy.name}")
        signals = strategy.run(
            close         = close,
            returns       = returns,
            high          = high_df,
            low           = low_df,
            funding_rates = funding,
            fear_greed    = fg,
        )
        signals_dict[strategy.name] = signals

    # ── 4. Backtest all strategies ─────────────────────────────────────────────
    logger.info("Step 4/6: Running walk-forward backtests")
    backtest_results = run_all_backtests(
        strategies   = strategies,
        signals_dict = signals_dict,
        close        = close,
        returns      = returns,
        optimizer    = max_sharpe_optimize,
    )

    # ── 5. Seasonality analysis ────────────────────────────────────────────────
    logger.info("Step 5/6: Seasonality and regime analysis")
    btc_close = close["BTCUSDT"] if "BTCUSDT" in close.columns else close.iloc[:, 0]

    analyzer = SeasonalityAnalyzer(
        backtest_results = backtest_results,
        btc_close        = btc_close,
    )
    monthly_df  = analyzer.compute_monthly_seasonality()
    regime_df   = analyzer.compute_regime_performance()
    selection   = analyzer.select_strategy(top_n=2)
    season_rpt  = analyzer.seasonality_report()

    logger.info(f"Current strategy selection: {selection}")

    # ── 6. Performance attribution ─────────────────────────────────────────────
    logger.info("Step 6/6: Performance attribution")
    factor_matrix = build_factor_matrix(close, returns, funding)
    attr_report   = full_attribution_report(backtest_results, factor_matrix)

    # Hedging analysis for the top strategy
    top_strategy_name = selection[0][0]
    top_returns       = backtest_results[top_strategy_name].portfolio_returns
    hedge_analysis    = analyze_hedging_impact(
        portfolio_returns = top_returns,
        factor_matrix     = factor_matrix,
        hedge_factor      = "btc_ret",
    )

    # ── Print + Save ───────────────────────────────────────────────────────────
    metrics_summary = {
        name: result.metrics for name, result in backtest_results.items()
    }
    print_summary_table(metrics_summary, attr_report, regime_df, hedge_analysis, backtest_results)
    save_results(
        backtest_results = backtest_results,
        monthly_df       = monthly_df,
        regime_df        = regime_df,
        attr_report      = attr_report,
        season_report    = season_rpt,
        output_dir       = RESULT_DIR,
    )

    return {
        "universe_data":      universe_data,
        "close":              close,
        "returns":            returns,
        "strategies":         strategies,
        "signals_dict":       signals_dict,
        "backtest_results":   backtest_results,
        "seasonality_analyzer": analyzer,
        "factor_matrix":      factor_matrix,
        "attr_report":        attr_report,
        "hedge_analysis":     hedge_analysis,
    }


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="QF623 Crypto Algo Trader")
    parser.add_argument(
        "--mode",
        choices=["backtest", "live", "full", "report"],
        default="backtest",
        help="Operating mode",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download of all market data",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Fire one live rebalance immediately on startup (skips waiting for 06:00 UTC)",
    )
    args = parser.parse_args()

    # Connectivity check
    logger.info("Checking Binance connectivity...")
    check_connectivity()

    try:
        if args.mode == "backtest":
            run_backtest_pipeline(use_cache=not args.no_cache)

        elif args.mode == "live":
            # Load saved signals if available, else run quick backtest first
            logger.warning("Starting live mode — ensure PAPER_TRADING=True in settings.py")
            artifacts = run_backtest_pipeline(use_cache=True)
            start_scheduler(
                strategies           = artifacts["strategies"],
                seasonality_analyzer = artifacts["seasonality_analyzer"],
                signals_dict         = artifacts["signals_dict"],
                run_now              = args.run_now,
            )

        elif args.mode == "full":
            artifacts = run_backtest_pipeline(use_cache=not args.no_cache)
            logger.info("Backtest complete. Launching live engine...")
            start_scheduler(
                strategies           = artifacts["strategies"],
                seasonality_analyzer = artifacts["seasonality_analyzer"],
                signals_dict         = artifacts["signals_dict"],
                run_now              = args.run_now,
            )

        elif args.mode == "report":
            import json
            state_file = RESULT_DIR / "live_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                nav_history = pd.DataFrame(state["nav_history"])
                logger.info(f"Current positions:\n{state['positions']}")
                logger.info(f"Cash: ${state['cash_usdt']:.2f} USDT")
                if not nav_history.empty:
                    logger.info(f"NAV history (last 7 days):\n{nav_history.tail(7)}")
            else:
                logger.warning("No live state file found. Run --mode live first.")
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Keyboard interrupt received — initiating graceful shutdown...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
    finally:
        logger.info("✅ Session complete.")


if __name__ == "__main__":
    main()
