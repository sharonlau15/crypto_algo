"""
seasonality/analyzer.py
========================
Seasonality and regime analysis.

Three layers
------------
1. Calendar seasonality   — which strategies perform better by month/DOW
2. Regime classification  — bull / bear / sideways via BTC 200-day MA + vol
3. Strategy selector      — given current regime + season → pick best strategy
                            (or blend of top-N strategies)

This is the "meta-strategy" layer: its output is fed to the live
execution engine to decide which alpha source to trade each day.
"""

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from config.settings import (
    SEASONALITY_MIN_PERIODS,
    REGIME_MA_WINDOW,
    STRATEGY_SELECT_METRIC,
    UNIVERSE,
)


# ── Regime classification ──────────────────────────────────────────────────────
def classify_regime(close_btc: pd.Series, vol_window: int = 30) -> pd.Series:
    """
    Three-state regime: bull / bear / sideways.

    Rules:
      - BTC > 200-day MA AND 30-day vol < 75th percentile → bull
      - BTC < 200-day MA AND 30-day vol < 75th percentile → bear
      - 30-day vol > 75th percentile → sideways (high vol / choppy)

    Returns
    -------
    pd.Series[str]: index=date, values ∈ {"bull", "bear", "sideways"}
    """
    ma       = close_btc.rolling(REGIME_MA_WINDOW).mean()
    log_ret  = np.log(close_btc / close_btc.shift(1))
    roll_vol = log_ret.rolling(vol_window).std() * np.sqrt(365)
    vol_75   = roll_vol.rolling(252).quantile(0.75)

    regime = pd.Series("unknown", index=close_btc.index)
    high_vol = roll_vol > vol_75

    regime[~high_vol & (close_btc > ma)] = "bull"
    regime[~high_vol & (close_btc < ma)] = "bear"
    regime[high_vol]                      = "sideways"

    return regime


# ── Calendar seasonality ───────────────────────────────────────────────────────
class SeasonalityAnalyzer:
    """
    Analyses backtest results to find seasonal patterns per strategy.

    Parameters
    ----------
    backtest_results : dict[str, BacktestResult]
    btc_close        : pd.Series — for regime classification
    """

    def __init__(self, backtest_results: dict, btc_close: pd.Series):
        self.results    = backtest_results
        self.btc_close  = btc_close
        self.regime_series = classify_regime(btc_close)
        self._monthly: pd.DataFrame | None = None
        self._regime:  pd.DataFrame | None = None

    # ── Monthly seasonality ────────────────────────────────────────────────────
    def compute_monthly_seasonality(self) -> pd.DataFrame:
        """
        For each strategy, compute average monthly Sharpe (or return)
        grouped by calendar month.

        Returns
        -------
        pd.DataFrame: index=month(1-12), columns=strategy_names
        """
        records = {}
        for name, result in self.results.items():
            r = result.portfolio_returns.copy()
            r.index = pd.to_datetime(r.index)
            monthly = r.resample("ME").apply(
                lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0
            )
            monthly.index = monthly.index.month
            records[name] = monthly

        df = pd.DataFrame(records)
        df.index.name = "month"
        self._monthly = df
        return df

    def compute_weekly_seasonality(self) -> pd.DataFrame:
        """Average daily Sharpe by day-of-week (0=Mon, 4=Fri)."""
        records = {}
        for name, result in self.results.items():
            r = result.portfolio_returns.copy()
            r.index = pd.to_datetime(r.index)
            by_dow = r.groupby(r.index.dayofweek).apply(
                lambda x: x.mean() / x.std() if x.std() > 0 else 0
            )
            records[name] = by_dow

        df = pd.DataFrame(records)
        df.index.name = "day_of_week"
        return df

    # ── Regime-conditioned performance ─────────────────────────────────────────
    def compute_regime_performance(self) -> pd.DataFrame:
        """
        For each strategy × regime combination, compute Sharpe.

        Returns
        -------
        pd.DataFrame: index=regime, columns=strategy_names
        """
        regimes = ["bull", "bear", "sideways"]
        records = {r: {} for r in regimes}

        for name, result in self.results.items():
            r = result.portfolio_returns.copy()
            r.index = pd.to_datetime(r.index)

            reg = self.regime_series.reindex(r.index, method="ffill")

            for regime in regimes:
                mask = reg == regime
                subset = r[mask]
                if len(subset) < SEASONALITY_MIN_PERIODS:
                    records[regime][name] = np.nan
                    continue
                ann_ret = subset.mean() * 365
                ann_vol = subset.std() * np.sqrt(365)
                sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
                records[regime][name] = round(sharpe, 3)

        self._regime = pd.DataFrame(records).T
        self._regime.index.name = "regime"
        return self._regime

    # ── Strategy selector ──────────────────────────────────────────────────────
    def select_strategy(
        self,
        current_date: pd.Timestamp | None = None,
        top_n: int = 2,
    ) -> list[tuple[str, float]]:
        """
        Given the current date, determine the current regime and season,
        then return the top-N strategies with their selection weights.

        Parameters
        ----------
        current_date : pd.Timestamp — defaults to today
        top_n        : int — how many strategies to blend

        Returns
        -------
        list of (strategy_name, blend_weight) tuples
        """
        if current_date is None:
            current_date = pd.Timestamp.utcnow().normalize()

        current_month  = current_date.month
        current_regime = self._get_current_regime(current_date)

        logger.info(f"Strategy selector: date={current_date.date()} | "
                    f"month={current_month} | regime={current_regime}")

        # Regime-based Sharpe scores
        if self._regime is None:
            self.compute_regime_performance()

        regime_scores = self._regime.loc[current_regime].dropna()

        # Monthly seasonality bonus
        if self._monthly is None:
            self.compute_monthly_seasonality()

        monthly_scores = self._monthly.loc[current_month].dropna() \
            if current_month in self._monthly.index else pd.Series()

        # Blend: 70% regime, 30% monthly seasonality
        combined = {}
        for strat in regime_scores.index:
            try:
                r_score = float(regime_scores.loc[strat]) if strat in regime_scores.index else 0.0
                m_score = float(monthly_scores.loc[strat]) if strat in monthly_scores.index else 0.0
                combined[strat] = 0.7 * r_score + 0.3 * m_score
            except (KeyError, ValueError, TypeError):
                combined[strat] = 0.0

        combined = pd.Series(combined)
        combined = combined.dropna().sort_values(ascending=False)

        if combined.empty:
            logger.warning("No strategy scored — defaulting to top strategy overall")
            return [("momentum", 1.0)]

        # Select top N and assign blend weights proportional to score
        top    = combined.head(top_n)
        scores = top.clip(lower=0)
        total  = scores.sum()
        if total == 0:
            blend_weights = [1.0 / top_n] * top_n
        else:
            blend_weights = (scores / total).tolist()

        selection = list(zip(top.index.tolist(), blend_weights))
        logger.info(f"Selected strategies: {selection}")
        return selection

    def _get_current_regime(self, dt: pd.Timestamp) -> str:
        """Look up or classify the regime at a given date."""
        try:
            return self.regime_series.asof(dt)
        except Exception:
            return "bull"

    # ── Summary report ─────────────────────────────────────────────────────────
    def seasonality_report(self) -> pd.DataFrame:
        """
        Consolidated report: strategy × (regime, month) performance.
        """
        if self._regime is None:
            self.compute_regime_performance()
        if self._monthly is None:
            self.compute_monthly_seasonality()

        regime_df  = self._regime.copy()
        monthly_df = self._monthly.T  # strategy × month

        regime_df.index  = ["regime_" + i for i in regime_df.index]
        monthly_df.index = monthly_df.index  # strategy rows already

        report = pd.concat([regime_df.T, monthly_df], axis=1)
        return report


def blend_signals(
    signals_dict:    dict,          # name → signal DataFrame
    selection:       list[tuple],   # [(name, weight), ...]
    close:           pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine multiple strategy signals into a single blended signal matrix.

    Parameters
    ----------
    signals_dict : dict mapping strategy_name → signal DataFrame
    selection    : list of (name, blend_weight) from select_strategy()
    close        : reference DataFrame for index/columns alignment

    Returns
    -------
    pd.DataFrame — blended signal matrix, same shape as close
    """
    blended = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    total_weight = sum(w for _, w in selection)

    for name, weight in selection:
        if name not in signals_dict:
            logger.warning(f"blend_signals: {name} not found in signals_dict")
            continue
        sig = signals_dict[name].reindex_like(close).fillna(0)
        blended += sig * (weight / total_weight)

    return blended.clip(-1, 1)