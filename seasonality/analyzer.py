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
    STRATEGY_RESELECT_DAYS,
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
        self._cached_selection: list | None = None
        self._selection_date:   pd.Timestamp | None = None

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
            monthly = r.resample("M").apply(
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

    # ── Risk-based weighting ──────────────────────────────────────────────────

    @staticmethod
    def _erc_weights(cov: np.ndarray) -> np.ndarray:
        """
        Equal Risk Contribution weights via SLSQP.
        Each strategy contributes equal marginal risk to the portfolio.
        Falls back to equal weight if optimization fails.
        """
        from scipy.optimize import minimize

        n = cov.shape[0]
        if n == 1:
            return np.array([1.0])

        def objective(w: np.ndarray) -> float:
            w  = np.maximum(w, 1e-10)
            pv = float(w @ cov @ w)
            if pv <= 0:
                return 1e10
            rc = w * (cov @ w) / pv
            return float(sum((rc[i] - rc[j]) ** 2
                             for i in range(n) for j in range(i + 1, n)))

        w0          = np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
        bounds      = [(1e-6, None)] * n
        result      = minimize(objective, w0, method="SLSQP",
                               bounds=bounds, constraints=constraints,
                               options={"ftol": 1e-10, "maxiter": 500})
        w = np.maximum(result.x, 0.0)
        s = w.sum()
        return w / s if s > 0 else w0

    @staticmethod
    def _daily_returns(nav_hist: list, vol_window_days: int) -> pd.Series | None:
        """Resample a NAV history list to daily returns; return None if insufficient."""
        try:
            dates = pd.to_datetime([h["date"] for h in nav_hist])
            navs  = pd.Series([h["nav"] for h in nav_hist], index=dates).sort_index()
            cutoff = navs.index[-1] - pd.Timedelta(days=vol_window_days)
            daily  = navs[navs.index >= cutoff].resample("1D").last().dropna()
            rets   = daily.pct_change().dropna()
            return rets if len(rets) >= 10 else None
        except Exception:
            return None

    def _compute_strategy_weights(
        self,
        candidates: list,
        hypotheticals: dict,
        vol_window_days: int = 90,
        max_weight: float    = 0.40,
    ) -> dict:
        """
        ERC portfolio weights for `candidates` with three safeguards:

        1. Activity gate  — zero trades OR near-zero NAV variance → weight 0.
        2. Volatility floor — clamp σ to max(σ, 0.25 × median_σ) before
                              computing the covariance matrix, so dormant-
                              adjacent strategies cannot absorb all weight.
        3. Per-strategy cap — no single strategy exceeds max_weight (40%).

        Also logs the full strategy × strategy correlation matrix (all
        strategies with sufficient data, not just candidates) and the final
        weight vector so the output is eyeball-able.
        """
        # ── Full correlation matrix across ALL strategies (for visibility) ─────
        all_rets: dict[str, pd.Series] = {}
        for name, hyp in hypotheticals.items():
            r = self._daily_returns(hyp.get("nav_history", []), vol_window_days)
            if r is not None:
                all_rets[name] = r

        if len(all_rets) >= 2:
            corr_df = pd.DataFrame(all_rets).dropna(how="all").corr()
            names_sorted = sorted(corr_df.columns)
            logger.info("─── Strategy return correlation matrix (90d daily) ───")
            header = f"{'':24}" + "  ".join(f"{n[:8]:>8}" for n in names_sorted)
            logger.info(header)
            for r_name in names_sorted:
                row = f"{r_name:<24}" + "  ".join(
                    f"{corr_df.loc[r_name, c]:>8.3f}" if c in corr_df.columns else f"{'—':>8}"
                    for c in names_sorted
                )
                logger.info(row)
            logger.info("─" * 60)

        # ── 1. Activity gate ──────────────────────────────────────────────────
        eligible: list[str] = []
        excluded: list[str] = []
        rets_map: dict[str, pd.Series] = {}

        for name in candidates:
            hyp        = hypotheticals.get(name, {})
            n_trades   = len(hyp.get("trade_history", []))
            nav_hist   = hyp.get("nav_history", [])
            r          = self._daily_returns(nav_hist, vol_window_days)

            if n_trades == 0 or r is None:
                excluded.append(name)
                logger.info(f"  ⊘ {name}: excluded (trades={n_trades}, "
                            f"nav_obs={len(nav_hist)})")
                continue

            # Near-zero variance check (deferred until median is known)
            rets_map[name] = r

        # Compute vols and apply variance floor
        if rets_map:
            vols       = {n: float(r.std()) for n, r in rets_map.items()}
            median_vol = float(np.median(list(vols.values())))
            vol_floor  = max(median_vol * 0.25, 1e-6)

            for name in list(rets_map):
                if vols[name] < vol_floor:
                    excluded.append(name)
                    rets_map.pop(name)
                    logger.info(f"  ⊘ {name}: excluded (σ={vols[name]:.5f} "
                                f"< floor={vol_floor:.5f})")
                else:
                    eligible.append(name)

        if not eligible:
            logger.warning("All candidates excluded — equal-weight fallback")
            w = 1.0 / len(candidates)
            return {n: w for n in candidates}

        # ── 2. Volatility floor in covariance matrix ──────────────────────────
        # Align returns on a common daily index, then floor each series' std.
        ret_df = pd.DataFrame({n: rets_map[n] for n in eligible}).dropna()
        if ret_df.empty or ret_df.shape[0] < 5:
            w = 1.0 / len(eligible)
            return {**{n: w for n in eligible}, **{n: 0.0 for n in excluded}}

        vols_aligned = ret_df.std()
        median_v     = float(vols_aligned.median())
        floor_v      = max(median_v * 0.25, 1e-6)
        for col in ret_df.columns:
            if vols_aligned[col] < floor_v:
                # Rescale the series so its std equals the floor
                scale = floor_v / vols_aligned[col] if vols_aligned[col] > 0 else 1
                ret_df[col] = ret_df[col] * scale

        cov_mat = ret_df.cov().values

        # ── 3. ERC optimization ───────────────────────────────────────────────
        raw_w   = self._erc_weights(cov_mat)
        weights = dict(zip(eligible, raw_w.tolist()))

        # ── 4. Per-strategy cap (redistribute proportionally) ─────────────────
        for _ in range(20):
            over_total = sum(max(0.0, w - max_weight) for w in weights.values())
            if over_total < 1e-8:
                break
            weights = {n: min(w, max_weight) for n, w in weights.items()}
            under   = {n: w for n, w in weights.items() if w < max_weight}
            if not under:
                break
            under_sum = sum(under.values())
            weights   = {
                n: (w + over_total * w / under_sum if w < max_weight else w)
                for n, w in weights.items()
            }

        total = sum(weights.values())
        weights = {n: w / total for n, w in weights.items()}

        # Zero weight for excluded strategies
        for name in excluded:
            weights[name] = 0.0

        # ── Log final weight vector ───────────────────────────────────────────
        logger.info("─── Strategy allocation weights ───")
        for name in sorted(weights, key=lambda x: -weights[x]):
            bar = "█" * max(0, int(weights[name] * 40))
            logger.info(f"  {name:<28} {weights[name]*100:5.1f}%  {bar}")
        logger.info("─" * 40)

        return weights

    # ── Strategy selector ──────────────────────────────────────────────────────
    def select_strategy(
        self,
        current_date: pd.Timestamp | None = None,
        top_n: int = 2,
        hypotheticals: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        Select top-N strategies and assign weights.

        Selection uses regime-conditioned backtest Sharpe.
        Weights use Equal Risk Contribution (ERC) over 90-day daily NAV
        returns with an activity gate (zero-trade strategies excluded),
        a volatility floor (prevents dormant strategies dominating), and
        a per-strategy cap of 40%.  Correlated strategies automatically
        share weight rather than stacking.

        The result is cached for STRATEGY_RESELECT_DAYS (default 30) so
        the portfolio does not rotate on every signal cycle.
        """
        if current_date is None:
            current_date = pd.Timestamp.utcnow().normalize()

        # ── Monthly rebalance gate ─────────────────────────────────────────────
        now = pd.Timestamp.utcnow()
        if (self._selection_date is not None
                and self._cached_selection
                and (now - self._selection_date).days < STRATEGY_RESELECT_DAYS):
            days_left = STRATEGY_RESELECT_DAYS - (now - self._selection_date).days
            logger.info(
                f"Cached selection ({days_left}d until rebalance): {self._cached_selection}"
            )
            return self._cached_selection

        current_month  = current_date.month
        current_regime = self._get_current_regime(current_date)

        logger.info(f"Strategy selector REBALANCING: date={current_date.date()} | "
                    f"month={current_month} | regime={current_regime}")

        # ── Regime-conditioned backtest Sharpe ────────────────────────────────
        if self._regime is None:
            self.compute_regime_performance()
        regime_scores = self._regime.loc[current_regime].dropna()

        if self._monthly is None:
            self.compute_monthly_seasonality()
        monthly_scores = (self._monthly.loc[current_month].dropna()
                          if current_month in self._monthly.index else pd.Series())

        combined = {}
        for strat in regime_scores.index:
            try:
                r_score = float(regime_scores.loc[strat]) if strat in regime_scores.index else 0.0
                m_score = float(monthly_scores.loc[strat]) if strat in monthly_scores.index else 0.0
                combined[strat] = 0.7 * r_score + 0.3 * m_score
            except (KeyError, ValueError, TypeError):
                combined[strat] = 0.0

        backtest_scores = pd.Series(combined).dropna().sort_values(ascending=False)

        if backtest_scores.empty:
            logger.warning("No strategy scored — defaulting to momentum")
            result = [("momentum", 1.0)]
            self._cached_selection = result
            self._selection_date   = now
            return result

        # ── Pick candidates by backtest score, weight by ERC ─────────────────
        candidates = backtest_scores.head(top_n).index.tolist()

        if hypotheticals:
            weights = self._compute_strategy_weights(candidates, hypotheticals)
        else:
            w       = 1.0 / len(candidates)
            weights = {n: w for n in candidates}
            logger.info("No hypothetical data — equal-weight fallback")

        selection = [(name, round(weights.get(name, 1.0 / top_n), 4))
                     for name in candidates]
        logger.info(f"Selected strategies: {selection}")

        self._cached_selection = selection
        self._selection_date   = now
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