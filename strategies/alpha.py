"""
strategies/alpha.py
===================
All 10 alpha strategies.

Strategy list
-------------
 1. MomentumStrategy          — 12-1 month time-series momentum
 2. MeanReversionStrategy     — z-score mean reversion
 3. CrossSectionalMomentum    — rank-based cross-sectional momentum
 4. VolBreakoutStrategy       — ATR channel breakout
 5. PairsTradingStrategy      — dynamic pair selection (correlation + cointegration + distance)
 6. MLSignalStrategy          — LightGBM on lagged features
 7. MacroRotationStrategy     — BTC regime risk-on/off
 8. CarryStrategy             — funding rate carry
 9. SentimentStrategy         — Fear & Greed index overlay
10. ExhaustionFadeStrategy    — BB breach + extreme funding + ADX ranging fade
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import lightgbm as lgb
from loguru import logger

from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


# ══════════════════════════════════════════════════════════════════════════════
# 1. MOMENTUM (12-1 month time-series)
# ══════════════════════════════════════════════════════════════════════════════
class MomentumStrategy(BaseStrategy):
    """
    Classic 12-1 month momentum.
    Return = cumulative return over [lookback_short, lookback_long] days.
    Top N → long, Bottom N → short.
    Economic justification: persistent winner/loser effect documented
    across all asset classes (Jegadeesh & Titman 1993, Asness 2014).
    """

    def __init__(self):
        super().__init__("momentum", STRATEGY_PARAMS["momentum"])

    def generate_signals(self, close, returns, **kwargs):
        p = self.params
        lb_long  = p["lookback_long"]
        lb_short = p["lookback_short"]
        top_n    = p["top_n"]
        bot_n    = p["bottom_n"]

        ret_long  = close / close.shift(lb_long)  - 1
        ret_short = close / close.shift(lb_short) - 1
        score = ret_long - ret_short

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for dt in score.index:
            row = score.loc[dt].dropna()
            if len(row) < top_n + bot_n:
                continue
            ranked = row.rank(ascending=True)
            n = len(ranked)
            signals.loc[dt, ranked[ranked >= n - top_n + 1].index]  =  1.0
            signals.loc[dt, ranked[ranked <= bot_n].index]           = -1.0

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 2. MEAN REVERSION (z-score)
# ══════════════════════════════════════════════════════════════════════════════
class MeanReversionStrategy(BaseStrategy):
    """
    Rolling z-score of price relative to its own moving average.
    Long when z < -entry_z, short when z > +entry_z.
    Economic justification: short-term crypto overreaction creates
    exploitable reversion windows (typically 3-20 days).
    """

    def __init__(self):
        super().__init__("mean_reversion", STRATEGY_PARAMS["mean_reversion"])

    def generate_signals(self, close, returns, **kwargs):
        p = self.params
        w  = p["zscore_window"]
        ez = p["entry_z"]

        roll_mean = close.rolling(w).mean()
        roll_std  = close.rolling(w).std()
        zscore    = (close - roll_mean) / roll_std.replace(0, np.nan)

        signals = (-zscore / (ez * 3)).clip(-1, 1).fillna(0)
        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-SECTIONAL MOMENTUM (rank-based)
# ══════════════════════════════════════════════════════════════════════════════
class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    Rank all tokens by their N-day return each day.
    Convert rank to a continuous signal in [-1, 1].
    Economic justification: relative momentum removes systematic beta,
    isolating idiosyncratic winner/loser dynamics.
    """

    def __init__(self):
        super().__init__("cross_sectional_momentum",
                         STRATEGY_PARAMS["cross_sectional_momentum"])

    def generate_signals(self, close, returns, **kwargs):
        lb     = self.params["lookback"]
        method = self.params["rank_method"]

        cum_ret = close / close.shift(lb) - 1
        ranked  = cum_ret.rank(axis=1, ascending=True, method=method)
        n_valid = cum_ret.notna().sum(axis=1)
        signals = (ranked.sub(1, axis=0)).div(n_valid - 1, axis=0) * 2 - 1
        signals = signals.where(cum_ret.notna(), np.nan)
        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 4. VOLATILITY BREAKOUT (ATR channels)
# ══════════════════════════════════════════════════════════════════════════════
class VolBreakoutStrategy(BaseStrategy):
    """
    ATR-based channel breakout. Continuous signal proportional to how far
    price has moved relative to the ATR midpoint, scaled by band half-width.
    Economic justification: volatility expansion after compression signals
    the start of a new directional move.
    """

    def __init__(self):
        super().__init__("vol_breakout", STRATEGY_PARAMS["vol_breakout"])

    @staticmethod
    def _atr(high, low, close, period):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def generate_signals(self, close, returns, **kwargs):
        high = kwargs.get("high", close)
        low  = kwargs.get("low",  close)
        p    = self.params

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for sym in close.columns:
            atr       = self._atr(high[sym], low[sym], close[sym], p["atr_period"])
            midpoint  = close[sym].shift(1)
            half_band = (p["atr_multiplier"] * atr).replace(0, np.nan)
            signals[sym] = ((close[sym] - midpoint) / half_band).clip(-1, 1).fillna(0)

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 5. PAIRS TRADING (dynamic pair selection)
# ══════════════════════════════════════════════════════════════════════════════
class PairsTradingStrategy(BaseStrategy):
    """
    Dynamically selects the best-fitting pairs from the full universe using
    three complementary measures, then trades z-score spread mean reversion.

    Pair selection (re-run every reselect_freq bars):
      1. Pearson correlation of log returns — screens for co-movement
      2. Engle-Granger cointegration test   — confirms long-run equilibrium
      3. Normalized price distance (SSD)    — rewards tight historical tracking

    Combined score = |ρ| × (1 − p_coint) / (1 + distance)

    Only pairs that pass both min_correlation and max_coint_pval thresholds
    are eligible. The top N scoring pairs are traded simultaneously.

    For each selected pair the signal is the rolling z-score of the OLS
    spread, with equal weight split across all active pairs.
    """

    def __init__(self):
        super().__init__("pairs_trading", STRATEGY_PARAMS["pairs_trading"])

    def _select_pairs(
        self,
        log_window: pd.DataFrame,
        top_n: int,
        min_corr: float,
        max_pval: float,
    ) -> list[dict]:
        """Score all candidate pairs and return the top N that pass filters."""
        cols = log_window.columns.tolist()
        candidates: list[dict] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                sym1, sym2 = cols[i], cols[j]
                aligned = log_window[[sym1, sym2]].dropna()

                if len(aligned) < 40:
                    continue

                s1, s2 = aligned[sym1], aligned[sym2]

                # 1. Correlation gate (cheap — run first to skip bad pairs early)
                corr = s1.corr(s2)
                if abs(corr) < min_corr:
                    continue

                # 2. Cointegration test (Engle-Granger)
                try:
                    _, pvalue, _ = coint(s1, s2)
                except Exception:
                    continue
                if pvalue > max_pval:
                    continue

                # 3. Normalized price distance (sum of squared differences)
                n1 = (s1 - s1.mean()) / (s1.std() + 1e-10)
                n2 = (s2 - s2.mean()) / (s2.std() + 1e-10)
                distance = ((n1 - n2) ** 2).mean()

                # OLS hedge ratio for spread construction
                beta = float(np.polyfit(s2.values, s1.values, 1)[0])

                score = abs(corr) * (1.0 - pvalue) / (1.0 + distance)
                candidates.append({
                    "sym1": sym1, "sym2": sym2,
                    "corr": corr, "pvalue": pvalue,
                    "distance": distance, "beta": beta,
                    "score": score,
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:top_n]

        if selected:
            summary = " | ".join(
                f"{p['sym1'].replace('USDT','')}/{p['sym2'].replace('USDT','')}"
                f"(ρ={p['corr']:.2f}, p={p['pvalue']:.3f})"
                for p in selected
            )
            logger.debug(f"PairsTrading selected: {summary}")

        return selected

    def generate_signals(self, close, returns, **kwargs):
        p             = self.params
        lb            = p["lookback"]
        ez            = p["entry_z"]
        top_n         = p["top_pairs"]
        min_corr      = p["min_correlation"]
        max_pval      = p["max_coint_pval"]
        reselect_freq = p["reselect_freq"]

        signals   = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        log_close = np.log(close.clip(lower=1e-10))

        if len(close) < lb + 30:
            return signals

        for chunk_start in range(lb, len(close), reselect_freq):
            # Select pairs using data strictly before this chunk
            selection_window = log_close.iloc[max(0, chunk_start - lb):chunk_start]
            pairs = self._select_pairs(selection_window, top_n, min_corr, max_pval)

            if not pairs:
                continue

            chunk_end   = min(chunk_start + reselect_freq, len(close))
            chunk_dates = close.index[chunk_start:chunk_end]
            pair_count  = len(pairs)

            for pair in pairs:
                sym1, sym2, beta = pair["sym1"], pair["sym2"], pair["beta"]

                # Vectorized rolling spread z-score over full history
                spread    = log_close[sym1] - beta * log_close[sym2]
                roll_mean = spread.rolling(lb).mean()
                roll_std  = spread.rolling(lb).std().replace(0, np.nan)
                zscore    = (spread - roll_mean) / roll_std

                pair_sig = (-zscore / (ez * 3)).clip(-1, 1).fillna(0)

                # Accumulate into signals, split weight equally across pairs
                signals.loc[chunk_dates, sym1] += pair_sig.loc[chunk_dates].values / pair_count
                signals.loc[chunk_dates, sym2] -= pair_sig.loc[chunk_dates].values / pair_count

        return signals.clip(-1, 1).fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# 6. ML SIGNAL (LightGBM)
# ══════════════════════════════════════════════════════════════════════════════
class MLSignalStrategy(BaseStrategy):
    """
    Walk-forward LightGBM classifier trained on lagged return features.
    Target: sign of next-day return (1 = up, 0 = down/flat).
    Features: lagged returns at [1, 3, 5, 10, 21] days + rolling vol/skew.
    Strictly no look-ahead: trained on [t-train_window, t-1], predicts t.
    """

    def __init__(self):
        super().__init__("ml_signal", STRATEGY_PARAMS["ml_signal"])
        self._models: dict = {}

    def _build_features(self, returns: pd.Series) -> pd.DataFrame:
        lags = self.params["feature_lookbacks"]
        feats = {}
        for lag in lags:
            feats[f"ret_{lag}d"] = returns.shift(lag)
        feats["vol_10d"]   = returns.rolling(10).std()
        feats["vol_21d"]   = returns.rolling(21).std()
        feats["skew_21d"]  = returns.rolling(21).skew()
        feats["vol_ratio"] = feats["vol_10d"] / feats["vol_21d"]
        return pd.DataFrame(feats)

    def generate_signals(self, close, returns, **kwargs):
        RETRAIN_FREQ = 30

        p       = self.params
        tw      = p["train_window"]
        signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        for sym in close.columns:
            ret = returns[sym].dropna()
            if len(ret) < tw + 30:
                continue

            feats   = self._build_features(ret)
            target  = (ret.shift(-1) > 0).astype(int)
            col_idx = close.columns.get_loc(sym)
            model   = None

            for i in range(tw, len(ret)):
                if model is None or (i - tw) % RETRAIN_FREQ == 0:
                    X_train = feats.iloc[i - tw : i].dropna()
                    y_train = target.iloc[i - tw : i - 1].loc[X_train.iloc[:-1].index]
                    X_train = X_train.iloc[:-1]

                    if len(X_train) < 50 or y_train.nunique() < 2:
                        model = None
                        continue
                    try:
                        model = lgb.LGBMClassifier(
                            n_estimators  = max(30, p["n_estimators"] // 8),
                            max_depth     = 3,
                            learning_rate = p["learning_rate"],
                            verbosity     = -1,
                            n_jobs        = 1,
                        )
                        model.fit(X_train, y_train)
                    except Exception as e:
                        logger.debug(f"MLSignal {sym} retrain@{i}: {e}")
                        model = None
                        continue

                if model is None:
                    continue

                X_pred = feats.iloc[[i]].dropna()
                if X_pred.empty:
                    continue
                try:
                    prob = model.predict_proba(X_pred)[0][1]
                    signals.iloc[i, col_idx] = prob * 2 - 1
                except Exception:
                    pass

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 7. MACRO ROTATION (BTC regime)
# ══════════════════════════════════════════════════════════════════════════════
class MacroRotationStrategy(BaseStrategy):
    """
    BTC 200-day MA regime filter. Continuous regime strength scored as
    BTC % deviation from MA, clipped ±20%, applied equally across all tokens.
    Economic justification: BTC is the crypto risk barometer — altcoins
    correlate strongly with BTC in bear regimes.
    """

    def __init__(self):
        super().__init__("macro_rotation", STRATEGY_PARAMS["macro_rotation"])

    def generate_signals(self, close, returns, **kwargs):
        p     = self.params
        proxy = p["btc_proxy"]

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        if proxy not in close.columns:
            logger.warning("MacroRotation: BTC proxy not in universe")
            return signals

        btc          = close[proxy]
        ma           = btc.rolling(200).mean()
        n            = close.shape[1]
        regime_score = ((btc / ma) - 1).clip(-0.20, 0.20) / 0.20

        for col in close.columns:
            signals[col] = regime_score / n

        return signals.fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# 8. CARRY (funding rate)
# ══════════════════════════════════════════════════════════════════════════════
class CarryStrategy(BaseStrategy):
    """
    Long tokens with highest positive funding (market pays you to hold longs).
    Short tokens with most negative funding.
    Economic justification: funding rate carry is a documented return
    premium in crypto perpetual markets.
    """

    def __init__(self):
        super().__init__("carry", STRATEGY_PARAMS["carry"])

    def generate_signals(self, close, returns, **kwargs):
        funding = kwargs.get("funding_rates")
        p       = self.params
        top_n   = p["top_n"]
        lb      = p["lookback"]

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        if funding is None or funding.empty:
            logger.warning("CarryStrategy: no funding rate data — returning flat signal")
            return signals

        avg_funding = funding.rolling(lb).mean().reindex(close.index, method="ffill")

        for dt in close.index:
            row = avg_funding.loc[dt].dropna()
            if len(row) < 2:
                continue
            ranked = row.rank(ascending=True)
            n = len(ranked)
            signals.loc[dt, ranked[ranked >= n - top_n + 1].index] =  1.0
            signals.loc[dt, ranked[ranked <= top_n].index]          = -1.0

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 9. SENTIMENT (Fear & Greed Index)
# ══════════════════════════════════════════════════════════════════════════════
class SentimentStrategy(BaseStrategy):
    """
    Contrarian overlay using the Crypto Fear & Greed Index (0–100).
    Extreme fear  (< fear_threshold)  → long all (capitulation signal).
    Extreme greed (> greed_threshold) → short all (sentiment overextended).
    Economic justification: retail overreaction creates short-term
    mispricings that institutions revert.
    """

    def __init__(self):
        super().__init__("sentiment", STRATEGY_PARAMS["sentiment"])

    def generate_signals(self, close, returns, **kwargs):
        fg_data = kwargs.get("fear_greed")
        p       = self.params

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        if fg_data is None or fg_data.empty:
            logger.warning("SentimentStrategy: no Fear & Greed data — returning flat")
            return signals

        fg = fg_data["fg_value"].reindex(close.index, method="ffill")
        n  = close.shape[1]

        signals.loc[fg < p["fear_threshold"],  :] =  1.0 / n
        signals.loc[fg > p["greed_threshold"], :] = -1.0 / n

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 10. EXHAUSTION FADE STRATEGY (EFS)
# ══════════════════════════════════════════════════════════════════════════════
class ExhaustionFadeStrategy(BaseStrategy):
    """
    Waits for a crypto futures market to overextend — confirmed by both price
    extremes and overcrowded positioning — then fades the snapback to the mean.

    Three conditions must all align on the same bar:
      1. Price breaches BB(bb_window, bb_std) and closes back inside
         (previous close outside band, current close inside band).
      2. Funding rate is extreme (|funding| > funding_threshold per 8hr)
         in the same direction as the breach — confirming overcrowded positioning.
      3. ADX(adx_period) < adx_threshold — market is ranging, not trending,
         so the fade has room to work.

    Signal direction:
      +1 → lower-band breach + close inside + negative funding (shorts overcrowded) → LONG
      -1 → upper-band breach + close inside + positive funding (longs overcrowded) → SHORT

    Signal strength is scaled by ADX distance from threshold and funding extremity,
    so cleaner setups receive larger allocations.

    Note: The 1.5×ATR hard stop and 12-bar time stop described in the strategy
    spec are enforced by the live engine's risk management layer (STOP_LOSS_PCT,
    trailing stop). The signal itself marks the entry; exits are handled upstream.
    """

    def __init__(self):
        super().__init__("exhaustion_fade", STRATEGY_PARAMS["exhaustion_fade"])

    @staticmethod
    def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    def _adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        up   = high - high.shift(1)
        down = low.shift(1) - low

        dm_plus  = up.where((up > down) & (up > 0), 0.0)
        dm_minus = down.where((down > up) & (down > 0), 0.0)

        atr_s    = self._wilder_smooth(tr, period)
        di_plus  = 100 * self._wilder_smooth(dm_plus,  period) / atr_s.replace(0, np.nan)
        di_minus = 100 * self._wilder_smooth(dm_minus, period) / atr_s.replace(0, np.nan)

        denom = (di_plus + di_minus).replace(0, np.nan)
        dx    = (di_plus - di_minus).abs() / denom * 100
        return self._wilder_smooth(dx, period)

    def generate_signals(self, close, returns, **kwargs):
        p       = self.params
        high    = kwargs.get("high", close)
        low     = kwargs.get("low",  close)
        funding = kwargs.get("funding_rates")

        bb_window   = p["bb_window"]
        bb_std      = p["bb_std"]
        adx_period  = p["adx_period"]
        adx_thresh  = p["adx_threshold"]
        fund_thresh = p["funding_threshold"]

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for sym in close.columns:
            # ── Bollinger Bands ───────────────────────────────────────────────
            ma    = close[sym].rolling(bb_window).mean()
            std   = close[sym].rolling(bb_window).std()
            upper = ma + bb_std * std
            lower = ma - bb_std * std

            # Breach-then-close-inside detection
            prev_above = close[sym].shift(1) > upper.shift(1)
            prev_below = close[sym].shift(1) < lower.shift(1)
            now_inside = (close[sym] <= upper) & (close[sym] >= lower)

            fade_short = prev_above & now_inside   # closed back from above → fade longs
            fade_long  = prev_below & now_inside   # closed back from below → fade shorts

            # ── ADX (ranging condition) ───────────────────────────────────────
            adx     = self._adx(high[sym], low[sym], close[sym], adx_period)
            ranging = adx < adx_thresh

            # ── Funding rate ──────────────────────────────────────────────────
            if (
                funding is not None
                and not funding.empty
                and sym in funding.columns
            ):
                fund = funding[sym].reindex(close.index, method="ffill").fillna(0)
            else:
                fund = pd.Series(0.0, index=close.index)

            # Funding confirms overcrowded positioning in the breach direction:
            #   upper breach + positive funding → longs overcrowded → fade short
            #   lower breach + negative funding → shorts overcrowded → fade long
            fund_confirms_short = fund >  fund_thresh
            fund_confirms_long  = fund < -fund_thresh

            # ── All three conditions ──────────────────────────────────────────
            long_entry  = fade_long  & ranging & fund_confirms_long
            short_entry = fade_short & ranging & fund_confirms_short

            # ── Signal strength ───────────────────────────────────────────────
            # Lower ADX = more room for the fade; more extreme funding = more overcrowding.
            adx_factor  = ((adx_thresh - adx.clip(upper=adx_thresh)) / adx_thresh).fillna(0)
            fund_abs    = fund.abs().clip(upper=fund_thresh * 4)
            fund_factor = (fund_abs / (fund_thresh * 4)).fillna(0)
            strength    = (0.4 + 0.3 * adx_factor + 0.3 * fund_factor).clip(0.3, 1.0)

            signals[sym] = 0.0
            signals.loc[long_entry,  sym] =  strength.loc[long_entry]
            signals.loc[short_entry, sym] = -strength.loc[short_entry]

        return signals.fillna(0)


# ── Factory ────────────────────────────────────────────────────────────────────
def get_all_strategies() -> list[BaseStrategy]:
    """Return instances of all 10 strategies."""
    return [
        MomentumStrategy(),
        MeanReversionStrategy(),
        CrossSectionalMomentumStrategy(),
        VolBreakoutStrategy(),
        PairsTradingStrategy(),
        MLSignalStrategy(),
        MacroRotationStrategy(),
        CarryStrategy(),
        SentimentStrategy(),
        ExhaustionFadeStrategy(),
    ]
