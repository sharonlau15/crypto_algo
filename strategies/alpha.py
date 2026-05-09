"""
strategies/alpha.py
===================
All 10 alpha strategies.

Strategy list
-------------
 1. MomentumStrategy          — 12-1 month time-series momentum
 2. MeanReversionStrategy     — z-score mean reversion
 3. RiskParityStrategy        — inverse volatility weighting
 4. CrossSectionalMomentum    — rank-based cross-sectional momentum
 5. VolBreakoutStrategy       — ATR channel breakout
 6. PairsTradingStrategy      — BTC/ETH cointegration spread
 7. MLSignalStrategy          — LightGBM on lagged features
 8. MacroRotationStrategy     — BTC regime risk-on/off
 9. CarryStrategy             — funding rate carry
10. SentimentStrategy         — Fear & Greed index overlay
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
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

        # Return = log(close[t] / close[t - lb_long]) - log(close[t] / close[t - lb_short])
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
    Exit when |z| < exit_z.
    Economic justification: short-term crypto overreaction to news
    creates exploitable reversion windows (typically 3-20 days).
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

        # Continuous signal: full [-1, +1] range proportional to z-score.
        # Negative z (oversold) → positive signal (long); positive z → short.
        # Clipped at ±3σ to avoid extreme outlier sizing.
        signals = (-zscore / (ez * 3)).clip(-1, 1).fillna(0)
        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 3. RISK PARITY (inverse volatility)
# ══════════════════════════════════════════════════════════════════════════════
class RiskParityStrategy(BaseStrategy):
    """
    Equal risk contribution: each token contributes the same amount
    of portfolio volatility. Weights ∝ 1 / rolling_vol.
    Long-only. Always fully invested.
    Economic justification: diversifies risk rather than capital,
    outperforms equal-weight in high-dispersion environments.
    """

    def __init__(self):
        super().__init__("risk_parity", STRATEGY_PARAMS["risk_parity"])

    def generate_signals(self, close, returns, **kwargs):
        lb = self.params["lookback"]
        roll_vol = returns.rolling(lb).std()

        # Replace inf (zero-vol / stale tokens like MATICUSDT post-migration) with NaN
        # so they are excluded from weighting rather than poisoning the entire row.
        inv_vol = (1.0 / roll_vol).replace([np.inf, -np.inf], np.nan)

        # Normalize per row using only the tokens that have valid vol estimates.
        # pandas sum(axis=1) skips NaN by default, so stale tokens get 0 weight.
        row_sum = inv_vol.sum(axis=1, skipna=True)
        signals = inv_vol.div(row_sum.replace(0, np.nan), axis=0).fillna(0)

        # Reindex to match close shape (returns has one fewer row due to dropna),
        # then forward-fill so the last row always has a valid signal.
        signals = signals.reindex(close.index, fill_value=0).ffill().fillna(0)
        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-SECTIONAL MOMENTUM (rank-based)
# ══════════════════════════════════════════════════════════════════════════════
class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    Rank all tokens by their N-day return each day.
    Convert rank to signal in [-1, 1] range.
    Continuous signal: top-ranked gets +1, bottom-ranked gets -1.
    Economic justification: relative momentum (winners vs losers
    within the same asset class) removes systematic market beta.
    """

    def __init__(self):
        super().__init__("cross_sectional_momentum",
                         STRATEGY_PARAMS["cross_sectional_momentum"])

    def generate_signals(self, close, returns, **kwargs):
        lb     = self.params["lookback"]
        method = self.params["rank_method"]

        cum_ret = close / close.shift(lb) - 1
        # Rank 0..N-1, rescale to [-1, +1]
        ranked  = cum_ret.rank(axis=1, ascending=True, method=method)
        n_valid = cum_ret.notna().sum(axis=1)
        signals = (ranked.sub(1, axis=0)).div(n_valid - 1, axis=0) * 2 - 1
        signals = signals.where(cum_ret.notna(), np.nan)
        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 5. VOLATILITY BREAKOUT (ATR channels)
# ══════════════════════════════════════════════════════════════════════════════
class VolBreakoutStrategy(BaseStrategy):
    """
    ATR-based Donchian channel breakout.
    Long when price closes above upper band (prev_close + atr_mult * ATR).
    Short when price closes below lower band.
    Economic justification: volatility expansion after compression
    signals the start of a new directional move.
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
            atr      = self._atr(high[sym], low[sym], close[sym], p["atr_period"])
            midpoint = close[sym].shift(1)
            half_band = (p["atr_multiplier"] * atr).replace(0, np.nan)
            # Continuous: how far price is above/below the ATR midpoint,
            # scaled by band half-width. Always produces a signal in [-1, +1].
            signals[sym] = ((close[sym] - midpoint) / half_band).clip(-1, 1).fillna(0)

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 6. PAIRS TRADING (BTC / ETH cointegration)
# ══════════════════════════════════════════════════════════════════════════════
class PairsTradingStrategy(BaseStrategy):
    """
    Engle-Granger cointegration between BTC and ETH.
    Trade the spread: spread = log(BTC) - β * log(ETH).
    Long spread when z < -entry_z, short when z > +entry_z.
    β is re-estimated on a rolling window.
    Economic justification: BTC/ETH share macro crypto risk factors;
    idiosyncratic divergence is mean-reverting.
    """

    def __init__(self):
        super().__init__("pairs_trading", STRATEGY_PARAMS["pairs_trading"])

    def generate_signals(self, close, returns, **kwargs):
        p    = self.params
        sym1, sym2 = p["pair"]
        lb   = p["lookback"]
        ez   = p["entry_z"]

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        if sym1 not in close.columns or sym2 not in close.columns:
            logger.warning("Pairs trading: one or both symbols missing from universe")
            return signals

        log1 = np.log(close[sym1])
        log2 = np.log(close[sym2])

        for i in range(lb, len(close)):
            window1 = log1.iloc[i-lb:i]
            window2 = log2.iloc[i-lb:i]

            # OLS hedge ratio — no cointegration gate so signal is always live
            beta   = np.polyfit(window2, window1, 1)[0]
            spread = window1 - beta * window2
            mu, sigma = spread.mean(), spread.std()
            if sigma == 0:
                continue

            current_spread = log1.iloc[i] - beta * log2.iloc[i]
            z = (current_spread - mu) / sigma

            dt = close.index[i]
            # Continuous signal proportional to z-score, clipped at ±3
            strength = float(np.clip(-z / (ez * 3), -1, 1))
            signals.loc[dt, sym1] =  strength
            signals.loc[dt, sym2] = -strength

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 7. ML SIGNAL (LightGBM)
# ══════════════════════════════════════════════════════════════════════════════
class MLSignalStrategy(BaseStrategy):
    """
    Walk-forward LightGBM classifier trained on lagged return features.
    Target: sign of next-day return (1 = up, 0 = down/flat).
    Features: lagged returns at [1, 3, 5, 10, 21] days + rolling vol/skew.
    Strictly no look-ahead: model trained on [t-train_window, t-1],
    predicts t, signal used at t+1.

    Economic justification: Non-linear interaction between short-term
    momentum and volatility regimes not captured by linear models.
    """

    def __init__(self):
        super().__init__("ml_signal", STRATEGY_PARAMS["ml_signal"])
        self._models: dict = {}

    def _build_features(self, returns: pd.Series) -> pd.DataFrame:
        lags = self.params["feature_lookbacks"]
        feats = {}
        for lag in lags:
            feats[f"ret_{lag}d"] = returns.shift(lag)
        feats["vol_10d"]  = returns.rolling(10).std()
        feats["vol_21d"]  = returns.rolling(21).std()
        feats["skew_21d"] = returns.rolling(21).skew()
        feats["vol_ratio"] = feats["vol_10d"] / feats["vol_21d"]
        return pd.DataFrame(feats)

    def generate_signals(self, close, returns, **kwargs):
        """
        Walk-forward LightGBM with periodic retraining every RETRAIN_FREQ days.

        For each prediction at index i:
          - Model is trained on returns[i - train_window : i]  (strictly past)
          - Prediction is made for returns[i]                  (current day)
          - Signal is applied at i+1 by base.py shift(1)       (T+1 execution)

        Retraining every 30 days (not every day) keeps runtime manageable
        while ensuring no future data ever enters the training set.
        """
        RETRAIN_FREQ = 30   # retrain model every N days

        p       = self.params
        tw      = p["train_window"]
        signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        for sym in close.columns:
            ret = returns[sym].dropna()
            if len(ret) < tw + 30:
                continue

            feats  = self._build_features(ret)
            # Target: sign of *next* day's return (what we are trying to predict)
            target = (ret.shift(-1) > 0).astype(int)

            col_idx = close.columns.get_loc(sym)
            model   = None

            for i in range(tw, len(ret)):
                # Retrain on first prediction and then every RETRAIN_FREQ steps
                if model is None or (i - tw) % RETRAIN_FREQ == 0:
                    X_train = feats.iloc[i - tw : i].dropna()
                    # target at row j = sign of ret[j+1]; exclude last row of
                    # window (ret[i-1]) because its label is ret[i] which is
                    # the value we are about to predict — that would be leakage.
                    y_train = target.iloc[i - tw : i - 1].loc[X_train.iloc[:-1].index]
                    X_train = X_train.iloc[:-1]   # align with y_train

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
                    prob = model.predict_proba(X_pred)[0][1]   # P(up)
                    signals.iloc[i, col_idx] = prob * 2 - 1    # rescale to [-1, +1]
                except Exception:
                    pass

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 8. MACRO ROTATION (BTC regime)
# ══════════════════════════════════════════════════════════════════════════════
class MacroRotationStrategy(BaseStrategy):
    """
    Simple regime filter: if BTC is above its 200-day MA → risk-on
    (equal-weight all tokens), else → risk-off (flat or short alts).
    Optionally overlaid with a momentum tilt in risk-on.
    Economic justification: BTC is the crypto risk barometer;
    altcoins correlate strongly with BTC in bear regimes.
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

        btc = close[proxy]
        ma  = btc.rolling(200).mean()
        n   = close.shape[1]

        # Continuous regime strength: BTC % deviation from 200d MA, clipped ±20%
        # Positive = risk-on (long all), Negative = risk-off (short all)
        regime_score = ((btc / ma) - 1).clip(-0.20, 0.20) / 0.20  # [-1, +1]

        for col in close.columns:
            signals[col] = regime_score / n

        return signals.fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# 9. CARRY (funding rate)
# ══════════════════════════════════════════════════════════════════════════════
class CarryStrategy(BaseStrategy):
    """
    Long tokens with highest positive funding rate (market pays you to hold).
    Short tokens with most negative funding rate.
    Funding rates reflect the market's directional bias; positive rate
    means longs pay shorts → contango → carry premium for shorts.
    We go long tokens where positive funding = sustained bull demand.

    Economic justification: Funding rate carry is a well-documented
    return premium in crypto perpetual markets.
    """

    def __init__(self):
        super().__init__("carry", STRATEGY_PARAMS["carry"])

    def generate_signals(self, close, returns, **kwargs):
        funding = kwargs.get("funding_rates")  # pd.DataFrame: index=date, cols=symbols
        p = self.params
        top_n = p["top_n"]
        lb    = p["lookback"]

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        if funding is None or funding.empty:
            logger.warning("CarryStrategy: no funding rate data — returning flat signal")
            return signals

        # Rolling mean funding rate
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
# 10. SENTIMENT (Fear & Greed Index)
# ══════════════════════════════════════════════════════════════════════════════
class SentimentStrategy(BaseStrategy):
    """
    Uses the Crypto Fear & Greed Index (0=extreme fear, 100=extreme greed).
    Contrarian: buy extreme fear, sell/short extreme greed.
    Momentum tilt: trend-follow moderate readings.

    fg > greed_threshold → short all (sentiment overextended)
    fg < fear_threshold  → long all (capitulation signal)
    else                 → flat

    Economic justification: retail sentiment overreaction creates
    short-term mispricings; institutional reversion provides the edge.
    """

    def __init__(self):
        super().__init__("sentiment", STRATEGY_PARAMS["sentiment"])

    def generate_signals(self, close, returns, **kwargs):
        fg_data = kwargs.get("fear_greed")  # pd.DataFrame from ingestion
        p       = self.params

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        if fg_data is None or fg_data.empty:
            logger.warning("SentimentStrategy: no Fear & Greed data — returning flat")
            return signals

        fg = fg_data["fg_value"].reindex(close.index, method="ffill")
        n  = close.shape[1]

        long_mask  = fg < p["fear_threshold"]
        short_mask = fg > p["greed_threshold"]

        signals.loc[long_mask,  :] =  1.0 / n
        signals.loc[short_mask, :] = -1.0 / n

        return signals


# ── Factory: instantiate all strategies at once ────────────────────────────────
def get_all_strategies() -> list[BaseStrategy]:
    """Return instances of all 10 strategies."""
    return [
        MomentumStrategy(),
        MeanReversionStrategy(),
        RiskParityStrategy(),
        CrossSectionalMomentumStrategy(),
        VolBreakoutStrategy(),
        PairsTradingStrategy(),
        MLSignalStrategy(),
        MacroRotationStrategy(),
        CarryStrategy(),
        SentimentStrategy(),
    ]
