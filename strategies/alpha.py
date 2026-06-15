"""
strategies/alpha.py
===================
10 core strategies + 5 research candidates.

Core strategies
---------------
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

Research candidates (paper only, excluded from LIVE_BOOK_STRATEGIES)
---------------------------------------------------------------------
R1. TSMOMVolScaledStrategy      — vol-scaled time-series momentum
R2. CarryNeutralStrategy        — cross-sectionally demeaned dollar-neutral carry
R3. ResidMomentumStrategy       — BTC-beta-neutralized residual momentum
R4. BTCDominanceStrategy        — signal from BTC's share of universe market action
R5. VolSpikeReversionStrategy   — fade |1d return| > 3×ATR + funding sign confirmation
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
        self.freeze_between_bands = True

    def generate_signals(self, close, returns, **kwargs):
        p     = self.params
        lb_long  = p["lookback_long"]
        lb_short = p["lookback_short"]
        top_n    = p["top_n"]
        bot_n    = p["bottom_n"]
        hyst     = p.get("hysteresis", 1)   # extra ranks to cross before exiting

        ret_long  = close / close.shift(lb_long)  - 1
        ret_short = close / close.shift(lb_short) - 1
        score = ret_long - ret_short

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        state   = {sym: 0 for sym in close.columns}

        for dt in score.index:
            row = score.loc[dt].dropna()
            if len(row) < top_n + bot_n:
                continue

            ranked = row.rank(ascending=True)
            n      = len(ranked)

            for sym in close.columns:
                if sym not in ranked.index:
                    state[sym] = 0
                    continue

                r = ranked[sym]

                if state[sym] == 0:
                    if r >= n - top_n + 1:        # top N  → enter long
                        state[sym] = 1
                    elif r <= bot_n:              # bottom N → enter short
                        state[sym] = -1
                elif state[sym] == 1:
                    # Exit long only when rank falls hyst positions below entry threshold
                    if r < n - top_n + 1 - hyst:
                        state[sym] = 0
                elif state[sym] == -1:
                    # Exit short only when rank rises hyst positions above entry threshold
                    if r > bot_n + hyst:
                        state[sym] = 0

            signals.loc[dt] = pd.Series(state)

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 2. MEAN REVERSION (z-score)
# ══════════════════════════════════════════════════════════════════════════════
class MeanReversionStrategy(BaseStrategy):
    """
    Banded entry/exit z-score mean reversion.
    Enter long  when z < -entry_z; hold flat until z >= -exit_z.
    Enter short when z >  entry_z; hold flat until z <=  exit_z.
    Position is flat (0) between bands — no continuous resizing.
    Economic justification: short-term crypto overreaction creates
    exploitable reversion windows (typically 3-20 days).
    """

    def __init__(self):
        super().__init__("mean_reversion", STRATEGY_PARAMS["mean_reversion"])
        self.freeze_between_bands = True

    def generate_signals(self, close, returns, **kwargs):
        p  = self.params
        w  = p["zscore_window"]
        ez = p["entry_z"]   # 2.0 — enter when |z| exceeds this
        xz = p["exit_z"]    # 0.5 — exit when |z| falls below this

        roll_mean = close.rolling(w).mean()
        roll_std  = close.rolling(w).std().replace(0, np.nan)
        zscore    = (close - roll_mean) / roll_std

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for sym in close.columns:
            z_vals  = zscore[sym].values
            col_idx = signals.columns.get_loc(sym)
            pos     = 0.0
            for i in range(len(z_vals)):
                zval = z_vals[i]
                if np.isnan(zval):
                    continue
                if pos == 0.0:
                    if zval > ez:       # price above mean → enter short
                        pos = -1.0
                    elif zval < -ez:    # price below mean → enter long
                        pos = 1.0
                elif pos == -1.0:
                    if zval <= xz:      # z reverted → close short
                        pos = 0.0
                elif pos == 1.0:
                    if zval >= -xz:     # z reverted → close long
                        pos = 0.0
                signals.iloc[i, col_idx] = pos

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
    ATR channel breakout: signal = distance of today's close from the
    rolling `lookback`-day mean, normalised by ATR * multiplier.
    Positive when price is extended above the channel; negative below.
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
            # Rolling mean as channel midpoint — measures breakout from a sustained level
            midpoint  = close[sym].rolling(p["lookback"]).mean().shift(1)
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

                # Skip degenerate series (constant price = zero variance)
                if s1.std() < 1e-8 or s2.std() < 1e-8:
                    continue

                # 1. Correlation gate (cheap — run first to skip bad pairs early)
                corr = s1.corr(s2)
                if not np.isfinite(corr) or abs(corr) < min_corr:
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
                try:
                    beta = float(np.polyfit(s2.values, s1.values, 1)[0])
                    if not np.isfinite(beta):
                        continue
                except (np.linalg.LinAlgError, ValueError):
                    continue

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
    SHORT tokens with highest positive funding (longs pay; shorts receive the premium).
    LONG tokens with most negative funding (shorts pay; longs receive the premium).
    Economic justification: funding rate carry is a documented return
    premium in crypto perpetual markets — be on the receiving side.
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
            # Highest positive funding → SHORT (you receive the premium as short holder)
            # Most negative funding → LONG (you receive the premium as long holder)
            signals.loc[dt, ranked[ranked >= n - top_n + 1].index] = -1.0
            signals.loc[dt, ranked[ranked <= top_n].index]          =  1.0

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


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH CANDIDATES — paper/hypothetical only, NOT in LIVE_BOOK_STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

# ── R1. VOL-SCALED TSMOM ──────────────────────────────────────────────────────
class TSMOMVolScaledStrategy(BaseStrategy):
    """
    Vol-scaled time-series momentum.

    For each token: sign(12-month trailing return) / realized_vol.
    Position size is inversely proportional to each token's own realized
    annualized vol — low-vol tokens get larger allocations.

    Key differences from the existing `momentum` strategy:
      1. Continuous signal (not binary ±1 with hysteresis)
      2. Sizing driven by realized vol, not rank order
      3. Optimizer mode (no band-freeze) — weights recomputed every bar
         using the Max-Sharpe optimizer with vol-scaled signals as μ

    Paper/research only.  Do NOT add to LIVE_BOOK_STRATEGIES until the
    gross OOS Sharpe passes the research gate.
    """

    def __init__(self):
        super().__init__("tsmom_volscaled", STRATEGY_PARAMS["tsmom_volscaled"])
        # freeze_between_bands = False (optimizer mode, not band-freeze)

    def generate_signals(self, close, returns, **kwargs):
        p          = self.params
        lookback   = p["lookback"]    # 252 bars — 12-month trailing return
        vol_window = p["vol_window"]  # 63 bars — 3-month realized vol
        min_vol    = p["min_vol"]     # 5% floor prevents 1/vol blow-up

        # ── Direction: sign of 12-month trailing return ────────────────────────
        trailing_ret = close / close.shift(lookback) - 1
        direction    = np.sign(trailing_ret)     # +1, 0, or -1

        # ── Vol estimate: 3-month annualized realized vol ──────────────────────
        realized_vol = returns.rolling(vol_window).std() * np.sqrt(365)
        realized_vol = realized_vol.clip(lower=min_vol)

        # ── Raw signal: direction ÷ vol (low-vol → larger magnitude) ──────────
        raw = direction / realized_vol

        # ── Cross-sectional normalization: max |signal| on each bar → ±1 ───────
        abs_max = raw.abs().max(axis=1).replace(0, np.nan)
        signals = raw.div(abs_max, axis=0).fillna(0).clip(-1, 1)

        # ── Zero-fill warm-up period (first `lookback` bars have no signal) ────
        signals = signals.where(trailing_ret.notna(), 0.0)

        return signals


# ── R2. CARRY NEUTRAL (cross-sectionally demeaned, dollar-neutral) ─────────────
class CarryNeutralStrategy(BaseStrategy):
    """
    Dollar-neutral funding carry — cross-sectional demeaning removes the
    market-wide funding level so only relative carry is traded.

    Construction (each bar):
      1. Smooth per-token funding over `funding_lookback` days.
      2. Subtract the cross-sectional mean: relative_i = funding_i − mean(funding).
      3. Signal = −relative_i  — high relative funding → SHORT (receive premium).

    Net signal sums to zero across the universe by construction (dollar-neutral).
    Signal normalized row-wise to [−1, +1].

    Research/paper only.  Do NOT add to LIVE_BOOK_STRATEGIES until the
    gross OOS Sharpe passes the research gate.
    """

    def __init__(self):
        super().__init__("carry_neutral", STRATEGY_PARAMS["carry_neutral"])

    def generate_signals(self, close, returns, **kwargs):
        funding = kwargs.get("funding_rates")
        p       = self.params
        lb      = p["funding_lookback"]

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        if funding is None or funding.empty:
            logger.warning("CarryNeutral: no funding data — returning flat signal")
            return signals

        # Align to price index, forward-fill daily gaps
        aligned = funding.reindex(close.index, method="ffill")

        # Short rolling smooth to reduce noise in daily funding prints
        smooth = aligned.rolling(lb, min_periods=max(1, lb // 2)).mean()

        # Cross-sectional demean: each token vs universe average on that day
        cross_mean = smooth.mean(axis=1)
        relative   = smooth.sub(cross_mean, axis=0)   # positive = above-avg funding

        # Direction: high positive relative funding → SHORT (we receive the premium)
        raw = -relative.reindex(columns=close.columns, fill_value=0.0)

        # Row-wise normalization to [−1, +1]
        abs_max = raw.abs().max(axis=1).replace(0, np.nan)
        signals = raw.div(abs_max, axis=0).fillna(0).clip(-1, 1)

        return signals


# ── R3. RESIDUAL MOMENTUM (BTC-beta-neutralized) ───────────────────────────────
class ResidMomentumStrategy(BaseStrategy):
    """
    Momentum on the BTC-factor-stripped residual return series.

    For each token on each bar:
      1. Estimate rolling BTC beta via OLS over `ols_window` days:
             beta_t = cov(r_token, r_btc, w) / var(r_btc, w)
      2. Strip the BTC factor:  resid_t = r_token − beta_t × r_btc
      3. Sum residuals over `mom_lookback` days for the idiosyncratic
         momentum signal.

    Result: correlated-with-BTC moves are removed; the signal captures
    token-specific winner/loser dynamics only.  Cross-sectionally
    normalized to [−1, +1].

    Research/paper only.  Do NOT add to LIVE_BOOK_STRATEGIES until the
    gross OOS Sharpe passes the research gate.
    """

    def __init__(self):
        super().__init__("resid_momentum", STRATEGY_PARAMS["resid_momentum"])

    def generate_signals(self, close, returns, **kwargs):
        p      = self.params
        ols_w  = p["ols_window"]
        mom_lb = p["mom_lookback"]
        proxy  = p.get("btc_proxy", "BTCUSDT")

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        if proxy not in returns.columns:
            logger.warning(f"ResidMomentum: BTC proxy '{proxy}' not in returns columns")
            return signals

        btc_ret      = returns[proxy]
        roll_var_btc = btc_ret.rolling(ols_w).var()

        # Vectorized rolling beta for each token
        resid_df = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for sym in returns.columns:
            roll_cov      = returns[sym].rolling(ols_w).cov(btc_ret)
            beta          = (roll_cov / roll_var_btc.replace(0, np.nan)).fillna(0)
            resid_df[sym] = returns[sym] - beta * btc_ret

        # Cumulative residual return as momentum signal
        cum_resid = resid_df.rolling(mom_lb).sum()

        # Row-wise cross-sectional normalization to [−1, +1]
        abs_max = cum_resid.abs().max(axis=1).replace(0, np.nan)
        signals = cum_resid.div(abs_max, axis=0).fillna(0).clip(-1, 1)

        # Zero-fill during warm-up (first ols_window + mom_lookback bars)
        signals = signals.where(cum_resid.notna(), 0.0)

        # EMA smoothing stabilises the daily signal and cuts excess turnover
        smooth = p.get("signal_smooth", 0)
        if smooth > 0:
            signals = signals.ewm(span=smooth, adjust=False).mean().clip(-1, 1)

        # returns is one row shorter than close — reindex to match close.index
        return signals.reindex(close.index, fill_value=0.0)


# ── R4. BTC DOMINANCE ─────────────────────────────────────────────────────────
class BTCDominanceStrategy(BaseStrategy):
    """
    Signal derived from BTC's share of total universe price-weighted market action.

    Construction:
      1. BTC dominance ratio = BTC_close / sum(all_close on each bar).
      2. Smooth with EMA (smooth_window) to remove daily noise.
      3. Signal = -change in smoothed dominance, applied uniformly to altcoins
         and inversely to BTC:
           • Rising BTC dominance → risk-off; go LONG BTC, SHORT altcoins.
           • Falling BTC dominance → risk-on; go SHORT BTC, LONG altcoins.
      4. Signal magnitude = |z-score of the smoothed change| clipped to [0, 1].

    Rationale: BTC dominance rising signals crypto-wide risk aversion
    (capital rotating from altcoins into BTC as the "safe haven" within
    crypto). Falling dominance signals speculative appetite expanding into
    altcoins.

    Research/paper only.  Do NOT add to LIVE_BOOK_STRATEGIES until the
    gross OOS Sharpe passes the research gate.
    """

    def __init__(self):
        super().__init__("btc_dominance", STRATEGY_PARAMS["btc_dominance"])

    def generate_signals(self, close, returns, **kwargs):
        p       = self.params
        proxy   = p.get("btc_proxy", "BTCUSDT")
        sm_w    = p["smooth_window"]
        sig_sm  = p.get("signal_smooth", 0)

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        if proxy not in close.columns:
            logger.warning(f"BTCDominance: '{proxy}' not in close columns")
            return signals

        # Normalize each token's close to its own first valid price (index-rebased)
        rebased = close.div(close.bfill().iloc[0])

        # BTC share of total universe "market cap proxy"
        total_mkt    = rebased.sum(axis=1).replace(0, np.nan)
        dom_raw      = rebased[proxy] / total_mkt
        dom_smooth   = dom_raw.ewm(span=sm_w, adjust=False).mean()

        # Daily change in smoothed dominance
        dom_change   = dom_smooth.diff()

        # Z-score of the dominance change (21-day rolling)
        dom_z = (
            (dom_change - dom_change.rolling(21).mean())
            / dom_change.rolling(21).std().replace(0, np.nan)
        ).clip(-3, 3).fillna(0)

        n_alts = len(close.columns) - 1   # everything except BTC

        for col in close.columns:
            if col == proxy:
                # BTC: long when dom rising (risk-off), short when dom falling
                signals[col] = dom_z.clip(-1, 1)
            else:
                if n_alts > 0:
                    # Altcoins: inverse of BTC signal, split equally
                    signals[col] = -dom_z.clip(-1, 1) / n_alts

        if sig_sm > 0:
            signals = signals.ewm(span=sig_sm, adjust=False).mean().clip(-1, 1)

        return signals.fillna(0)


# ── R5. VOL SPIKE REVERSION ───────────────────────────────────────────────────
class VolSpikeReversionStrategy(BaseStrategy):
    """
    Fade extreme single-day moves confirmed by overcrowded positioning.

    Entry conditions (all must be true on the same bar):
      1. |return_t| > spike_mult × ATR(atr_period)   — price spike above ATR band
      2. funding sign == sign(return_t)               — positioning crowded in
         the direction of the spike (longs paid on up-spike, shorts paid on down)
         [only checked when funding_confirm=True]

    Signal direction:
      +1 if return_t < 0 (down-spike → fade by going long)
      −1 if return_t > 0 (up-spike  → fade by going short)

    Signal strength = |return_t| / (spike_mult × ATR), clipped to [0, 1].
    Held for `hold_bars` bars then zeroed (time stop).

    Left-tail focus: this strategy intentionally takes positions AFTER
    extreme moves, so the strategy itself has low max single-day P&L
    variance but the entry bars are high-volatility by construction.

    Research/paper only.  Do NOT add to LIVE_BOOK_STRATEGIES until the
    gross OOS Sharpe passes the research gate.
    """

    def __init__(self):
        super().__init__("vol_spike_reversion", STRATEGY_PARAMS["vol_spike_reversion"])

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def generate_signals(self, close, returns, **kwargs):
        p            = self.params
        atr_period   = p["atr_period"]
        spike_mult   = p["spike_mult"]
        fund_confirm = p["funding_confirm"]
        hold_bars    = p["hold_bars"]
        sig_sm       = p.get("signal_smooth", 0)

        high    = kwargs.get("high", close)
        low     = kwargs.get("low",  close)
        funding = kwargs.get("funding_rates")

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for sym in close.columns:
            atr       = self._atr(high[sym], low[sym], close[sym], atr_period)
            threshold = spike_mult * atr                     # |ret| must exceed this
            ret       = returns[sym].reindex(close.index)   # align to close index

            if (
                fund_confirm
                and funding is not None
                and not funding.empty
                and sym in funding.columns
            ):
                fund = funding[sym].reindex(close.index, method="ffill").fillna(0)
            else:
                fund = pd.Series(0.0, index=close.index)

            # Spike detection: large move with funding confirming crowding
            is_up_spike   = (ret > 0) & (ret > threshold)
            is_down_spike = (ret < 0) & (ret.abs() > threshold)

            if fund_confirm:
                # Longs overcrowded on up-spike → fade short
                up_confirmed   = is_up_spike   & (fund > 0)
                # Shorts overcrowded on down-spike → fade long
                down_confirmed = is_down_spike & (fund < 0)
            else:
                up_confirmed   = is_up_spike
                down_confirmed = is_down_spike

            # Signal strength: size of the spike relative to threshold
            strength = (ret.abs() / threshold.replace(0, np.nan)).clip(0, 1).fillna(0)

            raw = pd.Series(0.0, index=close.index)
            raw[up_confirmed]   = -strength[up_confirmed]   # short after up-spike
            raw[down_confirmed] =  strength[down_confirmed] # long after down-spike

            # Hold for `hold_bars` after entry using a rolling max-magnitude window
            # Forward-fill: entry signal propagates for hold_bars bars
            held = raw.copy()
            for shift in range(1, hold_bars):
                held = held.where(held.abs() >= raw.shift(shift).abs(), raw.shift(shift))

            signals[sym] = held.fillna(0)

        if sig_sm > 0:
            signals = signals.ewm(span=sig_sm, adjust=False).mean().clip(-1, 1)

        return signals.fillna(0)


# ── Factory ────────────────────────────────────────────────────────────────────
def get_all_strategies() -> list[BaseStrategy]:
    """Return active strategy instances. VolBreakout disabled (OOS win_rate 30.5%)."""
    return [
        MomentumStrategy(),
        MeanReversionStrategy(),
        CrossSectionalMomentumStrategy(),
        PairsTradingStrategy(),
        MLSignalStrategy(),
        MacroRotationStrategy(),
        CarryStrategy(),
        SentimentStrategy(),
        ExhaustionFadeStrategy(),
        # ── Research candidates (paper only, excluded from LIVE_BOOK_STRATEGIES) ─
        TSMOMVolScaledStrategy(),
        CarryNeutralStrategy(),
        ResidMomentumStrategy(),
        BTCDominanceStrategy(),
        VolSpikeReversionStrategy(),
    ]
