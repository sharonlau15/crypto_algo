"""
tests/test_vt_equivalence.py
=============================
Proves that compute_live_vt_scale (incremental, Postgres-persisted) produces
scales identical to apply_vol_target_overlay (batch) given the same weights
and returns.

Run with:  python -m pytest tests/test_vt_equivalence.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from portfolio.optimizer import apply_vol_target_overlay, compute_live_vt_scale

VT_PARAMS = dict(target_vol=0.15, band=0.20, vol_window=63, max_scale=2.0)


def _make_fixtures(n_bars=300, n_assets=5, seed=42):
    rng     = np.random.default_rng(seed)
    dates   = pd.date_range("2023-01-01", periods=n_bars, freq="D", tz="UTC")
    weights = pd.DataFrame(
        np.clip(rng.standard_normal((n_bars, n_assets)) * 0.10, -0.20, 0.20),
        index=dates,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    returns = pd.DataFrame(
        rng.standard_normal((n_bars, n_assets)) * 0.02,
        index=dates,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    return weights, returns


def _extract_batch_scales(weights, returns, vp):
    """Re-run the batch loop to get the scale at each bar (for direct comparison)."""
    common   = weights.index.intersection(returns.index)
    w        = weights.reindex(common).fillna(0.0)
    r        = returns.reindex(common).fillna(0.0)
    port_ret = (w * r).sum(axis=1)
    rv_series = port_ret.rolling(vp["vol_window"]).std() * np.sqrt(365)
    low, high = vp["target_vol"] * (1 - vp["band"]), vp["target_vol"] * (1 + vp["band"])

    scales = np.ones(len(common))
    s = 1.0
    for i in range(len(common)):
        rv = rv_series.iloc[i]
        if pd.isna(rv) or rv <= 0.0:
            scales[i] = s
            continue
        if rv < low or rv > high:
            s = min(vp["target_vol"] / rv, vp["max_scale"])
        scales[i] = s
    return scales


def test_incremental_scales_match_batch():
    """
    Bar-by-bar incremental compute_live_vt_scale matches apply_vol_target_overlay
    to floating-point precision for all bars at or beyond vol_window.
    """
    weights, returns = _make_fixtures()
    vp               = VT_PARAMS
    vol_window       = vp["vol_window"]

    batch_scales = _extract_batch_scales(weights, returns, vp)

    # Simulate live engine: one call per bar, state thread (loaded_scale) persisted
    current_scale = 1.0
    incr_scales   = np.ones(len(weights))

    for i in range(len(weights)):
        start = max(0, i - vol_window + 1)
        current_scale = compute_live_vt_scale(
            weights_window=weights.iloc[start : i + 1],
            returns_window=returns.iloc[start : i + 1],
            loaded_scale=current_scale,
            **vp,
        )
        incr_scales[i] = current_scale

    # Compare from bar vol_window-1 onward (warmup bars have rv=NaN → both stay 1.0)
    np.testing.assert_allclose(
        incr_scales[vol_window - 1 :],
        batch_scales[vol_window - 1 :],
        rtol=1e-12,
        err_msg="Incremental scales diverge from batch scales",
    )


def test_scaled_weights_match_overlay_output():
    """
    Applying the incremental scale to raw weights produces the same DataFrame
    as apply_vol_target_overlay, within floating-point tolerance.
    """
    from config.settings import MAX_POSITION_SIZE

    weights, returns = _make_fixtures(n_bars=200)
    vp               = VT_PARAMS
    vol_window       = vp["vol_window"]

    expected = apply_vol_target_overlay(weights, returns, **vp)

    current_scale = 1.0
    for i in range(len(weights)):
        start = max(0, i - vol_window + 1)
        current_scale = compute_live_vt_scale(
            weights_window=weights.iloc[start : i + 1],
            returns_window=returns.iloc[start : i + 1],
            loaded_scale=current_scale,
            **vp,
        )

    # Final bar: apply scale and clip exactly as apply_vol_target_overlay does
    actual_last = (weights.iloc[-1] * current_scale).clip(-MAX_POSITION_SIZE, MAX_POSITION_SIZE)
    pd.testing.assert_series_equal(
        actual_last,
        expected.iloc[-1],
        check_names=False,
        rtol=1e-12,
    )


def test_persistence_semantics():
    """
    Resuming mid-history with a persisted scale (instead of starting from 1.0)
    gives the same result as a continuous run from bar 0.
    """
    weights, returns = _make_fixtures(n_bars=250)
    vp               = VT_PARAMS
    vol_window       = vp["vol_window"]
    split            = 150  # simulate engine restart at bar 150

    # Full continuous run from bar 0
    s_full = 1.0
    for i in range(len(weights)):
        start  = max(0, i - vol_window + 1)
        s_full = compute_live_vt_scale(
            weights.iloc[start : i + 1], returns.iloc[start : i + 1], s_full, **vp
        )

    # Run up to split, save scale, then resume
    s_before_split = 1.0
    for i in range(split):
        start = max(0, i - vol_window + 1)
        s_before_split = compute_live_vt_scale(
            weights.iloc[start : i + 1], returns.iloc[start : i + 1], s_before_split, **vp
        )

    # "Load from Postgres" and continue
    s_after_restart = s_before_split
    for i in range(split, len(weights)):
        start = max(0, i - vol_window + 1)
        s_after_restart = compute_live_vt_scale(
            weights.iloc[start : i + 1], returns.iloc[start : i + 1], s_after_restart, **vp
        )

    assert abs(s_full - s_after_restart) < 1e-12, (
        f"Scale after restart ({s_after_restart:.8f}) differs from "
        f"continuous run ({s_full:.8f})"
    )


def test_no_optimizer_in_live_weight_path():
    """
    Confirm compute_target_weights never calls the optimizer.
    MIN_ANNUALIZED_VOL remains inert in the live path.
    """
    import inspect
    from execution.live_engine import compute_target_weights

    src = inspect.getsource(compute_target_weights)
    assert "max_sharpe_optimize" not in src, (
        "compute_target_weights must not call max_sharpe_optimize — "
        "optimizer is backtest-only; MIN_ANNUALIZED_VOL must stay inert live"
    )
    assert "from portfolio.optimizer import" in src or "portfolio.optimizer" in src, (
        "VT overlay import missing from compute_target_weights"
    )
