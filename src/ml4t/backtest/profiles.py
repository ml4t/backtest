"""Centralized profile definitions for framework-aligned behavior."""

from __future__ import annotations

from copy import deepcopy

DEFAULT_PROFILE = {
    "account": {
        "allow_short_selling": False,
        "allow_leverage": False,
        "short_cash_policy": "credit",
    },
    "execution": {
        "execution_price": "open",
        "execution_mode": "next_bar",
    },
    "stops": {
        "stop_fill_mode": "stop_price",
        "stop_level_basis": "fill_price",
        "trail_hwm_source": "close",
        "trail_stop_timing": "lagged",
    },
    "position_sizing": {
        "share_type": "fractional",
    },
    "commission": {
        "model": "percentage",
        "rate": 0.001,
    },
    "slippage": {
        "model": "percentage",
        "rate": 0.001,
    },
    "cash": {
        "initial": 100000.0,
        "buffer_pct": 0.0,
    },
    "orders": {
        "reject_on_insufficient_cash": True,
        "partial_fills_allowed": False,
        "fill_ordering": "exit_first",
        "entry_order_priority": "submission",
        "rebalance_mode": "incremental",
        "rebalance_headroom_pct": 1.0,
        "missing_price_policy": "skip",
        "late_asset_policy": "allow",
        "late_asset_min_bars": 1,
    },
}

BACKTRADER_PROFILE = {
    "account": {
        "allow_short_selling": True,
        "allow_leverage": True,
        "initial_margin": 0.5,
        "long_maintenance_margin": 0.25,
        "short_maintenance_margin": 0.30,
        "short_cash_policy": "credit",
    },
    "execution": {
        "execution_price": "open",
        "execution_mode": "next_bar",
    },
    "stops": {
        "stop_fill_mode": "stop_price",
        "stop_level_basis": "signal_price",
        "trail_hwm_source": "close",
        "trail_stop_timing": "lagged",
    },
    "position_sizing": {
        "share_type": "integer",
    },
    "commission": {
        "model": "percentage",
        "rate": 0.001,
    },
    "slippage": {
        "model": "percentage",
        "rate": 0.001,
    },
    "cash": {
        "initial": 100000.0,
        "buffer_pct": 0.0,
    },
    "orders": {
        "reject_on_insufficient_cash": True,
        "partial_fills_allowed": False,
        "fill_ordering": "fifo",
        "entry_order_priority": "submission",
        "rebalance_mode": "snapshot",
        "rebalance_headroom_pct": 0.998,
        "missing_price_policy": "use_last",
        "late_asset_policy": "require_history",
        "late_asset_min_bars": 2,
    },
}

VECTORBT_PROFILE = {
    "account": {
        "allow_short_selling": True,
        "allow_leverage": False,
        "short_cash_policy": "credit",
    },
    "execution": {
        "execution_price": "close",
        "execution_mode": "same_bar",
    },
    "stops": {
        "stop_fill_mode": "stop_price",
        "stop_level_basis": "fill_price",
        "trail_hwm_source": "bar_extreme",
        "initial_hwm_source": "bar_high",
        "trail_stop_timing": "intrabar",
    },
    "position_sizing": {
        "share_type": "fractional",
    },
    "commission": {
        "model": "none",
        "rate": 0.0,
    },
    "slippage": {
        "model": "none",
        "rate": 0.0,
    },
    "cash": {
        "initial": 100000.0,
        "buffer_pct": 0.0,
    },
    "orders": {
        "reject_on_insufficient_cash": False,
        "partial_fills_allowed": True,
        "fill_ordering": "exit_first",
        "entry_order_priority": "submission",
        "rebalance_mode": "hybrid",
        "rebalance_headroom_pct": 1.0,
        "missing_price_policy": "use_last",
        "late_asset_policy": "allow",
        "late_asset_min_bars": 1,
    },
}

ZIPLINE_PROFILE = {
    "account": {
        "allow_short_selling": False,
        "allow_leverage": False,
        "short_cash_policy": "credit",
    },
    "execution": {
        "execution_price": "open",
        "execution_mode": "next_bar",
    },
    "stops": {
        "stop_fill_mode": "stop_price",
        "stop_level_basis": "fill_price",
        "trail_hwm_source": "close",
        "trail_stop_timing": "lagged",
    },
    "position_sizing": {
        "share_type": "integer",
    },
    "commission": {
        "model": "per_share",
        "rate": 0.0,
        "per_share": 0.005,
        "minimum": 1.0,
    },
    "slippage": {
        "model": "volume_based",
        "rate": 0.1,
    },
    "cash": {
        "initial": 100000.0,
        "buffer_pct": 0.0,
    },
    "orders": {
        "reject_on_insufficient_cash": True,
        "partial_fills_allowed": True,
        "fill_ordering": "exit_first",
        "entry_order_priority": "submission",
        "rebalance_mode": "snapshot",
        "rebalance_headroom_pct": 0.998,
        "missing_price_policy": "use_last",
        "late_asset_policy": "allow",
        "late_asset_min_bars": 1,
    },
}

REALISTIC_PROFILE = {
    "account": {
        "allow_short_selling": False,
        "allow_leverage": False,
        "short_cash_policy": "credit",
    },
    "execution": {
        "execution_price": "open",
        "execution_mode": "next_bar",
    },
    "stops": {
        "stop_fill_mode": "next_bar_open",
        "stop_level_basis": "fill_price",
        "trail_hwm_source": "close",
        "trail_stop_timing": "lagged",
    },
    "position_sizing": {
        "share_type": "integer",
    },
    "commission": {
        "model": "percentage",
        "rate": 0.002,
    },
    "slippage": {
        "model": "percentage",
        "rate": 0.002,
        "stop_rate": 0.001,
    },
    "cash": {
        "initial": 100000.0,
        "buffer_pct": 0.02,
    },
    "orders": {
        "reject_on_insufficient_cash": True,
        "partial_fills_allowed": False,
        "fill_ordering": "exit_first",
        "entry_order_priority": "submission",
        "rebalance_mode": "incremental",
        "rebalance_headroom_pct": 1.0,
        "missing_price_policy": "skip",
        "late_asset_policy": "allow",
        "late_asset_min_bars": 1,
    },
}

VECTORBT_STRICT_PROFILE = deepcopy(VECTORBT_PROFILE)
VECTORBT_STRICT_PROFILE["account"]["short_cash_policy"] = "lock_notional"
VECTORBT_STRICT_PROFILE["orders"]["reject_on_insufficient_cash"] = True
VECTORBT_STRICT_PROFILE["orders"]["partial_fills_allowed"] = True
VECTORBT_STRICT_PROFILE["orders"]["fill_ordering"] = "fifo"
VECTORBT_STRICT_PROFILE["orders"]["entry_order_priority"] = "submission"

BACKTRADER_STRICT_PROFILE = deepcopy(BACKTRADER_PROFILE)
BACKTRADER_STRICT_PROFILE["orders"]["entry_order_priority"] = "submission"
BACKTRADER_STRICT_PROFILE["orders"]["next_bar_submission_precheck"] = True
BACKTRADER_STRICT_PROFILE["orders"]["next_bar_simple_cash_check"] = True

ZIPLINE_STRICT_PROFILE = deepcopy(ZIPLINE_PROFILE)
ZIPLINE_STRICT_PROFILE["account"]["allow_short_selling"] = True
ZIPLINE_STRICT_PROFILE["account"]["short_cash_policy"] = "credit"
ZIPLINE_STRICT_PROFILE["orders"]["skip_cash_validation"] = True
ZIPLINE_STRICT_PROFILE["orders"]["entry_order_priority"] = "submission"

LEAN_PROFILE = {
    "account": {
        "allow_short_selling": True,
        "allow_leverage": False,
        "short_cash_policy": "credit",
    },
    "execution": {
        "execution_price": "close",
        "execution_mode": "same_bar",
    },
    "stops": {
        "stop_fill_mode": "stop_price",
        "stop_level_basis": "fill_price",
        "trail_hwm_source": "close",
        "trail_stop_timing": "lagged",
    },
    "position_sizing": {
        "share_type": "integer",
    },
    "commission": {
        "model": "per_share",
        "rate": 0.0,
        "per_share": 0.005,
        "minimum": 1.0,
    },
    "slippage": {
        "model": "percentage",
        "rate": 0.001,
    },
    "cash": {
        "initial": 100000.0,
        "buffer_pct": 0.0,
    },
    "orders": {
        "reject_on_insufficient_cash": True,
        "partial_fills_allowed": False,
        "fill_ordering": "exit_first",
        "entry_order_priority": "submission",
        "rebalance_mode": "snapshot",
        "rebalance_headroom_pct": 1.0,
        "missing_price_policy": "use_last",
        "late_asset_policy": "allow",
        "late_asset_min_bars": 1,
    },
}

LEAN_STRICT_PROFILE = deepcopy(LEAN_PROFILE)
LEAN_STRICT_PROFILE["orders"]["buying_power_reservation"] = True
LEAN_STRICT_PROFILE["settlement"] = {"delay": 2}  # T+2 for US equities

FAST_PROFILE = {
    "account": {
        "allow_short_selling": True,
        "allow_leverage": False,
        "short_cash_policy": "credit",
    },
    "execution": {
        "execution_price": "close",
        "execution_mode": "same_bar",
    },
    "stops": {
        "stop_fill_mode": "stop_price",
        "stop_level_basis": "fill_price",
        "trail_hwm_source": "close",
        "trail_stop_timing": "lagged",
    },
    "position_sizing": {
        "share_type": "fractional",
    },
    "commission": {
        "model": "none",
        "rate": 0.0,
    },
    "slippage": {
        "model": "none",
        "rate": 0.0,
    },
    "cash": {
        "initial": 100000.0,
        "buffer_pct": 0.0,
    },
    "orders": {
        "reject_on_insufficient_cash": False,
        "partial_fills_allowed": False,
        "fill_ordering": "exit_first",
        "entry_order_priority": "submission",
        "rebalance_mode": "snapshot",
        "rebalance_headroom_pct": 1.0,
        "missing_price_policy": "skip",
        "late_asset_policy": "allow",
        "late_asset_min_bars": 1,
    },
}

_PROFILES = {
    "default": DEFAULT_PROFILE,
    "fast": FAST_PROFILE,
    "backtrader": BACKTRADER_PROFILE,
    "vectorbt": VECTORBT_PROFILE,
    "zipline": ZIPLINE_PROFILE,
    "lean": LEAN_PROFILE,
    "realistic": REALISTIC_PROFILE,
    "vectorbt_strict": VECTORBT_STRICT_PROFILE,
    "backtrader_strict": BACKTRADER_STRICT_PROFILE,
    "zipline_strict": ZIPLINE_STRICT_PROFILE,
    "lean_strict": LEAN_STRICT_PROFILE,
}

_ALIASES = {
    "vectorbt_pro": "vectorbt",
    "vectorbt_oss": "vectorbt",
    "quantconnect": "lean",
    "vectorbt_compare": "vectorbt_strict",
    "backtrader_compare": "backtrader_strict",
    "zipline_compare": "zipline_strict",
    "lean_compare": "lean_strict",
}

_CORE_PROFILE_NAMES = ["backtrader", "default", "lean", "realistic", "vectorbt", "zipline"]


def get_profile_config(name: str) -> dict:
    """Return a deep copy of nested config data for the named profile."""
    key = _ALIASES.get(name, name)
    if key not in _PROFILES:
        available = ", ".join(sorted(_PROFILES.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return deepcopy(_PROFILES[key])


def list_profiles() -> list[str]:
    """List canonical preset names."""
    return _CORE_PROFILE_NAMES.copy()
