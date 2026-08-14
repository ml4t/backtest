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
        "share_type": "integer",
    },
    "commission": {
        "model": "none",
        "rate": 0.0,
        "per_share": 0.0,
        "minimum": 0.0,
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
        "allow_leverage": False,
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
        "initial_hwm_source": "signal_price",
        "trail_stop_timing": "lagged",
    },
    "position_sizing": {
        "share_type": "integer",
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
        "reject_on_insufficient_cash": True,
        "partial_fills_allowed": False,
        "fill_ordering": "fifo",
        "entry_order_priority": "submission",
        "rebalance_mode": "snapshot",
        "rebalance_headroom_pct": 1.0,
        "missing_price_policy": "use_last",
        "late_asset_policy": "allow",
        "late_asset_min_bars": 1,
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
        "immediate_fill": True,
        "rebalance_mode": "hybrid",
        "rebalance_headroom_pct": 1.0,
        "missing_price_policy": "skip",
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
        "next_bar_queue_shadow_validation": True,
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

IBKR_US_STOCKS_FIXED_PROFILE = {
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
        "model": "none",
        "rate": 0.0,
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

VECTORBT_STRICT_PROFILE = deepcopy(VECTORBT_PROFILE)

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
        "next_bar_queue_shadow_validation": True,
        "rebalance_mode": "snapshot",
        "rebalance_headroom_pct": 1.0,
        "missing_price_policy": "use_last",
        "late_asset_policy": "allow",
        "late_asset_min_bars": 1,
    },
}

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
        "share_type": "integer",
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
    "ibkr_us_stocks_fixed": IBKR_US_STOCKS_FIXED_PROFILE,
    "vectorbt_strict": VECTORBT_STRICT_PROFILE,
    "backtrader_strict": BACKTRADER_STRICT_PROFILE,
    "zipline_strict": ZIPLINE_STRICT_PROFILE,
}

_ALIASES = {
    "vectorbt_pro": "vectorbt",
    "vectorbt_oss": "vectorbt",
    "quantconnect": "lean",
    "ibkr:us:stocks:fixed": "ibkr_us_stocks_fixed",
    "vectorbt_compare": "vectorbt_strict",
    "backtrader_compare": "backtrader_strict",
    "zipline_compare": "zipline_strict",
    "lean_compare": "lean",
}

_CORE_PROFILE_NAMES = ["backtrader", "default", "lean", "realistic", "vectorbt", "zipline"]

_BROKER_ALIASES = {
    "ibkr": "ibkr",
    "interactive_brokers": "ibkr",
    "interactivebrokers": "ibkr",
}

_REGION_ALIASES = {
    "us": "us",
    "usa": "us",
}

_ASSET_CLASS_ALIASES = {
    "stocks": "stocks",
    "equities": "stocks",
}

_PLAN_ALIASES = {
    "fixed": "fixed",
}

_ASSUMPTION_PRESETS = {
    ("ibkr", "us", "stocks", "fixed"): "ibkr_us_stocks_fixed",
}


def _normalize_component(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


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


def get_assumption_preset(
    *,
    broker: str,
    region: str,
    asset_class: str,
    plan: str,
) -> str:
    """Resolve structured broker assumptions to a canonical preset name."""
    broker_key = _BROKER_ALIASES.get(_normalize_component(broker))
    region_key = _REGION_ALIASES.get(_normalize_component(region))
    asset_class_key = _ASSET_CLASS_ALIASES.get(_normalize_component(asset_class))
    plan_key = _PLAN_ALIASES.get(_normalize_component(plan))

    if None in (broker_key, region_key, asset_class_key, plan_key):
        raise ValueError(
            "Unknown broker assumptions. Supported combinations currently include "
            "broker='ibkr', region='us', asset_class='stocks', plan='fixed'."
        )

    key = (broker_key, region_key, asset_class_key, plan_key)
    if key not in _ASSUMPTION_PRESETS:
        raise ValueError(
            "Unsupported broker assumptions combination: "
            f"broker={broker!r}, region={region!r}, asset_class={asset_class!r}, plan={plan!r}"
        )
    return _ASSUMPTION_PRESETS[key]
