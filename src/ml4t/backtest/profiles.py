"""Centralized profile definitions for framework-aligned behavior."""

from __future__ import annotations

from copy import deepcopy

DEFAULT_PROFILE = {
    "account": {
        "allow_short_selling": False,
        "allow_leverage": False,
    },
    "execution": {
        "fill_timing": "next_bar_open",
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
        "default_position_pct": 0.10,
    },
    "signals": {
        "signal_processing": "check_position",
        "accumulate_positions": False,
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
        "rebalance_mode": "incremental",
    },
}

BACKTRADER_PROFILE = {
    "account": {
        "allow_short_selling": True,
        "allow_leverage": True,
        "initial_margin": 0.5,
        "long_maintenance_margin": 0.25,
        "short_maintenance_margin": 0.30,
    },
    "execution": {
        "fill_timing": "next_bar_open",
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
        "default_position_pct": 0.10,
    },
    "signals": {
        "signal_processing": "check_position",
        "accumulate_positions": False,
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
        "rebalance_mode": "snapshot",
    },
}

VECTORBT_PROFILE = {
    "account": {
        "allow_short_selling": True,
        "allow_leverage": False,
    },
    "execution": {
        "fill_timing": "same_bar",
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
        "default_position_pct": 0.10,
    },
    "signals": {
        "signal_processing": "process_all",
        "accumulate_positions": False,
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
        "rebalance_mode": "hybrid",
    },
}

ZIPLINE_PROFILE = {
    "account": {
        "allow_short_selling": False,
        "allow_leverage": False,
    },
    "execution": {
        "fill_timing": "next_bar_open",
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
        "default_position_pct": 0.10,
    },
    "signals": {
        "signal_processing": "check_position",
        "accumulate_positions": False,
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
        "rebalance_mode": "snapshot",
    },
}

REALISTIC_PROFILE = {
    "account": {
        "allow_short_selling": False,
        "allow_leverage": False,
    },
    "execution": {
        "fill_timing": "next_bar_open",
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
        "default_position_pct": 0.05,
    },
    "signals": {
        "signal_processing": "check_position",
        "accumulate_positions": False,
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
        "rebalance_mode": "incremental",
    },
}

_PROFILES = {
    "default": DEFAULT_PROFILE,
    "backtrader": BACKTRADER_PROFILE,
    "vectorbt": VECTORBT_PROFILE,
    "zipline": ZIPLINE_PROFILE,
    "realistic": REALISTIC_PROFILE,
}

_ALIASES = {
    "vectorbt_pro": "vectorbt",
    "vectorbt_oss": "vectorbt",
}


def get_profile_config(name: str) -> dict:
    """Return a deep copy of nested config data for the named profile."""
    key = _ALIASES.get(name, name)
    if key not in _PROFILES:
        available = ", ".join(sorted(_PROFILES.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return deepcopy(_PROFILES[key])


def list_profiles() -> list[str]:
    """List canonical preset names."""
    return sorted(_PROFILES.keys())
