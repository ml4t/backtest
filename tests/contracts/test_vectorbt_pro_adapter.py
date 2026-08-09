"""Contracts for the pinned VectorBT Pro validation adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parents[2]
_VALIDATION_DIR = _PROJECT_ROOT / "validation"
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

from scenarios.definitions import SCENARIOS  # noqa: E402


def _load_adapter():
    module_name = "ml4t_vectorbt_pro_contract_adapter"
    spec = importlib.util.spec_from_file_location(
        module_name, _VALIDATION_DIR / "frameworks" / "vectorbt_pro.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_supported_vectorbt_pro_version_is_explicit() -> None:
    adapter = _load_adapter()

    assert adapter.SUPPORTED_VECTORBT_PRO_VERSION == "2025.12.31"
    adapter._require_supported_version("2025.12.31")


def test_unsupported_vectorbt_pro_version_fails_precisely() -> None:
    adapter = _load_adapter()

    with pytest.raises(
        RuntimeError,
        match=r"requires version 2025\.12\.31, found 2026\.1\.1",
    ):
        adapter._require_supported_version("2026.1.1")


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1_000.0, 1_000.0],
        },
        index=pd.date_range("2024-01-02", periods=2),
    )


def test_portfolio_kwargs_use_current_stop_api_and_ohlc() -> None:
    adapter = _load_adapter()
    prices = _prices()
    kwargs = adapter._build_portfolio_kwargs(SCENARIOS["15"], prices, np.array([True, False]), None)

    pd.testing.assert_series_equal(kwargs["open"], prices["open"])
    pd.testing.assert_series_equal(kwargs["high"], prices["high"])
    pd.testing.assert_series_equal(kwargs["low"], prices["low"])
    assert kwargs["tsl_stop"] == 0.03
    assert kwargs["sl_stop"] == 0.05
    assert kwargs["tp_stop"] == 0.10
    assert "sl_trail" not in kwargs


def test_per_share_commission_maps_to_fixed_order_fee() -> None:
    adapter = _load_adapter()
    scenario = SCENARIOS["06"]
    kwargs = adapter._build_portfolio_kwargs(
        scenario, _prices(), np.array([True, False]), np.array([False, True])
    )

    assert kwargs["fees"] == 0.0
    assert kwargs["fixed_fees"] == scenario.constants["per_share_rate"] * scenario.shares


def test_short_risk_scenario_uses_short_entry_channel() -> None:
    adapter = _load_adapter()
    entries = np.array([True, False])
    kwargs = adapter._build_portfolio_kwargs(SCENARIOS["12"], _prices(), entries, None)

    assert kwargs["short_entries"] is entries
    assert "entries" not in kwargs
