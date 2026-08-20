"""Behavior contracts for the frozen real-strategy ML4T runner."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl


def _load_runner():
    path = Path(__file__).parents[2] / "validation" / "real_strategy_runner.py"
    spec = importlib.util.spec_from_file_location("ml4t_real_strategy_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _load_input():
    path = Path(__file__).parents[2] / "validation" / "real_strategy_input.py"
    spec = importlib.util.spec_from_file_location("ml4t_real_strategy_input", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_evidence():
    path = Path(__file__).parents[2] / "validation" / "real_strategy_evidence.py"
    spec = importlib.util.spec_from_file_location("ml4t_real_strategy_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_benchmark():
    path = Path(__file__).parents[2] / "validation" / "real_strategy_benchmark.py"
    spec = importlib.util.spec_from_file_location("ml4t_real_strategy_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Broker:
    def __init__(self) -> None:
        self.cash = 1_000.0
        self.positions = {"PERP": SimpleNamespace(quantity=2.0, multiplier=3.0)}

    @staticmethod
    def get_mark_price(_symbol: str, *, quantity: float) -> float:
        assert quantity == 2.0
        return 10.0


def test_lean_funding_schedule_skips_entry_boundary_then_settles() -> None:
    runner = _load_runner()
    funding = pl.DataFrame(
        {
            "symbol": ["PERP", "PERP"],
            "timestamp": [
                datetime(2024, 1, 1, 8, tzinfo=UTC),
                datetime(2024, 1, 1, 16, tzinfo=UTC),
            ],
            "funding_rate": [0.01, 0.02],
        }
    )
    ledger = runner.FundingSettlementLedger(funding, skip_first_after_entry=True)
    broker = _Broker()

    ledger.settle(datetime(2024, 1, 1, 8, tzinfo=UTC), broker)
    assert broker.cash == 1_000.0

    ledger.settle(datetime(2024, 1, 1, 16, tzinfo=UTC), broker)
    assert broker.cash == 998.8
    assert ledger.metrics() == {
        "funding_pnl": -1.2,
        "funding_events": 1,
        "funding_settlements": 2,
    }


def test_lean_funding_schedule_observes_flat_positions_between_settlements() -> None:
    runner = _load_runner()
    funding = pl.DataFrame(
        {
            "symbol": ["PERP", "PERP"],
            "timestamp": [
                datetime(2024, 1, 1, 8, tzinfo=UTC),
                datetime(2024, 1, 1, 16, tzinfo=UTC),
            ],
            "funding_rate": [0.01, 0.02],
        }
    )
    ledger = runner.FundingSettlementLedger(funding, skip_first_after_entry=True)
    broker = _Broker()

    ledger.settle(datetime(2024, 1, 1, 8, tzinfo=UTC), broker)
    broker.positions["PERP"].quantity = 0.0
    ledger.settle(datetime(2024, 1, 1, 12, tzinfo=UTC), broker)
    broker.positions["PERP"].quantity = 2.0
    ledger.settle(datetime(2024, 1, 1, 16, tzinfo=UTC), broker)

    assert broker.cash == 1_000.0
    assert ledger.metrics()["funding_events"] == 0


def test_lean_comparison_preserves_margin_account_and_disables_equity_costs() -> None:
    runner = _load_runner()
    source = {
        "account": {"allow_short_selling": True, "allow_leverage": False},
        "cash": {"initial": 125_000.0},
        "calendar": {"calendar": "NYSE", "data_frequency": "daily"},
        "feed": {"calendar": "NYSE", "data_frequency": "daily"},
        "commission": {"model": "percentage", "rate": 0.001},
        "slippage": {"model": "percentage", "rate": 0.001},
        "orders": {"rebalance_headroom_pct": 0.99},
        "metadata": {},
    }

    config = runner._comparison_config({"backtest_config": source}, "lean")

    assert config.allow_leverage is True
    assert config.commission_rate == 0.0
    assert config.slippage_rate == 0.0
    assert config.rebalance_headroom_pct == 1.0


def test_real_strategy_comparator_detects_one_quantum_mutation() -> None:
    evidence = _load_evidence()
    framework = [{"timestamp": "2024-01-01", "price": "10.00000000"}]
    mutated = [{"timestamp": "2024-01-01", "price": "10.00000001"}]

    result = evidence.compare_records(framework, mutated, numeric_fields=("price",))

    assert result["passed"] is False
    assert result["first_divergence"] == {
        "kind": "record",
        "index": 0,
        "field": "price",
        "framework": framework[0],
        "ml4t": mutated[0],
        "canonical_gap": "0.00000001",
    }


def test_real_strategy_comparator_rejects_missing_valuation_timestamp() -> None:
    evidence = _load_evidence()
    framework = [
        {"timestamp": "2024-01-01", "equity": "1000.0"},
        {"timestamp": "2024-01-02", "equity": "1000.0"},
    ]
    ml4t = [{"timestamp": "2024-01-01", "equity": "1000.0"}]

    surface, _, _ = evidence._value_surface(framework, ml4t, field="equity")

    assert surface["passed"] is False
    assert surface["coverage_passed"] is False
    assert surface["first_divergence"] == {
        "kind": "timestamp_coverage",
        "framework_only": "2024-01-02",
        "ml4t_only": None,
    }


def test_comparison_market_applies_production_daily_calendar() -> None:
    comparison_input = _load_input()
    market = pl.DataFrame(
        {
            "timestamp": [datetime(2018, 12, 4), datetime(2018, 12, 5)],
            "symbol": ["ES", "ES"],
            "close": [100.0, 101.0],
        }
    )
    spec = {
        "backtest_config": {
            "calendar": {
                "calendar": "CME",
                "data_frequency": "daily",
                "enforce_sessions": True,
                "timezone": "UTC",
            }
        }
    }

    filtered = comparison_input.filter_comparison_market(market, spec)

    assert filtered["timestamp"].to_list() == [datetime(2018, 12, 4)]


def test_real_strategy_benchmark_interval_is_deterministic() -> None:
    benchmark = _load_benchmark()
    values = [float(value) for value in range(1, 11)]

    first = benchmark.bootstrap_median_interval(values, draws=1_000, seed=7)
    second = benchmark.bootstrap_median_interval(values, draws=1_000, seed=7)

    assert first == second
    assert first[0] <= 5.5 <= first[1]
