"""Behavior contracts for the frozen real-strategy ML4T runner."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl


def _load_runner():
    path = Path(__file__).parents[2] / "validation" / "real_strategy_runner.py"
    spec = importlib.util.spec_from_file_location("ml4t_real_strategy_runner", path)
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
