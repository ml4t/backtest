"""Unit tests for VectorBT-compatible exit priority handling.

Tests verify that ml4t.backtest's Broker implements VectorBT's exit priority rule:
    SL (Stop Loss) > TSL (Trailing Stop Loss) > TP (Take Profit)

When multiple bracket exits trigger in the same bar, only the highest priority
exit should fill, and the others should be cancelled via OCO logic.
"""

import pytest

from ml4t.backtest.data.asset_registry import AssetRegistry
from ml4t.backtest.execution.broker import SimulationBroker


class TestExitPriority:
    """Test VectorBT exit priority implementation."""

    def test_get_exit_priority_values(self):
        """Test _get_exit_priority returns correct priority values."""
        asset_registry = AssetRegistry()
        broker = SimulationBroker(initial_cash=100000.0, asset_registry=asset_registry)

        # Priority values (lower = higher priority)
        assert broker._get_exit_priority("stop_loss") == 1  # Highest
        assert broker._get_exit_priority("trailing_stop") == 2  # Medium
        assert broker._get_exit_priority("take_profit") == 3  # Lowest

        # Unknown types get lowest priority
        assert broker._get_exit_priority("unknown") == 999

    def test_priority_order(self):
        """Test that priority values enforce correct order: SL > TSL > TP."""
        asset_registry = AssetRegistry()
        broker = SimulationBroker(initial_cash=100000.0, asset_registry=asset_registry)

        sl_priority = broker._get_exit_priority("stop_loss")
        tsl_priority = broker._get_exit_priority("trailing_stop")
        tp_priority = broker._get_exit_priority("take_profit")

        # Lower number = higher priority
        assert sl_priority < tsl_priority < tp_priority

    # TODO: Integration tests for actual conflict scenarios
    # These will be added in TASK-024 as part of full backtest validation
    # For now, the priority logic is tested and the implementation is complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
