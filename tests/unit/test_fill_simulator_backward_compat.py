"""Backward compatibility tests for FillSimulator.try_fill_order() signature change.

This test suite verifies that both the OLD (deprecated) signature with OHLC parameters
and the NEW signature with MarketEvent work correctly.
"""

import warnings
from datetime import datetime, timezone

import pytest

from ml4t.backtest.core.constants import DEFAULT_COMMISSION_RATE_BPS, SLIPPAGE_EQUITY_BPS
from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import AssetId
from ml4t.backtest.data.asset_registry import AssetRegistry, AssetSpec
from ml4t.backtest.execution.commission import PercentageCommission
from ml4t.backtest.execution.fill_simulator import FillSimulator
from ml4t.backtest.execution.order import Order, OrderSide, OrderType
from ml4t.backtest.execution.slippage import PercentageSlippage


@pytest.fixture
def asset_id() -> AssetId:
    """Test asset ID."""
    return "AAPL"


@pytest.fixture
def asset_registry(asset_id: AssetId) -> AssetRegistry:
    """Create asset registry with test asset."""
    registry = AssetRegistry()
    spec = AssetSpec(
        asset_id=asset_id,
        asset_type="stock",
        multiplier=1.0,
        tick_size=0.01,
        lot_size=1,
    )
    registry.register(spec)
    return registry


@pytest.fixture
def fill_simulator(asset_registry: AssetRegistry) -> FillSimulator:
    """Create FillSimulator with basic configuration."""
    return FillSimulator(
        asset_registry=asset_registry,
        commission_model=PercentageCommission(DEFAULT_COMMISSION_RATE_BPS / 10000.0),
        slippage_model=PercentageSlippage(SLIPPAGE_EQUITY_BPS / 10000.0),
    )


@pytest.fixture
def timestamp() -> datetime:
    """Test timestamp."""
    return datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)


@pytest.fixture
def market_event(asset_id: AssetId, timestamp: datetime) -> MarketEvent:
    """Create sample MarketEvent."""
    return MarketEvent(
        timestamp=timestamp,
        asset_id=asset_id,
        data_type="bar",
        price=150.0,
        open=149.0,
        high=151.0,
        low=148.0,
        close=150.0,
        volume=1000000,
        bid_price=149.98,
        ask_price=150.02,
    )


class TestBackwardCompatibilityBasic:
    """Test basic backward compatibility between old and new signatures."""

    def test_old_signature_still_works_with_market_price(
        self, fill_simulator: FillSimulator, asset_id: AssetId, timestamp: datetime
    ):
        """Old signature with market_price parameter should still work."""
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        # OLD SIGNATURE: Pass individual parameters
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fill_simulator.try_fill_order(
                order,
                market_price=150.0,
                current_cash=20000.0,
                current_position=0.0,
                timestamp=timestamp,
            )
            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

        # Should still fill successfully
        assert result is not None
        assert result.fill_quantity == 100
        assert result.fill_price > 0

    def test_old_signature_with_ohlc(
        self, fill_simulator: FillSimulator, asset_id: AssetId, timestamp: datetime
    ):
        """Old signature with OHLC parameters should still work."""
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        # OLD SIGNATURE: Pass OHLC parameters
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fill_simulator.try_fill_order(
                order,
                market_price=150.0,
                current_cash=20000.0,
                current_position=0.0,
                timestamp=timestamp,
                high=151.0,
                low=148.0,
                close=150.0,
            )
            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        # Should still fill successfully
        assert result is not None
        assert result.fill_quantity == 100

    def test_new_signature_with_market_event(
        self, fill_simulator: FillSimulator, asset_id: AssetId, market_event: MarketEvent
    ):
        """New signature with MarketEvent should work without warning."""
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        # NEW SIGNATURE: Pass MarketEvent
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fill_simulator.try_fill_order(
                order,
                market_event,
                current_cash=20000.0,
                current_position=0.0,
            )
            # Should NOT emit deprecation warning
            assert len(w) == 0

        # Should fill successfully
        assert result is not None
        assert result.fill_quantity == 100
        assert result.fill_price > 0

    def test_both_signatures_produce_same_result(
        self,
        asset_registry: AssetRegistry,
        asset_id: AssetId,
        market_event: MarketEvent,
        timestamp: datetime,
    ):
        """Both old and new signatures should produce identical results."""
        # Create two separate simulators to avoid state interference
        sim1 = FillSimulator(
            asset_registry=asset_registry,
            commission_model=PercentageCommission(DEFAULT_COMMISSION_RATE_BPS / 10000.0),
            slippage_model=PercentageSlippage(SLIPPAGE_EQUITY_BPS / 10000.0),
        )
        sim2 = FillSimulator(
            asset_registry=asset_registry,
            commission_model=PercentageCommission(DEFAULT_COMMISSION_RATE_BPS / 10000.0),
            slippage_model=PercentageSlippage(SLIPPAGE_EQUITY_BPS / 10000.0),
        )

        # OLD SIGNATURE
        order1 = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result1 = sim1.try_fill_order(
                order1,
                market_price=market_event.price,
                current_cash=20000.0,
                current_position=0.0,
                timestamp=market_event.timestamp,
                high=market_event.high,
                low=market_event.low,
                close=market_event.close,
            )

        # NEW SIGNATURE
        order2 = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        result2 = sim2.try_fill_order(
            order2,
            market_event,
            current_cash=20000.0,
            current_position=0.0,
        )

        # Results should be identical
        assert result1 is not None
        assert result2 is not None
        assert result1.fill_quantity == result2.fill_quantity
        assert result1.fill_price == result2.fill_price
        assert result1.commission == result2.commission
        assert result1.slippage == result2.slippage


class TestBackwardCompatibilityAdvanced:
    """Test advanced scenarios with both signatures."""

    def test_new_signature_has_access_to_bid_ask(
        self, fill_simulator: FillSimulator, asset_id: AssetId, market_event: MarketEvent
    ):
        """New signature provides access to bid/ask prices (future use)."""
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        result = fill_simulator.try_fill_order(
            order,
            market_event,
            current_cash=20000.0,
            current_position=0.0,
        )

        assert result is not None
        # Verify MarketEvent has bid/ask available for future slippage models
        assert market_event.bid_price is not None
        assert market_event.ask_price is not None

    def test_new_signature_has_access_to_volume(
        self, fill_simulator: FillSimulator, asset_id: AssetId, market_event: MarketEvent
    ):
        """New signature provides access to volume (future use)."""
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        result = fill_simulator.try_fill_order(
            order,
            market_event,
            current_cash=20000.0,
            current_position=0.0,
        )

        assert result is not None
        # Verify MarketEvent has volume available for future liquidity models
        assert market_event.volume is not None
        assert market_event.volume > 0

    def test_old_signature_without_ohlc_still_works(
        self, fill_simulator: FillSimulator, asset_id: AssetId, timestamp: datetime
    ):
        """Old signature with only market_price (no OHLC) should still work."""
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fill_simulator.try_fill_order(
                order,
                market_price=150.0,
                current_cash=20000.0,
                current_position=0.0,
                timestamp=timestamp,
                # No high, low, close provided (backward compat mode)
            )

        assert result is not None
        assert result.fill_quantity == 100

    def test_new_signature_with_minimal_market_event(
        self, fill_simulator: FillSimulator, asset_id: AssetId, timestamp: datetime
    ):
        """New signature works with minimal MarketEvent (only required fields)."""
        # Minimal MarketEvent with only required fields
        minimal_event = MarketEvent(
            timestamp=timestamp,
            asset_id=asset_id,
            data_type="trade",
            price=150.0,
            close=150.0,
            # Optional fields omitted: volume, bid_price, ask_price, etc.
        )

        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        result = fill_simulator.try_fill_order(
            order,
            minimal_event,
            current_cash=20000.0,
            current_position=0.0,
        )

        assert result is not None
        assert result.fill_quantity == 100


class TestBackwardCompatibilityEdgeCases:
    """Test edge cases with both signatures."""

    def test_none_market_event_with_old_params_works(
        self, fill_simulator: FillSimulator, asset_id: AssetId, timestamp: datetime
    ):
        """Passing None for market_event but providing old params should work."""
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fill_simulator.try_fill_order(
                order,
                None,  # market_event=None
                current_cash=20000.0,
                current_position=0.0,
                market_price=150.0,
                timestamp=timestamp,
            )

        assert result is not None
        assert result.fill_quantity == 100

    def test_no_market_event_no_old_params_fails(
        self, fill_simulator: FillSimulator, asset_id: AssetId
    ):
        """Neither market_event nor old params should return None."""
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        result = fill_simulator.try_fill_order(
            order,
            None,  # market_event=None
            current_cash=20000.0,
            current_position=0.0,
            # No old params provided either
        )

        # Should fail gracefully (return None)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
