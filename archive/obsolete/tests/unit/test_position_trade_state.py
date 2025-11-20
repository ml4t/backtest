"""Unit tests for PositionTradeState tracking.

Tests bar counting, MFE/MAE calculation, and entry state management.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from ml4t.backtest.risk.manager import PositionTradeState


class TestPositionTradeStateCore:
    """Test core PositionTradeState functionality."""

    def test_initialization(self):
        """Test PositionTradeState initialization."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=entry_time,
            entry_price=Decimal("100.00"),
            entry_quantity=10.0,
        )

        assert state.asset_id == "AAPL"
        assert state.entry_time == entry_time
        assert state.entry_price == Decimal("100.00")
        assert state.entry_quantity == 10.0
        assert state.bars_held == 0
        assert state.max_favorable_excursion == Decimal("0.0")
        assert state.max_adverse_excursion == Decimal("0.0")

    def test_bars_held_increment(self):
        """Test that bars_held increments on each market event."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=10.0,
        )

        # Bars held should increment on each update
        assert state.bars_held == 0

        state.update_on_market_event(Decimal("100.00"))
        assert state.bars_held == 1

        state.update_on_market_event(Decimal("101.00"))
        assert state.bars_held == 2

        state.update_on_market_event(Decimal("99.00"))
        assert state.bars_held == 3


class TestMFELongPosition:
    """Test MFE tracking for long positions."""

    def test_mfe_long_price_increases(self):
        """Test MFE tracking when price moves favorably (up) for long position."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=10.0,  # Long position
        )

        # Price goes up - favorable for long
        state.update_on_market_event(Decimal("105.00"))
        assert state.max_favorable_excursion == Decimal("5.00")  # 105 - 100
        assert state.max_adverse_excursion == Decimal("0.0")  # No adverse move

        # Price goes up more - MFE increases
        state.update_on_market_event(Decimal("108.00"))
        assert state.max_favorable_excursion == Decimal("8.00")  # 108 - 100
        assert state.max_adverse_excursion == Decimal("0.0")

        # Price comes back down but MFE stays at peak
        state.update_on_market_event(Decimal("103.00"))
        assert state.max_favorable_excursion == Decimal("8.00")  # Still 108 - 100 (peak)
        assert state.max_adverse_excursion == Decimal("0.0")  # Still above entry

    def test_mfe_long_price_flat(self):
        """Test MFE when price stays at entry (no excursion)."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=10.0,
        )

        # Price stays flat
        state.update_on_market_event(Decimal("100.00"))
        assert state.max_favorable_excursion == Decimal("0.0")
        assert state.max_adverse_excursion == Decimal("0.0")


class TestMAELongPosition:
    """Test MAE tracking for long positions."""

    def test_mae_long_price_decreases(self):
        """Test MAE tracking when price moves adversely (down) for long position."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=10.0,  # Long position
        )

        # Price goes down - adverse for long
        state.update_on_market_event(Decimal("95.00"))
        assert state.max_favorable_excursion == Decimal("0.0")  # No favorable move
        assert state.max_adverse_excursion == Decimal("5.00")  # 100 - 95

        # Price goes down more - MAE increases
        state.update_on_market_event(Decimal("92.00"))
        assert state.max_favorable_excursion == Decimal("0.0")
        assert state.max_adverse_excursion == Decimal("8.00")  # 100 - 92

        # Price recovers but MAE stays at worst point
        state.update_on_market_event(Decimal("97.00"))
        assert state.max_favorable_excursion == Decimal("0.0")  # Still below entry
        assert state.max_adverse_excursion == Decimal("8.00")  # Still 100 - 92 (worst)


class TestMFEMAELongMixedMoves:
    """Test MFE/MAE tracking with mixed price movements for long positions."""

    def test_long_position_both_mfe_and_mae(self):
        """Test tracking both MFE and MAE when price swings both ways."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=10.0,  # Long position
        )

        # Start at entry
        assert state.max_favorable_excursion == Decimal("0.0")
        assert state.max_adverse_excursion == Decimal("0.0")

        # Price goes up (favorable)
        state.update_on_market_event(Decimal("110.00"))
        assert state.max_favorable_excursion == Decimal("10.00")
        assert state.max_adverse_excursion == Decimal("0.0")

        # Price drops below entry (adverse)
        state.update_on_market_event(Decimal("95.00"))
        assert state.max_favorable_excursion == Decimal("10.00")  # Stays at peak
        assert state.max_adverse_excursion == Decimal("5.00")  # 100 - 95

        # Price recovers above entry but below peak
        state.update_on_market_event(Decimal("105.00"))
        assert state.max_favorable_excursion == Decimal("10.00")  # Still at peak
        assert state.max_adverse_excursion == Decimal("5.00")  # Still at worst

        # Price drops to new low
        state.update_on_market_event(Decimal("90.00"))
        assert state.max_favorable_excursion == Decimal("10.00")  # Still at peak
        assert state.max_adverse_excursion == Decimal("10.00")  # New worst: 100 - 90

        # Price rallies to new high
        state.update_on_market_event(Decimal("115.00"))
        assert state.max_favorable_excursion == Decimal("15.00")  # New peak: 115 - 100
        assert state.max_adverse_excursion == Decimal("10.00")  # Still at worst


class TestMFEShortPosition:
    """Test MFE tracking for short positions."""

    def test_mfe_short_price_decreases(self):
        """Test MFE tracking when price moves favorably (down) for short position."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=-10.0,  # Short position
        )

        # Price goes down - favorable for short
        state.update_on_market_event(Decimal("95.00"))
        assert state.max_favorable_excursion == Decimal("5.00")  # 100 - 95
        assert state.max_adverse_excursion == Decimal("0.0")

        # Price goes down more - MFE increases
        state.update_on_market_event(Decimal("92.00"))
        assert state.max_favorable_excursion == Decimal("8.00")  # 100 - 92
        assert state.max_adverse_excursion == Decimal("0.0")

        # Price comes back up but MFE stays at best point
        state.update_on_market_event(Decimal("97.00"))
        assert state.max_favorable_excursion == Decimal("8.00")  # Still 100 - 92 (best)
        assert state.max_adverse_excursion == Decimal("0.0")  # Still below entry


class TestMAEShortPosition:
    """Test MAE tracking for short positions."""

    def test_mae_short_price_increases(self):
        """Test MAE tracking when price moves adversely (up) for short position."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=-10.0,  # Short position
        )

        # Price goes up - adverse for short
        state.update_on_market_event(Decimal("105.00"))
        assert state.max_favorable_excursion == Decimal("0.0")
        assert state.max_adverse_excursion == Decimal("5.00")  # 105 - 100

        # Price goes up more - MAE increases
        state.update_on_market_event(Decimal("108.00"))
        assert state.max_favorable_excursion == Decimal("0.0")
        assert state.max_adverse_excursion == Decimal("8.00")  # 108 - 100

        # Price comes back down but MAE stays at worst point
        state.update_on_market_event(Decimal("103.00"))
        assert state.max_favorable_excursion == Decimal("0.0")  # Still above entry
        assert state.max_adverse_excursion == Decimal("8.00")  # Still 108 - 100 (worst)


class TestMFEMAEShortMixedMoves:
    """Test MFE/MAE tracking with mixed price movements for short positions."""

    def test_short_position_both_mfe_and_mae(self):
        """Test tracking both MFE and MAE when price swings both ways."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=-10.0,  # Short position
        )

        # Price goes down (favorable for short)
        state.update_on_market_event(Decimal("90.00"))
        assert state.max_favorable_excursion == Decimal("10.00")  # 100 - 90
        assert state.max_adverse_excursion == Decimal("0.0")

        # Price rises above entry (adverse for short)
        state.update_on_market_event(Decimal("105.00"))
        assert state.max_favorable_excursion == Decimal("10.00")  # Stays at best
        assert state.max_adverse_excursion == Decimal("5.00")  # 105 - 100

        # Price drops below entry but above best
        state.update_on_market_event(Decimal("95.00"))
        assert state.max_favorable_excursion == Decimal("10.00")  # Still at best
        assert state.max_adverse_excursion == Decimal("5.00")  # Still at worst

        # Price rallies to new high
        state.update_on_market_event(Decimal("110.00"))
        assert state.max_favorable_excursion == Decimal("10.00")  # Still at best
        assert state.max_adverse_excursion == Decimal("10.00")  # New worst: 110 - 100

        # Price drops to new low
        state.update_on_market_event(Decimal("85.00"))
        assert state.max_favorable_excursion == Decimal("15.00")  # New best: 100 - 85
        assert state.max_adverse_excursion == Decimal("10.00")  # Still at worst


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_quantity_position(self):
        """Test that zero quantity doesn't break calculations."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=0.0,  # Zero quantity (shouldn't happen but test anyway)
        )

        # Should not crash
        state.update_on_market_event(Decimal("105.00"))
        assert state.bars_held == 1

    def test_very_small_price_changes(self):
        """Test with very small price changes (precision)."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.000"),
            entry_quantity=10.0,
        )

        # Tiny move up
        state.update_on_market_event(Decimal("100.001"))
        assert state.max_favorable_excursion == Decimal("0.001")
        assert state.max_adverse_excursion == Decimal("0.0")

        # Tiny move down
        state.update_on_market_event(Decimal("99.999"))
        assert state.max_favorable_excursion == Decimal("0.001")  # Stays at peak
        assert state.max_adverse_excursion == Decimal("0.001")  # 100.000 - 99.999

    def test_large_quantity(self):
        """Test with large position quantity."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=100000.0,  # Large position
        )

        state.update_on_market_event(Decimal("105.00"))
        assert state.max_favorable_excursion == Decimal("5.00")
        # Note: excursion is in price units, not currency units
        # Currency units would be 5.00 * 100000 = 500,000

    def test_multiple_bars_same_price(self):
        """Test that bars increment even when price doesn't change."""
        state = PositionTradeState(
            asset_id="AAPL",
            entry_time=datetime.now(),
            entry_price=Decimal("100.00"),
            entry_quantity=10.0,
        )

        # Price stays the same for multiple bars
        for i in range(10):
            state.update_on_market_event(Decimal("100.00"))

        assert state.bars_held == 10
        assert state.max_favorable_excursion == Decimal("0.0")
        assert state.max_adverse_excursion == Decimal("0.0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
