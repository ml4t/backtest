"""Tests for execution limits (volume participation)."""

from ml4t.backtest.execution.limits import (
    AdaptiveParticipationLimit,
    NoLimits,
    VolumeParticipationLimit,
)


class TestNoLimits:
    """Test NoLimits execution limit."""

    def test_fill_entire_order(self):
        """Test that entire order is filled."""
        limit = NoLimits()
        result = limit.calculate(order_quantity=1000.0, bar_volume=10000.0, price=100.0)

        assert result.fillable_quantity == 1000.0
        assert result.remaining_quantity == 0.0
        assert result.adjusted_price == 100.0
        assert result.impact_cost == 0.0

    def test_participation_rate_with_volume(self):
        """Test participation rate calculation."""
        limit = NoLimits()
        result = limit.calculate(order_quantity=1000.0, bar_volume=10000.0, price=50.0)

        assert result.participation_rate == 0.1  # 1000 / 10000

    def test_participation_rate_no_volume(self):
        """Test participation rate with None volume."""
        limit = NoLimits()
        result = limit.calculate(order_quantity=500.0, bar_volume=None, price=75.0)

        assert result.participation_rate == 0.0
        assert result.fillable_quantity == 500.0

    def test_participation_rate_zero_volume(self):
        """Test participation rate with zero volume."""
        limit = NoLimits()
        result = limit.calculate(order_quantity=500.0, bar_volume=0.0, price=75.0)

        assert result.participation_rate == 0.0
        assert result.fillable_quantity == 500.0


class TestVolumeParticipationLimit:
    """Test VolumeParticipationLimit execution limit."""

    def test_default_participation(self):
        """Test default 10% participation rate."""
        limit = VolumeParticipationLimit()
        assert limit.max_participation == 0.10
        assert limit.min_volume == 0.0

    def test_fill_within_limit(self):
        """Test order smaller than limit fills completely."""
        limit = VolumeParticipationLimit(max_participation=0.10)
        result = limit.calculate(order_quantity=500.0, bar_volume=10000.0, price=100.0)

        # 10% of 10000 = 1000, so 500 fills completely
        assert result.fillable_quantity == 500.0
        assert result.remaining_quantity == 0.0
        assert result.participation_rate == 0.05  # 500 / 10000

    def test_fill_exceeds_limit(self):
        """Test order larger than limit is partially filled."""
        limit = VolumeParticipationLimit(max_participation=0.10)
        result = limit.calculate(order_quantity=2000.0, bar_volume=10000.0, price=100.0)

        # 10% of 10000 = 1000, so only 1000 fills
        assert result.fillable_quantity == 1000.0
        assert result.remaining_quantity == 1000.0
        assert result.participation_rate == 0.10

    def test_no_volume_data(self):
        """Test that None volume allows full fill."""
        limit = VolumeParticipationLimit(max_participation=0.10)
        result = limit.calculate(order_quantity=5000.0, bar_volume=None, price=100.0)

        assert result.fillable_quantity == 5000.0
        assert result.remaining_quantity == 0.0
        assert result.participation_rate == 0.0

    def test_volume_below_minimum(self):
        """Test that low volume prevents any fill."""
        limit = VolumeParticipationLimit(max_participation=0.10, min_volume=5000.0)
        result = limit.calculate(order_quantity=1000.0, bar_volume=1000.0, price=100.0)

        assert result.fillable_quantity == 0.0
        assert result.remaining_quantity == 1000.0
        assert result.participation_rate == 0.0

    def test_volume_at_minimum(self):
        """Test volume exactly at minimum allows fill."""
        limit = VolumeParticipationLimit(max_participation=0.10, min_volume=5000.0)
        result = limit.calculate(order_quantity=300.0, bar_volume=5000.0, price=100.0)

        # 10% of 5000 = 500, so 300 fills completely
        assert result.fillable_quantity == 300.0
        assert result.remaining_quantity == 0.0

    def test_custom_participation_rate(self):
        """Test custom participation rate."""
        limit = VolumeParticipationLimit(max_participation=0.25)
        result = limit.calculate(order_quantity=5000.0, bar_volume=10000.0, price=50.0)

        # 25% of 10000 = 2500
        assert result.fillable_quantity == 2500.0
        assert result.remaining_quantity == 2500.0
        assert result.participation_rate == 0.25

    def test_zero_volume(self):
        """Test behavior with zero volume bar."""
        limit = VolumeParticipationLimit(max_participation=0.10, min_volume=100.0)
        result = limit.calculate(order_quantity=100.0, bar_volume=0.0, price=100.0)

        assert result.fillable_quantity == 0.0
        assert result.remaining_quantity == 100.0


class TestAdaptiveParticipationLimit:
    """Test AdaptiveParticipationLimit execution limit."""

    def test_default_values(self):
        """Test default configuration."""
        limit = AdaptiveParticipationLimit()
        assert limit.base_participation == 0.10
        assert limit.volatility_factor == 0.5
        assert limit.max_participation == 0.25
        assert limit.min_participation == 0.02
        assert limit.avg_volatility == 0.02

    def test_no_volume_full_fill(self):
        """Test that None volume allows full fill."""
        limit = AdaptiveParticipationLimit()
        result = limit.calculate(order_quantity=1000.0, bar_volume=None, price=100.0)

        assert result.fillable_quantity == 1000.0
        assert result.remaining_quantity == 0.0

    def test_base_participation_no_volatility(self):
        """Test base participation without volatility adjustment."""
        limit = AdaptiveParticipationLimit(base_participation=0.15)
        result = limit.calculate(
            order_quantity=2000.0, bar_volume=10000.0, price=100.0, volatility=None
        )

        # 15% of 10000 = 1500
        assert result.fillable_quantity == 1500.0
        assert result.remaining_quantity == 500.0

    def test_high_volatility_reduces_participation(self):
        """Test that high volatility reduces participation rate."""
        limit = AdaptiveParticipationLimit(
            base_participation=0.10,
            volatility_factor=0.5,
            avg_volatility=0.02,
        )
        # High volatility (2x average)
        result = limit.calculate(
            order_quantity=2000.0, bar_volume=10000.0, price=100.0, volatility=0.04
        )

        # vol_ratio = 0.04 / 0.02 = 2.0
        # adjustment = 0.5 * (2.0 - 1.0) = 0.5
        # participation = 0.10 * (1.0 - 0.5) = 0.05
        # max_fillable = 10000 * 0.05 = 500
        assert result.fillable_quantity == 500.0
        assert result.remaining_quantity == 1500.0

    def test_low_volatility_increases_participation(self):
        """Test that low volatility increases participation rate."""
        limit = AdaptiveParticipationLimit(
            base_participation=0.10,
            volatility_factor=0.5,
            avg_volatility=0.02,
        )
        # Low volatility (half of average)
        result = limit.calculate(
            order_quantity=3000.0, bar_volume=10000.0, price=100.0, volatility=0.01
        )

        # vol_ratio = 0.01 / 0.02 = 0.5
        # adjustment = 0.5 * (0.5 - 1.0) = -0.25
        # participation = 0.10 * (1.0 - (-0.25)) = 0.125
        # max_fillable = 10000 * 0.125 = 1250
        assert result.fillable_quantity == 1250.0
        assert result.remaining_quantity == 1750.0

    def test_max_participation_cap(self):
        """Test that participation doesn't exceed max."""
        limit = AdaptiveParticipationLimit(
            base_participation=0.30,
            volatility_factor=0.5,
            max_participation=0.25,
        )
        result = limit.calculate(
            order_quantity=5000.0, bar_volume=10000.0, price=100.0, volatility=0.01
        )

        # Even with low vol boost, capped at 25%
        assert result.fillable_quantity == 2500.0
        assert result.remaining_quantity == 2500.0

    def test_min_participation_floor(self):
        """Test that participation doesn't go below min."""
        limit = AdaptiveParticipationLimit(
            base_participation=0.05,
            volatility_factor=0.8,
            min_participation=0.02,
            avg_volatility=0.02,
        )
        # Very high volatility (5x)
        result = limit.calculate(
            order_quantity=1000.0, bar_volume=10000.0, price=100.0, volatility=0.10
        )

        # Should be floored at min_participation (0.02)
        # max_fillable = 10000 * 0.02 = 200
        assert result.fillable_quantity == 200.0
        assert result.remaining_quantity == 800.0

    def test_participation_rate_calculated(self):
        """Test participation rate in result."""
        limit = AdaptiveParticipationLimit(base_participation=0.10)
        result = limit.calculate(
            order_quantity=500.0, bar_volume=10000.0, price=100.0, volatility=None
        )

        # 500 / 10000 = 0.05
        assert result.participation_rate == 0.05

    def test_zero_volume(self):
        """Test behavior with zero volume - prevents fill."""
        limit = AdaptiveParticipationLimit()
        result = limit.calculate(order_quantity=100.0, bar_volume=0.0, price=100.0)

        # Zero volume means no fill possible (different from None)
        assert result.fillable_quantity == 0.0
        assert result.remaining_quantity == 100.0
