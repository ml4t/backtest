"""Tests for market impact models."""

import math
from datetime import datetime, timedelta

import pytest

from ml4t.backtest.execution.market_impact import (
    AlmgrenChrissImpact,
    ImpactState,
    IntraDayMomentum,
    LinearMarketImpact,
    NoMarketImpact,
    ObizhaevWangImpact,
    PropagatorImpact,
)
from ml4t.backtest.execution.order import Order, OrderSide, OrderType


@pytest.fixture
def buy_order():
    """Create a sample buy order."""
    return Order(
        order_id="TEST001",
        asset_id="AAPL",
        quantity=1000,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        created_time=datetime.now(),
    )


@pytest.fixture
def sell_order():
    """Create a sample sell order."""
    return Order(
        order_id="TEST002",
        asset_id="AAPL",
        quantity=1000,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        created_time=datetime.now(),
    )


class TestImpactState:
    """Test ImpactState dataclass."""

    def test_initial_state(self):
        """Test initial impact state."""
        state = ImpactState()
        assert state.permanent_impact == 0.0
        assert state.temporary_impact == 0.0
        assert state.last_update is None
        assert state.volume_traded == 0.0

    def test_total_impact(self):
        """Test total impact calculation."""
        state = ImpactState(permanent_impact=0.5, temporary_impact=0.3)
        assert state.get_total_impact() == 0.8

    def test_decay_temporary_impact(self):
        """Test temporary impact decay."""
        state = ImpactState(temporary_impact=1.0)

        # Test decay with time
        state.decay_temporary_impact(decay_rate=0.1, time_elapsed=10.0)
        assert pytest.approx(state.temporary_impact) == math.exp(-1.0)

        # Test no decay with zero time
        initial = state.temporary_impact
        state.decay_temporary_impact(decay_rate=0.1, time_elapsed=0.0)
        assert state.temporary_impact == initial

    def test_cleanup_near_zero(self):
        """Test cleanup of near-zero temporary impact."""
        state = ImpactState(temporary_impact=1e-11)
        state.decay_temporary_impact(decay_rate=0.1, time_elapsed=1.0)
        assert state.temporary_impact == 0.0


class TestNoMarketImpact:
    """Test NoMarketImpact model."""

    def test_zero_impact(self, buy_order):
        """Test that no impact model returns zeros."""
        model = NoMarketImpact()
        permanent, temporary = model.calculate_impact(
            buy_order,
            1000,
            150.0,
            datetime.now(),
        )
        assert permanent == 0.0
        assert temporary == 0.0

    def test_get_current_impact(self, buy_order):
        """Test getting current impact (should be zero)."""
        model = NoMarketImpact()
        model.calculate_impact(buy_order, 1000, 150.0, datetime.now())
        assert model.get_current_impact("AAPL") == 0.0


class TestLinearMarketImpact:
    """Test LinearMarketImpact model."""

    def test_buy_impact(self, buy_order):
        """Test buy order creates positive impact."""
        model = LinearMarketImpact(
            permanent_impact_factor=0.1,
            temporary_impact_factor=0.5,
            avg_daily_volume=100_000,
        )

        permanent, temporary = model.calculate_impact(
            buy_order,
            1000,
            100.0,
            datetime.now(),
        )

        # 1000/100000 = 0.01 volume fraction
        assert permanent == pytest.approx(0.1)  # 100 * 0.1 * 0.01
        assert temporary == pytest.approx(0.5)  # 100 * 0.5 * 0.01

    def test_sell_impact(self, sell_order):
        """Test sell order creates negative impact."""
        model = LinearMarketImpact(
            permanent_impact_factor=0.1,
            temporary_impact_factor=0.5,
            avg_daily_volume=100_000,
        )

        permanent, temporary = model.calculate_impact(
            sell_order,
            1000,
            100.0,
            datetime.now(),
        )

        assert permanent == pytest.approx(-0.1)
        assert temporary == pytest.approx(-0.5)

    def test_state_tracking(self, buy_order):
        """Test market state tracking."""
        model = LinearMarketImpact()
        timestamp = datetime.now()

        permanent, temporary = model.calculate_impact(
            buy_order,
            1000,
            100.0,
            timestamp,
        )

        model.update_market_state("AAPL", permanent, temporary, timestamp)

        # Check state was updated
        assert "AAPL" in model.impact_states
        state = model.impact_states["AAPL"]
        assert state.permanent_impact == permanent
        assert state.temporary_impact == temporary

    def test_impact_decay(self, buy_order):
        """Test temporary impact decay over time."""
        model = LinearMarketImpact(decay_rate=1.0)  # Fast decay for testing
        timestamp = datetime.now()

        # Create initial impact
        permanent, temporary = model.calculate_impact(
            buy_order,
            1000,
            100.0,
            timestamp,
        )
        model.update_market_state("AAPL", permanent, temporary, timestamp)

        # Check impact after 1 second
        future_time = timestamp + timedelta(seconds=1)
        current_impact = model.get_current_impact("AAPL", future_time)

        # Permanent should remain, temporary should decay
        expected = permanent + temporary * math.exp(-1.0)
        assert pytest.approx(current_impact) == expected


class TestAlmgrenChrissImpact:
    """Test AlmgrenChrissImpact model."""

    def test_square_root_permanent_impact(self, buy_order):
        """Test square-root scaling of permanent impact."""
        model = AlmgrenChrissImpact(
            permanent_impact_const=0.01,
            temporary_impact_const=0.1,
            daily_volatility=0.02,
            avg_daily_volume=100_000,
        )

        # Test with different volumes
        perm1, _ = model.calculate_impact(buy_order, 100, 100.0, datetime.now())
        perm2, _ = model.calculate_impact(buy_order, 400, 100.0, datetime.now())

        # Square root relationship: 400 is 4x volume, sqrt(4) = 2
        assert pytest.approx(perm2 / perm1) == 2.0

    def test_linear_temporary_impact(self, buy_order):
        """Test linear scaling of temporary impact."""
        model = AlmgrenChrissImpact(
            permanent_impact_const=0.01,
            temporary_impact_const=0.1,
            daily_volatility=0.02,
            avg_daily_volume=100_000,
        )

        # Test with different volumes
        _, temp1 = model.calculate_impact(buy_order, 100, 100.0, datetime.now())
        _, temp2 = model.calculate_impact(buy_order, 400, 100.0, datetime.now())

        # Linear relationship
        assert pytest.approx(temp2 / temp1) == 4.0

    def test_volatility_scaling(self, buy_order):
        """Test that impact scales with volatility."""
        model1 = AlmgrenChrissImpact(daily_volatility=0.01)
        model2 = AlmgrenChrissImpact(daily_volatility=0.02)

        perm1, temp1 = model1.calculate_impact(buy_order, 1000, 100.0, datetime.now())
        perm2, temp2 = model2.calculate_impact(buy_order, 1000, 100.0, datetime.now())

        # Both should scale linearly with volatility
        assert pytest.approx(perm2 / perm1) == 2.0
        assert pytest.approx(temp2 / temp1) == 2.0


class TestPropagatorImpact:
    """Test PropagatorImpact model."""

    def test_power_law_impact(self, buy_order):
        """Test power law scaling of impact."""
        model = PropagatorImpact(
            impact_coefficient=0.1,
            propagator_exponent=0.5,
            avg_daily_volume=100_000,
        )

        perm1, _ = model.calculate_impact(buy_order, 100, 100.0, datetime.now())
        perm2, _ = model.calculate_impact(buy_order, 400, 100.0, datetime.now())

        # Power law with exponent 0.5
        # Note: Only checking instantaneous part (0.2 factor)
        ratio = abs(perm2 / perm1)
        assert pytest.approx(ratio, rel=0.1) == 2.0  # sqrt(4) = 2

    def test_order_history(self, buy_order, sell_order):
        """Test order history tracking."""
        model = PropagatorImpact()
        timestamp = datetime.now()

        # Execute multiple orders
        model.calculate_impact(buy_order, 1000, 100.0, timestamp)
        model.calculate_impact(sell_order, 500, 100.0, timestamp + timedelta(seconds=1))

        assert len(model.order_history) == 2
        assert model.order_history[0][1] == 1000  # Buy volume
        assert model.order_history[1][1] == -500  # Sell volume (negative)

    def test_history_cleanup(self, buy_order):
        """Test that old history is cleaned up."""
        model = PropagatorImpact()

        # Add many orders
        for i in range(1100):
            model.calculate_impact(
                buy_order,
                100,
                100.0,
                datetime.now() + timedelta(seconds=i),
            )

        # Should keep only recent history (trimmed to 500 when > 1000)
        # After 1100 orders: trimmed at 1001 to 500, then added 99 more
        assert len(model.order_history) == 599

    def test_reset(self, buy_order):
        """Test reset functionality."""
        model = PropagatorImpact()
        model.calculate_impact(buy_order, 1000, 100.0, datetime.now())

        assert len(model.order_history) > 0
        model.reset()
        assert len(model.order_history) == 0
        assert len(model.impact_states) == 0


class TestIntraDayMomentum:
    """Test IntraDayMomentum model."""

    def test_momentum_buildup(self, buy_order):
        """Test momentum builds up with same-direction trades."""
        model = IntraDayMomentum(
            base_impact=0.05,
            momentum_factor=0.3,
            momentum_decay=0.2,
        )

        timestamp = datetime.now()

        # First buy order
        perm1, temp1 = model.calculate_impact(buy_order, 1000, 100.0, timestamp)
        impact1 = perm1 + temp1

        # Second buy order (same direction)
        perm2, temp2 = model.calculate_impact(
            buy_order,
            1000,
            100.0,
            timestamp + timedelta(seconds=1),
        )
        impact2 = perm2 + temp2

        # Second order should have larger impact due to momentum
        assert abs(impact2) > abs(impact1)

    def test_momentum_reversal(self, buy_order, sell_order):
        """Test momentum reversal with opposite trades."""
        model = IntraDayMomentum(
            base_impact=0.05,
            momentum_factor=0.3,
            momentum_decay=0.2,
        )

        timestamp = datetime.now()

        # Build buy momentum
        model.calculate_impact(buy_order, 1000, 100.0, timestamp)

        # Sell against momentum
        perm, temp = model.calculate_impact(
            sell_order,
            1000,
            100.0,
            timestamp + timedelta(seconds=1),
        )

        # Should have reduced impact when trading against momentum
        base_impact = 0.05 * 100.0 * (1000 / 1_000_000)
        assert abs(perm + temp) < base_impact * 1.3

    def test_momentum_state(self, buy_order):
        """Test momentum state tracking."""
        model = IntraDayMomentum()

        model.calculate_impact(buy_order, 1000, 100.0, datetime.now())

        assert "AAPL" in model.momentum_states
        assert model.momentum_states["AAPL"] != 0.0

    def test_reset(self, buy_order):
        """Test reset clears momentum."""
        model = IntraDayMomentum()
        model.calculate_impact(buy_order, 1000, 100.0, datetime.now())

        model.reset()
        assert len(model.momentum_states) == 0
        assert len(model.impact_states) == 0


class TestObizhaevWangImpact:
    """Test ObizhaevWangImpact model."""

    def test_information_split(self, buy_order):
        """Test split between permanent and temporary impact."""
        model = ObizhaevWangImpact(
            price_impact_const=0.1,
            information_share=0.3,
            book_depth=100_000,
        )

        permanent, temporary = model.calculate_impact(
            buy_order,
            1000,
            100.0,
            datetime.now(),
        )

        # Permanent should be 30% of total linear impact
        # Temporary should be 70% of total linear impact
        total_impact = abs(permanent) + abs(temporary)
        assert pytest.approx(abs(permanent) / total_impact) == 0.3
        assert pytest.approx(abs(temporary) / total_impact) == 0.7

    def test_book_depth_scaling(self, buy_order):
        """Test impact scales with book depth."""
        model1 = ObizhaevWangImpact(book_depth=100_000)
        model2 = ObizhaevWangImpact(book_depth=200_000)

        perm1, temp1 = model1.calculate_impact(buy_order, 1000, 100.0, datetime.now())
        perm2, temp2 = model2.calculate_impact(buy_order, 1000, 100.0, datetime.now())

        # Deeper book should have less impact
        assert abs(perm2) < abs(perm1)
        assert abs(temp2) < abs(temp1)

    def test_resilience_rate(self, buy_order):
        """Test resilience rate affects decay."""
        model = ObizhaevWangImpact(resilience_rate=2.0)  # Fast resilience

        timestamp = datetime.now()
        permanent, temporary = model.calculate_impact(
            buy_order,
            1000,
            100.0,
            timestamp,
        )

        model.update_market_state("AAPL", permanent, temporary, timestamp)

        # Check fast decay
        future_time = timestamp + timedelta(seconds=1)
        current_impact = model.get_current_impact("AAPL", future_time)

        # Should decay quickly
        expected = permanent + temporary * math.exp(-2.0)
        assert pytest.approx(current_impact) == expected


class TestMarketImpactIntegration:
    """Test market impact integration scenarios."""

    def test_multiple_assets(self, buy_order, sell_order):
        """Test impact tracking for multiple assets."""
        model = LinearMarketImpact()
        timestamp = datetime.now()

        # Impact on AAPL
        buy_order.asset_id = "AAPL"
        perm1, temp1 = model.calculate_impact(buy_order, 1000, 100.0, timestamp)
        model.update_market_state("AAPL", perm1, temp1, timestamp)

        # Impact on GOOGL
        sell_order.asset_id = "GOOGL"
        perm2, temp2 = model.calculate_impact(sell_order, 500, 200.0, timestamp)
        model.update_market_state("GOOGL", perm2, temp2, timestamp)

        # Check separate tracking
        assert model.get_current_impact("AAPL") > 0
        assert model.get_current_impact("GOOGL") < 0
        assert "AAPL" in model.impact_states
        assert "GOOGL" in model.impact_states

    def test_cumulative_impact(self, buy_order):
        """Test cumulative impact from multiple orders."""
        model = LinearMarketImpact(
            permanent_impact_factor=0.1,
            temporary_impact_factor=0.5,
            decay_rate=0.0,  # No decay for simplicity
        )

        timestamp = datetime.now()

        # Execute multiple buy orders
        total_permanent = 0
        total_temporary = 0

        for i in range(3):
            perm, temp = model.calculate_impact(
                buy_order,
                1000,
                100.0,
                timestamp + timedelta(seconds=i),
            )
            model.update_market_state(
                "AAPL",
                perm,
                temp,
                timestamp + timedelta(seconds=i),
            )
            total_permanent += perm
            total_temporary += temp

        # Check cumulative effect
        final_impact = model.get_current_impact("AAPL")
        assert pytest.approx(final_impact) == total_permanent + total_temporary
