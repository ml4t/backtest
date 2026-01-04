"""Tests for market impact models."""

import math

from ml4t.backtest.execution.impact import (
    LinearImpact,
    NoImpact,
    PowerLawImpact,
    SquareRootImpact,
)


class TestNoImpact:
    """Test NoImpact market impact model."""

    def test_returns_zero_for_buy(self):
        """Test no impact for buy orders."""
        model = NoImpact()
        impact = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)
        assert impact == 0.0

    def test_returns_zero_for_sell(self):
        """Test no impact for sell orders."""
        model = NoImpact()
        impact = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=False)
        assert impact == 0.0

    def test_returns_zero_no_volume(self):
        """Test no impact with None volume."""
        model = NoImpact()
        impact = model.calculate(quantity=5000.0, price=50.0, volume=None, is_buy=True)
        assert impact == 0.0


class TestLinearImpact:
    """Test LinearImpact market impact model."""

    def test_default_values(self):
        """Test default configuration."""
        model = LinearImpact()
        assert model.coefficient == 0.1
        assert model.permanent_fraction == 0.5

    def test_buy_positive_impact(self):
        """Test that buy orders have positive impact (price goes up)."""
        model = LinearImpact(coefficient=1.0)  # coefficient=1 for easier math
        # 10% participation (1000/10000) at $100
        impact = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)

        # impact = 1.0 * (1000/10000) * 100 = 1.0 * 0.1 * 100 = 10.0
        assert impact == 10.0

    def test_sell_negative_impact(self):
        """Test that sell orders have negative impact (price goes down)."""
        model = LinearImpact(coefficient=1.0)  # coefficient=1 for easier math
        impact = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=False)

        # Same magnitude but negative
        assert impact == -10.0

    def test_larger_orders_more_impact(self):
        """Test that larger orders have more impact."""
        model = LinearImpact(coefficient=0.1)

        small_impact = model.calculate(quantity=500.0, price=100.0, volume=10000.0, is_buy=True)
        large_impact = model.calculate(quantity=2000.0, price=100.0, volume=10000.0, is_buy=True)

        assert large_impact > small_impact
        assert large_impact == 4 * small_impact  # Linear relationship

    def test_no_volume_returns_zero(self):
        """Test that None volume returns zero impact."""
        model = LinearImpact(coefficient=0.1)
        impact = model.calculate(quantity=1000.0, price=100.0, volume=None, is_buy=True)
        assert impact == 0.0

    def test_zero_volume_returns_zero(self):
        """Test that zero volume returns zero impact."""
        model = LinearImpact(coefficient=0.1)
        impact = model.calculate(quantity=1000.0, price=100.0, volume=0.0, is_buy=True)
        assert impact == 0.0

    def test_custom_coefficient(self):
        """Test custom coefficient."""
        model = LinearImpact(coefficient=0.5)
        impact = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)

        # impact = 0.5 * 0.1 * 100 = 5.0
        assert impact == 5.0


class TestSquareRootImpact:
    """Test SquareRootImpact market impact model (Almgren-Chriss)."""

    def test_default_values(self):
        """Test default configuration."""
        model = SquareRootImpact()
        assert model.coefficient == 0.5
        assert model.volatility == 0.02
        assert model.adv_factor == 1.0

    def test_buy_positive_impact(self):
        """Test that buy orders have positive impact."""
        model = SquareRootImpact(coefficient=0.5, volatility=0.02, adv_factor=1.0)
        # participation = 1000/10000 = 0.1
        # impact = 0.5 * 0.02 * sqrt(0.1) * 100
        impact = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)

        expected = 0.5 * 0.02 * math.sqrt(0.1) * 100
        assert abs(impact - expected) < 0.0001

    def test_sell_negative_impact(self):
        """Test that sell orders have negative impact."""
        model = SquareRootImpact(coefficient=0.5, volatility=0.02)
        impact = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=False)

        expected = -0.5 * 0.02 * math.sqrt(0.1) * 100
        assert abs(impact - expected) < 0.0001

    def test_square_root_scaling(self):
        """Test that impact scales with square root of size."""
        model = SquareRootImpact(coefficient=1.0, volatility=0.01)

        impact_1000 = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)
        impact_4000 = model.calculate(quantity=4000.0, price=100.0, volume=10000.0, is_buy=True)

        # 4x quantity should give 2x impact (sqrt(4) = 2)
        assert abs(impact_4000 / impact_1000 - 2.0) < 0.01

    def test_no_volume_returns_zero(self):
        """Test that None volume returns zero impact."""
        model = SquareRootImpact()
        impact = model.calculate(quantity=1000.0, price=100.0, volume=None, is_buy=True)
        assert impact == 0.0

    def test_zero_volume_returns_zero(self):
        """Test that zero volume returns zero impact."""
        model = SquareRootImpact()
        impact = model.calculate(quantity=1000.0, price=100.0, volume=0.0, is_buy=True)
        assert impact == 0.0

    def test_adv_factor(self):
        """Test ADV factor adjusts volume normalization."""
        # For minute bars, adv_factor might be 390 (trading minutes per day)
        model = SquareRootImpact(coefficient=0.5, volatility=0.02, adv_factor=10.0)

        # With adv_factor=10, effective volume is 10000 * 10 = 100000
        # participation = 1000/100000 = 0.01
        impact = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)

        expected = 0.5 * 0.02 * math.sqrt(0.01) * 100
        assert abs(impact - expected) < 0.0001


class TestPowerLawImpact:
    """Test PowerLawImpact market impact model."""

    def test_default_values(self):
        """Test default configuration."""
        model = PowerLawImpact()
        assert model.coefficient == 0.1
        assert model.exponent == 0.5
        assert model.min_impact == 0.0

    def test_buy_positive_impact(self):
        """Test that buy orders have positive impact."""
        model = PowerLawImpact(coefficient=0.1, exponent=0.5)
        # participation = 0.1, impact = 0.1 * 0.1^0.5 * 100
        impact = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)

        expected = 0.1 * (0.1**0.5) * 100
        assert abs(impact - expected) < 0.0001

    def test_sell_negative_impact(self):
        """Test that sell orders have negative impact."""
        model = PowerLawImpact(coefficient=0.1, exponent=0.5)
        impact = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=False)

        expected = -0.1 * (0.1**0.5) * 100
        assert abs(impact - expected) < 0.0001

    def test_linear_exponent(self):
        """Test that exponent=1.0 gives linear behavior."""
        model = PowerLawImpact(coefficient=0.1, exponent=1.0)

        impact_1000 = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)
        impact_2000 = model.calculate(quantity=2000.0, price=100.0, volume=10000.0, is_buy=True)

        # With linear exponent, 2x quantity = 2x impact
        assert abs(impact_2000 / impact_1000 - 2.0) < 0.01

    def test_square_root_exponent(self):
        """Test that exponent=0.5 matches SquareRootImpact behavior."""
        model = PowerLawImpact(coefficient=0.1, exponent=0.5)

        impact_1000 = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)
        impact_4000 = model.calculate(quantity=4000.0, price=100.0, volume=10000.0, is_buy=True)

        # 4x quantity should give 2x impact (4^0.5 = 2)
        assert abs(impact_4000 / impact_1000 - 2.0) < 0.01

    def test_no_volume_returns_min_impact(self):
        """Test that None volume returns min_impact."""
        model = PowerLawImpact(min_impact=0.5)
        impact = model.calculate(quantity=1000.0, price=100.0, volume=None, is_buy=True)
        assert impact == 0.5

    def test_no_volume_sell_negative_min(self):
        """Test that None volume on sell returns negative min_impact."""
        model = PowerLawImpact(min_impact=0.5)
        impact = model.calculate(quantity=1000.0, price=100.0, volume=None, is_buy=False)
        assert impact == -0.5

    def test_zero_volume_returns_min_impact(self):
        """Test that zero volume returns min_impact."""
        model = PowerLawImpact(min_impact=0.25)
        impact = model.calculate(quantity=1000.0, price=100.0, volume=0.0, is_buy=True)
        assert impact == 0.25

    def test_min_impact_floor(self):
        """Test that calculated impact is floored at min_impact."""
        model = PowerLawImpact(coefficient=0.001, exponent=0.5, min_impact=0.10)
        # Very small coefficient should give impact < min_impact
        impact = model.calculate(quantity=100.0, price=100.0, volume=100000.0, is_buy=True)

        # Should be at least min_impact
        assert impact >= 0.10

    def test_convex_exponent(self):
        """Test exponent > 1 gives convex (accelerating) impact."""
        model = PowerLawImpact(coefficient=0.1, exponent=2.0)

        impact_1000 = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)
        impact_2000 = model.calculate(quantity=2000.0, price=100.0, volume=10000.0, is_buy=True)

        # With exponent=2, 2x quantity = 4x impact
        assert abs(impact_2000 / impact_1000 - 4.0) < 0.1

    def test_concave_exponent(self):
        """Test exponent < 0.5 gives concave (flattening) impact."""
        model = PowerLawImpact(coefficient=0.1, exponent=0.25)

        impact_1000 = model.calculate(quantity=1000.0, price=100.0, volume=10000.0, is_buy=True)
        impact_16000 = model.calculate(quantity=16000.0, price=100.0, volume=10000.0, is_buy=True)

        # With exponent=0.25, 16x quantity = 2x impact (16^0.25 = 2)
        assert abs(impact_16000 / impact_1000 - 2.0) < 0.1
