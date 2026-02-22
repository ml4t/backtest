"""Tests for futures notional exposure and leverage calculations.

Validates notional/leverage calculations against PySystemTrade:
    notional_exposure = abs(qty) * price * multiplier
    leverage = notional_exposure / capital

Reference: systems/accounts/curves/account_curve.py
"""

from datetime import datetime

from ml4t.backtest.types import ContractSpec, Position


class TestNotionalExposure:
    """Test notional exposure calculations."""

    def test_notional_long(self, es_contract: ContractSpec):
        """10 ES at 4000: 10 * 4000 * 50 = $2,000,000."""
        qty = 10
        price = 4000.0
        multiplier = es_contract.multiplier

        notional = abs(qty) * price * multiplier
        assert notional == 2_000_000.0

    def test_notional_short(self, cl_contract: ContractSpec):
        """Short -5 CL at 80: |-5| * 80 * 1000 = $400,000.

        Notional uses absolute value for shorts.
        """
        qty = -5
        price = 80.0
        multiplier = cl_contract.multiplier

        notional = abs(qty) * price * multiplier
        assert notional == 400_000.0

    def test_notional_via_position(self, es_contract: ContractSpec):
        """Position.notional_value() computes correctly."""
        pos = Position(
            asset="ES",
            quantity=10.0,
            entry_price=4000.0,
            entry_time=datetime.now(),
            multiplier=es_contract.multiplier,
        )

        assert pos.notional_value(4000.0) == 2_000_000.0

    def test_notional_short_via_position(self, cl_contract: ContractSpec):
        """Position.notional_value() for shorts uses abs quantity."""
        pos = Position(
            asset="CL",
            quantity=-5.0,
            entry_price=80.0,
            entry_time=datetime.now(),
            multiplier=cl_contract.multiplier,
        )

        assert pos.notional_value(80.0) == 400_000.0


class TestLeverage:
    """Test leverage calculations."""

    def test_leverage_basic(self, es_contract: ContractSpec):
        """Leverage = notional / capital.

        $2M notional / $100k capital = 20x leverage
        """
        capital = 100_000.0
        notional = 10 * 4000.0 * es_contract.multiplier  # $2M

        leverage = notional / capital
        assert leverage == 20.0

    def test_leverage_single_contract(self, es_contract: ContractSpec):
        """Single ES contract leverage.

        Notional: 1 * 4000 * 50 = $200,000
        Capital: $100,000
        Leverage: 2.0x
        """
        capital = 100_000.0
        notional = 1 * 4000.0 * es_contract.multiplier

        leverage = notional / capital
        assert leverage == 2.0

    def test_leverage_with_margin(self, es_contract: ContractSpec):
        """Margin-based leverage calculation.

        ES margin: $15,000 per contract
        Capital: $100,000
        Max contracts = 100,000 / 15,000 = 6 contracts
        Max notional = 6 * 4000 * 50 = $1,200,000
        Effective leverage = 1,200,000 / 100,000 = 12x
        """
        capital = 100_000.0
        margin_per_contract = es_contract.margin
        assert margin_per_contract is not None

        max_contracts = int(capital / margin_per_contract)
        assert max_contracts == 6

        max_notional = max_contracts * 4000.0 * es_contract.multiplier
        effective_leverage = max_notional / capital
        assert effective_leverage == 12.0


class TestPortfolioNotional:
    """Test portfolio-level notional calculations."""

    def test_portfolio_notional(self, es_contract: ContractSpec, cl_contract: ContractSpec):
        """Portfolio notional: sum of individual notionals.

        5 ES at 4000: 5 * 4000 * 50 = $1,000,000
        2 CL at 80: 2 * 80 * 1000 = $160,000
        Total: $1,160,000
        """
        es_notional = 5 * 4000.0 * es_contract.multiplier
        cl_notional = 2 * 80.0 * cl_contract.multiplier

        total = es_notional + cl_notional
        assert total == 1_160_000.0

    def test_portfolio_leverage(self, es_contract: ContractSpec, cl_contract: ContractSpec):
        """Portfolio leverage = total notional / capital."""
        capital = 100_000.0

        es_notional = 5 * 4000.0 * es_contract.multiplier
        cl_notional = 2 * 80.0 * cl_contract.multiplier
        total_notional = es_notional + cl_notional

        leverage = total_notional / capital
        assert leverage == 11.6

    def test_mixed_long_short_notional(self, es_contract: ContractSpec, cl_contract: ContractSpec):
        """Mixed positions: gross notional uses absolute values.

        Long 5 ES: $1,000,000
        Short 2 CL: |-2| * 80 * 1000 = $160,000
        Gross notional: $1,160,000 (not net)
        """
        es_notional = abs(5) * 4000.0 * es_contract.multiplier
        cl_notional = abs(-2) * 80.0 * cl_contract.multiplier

        gross_notional = es_notional + cl_notional
        assert gross_notional == 1_160_000.0
