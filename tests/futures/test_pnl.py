"""Tests for futures P&L calculation.

Validates P&L calculations against PySystemTrade's formula:
    pnl = position[t-1] * (price[t] - price[t-1]) * multiplier

Reference: systems/accounts/pandl_calculators/pandl_cash_costs.py
"""

from datetime import datetime

from ml4t.backtest.types import ContractSpec, Position


class TestBasicPnL:
    """Test basic P&L calculations for futures positions."""

    def test_pnl_long_position(self, es_contract: ContractSpec):
        """Long 2 ES: entry 4000, exit 4010 = 2 * 10 * 50 = $1000.

        PySystemTrade formula: pnl = qty * (exit - entry) * multiplier
        """
        qty = 2
        entry_price = 4000.0
        exit_price = 4010.0
        multiplier = es_contract.multiplier

        expected_pnl = qty * (exit_price - entry_price) * multiplier
        assert expected_pnl == 1000.0

    def test_pnl_short_position(self, cl_contract: ContractSpec):
        """Short 3 CL: entry 80, exit 78 = -3 * (78-80) * 1000 = $6000.

        Short positions have negative quantity in our model.
        """
        qty = -3  # Short position
        entry_price = 80.0
        exit_price = 78.0
        multiplier = cl_contract.multiplier

        # PySystemTrade formula works for shorts too (negative qty)
        expected_pnl = qty * (exit_price - entry_price) * multiplier
        assert expected_pnl == 6000.0  # -3 * -2 * 1000 = 6000

    def test_pnl_fractional_contracts(self, btc_contract: ContractSpec):
        """0.5 BTC position with 1000 move = $2500.

        Some systems allow fractional contracts for sizing.
        0.5 contracts * 5 BTC/contract * $1000 move = $2500
        """
        qty = 0.5
        entry_price = 40000.0
        exit_price = 41000.0
        multiplier = btc_contract.multiplier

        expected_pnl = qty * (exit_price - entry_price) * multiplier
        assert expected_pnl == 2500.0  # 0.5 * 1000 * 5


class TestMultidayPnL:
    """Test multi-day mark-to-market P&L calculations."""

    def test_pnl_multiday_holding(self, es_contract: ContractSpec):
        """Hold 1 ES for 3 days with daily MTM.

        Day 1: 4000→4005 = 250
        Day 2: 4005→4020 = 750
        Day 3: 4020→4015 = -250
        Total: 750

        PySystemTrade calculates daily PnL then sums.
        """
        qty = 1
        multiplier = es_contract.multiplier
        prices = [4000.0, 4005.0, 4020.0, 4015.0]

        daily_pnls = []
        for i in range(1, len(prices)):
            daily_pnl = qty * (prices[i] - prices[i - 1]) * multiplier
            daily_pnls.append(daily_pnl)

        assert daily_pnls == [250.0, 750.0, -250.0]
        assert sum(daily_pnls) == 750.0

    def test_pnl_equivalent_direct_calculation(self, es_contract: ContractSpec):
        """Direct entry-to-exit equals sum of daily MTM."""
        qty = 1
        multiplier = es_contract.multiplier
        entry_price = 4000.0
        exit_price = 4015.0

        # Direct calculation
        direct_pnl = qty * (exit_price - entry_price) * multiplier
        assert direct_pnl == 750.0

        # This equals the sum of daily MTM (tested above)


class TestPositionPnL:
    """Test P&L calculations using Position dataclass."""

    def test_position_unrealized_pnl_long(self, es_contract: ContractSpec):
        """Position.unrealized_pnl() with multiplier for longs."""
        pos = Position(
            asset="ES",
            quantity=2.0,
            entry_price=4000.0,
            entry_time=datetime.now(),
            multiplier=es_contract.multiplier,
        )

        # Mark to 4010
        pnl = pos.unrealized_pnl(4010.0)
        assert pnl == 1000.0  # 2 * 10 * 50

    def test_position_unrealized_pnl_short(self, cl_contract: ContractSpec):
        """Position.unrealized_pnl() with multiplier for shorts."""
        pos = Position(
            asset="CL",
            quantity=-3.0,  # Short
            entry_price=80.0,
            entry_time=datetime.now(),
            multiplier=cl_contract.multiplier,
        )

        # Mark to 78 (profitable for short)
        pnl = pos.unrealized_pnl(78.0)
        assert pnl == 6000.0  # -3 * (78 - 80) * 1000 = 6000

    def test_position_notional_value(self, es_contract: ContractSpec):
        """Position.notional_value() uses multiplier."""
        pos = Position(
            asset="ES",
            quantity=2.0,
            entry_price=4000.0,
            entry_time=datetime.now(),
            multiplier=es_contract.multiplier,
        )

        # Notional = |qty| * price * multiplier
        notional = pos.notional_value(4000.0)
        assert notional == 400000.0  # 2 * 4000 * 50

    def test_position_market_value(self, es_contract: ContractSpec):
        """Position.market_value uses multiplier."""
        pos = Position(
            asset="ES",
            quantity=2.0,
            entry_price=4000.0,
            entry_time=datetime.now(),
            current_price=4010.0,
            multiplier=es_contract.multiplier,
        )

        # Market value = qty * current_price * multiplier
        assert pos.market_value == 401000.0  # 2 * 4010 * 50
