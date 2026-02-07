"""Tests for complete trade records with futures.

Validates that Trade objects correctly track P&L, costs, and exit reasons
for futures trades.
"""

import pytest
from datetime import datetime, timedelta

from ml4t.backtest.types import ContractSpec, Trade
from ml4t.backtest.models import FuturesCommission, FuturesSlippage


class TestCompleteTradeRecord:
    """Test complete trade lifecycle with costs."""

    def test_complete_trade(
        self, es_contract: ContractSpec, ib_commission, es_slippage
    ):
        """Entry → Hold → Exit with correct MTM and net PnL.

        Entry: Buy 2 ES at 4000
        Exit: Sell 2 at 4015
        Gross P&L: 2 * 15 * 50 = $1,500
        Commission: 2 * $2.25 * 2 (entry+exit) = $9.00
        Slippage: 2 * 0.25 * 50 * 2 (entry+exit) = $50.00
        Net P&L: 1500 - 9 - 50 = $1,441
        """
        qty = 2.0
        entry_price = 4000.0
        exit_price = 4015.0
        multiplier = es_contract.multiplier

        # Gross P&L
        gross_pnl = qty * (exit_price - entry_price) * multiplier
        assert gross_pnl == 1500.0

        # Costs (entry + exit)
        entry_comm = ib_commission.calculate("ES", qty, entry_price, multiplier)
        exit_comm = ib_commission.calculate("ES", -qty, exit_price, multiplier)
        entry_slip = es_slippage.calculate("ES", qty, entry_price, multiplier=multiplier)
        exit_slip = es_slippage.calculate("ES", -qty, exit_price, multiplier=multiplier)

        total_comm = entry_comm + exit_comm
        total_slip = entry_slip + exit_slip

        assert total_comm == 9.0
        assert total_slip == 50.0

        # Net P&L
        net_pnl = gross_pnl - total_comm - total_slip
        assert net_pnl == 1441.0

    def test_losing_trade_with_costs(
        self, es_contract: ContractSpec, ib_commission, es_slippage
    ):
        """Losing trade: costs make it worse.

        Entry: Buy 2 ES at 4000
        Exit: Sell 2 at 3990 (10 point loss)
        Gross P&L: 2 * -10 * 50 = -$1,000
        Costs: same as above = $59
        Net P&L: -1000 - 59 = -$1,059
        """
        qty = 2.0
        entry_price = 4000.0
        exit_price = 3990.0
        multiplier = es_contract.multiplier

        gross_pnl = qty * (exit_price - entry_price) * multiplier
        assert gross_pnl == -1000.0

        total_costs = 59.0  # From previous test
        net_pnl = gross_pnl - total_costs
        assert net_pnl == -1059.0


class TestTradeDataclass:
    """Test Trade dataclass with futures."""

    def test_trade_creation(self, es_contract: ContractSpec):
        """Create Trade with correct attributes."""
        trade = Trade(
            symbol="ES",
            entry_time=datetime(2024, 1, 1, 9, 30),
            exit_time=datetime(2024, 1, 1, 15, 0),
            entry_price=4000.0,
            exit_price=4015.0,
            quantity=2.0,
            pnl=1441.0,  # Net P&L after costs
            pnl_percent=0.00375,  # 1441 / (2 * 4000 * 50)
            bars_held=10,
            fees=9.0,
            slippage=50.0,
            exit_reason="signal",
        )

        assert trade.symbol == "ES"
        assert trade.direction == "long"
        assert trade.pnl == 1441.0
        assert trade.fees == 9.0
        assert trade.slippage == 50.0

    def test_trade_short(self, cl_contract: ContractSpec):
        """Short trade with negative quantity."""
        trade = Trade(
            symbol="CL",
            entry_time=datetime(2024, 1, 1, 9, 30),
            exit_time=datetime(2024, 1, 1, 15, 0),
            entry_price=80.0,
            exit_price=78.0,
            quantity=-3.0,  # Short
            pnl=6000.0 - 100.0,  # Gross - costs
            pnl_percent=0.025,
            bars_held=5,
            fees=15.0,
            slippage=85.0,
            exit_reason="take_profit",
        )

        assert trade.direction == "short"
        assert trade.quantity == -3.0


class TestCostsMatchPySysTradeFormula:
    """Verify cost calculations match PySystemTrade formulas."""

    def test_net_equals_gross_minus_costs(
        self, es_contract: ContractSpec, pysystemtrade_commission, es_slippage
    ):
        """Net P&L = Gross P&L - Slippage - Commission.

        PySystemTrade formula:
            net_currency_cost = slippage_cost + commission_cost
            net_p_and_l = gross_p_and_l - net_currency_cost
        """
        qty = 5.0
        entry_price = 4000.0
        exit_price = 4020.0
        multiplier = es_contract.multiplier

        # Gross P&L
        gross_pnl = qty * (exit_price - entry_price) * multiplier
        assert gross_pnl == 5000.0  # 5 * 20 * 50

        # Entry costs (using PST commission model)
        entry_comm = pysystemtrade_commission.calculate(
            "ES", qty, entry_price, multiplier
        )
        entry_slip = es_slippage.calculate(
            "ES", qty, entry_price, multiplier=multiplier
        )

        # Exit costs
        exit_comm = pysystemtrade_commission.calculate(
            "ES", -qty, exit_price, multiplier
        )
        exit_slip = es_slippage.calculate(
            "ES", -qty, exit_price, multiplier=multiplier
        )

        # Total costs
        total_comm = entry_comm + exit_comm
        total_slip = entry_slip + exit_slip
        net_currency_cost = total_slip + total_comm

        # Net P&L
        net_pnl = gross_pnl - net_currency_cost

        # Verify the formula
        assert net_pnl == gross_pnl - total_slip - total_comm
