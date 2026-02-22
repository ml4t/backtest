"""Tests for FuturesSlippage model.

Validates slippage calculations against PySystemTrade's formula:
    slippage = abs(qty) * price_slippage_points * multiplier

Reference: sysobjects/instruments.py in PySystemTrade
"""

from ml4t.backtest.models import FuturesSlippage


class TestFuturesSlippageBasic:
    """Test basic slippage calculations."""

    def test_single_contract_es(self, es_slippage: FuturesSlippage):
        """1 ES with 0.25 tick slippage: 1 * 0.25 * 50 = $12.50."""
        cost = es_slippage.calculate("ES", 1.0, 4000.0, multiplier=50.0)
        assert cost == 12.50

    def test_multiple_contracts_es(self, es_slippage: FuturesSlippage):
        """5 ES with 0.25 tick slippage: 5 * 0.25 * 50 = $62.50."""
        cost = es_slippage.calculate("ES", 5.0, 4000.0, multiplier=50.0)
        assert cost == 62.50

    def test_short_position_uses_abs(self, es_slippage: FuturesSlippage):
        """Slippage is absolute: |-3| * 0.25 * 50 = $37.50."""
        cost = es_slippage.calculate("ES", -3.0, 4000.0, multiplier=50.0)
        assert cost == 37.50

    def test_cl_slippage(self, cl_slippage: FuturesSlippage):
        """CL with 0.02 slippage: 5 * 0.02 * 1000 = $100."""
        cost = cl_slippage.calculate("CL", 5.0, 80.0, multiplier=1000.0)
        assert cost == 100.0


class TestFuturesSlippageWithMultiplier:
    """Verify slippage uses multiplier correctly."""

    def test_multiplier_essential(self):
        """Slippage without multiplier would be wrong.

        ES tick = 0.25 points
        Without multiplier: 0.25 (meaningless)
        With multiplier 50: 0.25 * 50 = $12.50 per tick
        """
        slip = FuturesSlippage(slippage_points=0.25)

        # With correct multiplier
        correct = slip.calculate("ES", 1.0, 4000.0, multiplier=50.0)
        assert correct == 12.50

        # Without multiplier (default=1)
        wrong = slip.calculate("ES", 1.0, 4000.0)
        assert wrong == 0.25  # Meaningless without multiplier

    def test_different_multipliers(self):
        """Same slippage points, different multipliers."""
        slip = FuturesSlippage(slippage_points=1.0)

        es_cost = slip.calculate("ES", 1.0, 4000.0, multiplier=50.0)
        nq_cost = slip.calculate("NQ", 1.0, 15000.0, multiplier=20.0)
        cl_cost = slip.calculate("CL", 1.0, 80.0, multiplier=1000.0)

        assert es_cost == 50.0  # 1 * 1 * 50
        assert nq_cost == 20.0  # 1 * 1 * 20
        assert cl_cost == 1000.0  # 1 * 1 * 1000


class TestTotalTransactionCost:
    """Test combined commission + slippage costs."""

    def test_total_transaction_cost(self, ib_commission, es_slippage, es_contract):
        """Total cost = commission + slippage.

        2 ES:
        - Commission: 2 * $2.25 = $4.50
        - Slippage: 2 * 0.25 * 50 = $25.00
        - Total: $29.50
        """
        qty = 2.0
        price = 4000.0
        mult = es_contract.multiplier

        commission = ib_commission.calculate("ES", qty, price, multiplier=mult)
        slippage = es_slippage.calculate("ES", qty, price, multiplier=mult)

        assert commission == 4.50
        assert slippage == 25.0
        assert commission + slippage == 29.50

    def test_round_trip_total_cost(self, ib_commission, es_slippage, es_contract):
        """Round trip: 2x slippage, 2x commission.

        Entry + Exit for 2 ES:
        - Commission: 2 * $4.50 = $9.00
        - Slippage: 2 * $25.00 = $50.00
        - Total: $59.00
        """
        qty = 2.0
        price = 4000.0
        mult = es_contract.multiplier

        entry_comm = ib_commission.calculate("ES", qty, price, multiplier=mult)
        exit_comm = ib_commission.calculate("ES", -qty, price, multiplier=mult)
        entry_slip = es_slippage.calculate("ES", qty, price, multiplier=mult)
        exit_slip = es_slippage.calculate("ES", -qty, price, multiplier=mult)

        total = entry_comm + exit_comm + entry_slip + exit_slip
        assert total == 59.00


class TestZeroSlippage:
    """Test zero slippage behavior."""

    def test_zero_slippage_points(self):
        """Zero slippage returns zero cost."""
        slip = FuturesSlippage(slippage_points=0.0)
        cost = slip.calculate("ES", 10.0, 4000.0, multiplier=50.0)
        assert cost == 0.0
