"""Tests for FuturesCommission model.

Validates commission calculations against PySystemTrade's formula:
    total = max(per_trade, per_block * abs(qty), percentage * notional)
    where: notional = qty * price * multiplier

Reference: sysobjects/instruments.py in PySystemTrade
"""

from ml4t.backtest.models import FuturesCommission


class TestFuturesCommissionMax:
    """Test MAX formula for commission calculation."""

    def test_per_trade_dominates(self):
        """Fixed $5 per trade dominates for small orders.

        1 contract at $2.50/block: max(5, 2.5*1, 0) = 5
        """
        comm = FuturesCommission(
            per_trade=5.0,
            per_block=2.50,
            percentage=0.0,
        )
        cost = comm.calculate("ES", 1.0, 4000.0, multiplier=50.0)
        assert cost == 5.0

    def test_per_block_dominates(self):
        """Per-block cost dominates for larger orders.

        10 contracts at $2.50/block: max(5, 2.5*10, 0) = 25
        """
        comm = FuturesCommission(
            per_trade=5.0,
            per_block=2.50,
            percentage=0.0,
        )
        cost = comm.calculate("ES", 10.0, 4000.0, multiplier=50.0)
        assert cost == 25.0

    def test_percentage_dominates(self):
        """Percentage cost dominates for large notional.

        10 ES at 4000: notional = 10 * 4000 * 50 = $2,000,000
        1bp = $200 > max(5, 25)
        """
        comm = FuturesCommission(
            per_trade=5.0,
            per_block=2.50,
            percentage=0.0001,  # 1 basis point
        )
        cost = comm.calculate("ES", 10.0, 4000.0, multiplier=50.0)
        assert cost == 200.0

    def test_negative_quantity_uses_abs(self):
        """Short positions use absolute quantity.

        -10 contracts should cost same as +10 contracts.
        """
        comm = FuturesCommission(per_block=2.50)
        long_cost = comm.calculate("ES", 10.0, 4000.0, multiplier=50.0)
        short_cost = comm.calculate("ES", -10.0, 4000.0, multiplier=50.0)
        assert long_cost == short_cost == 25.0


class TestFuturesCommissionRoundTrip:
    """Test round-trip (entry + exit) commission."""

    def test_round_trip_per_block(self):
        """Round trip commission: entry + exit.

        5 contracts at $2.50/block: 12.50 + 12.50 = 25.00
        """
        comm = FuturesCommission(per_block=2.50)
        entry_cost = comm.calculate("ES", 5.0, 4000.0, multiplier=50.0)
        exit_cost = comm.calculate("ES", -5.0, 4000.0, multiplier=50.0)

        total = entry_cost + exit_cost
        assert total == 25.0

    def test_round_trip_percentage(self):
        """Round trip with percentage commission.

        Entry and exit at same price: 2 * notional * rate
        """
        comm = FuturesCommission(percentage=0.0001)  # 1bp
        entry_cost = comm.calculate("ES", 1.0, 4000.0, multiplier=50.0)
        exit_cost = comm.calculate("ES", -1.0, 4000.0, multiplier=50.0)

        # Notional per leg: 1 * 4000 * 50 = 200,000
        # Cost per leg: 200,000 * 0.0001 = 20
        assert entry_cost == 20.0
        assert exit_cost == 20.0
        assert entry_cost + exit_cost == 40.0


class TestFuturesCommissionNotionalWithMultiplier:
    """Verify notional calculation includes multiplier."""

    def test_notional_includes_multiplier(self):
        """Notional = qty * price * multiplier (not just qty * price).

        Without multiplier: 10 * 4000 = 40,000
        With multiplier:    10 * 4000 * 50 = 2,000,000

        1bp on $2M = $200 (correct)
        1bp on $40k = $4 (wrong)
        """
        comm = FuturesCommission(percentage=0.0001)

        # With multiplier = 50 (ES)
        cost = comm.calculate("ES", 10.0, 4000.0, multiplier=50.0)
        assert cost == 200.0  # 2,000,000 * 0.0001

    def test_multiplier_default_is_one(self):
        """Default multiplier is 1.0 for equity-like assets."""
        comm = FuturesCommission(percentage=0.0001)
        cost = comm.calculate("AAPL", 100.0, 150.0)  # No multiplier arg
        # Notional: 100 * 150 * 1 = 15,000
        # Cost: 15,000 * 0.0001 = 1.50
        assert cost == 1.50


class TestFuturesCommissionIBStyle:
    """Test Interactive Brokers style per-contract commission."""

    def test_ib_es_commission(self, ib_commission: FuturesCommission):
        """IB ES commission: $2.25 per contract each way.

        10 contracts = $22.50
        """
        cost = ib_commission.calculate("ES", 10.0, 4000.0, multiplier=50.0)
        assert cost == 22.50

    def test_ib_single_contract(self, ib_commission: FuturesCommission):
        """Single contract minimum."""
        cost = ib_commission.calculate("ES", 1.0, 4000.0, multiplier=50.0)
        assert cost == 2.25
