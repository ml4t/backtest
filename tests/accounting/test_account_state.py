"""Unit tests for AccountState with short position tracking."""

from datetime import datetime

import pytest

from src.ml4t.backtest.accounting import (
    AccountState,
    CashAccountPolicy,
    MarginAccountPolicy,
)


class TestAccountStateApplyFillLongPositions:
    """Tests for apply_fill with long positions."""

    def test_open_long_position(self):
        """Test opening a new long position decreases cash."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Buy 100 shares @ $150
        cash_change = account.apply_fill(
            asset="AAPL",
            quantity_delta=100.0,
            fill_price=150.0,
            timestamp=datetime.now(),
        )

        assert cash_change == -15_000.0  # Cash out
        assert account.cash == 85_000.0  # 100k - 15k
        assert "AAPL" in account.positions
        pos = account.positions["AAPL"]
        assert pos.quantity == 100.0
        assert pos.avg_entry_price == 150.0

    def test_add_to_long_position(self):
        """Test adding to existing long position updates cost basis."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Buy 100 @ $150
        account.apply_fill("AAPL", 100.0, 150.0, datetime.now())
        # Buy another 50 @ $160
        cash_change = account.apply_fill("AAPL", 50.0, 160.0, datetime.now())

        assert cash_change == -8_000.0  # 50 × $160
        assert account.cash == 77_000.0  # 100k - 15k - 8k
        pos = account.positions["AAPL"]
        assert pos.quantity == 150.0
        # Weighted average: (100×150 + 50×160) / 150 = 23,000 / 150 = 153.33
        assert abs(pos.avg_entry_price - 153.333) < 0.01

    def test_close_long_position(self):
        """Test closing long position increases cash."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Buy 100 @ $150
        account.apply_fill("AAPL", 100.0, 150.0, datetime.now())
        # Sell 100 @ $160
        cash_change = account.apply_fill("AAPL", -100.0, 160.0, datetime.now())

        assert cash_change == 16_000.0  # Cash in (profit)
        assert account.cash == 101_000.0  # 100k - 15k + 16k = +1k profit
        assert "AAPL" not in account.positions  # Position closed

    def test_partial_close_long(self):
        """Test partial close of long position."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Buy 100 @ $150
        account.apply_fill("AAPL", 100.0, 150.0, datetime.now())
        # Sell 60 @ $160
        cash_change = account.apply_fill("AAPL", -60.0, 160.0, datetime.now())

        assert cash_change == 9_600.0  # 60 × $160 cash in
        assert account.cash == 94_600.0  # 100k - 15k + 9.6k
        pos = account.positions["AAPL"]
        assert pos.quantity == 40.0  # 100 - 60
        assert pos.avg_entry_price == 150.0  # Unchanged for partial close


class TestAccountStateApplyFillShortPositions:
    """Tests for apply_fill with short positions (TASK-012 focus)."""

    def test_open_short_position_increases_cash(self):
        """Test opening short position increases cash (proceeds received).

        TASK-012 acceptance criteria:
        - Cash increases when opening shorts (proceeds received)
        """
        policy = MarginAccountPolicy()  # Only margin accounts allow shorts
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Short 100 shares @ $150 (sell shares we don't own)
        cash_change = account.apply_fill(
            asset="AAPL",
            quantity_delta=-100.0,  # Negative = short
            fill_price=150.0,
            timestamp=datetime.now(),
        )

        assert cash_change == 15_000.0  # Cash IN (proceeds from short sale)
        assert account.cash == 115_000.0  # 100k + 15k
        assert "AAPL" in account.positions
        pos = account.positions["AAPL"]
        assert pos.quantity == -100.0  # Negative quantity
        assert pos.avg_entry_price == 150.0
        assert pos.market_value == -15_000.0  # Negative market value (liability)

    def test_add_to_short_position_updates_cost_basis(self):
        """Test adding to existing short position.

        TASK-012 acceptance criteria:
        - Cost basis calculation correct for adding to shorts
        """
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Short 100 @ $150
        account.apply_fill("AAPL", -100.0, 150.0, datetime.now())
        # Short another 50 @ $160
        cash_change = account.apply_fill("AAPL", -50.0, 160.0, datetime.now())

        assert cash_change == 8_000.0  # 50 × $160 cash in
        assert account.cash == 123_000.0  # 100k + 15k + 8k
        pos = account.positions["AAPL"]
        assert pos.quantity == -150.0  # Total short position
        # Weighted average: (100×150 + 50×160) / 150 = 23,000 / 150 = 153.33
        assert abs(pos.avg_entry_price - 153.333) < 0.01

    def test_close_short_position_decreases_cash(self):
        """Test covering short position decreases cash.

        TASK-012 acceptance criteria:
        - Cash decreases when covering shorts (cost to close)
        """
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Short 100 @ $150
        account.apply_fill("AAPL", -100.0, 150.0, datetime.now())
        # Cover short @ $145 (buy back shares)
        cash_change = account.apply_fill("AAPL", 100.0, 145.0, datetime.now())

        assert cash_change == -14_500.0  # Cash OUT (paid to cover)
        assert account.cash == 100_500.0  # 100k + 15k - 14.5k = +500 profit
        assert "AAPL" not in account.positions  # Position closed

    def test_close_short_at_loss(self):
        """Test covering short at a loss."""
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Short 100 @ $150
        account.apply_fill("AAPL", -100.0, 150.0, datetime.now())
        # Cover short @ $160 (price went up, loss on short)
        cash_change = account.apply_fill("AAPL", 100.0, 160.0, datetime.now())

        assert cash_change == -16_000.0  # Cash OUT
        assert account.cash == 99_000.0  # 100k + 15k - 16k = -1k loss
        assert "AAPL" not in account.positions

    def test_partial_close_short(self):
        """Test partial cover of short position."""
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Short 100 @ $150
        account.apply_fill("AAPL", -100.0, 150.0, datetime.now())
        # Cover 40 shares @ $145
        cash_change = account.apply_fill("AAPL", 40.0, 145.0, datetime.now())

        assert cash_change == -5_800.0  # 40 × $145 cash out
        assert account.cash == 109_200.0  # 100k + 15k - 5.8k
        pos = account.positions["AAPL"]
        assert pos.quantity == -60.0  # Still short 60
        assert pos.avg_entry_price == 150.0  # Unchanged for partial close

    def test_short_position_market_value_negative(self):
        """Test that short position market value is negative (liability).

        TASK-012 acceptance criteria:
        - Market value calculation correct for shorts
        """
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Short 100 @ $150
        account.apply_fill("AAPL", -100.0, 150.0, datetime.now())
        pos = account.positions["AAPL"]

        # Market value should be negative (it's a liability)
        assert pos.market_value == -15_000.0  # quantity × current_price
        assert pos.unrealized_pnl() == 0.0  # No price change yet

        # Simulate price increase to $160 (bad for shorts)
        pos.current_price = 160.0
        assert pos.market_value == -16_000.0  # More liability
        assert pos.unrealized_pnl() == -1_000.0  # Loss (160-150)×(-100)

        # Simulate price decrease to $140 (good for shorts)
        pos.current_price = 140.0
        assert pos.market_value == -14_000.0  # Less liability
        assert pos.unrealized_pnl() == 1_000.0  # Profit (140-150)×(-100)


class TestAccountStateApplyFillPositionReversals:
    """Tests for position reversals (long → short, short → long)."""

    def test_reversal_long_to_short(self):
        """Test reversing from long to short position."""
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Open long 100 @ $150
        account.apply_fill("AAPL", 100.0, 150.0, datetime.now())
        assert account.cash == 85_000.0  # 100k - 15k

        # Sell 200 @ $160 (close long, open short 100)
        cash_change = account.apply_fill("AAPL", -200.0, 160.0, datetime.now())

        assert cash_change == 32_000.0  # 200 × $160 cash in
        assert account.cash == 117_000.0  # 85k + 32k
        pos = account.positions["AAPL"]
        assert pos.quantity == -100.0  # Now short 100
        assert pos.avg_entry_price == 160.0  # New entry price for short

    def test_reversal_short_to_long(self):
        """Test reversing from short to long position."""
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Open short 100 @ $150
        account.apply_fill("AAPL", -100.0, 150.0, datetime.now())
        assert account.cash == 115_000.0  # 100k + 15k

        # Buy 200 @ $145 (cover short, open long 100)
        cash_change = account.apply_fill("AAPL", 200.0, 145.0, datetime.now())

        assert cash_change == -29_000.0  # 200 × $145 cash out
        assert account.cash == 86_000.0  # 115k - 29k
        pos = account.positions["AAPL"]
        assert pos.quantity == 100.0  # Now long 100
        assert pos.avg_entry_price == 145.0  # New entry price for long


class TestAccountStateApplyFillEquityCalculation:
    """Tests for total equity calculation with shorts."""

    def test_equity_with_long_position(self):
        """Test equity calculation with long position."""
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Buy 100 @ $150
        account.apply_fill("AAPL", 100.0, 150.0, datetime.now())
        pos = account.positions["AAPL"]
        pos.current_price = 160.0  # Price goes up

        # NLV = cash + position market value
        # NLV = $85,000 + (100 × $160) = $85,000 + $16,000 = $101,000
        assert account.total_equity == 101_000.0

    def test_equity_with_short_position(self):
        """Test equity calculation with short position."""
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Short 100 @ $150
        account.apply_fill("AAPL", -100.0, 150.0, datetime.now())
        pos = account.positions["AAPL"]
        pos.current_price = 140.0  # Price goes down (good for short)

        # NLV = cash + position market value
        # NLV = $115,000 + (-100 × $140) = $115,000 - $14,000 = $101,000
        assert account.total_equity == 101_000.0

    def test_equity_with_short_at_loss(self):
        """Test equity with short position at a loss."""
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Short 100 @ $150
        account.apply_fill("AAPL", -100.0, 150.0, datetime.now())
        pos = account.positions["AAPL"]
        pos.current_price = 160.0  # Price goes up (bad for short)

        # NLV = cash + position market value
        # NLV = $115,000 + (-100 × $160) = $115,000 - $16,000 = $99,000
        assert account.total_equity == 99_000.0

    def test_equity_with_multiple_positions(self):
        """Test equity with both long and short positions."""
        policy = MarginAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Long AAPL 100 @ $150
        account.apply_fill("AAPL", 100.0, 150.0, datetime.now())
        # Short MSFT 50 @ $200
        account.apply_fill("MSFT", -50.0, 200.0, datetime.now())

        aapl_pos = account.positions["AAPL"]
        msft_pos = account.positions["MSFT"]
        aapl_pos.current_price = 160.0  # AAPL up 10
        msft_pos.current_price = 190.0  # MSFT down 10

        # Cash: 100k - 15k (AAPL buy) + 10k (MSFT short) = 95k
        assert account.cash == 95_000.0

        # AAPL market value: 100 × 160 = +16k
        # MSFT market value: -50 × 190 = -9.5k
        # NLV = 95k + 16k - 9.5k = 101.5k
        assert account.total_equity == 101_500.0


class TestAccountStateApplyFillEdgeCases:
    """Edge case tests for apply_fill."""

    def test_zero_quantity_delta_no_change(self):
        """Test that zero quantity delta does nothing."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        cash_change = account.apply_fill("AAPL", 0.0, 150.0, datetime.now())

        assert cash_change == 0.0
        assert account.cash == 100_000.0
        assert "AAPL" not in account.positions

    def test_fractional_shares(self):
        """Test fractional share positions (crypto, fractional stocks)."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Buy 0.5 BTC @ $50,000
        account.apply_fill("BTC", 0.5, 50_000.0, datetime.now())

        assert account.cash == 75_000.0  # 100k - 25k
        pos = account.positions["BTC"]
        assert pos.quantity == 0.5
        assert pos.avg_entry_price == 50_000.0

    def test_very_small_price(self):
        """Test with very small prices (penny stocks, crypto)."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100_000.0, policy=policy)

        # Buy 10,000 shares @ $0.01
        account.apply_fill("PENNY", 10_000.0, 0.01, datetime.now())

        assert account.cash == 99_900.0  # 100k - 100
        pos = account.positions["PENNY"]
        assert pos.quantity == 10_000.0
        assert pos.avg_entry_price == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
