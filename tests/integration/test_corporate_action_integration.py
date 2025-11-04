"""Integration test for corporate action processing in BacktestEngine."""

from datetime import date, datetime

import polars as pl
import pytest

from qengine.core.event import MarketEvent
from qengine.data.feed import CSVDataFeed
from qengine.engine import BacktestEngine
from qengine.execution.corporate_actions import CashDividend, StockSplit
from qengine.strategy.base import Strategy


class BuyAndHoldStrategy(Strategy):
    """Simple strategy that buys on first market event."""

    def __init__(self):
        super().__init__()
        self.has_bought = False

    def on_start(self, portfolio, event_bus):
        self.portfolio = portfolio
        self.event_bus = event_bus

    def on_event(self, event):
        if isinstance(event, MarketEvent) and not self.has_bought:
            # Buy 100 shares
            from qengine.core.event import OrderEvent
            from qengine.core.types import OrderSide, OrderType

            order_event = OrderEvent(
                timestamp=event.timestamp,
                order_id="BUY001",
                asset_id=event.asset_id,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=100.0,
            )

            self.event_bus.publish(order_event)
            self.has_bought = True


@pytest.fixture
def sample_market_data():
    """Create sample market data spanning multiple days."""
    data = pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 10, 9, 30),  # Before split
            datetime(2024, 1, 11, 9, 30),  # Day of split
            datetime(2024, 1, 15, 9, 30),  # After split
            datetime(2024, 1, 16, 9, 30),  # Day of dividend
            datetime(2024, 1, 17, 9, 30),  # After dividend
        ],
        "asset_id": ["AAPL"] * 5,
        "price": [150.0, 149.0, 75.0, 76.0, 77.0],  # Price adjusts after split
        "volume": [1000000] * 5,
    })
    return data


@pytest.fixture
def corporate_actions():
    """Create sample corporate actions."""
    return [
        StockSplit(
            action_id="SPLIT001",
            asset_id="AAPL",
            ex_date=date(2024, 1, 11),
            split_ratio=2.0,  # 2-for-1 split
        ),
        CashDividend(
            action_id="DIV001",
            asset_id="AAPL",
            ex_date=date(2024, 1, 16),
            dividend_per_share=0.50,  # $0.50 per share dividend
        ),
    ]


def test_stock_split_integration(tmp_path, sample_market_data, corporate_actions):
    """Test that stock splits are properly processed during backtest."""

    # Create temporary CSV file
    csv_file = tmp_path / "market_data.csv"
    sample_market_data.write_csv(csv_file)

    # Create data feed
    data_feed = CSVDataFeed(str(csv_file), asset_id="AAPL")

    # Create strategy
    strategy = BuyAndHoldStrategy()

    # Create engine with corporate actions
    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        initial_capital=50000.0,
        corporate_actions=corporate_actions,
    )

    # Run backtest
    results = engine.run()

    # Verify that corporate actions were processed
    assert engine.corporate_action_processor.processed_actions

    # Check that we have 2 processed actions (split and dividend)
    processed_actions = engine.corporate_action_processor.processed_actions
    assert len(processed_actions) == 2

    # Verify split was processed
    split_action = next((a for a in processed_actions if isinstance(a, StockSplit)), None)
    assert split_action is not None
    assert split_action.asset_id == "AAPL"
    assert split_action.split_ratio == 2.0

    # Verify dividend was processed
    dividend_action = next((a for a in processed_actions if isinstance(a, CashDividend)), None)
    assert dividend_action is not None
    assert dividend_action.asset_id == "AAPL"
    assert dividend_action.dividend_per_share == 0.50


def test_position_adjustment_after_split(tmp_path, sample_market_data, corporate_actions):
    """Test that positions are correctly adjusted after stock split."""

    # Create temporary CSV file
    csv_file = tmp_path / "market_data.csv"
    sample_market_data.write_csv(csv_file)

    # Create data feed
    data_feed = CSVDataFeed(str(csv_file), asset_id="AAPL")

    # Create strategy
    strategy = BuyAndHoldStrategy()

    # Create engine with corporate actions
    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        initial_capital=50000.0,
        corporate_actions=corporate_actions,
    )

    # Run backtest
    results = engine.run()

    # Check that positions exist
    final_positions = engine.portfolio.get_positions()
    assert not final_positions.is_empty()

    # Get AAPL position
    aapl_position = final_positions.filter(pl.col("asset_id") == "AAPL")
    assert len(aapl_position) == 1

    # After a 2-for-1 split, should have 200 shares instead of 100
    # (assuming the split adjustment worked correctly)
    position_qty = aapl_position["quantity"][0]
    assert position_qty == 200.0  # 100 shares * 2.0 split ratio


def test_cash_adjustment_after_dividend(tmp_path, sample_market_data, corporate_actions):
    """Test that cash is correctly adjusted after dividend payment."""

    # Create temporary CSV file
    csv_file = tmp_path / "market_data.csv"
    sample_market_data.write_csv(csv_file)

    # Create data feed
    data_feed = CSVDataFeed(str(csv_file), asset_id="AAPL")

    # Create strategy
    strategy = BuyAndHoldStrategy()

    # Create engine with corporate actions
    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        initial_capital=50000.0,
        corporate_actions=corporate_actions,
    )

    # Run backtest
    results = engine.run()

    # After buying 100 shares and receiving a 2-for-1 split, we have 200 shares
    # Dividend of $0.50 per share should add $100 to cash (200 * $0.50)
    final_cash = engine.portfolio.cash

    # Should have initial cash - cost of 100 shares + dividend payment
    # Cost ≈ 100 * $150 = $15,000 (plus commission)
    # Dividend = 200 * $0.50 = $100
    # Final cash should be significantly more than $35,000 due to dividend
    assert final_cash > 35000  # Basic sanity check
