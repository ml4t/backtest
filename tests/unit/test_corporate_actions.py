"""Tests for corporate actions handling."""

from datetime import date

import pytest

from qengine.execution.corporate_actions import (
    CashDividend,
    CorporateAction,
    CorporateActionDataProvider,
    CorporateActionProcessor,
    Merger,
    RightsOffering,
    SpinOff,
    StockDividend,
    StockSplit,
    SymbolChange,
)
from qengine.execution.order import Order, OrderSide, OrderType


@pytest.fixture
def sample_positions():
    """Sample position dictionary."""
    return {
        "AAPL": 1000.0,
        "GOOGL": 500.0,
        "TSLA": 200.0,
    }


@pytest.fixture
def sample_orders():
    """Sample orders list."""
    return [
        Order(
            order_id="ORDER1",
            asset_id="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        ),
        Order(
            order_id="ORDER2",
            asset_id="GOOGL",
            quantity=50,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=2800.0,
        ),
    ]


class TestCorporateAction:
    """Test base CorporateAction class."""

    def test_creation(self):
        """Test creating a basic corporate action."""
        action = CorporateAction(
            action_id="TEST001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            record_date=date(2024, 3, 13),
            payment_date=date(2024, 3, 28),
        )

        assert action.action_id == "TEST001"
        assert action.asset_id == "AAPL"
        assert action.ex_date == date(2024, 3, 15)
        assert action.record_date == date(2024, 3, 13)

    def test_invalid_dates(self):
        """Test validation of invalid dates."""
        with pytest.raises(ValueError):
            CorporateAction(
                action_id="TEST002",
                asset_id="AAPL",
                ex_date=date(2024, 3, 15),
                record_date=date(2024, 3, 16),  # After ex-date - should be before
            )


class TestCashDividend:
    """Test CashDividend corporate action."""

    def test_creation(self):
        """Test creating cash dividend."""
        dividend = CashDividend(
            action_id="DIV001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            dividend_per_share=0.25,
        )

        assert dividend.action_type == "DIVIDEND"
        assert dividend.dividend_per_share == 0.25
        assert dividend.currency == "USD"

    def test_with_currency(self):
        """Test dividend with non-USD currency."""
        dividend = CashDividend(
            action_id="DIV002",
            asset_id="ASML",
            ex_date=date(2024, 3, 15),
            dividend_per_share=1.50,
            currency="EUR",
        )

        assert dividend.currency == "EUR"


class TestStockSplit:
    """Test StockSplit corporate action."""

    def test_creation(self):
        """Test creating stock split."""
        split = StockSplit(
            action_id="SPLIT001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            split_ratio=2.0,
        )

        assert split.action_type == "SPLIT"
        assert split.split_ratio == 2.0

    def test_invalid_ratio(self):
        """Test invalid split ratio."""
        with pytest.raises(ValueError):
            StockSplit(
                action_id="SPLIT002",
                asset_id="AAPL",
                ex_date=date(2024, 3, 15),
                split_ratio=-1.0,
            )


class TestMerger:
    """Test Merger corporate action."""

    def test_cash_merger(self):
        """Test cash merger."""
        merger = Merger(
            action_id="MERGER001",
            asset_id="TARGET",
            target_asset_id="ACQUIRER",
            ex_date=date(2024, 3, 15),
            cash_consideration=50.0,
        )

        assert merger.action_type == "MERGER"
        assert merger.cash_consideration == 50.0
        assert merger.stock_consideration == 0.0

    def test_stock_merger(self):
        """Test stock-for-stock merger."""
        merger = Merger(
            action_id="MERGER002",
            asset_id="TARGET",
            target_asset_id="ACQUIRER",
            ex_date=date(2024, 3, 15),
            stock_consideration=0.5,
        )

        assert merger.stock_consideration == 0.5

    def test_mixed_merger(self):
        """Test mixed cash and stock merger."""
        merger = Merger(
            action_id="MERGER003",
            asset_id="TARGET",
            target_asset_id="ACQUIRER",
            ex_date=date(2024, 3, 15),
            cash_consideration=25.0,
            stock_consideration=0.25,
        )

        assert merger.cash_consideration == 25.0
        assert merger.stock_consideration == 0.25

    def test_invalid_merger(self):
        """Test merger with no consideration."""
        with pytest.raises(ValueError):
            Merger(
                action_id="MERGER004",
                asset_id="TARGET",
                target_asset_id="ACQUIRER",
                ex_date=date(2024, 3, 15),
            )


class TestCorporateActionProcessor:
    """Test CorporateActionProcessor."""

    def test_add_action(self):
        """Test adding corporate actions."""
        processor = CorporateActionProcessor()

        dividend = CashDividend(
            action_id="DIV001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            dividend_per_share=0.25,
        )

        processor.add_action(dividend)
        assert len(processor.pending_actions) == 1
        assert processor.pending_actions[0] == dividend

    def test_get_pending_actions(self):
        """Test getting pending actions by date."""
        processor = CorporateActionProcessor()

        # Add actions with different dates
        early_action = CashDividend(
            action_id="DIV001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 10),
            dividend_per_share=0.25,
        )

        later_action = CashDividend(
            action_id="DIV002",
            asset_id="AAPL",
            ex_date=date(2024, 3, 20),
            dividend_per_share=0.30,
        )

        processor.add_action(later_action)
        processor.add_action(early_action)

        # Should be sorted by ex-date
        assert processor.pending_actions[0] == early_action
        assert processor.pending_actions[1] == later_action

        # Get pending as of middle date
        pending = processor.get_pending_actions(date(2024, 3, 15))
        assert len(pending) == 1
        assert pending[0] == early_action

    def test_process_cash_dividend(self, sample_positions):
        """Test processing cash dividend."""
        processor = CorporateActionProcessor()

        dividend = CashDividend(
            action_id="DIV001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            dividend_per_share=0.25,
        )

        processor.add_action(dividend)

        initial_cash = 10000.0
        updated_positions, updated_orders, updated_cash, notifications = processor.process_actions(
            date(2024, 3, 15),
            sample_positions,
            [],
            initial_cash,
        )

        # Should receive dividend: 1000 shares × $0.25 = $250
        assert updated_cash == initial_cash + 250.0
        assert len(notifications) == 1
        assert "250.00" in notifications[0]
        assert len(processor.processed_actions) == 1

    def test_process_stock_split(self, sample_positions, sample_orders):
        """Test processing stock split."""
        processor = CorporateActionProcessor()

        split = StockSplit(
            action_id="SPLIT001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            split_ratio=2.0,
        )

        processor.add_action(split)

        updated_positions, updated_orders, updated_cash, notifications = processor.process_actions(
            date(2024, 3, 15),
            sample_positions,
            sample_orders,
            10000.0,
        )

        # Position should double
        assert updated_positions["AAPL"] == 2000.0

        # Orders should be adjusted
        aapl_order = next(o for o in updated_orders if o.asset_id == "AAPL")
        assert aapl_order.quantity == 200.0  # Was 100, now 200
        assert aapl_order.limit_price == 75.0  # Was 150, now 75

        assert len(notifications) == 1
        assert "2000" in notifications[0]

    def test_process_stock_dividend(self, sample_positions):
        """Test processing stock dividend."""
        processor = CorporateActionProcessor()

        stock_div = StockDividend(
            action_id="STOCKDIV001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            dividend_ratio=0.05,  # 5% stock dividend
        )

        processor.add_action(stock_div)

        updated_positions, _, _, notifications = processor.process_actions(
            date(2024, 3, 15),
            sample_positions,
            [],
            10000.0,
        )

        # Should receive 5% more shares: 1000 × 0.05 = 50
        assert updated_positions["AAPL"] == 1050.0
        assert "50" in notifications[0]

    def test_process_cash_merger(self, sample_positions):
        """Test processing cash merger."""
        processor = CorporateActionProcessor()

        merger = Merger(
            action_id="MERGER001",
            asset_id="AAPL",
            target_asset_id="MERGED_CO",
            ex_date=date(2024, 3, 15),
            cash_consideration=175.0,
        )

        processor.add_action(merger)

        initial_cash = 10000.0
        updated_positions, _, updated_cash, notifications = processor.process_actions(
            date(2024, 3, 15),
            sample_positions,
            [],
            initial_cash,
        )

        # AAPL position should be gone
        assert "AAPL" not in updated_positions

        # Should receive cash: 1000 shares × $175 = $175,000
        assert updated_cash == initial_cash + 175000.0

        assert "175000.00" in notifications[0]

    def test_process_stock_merger(self, sample_positions):
        """Test processing stock merger."""
        processor = CorporateActionProcessor()

        merger = Merger(
            action_id="MERGER002",
            asset_id="AAPL",
            target_asset_id="MERGED_CO",
            ex_date=date(2024, 3, 15),
            stock_consideration=0.5,
        )

        processor.add_action(merger)

        updated_positions, _, _, notifications = processor.process_actions(
            date(2024, 3, 15),
            sample_positions,
            [],
            10000.0,
        )

        # AAPL position should be gone
        assert "AAPL" not in updated_positions

        # Should receive stock: 1000 shares × 0.5 = 500 shares
        assert updated_positions["MERGED_CO"] == 500.0

        assert "500" in notifications[0]

    def test_process_spinoff(self, sample_positions):
        """Test processing spin-off."""
        processor = CorporateActionProcessor()

        spinoff = SpinOff(
            action_id="SPINOFF001",
            asset_id="AAPL",
            new_asset_id="AAPL_SPINOFF",
            ex_date=date(2024, 3, 15),
            distribution_ratio=0.1,
        )

        processor.add_action(spinoff)

        updated_positions, _, _, notifications = processor.process_actions(
            date(2024, 3, 15),
            sample_positions,
            [],
            10000.0,
        )

        # Original position should remain
        assert updated_positions["AAPL"] == 1000.0

        # Should receive spinoff shares: 1000 × 0.1 = 100
        assert updated_positions["AAPL_SPINOFF"] == 100.0

        assert "100" in notifications[0]

    def test_process_symbol_change(self, sample_positions, sample_orders):
        """Test processing symbol change."""
        processor = CorporateActionProcessor()

        symbol_change = SymbolChange(
            action_id="SYMBOL001",
            asset_id="AAPL",
            new_asset_id="AAPL_NEW",
            ex_date=date(2024, 3, 15),
        )

        processor.add_action(symbol_change)

        updated_positions, updated_orders, _, notifications = processor.process_actions(
            date(2024, 3, 15),
            sample_positions,
            sample_orders,
            10000.0,
        )

        # Position should move to new symbol
        assert "AAPL" not in updated_positions
        assert updated_positions["AAPL_NEW"] == 1000.0

        # Order should be updated
        aapl_order = next(o for o in updated_orders if o.asset_id == "AAPL_NEW")
        assert aapl_order.asset_id == "AAPL_NEW"

        assert "AAPL_NEW" in notifications[0]

    def test_adjust_price_for_actions(self):
        """Test price adjustment for corporate actions."""
        processor = CorporateActionProcessor()

        # Add a processed split
        split = StockSplit(
            action_id="SPLIT001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            split_ratio=2.0,
        )
        processor.processed_actions.append(split)

        # Price before split should be adjusted down
        historical_price = 200.0
        adjusted_price = processor.adjust_price_for_actions(
            "AAPL",
            historical_price,
            date(2024, 3, 10),
        )

        # Should be adjusted for the 2:1 split
        assert adjusted_price == 100.0

        # Price after split should not be adjusted
        adjusted_price_after = processor.adjust_price_for_actions(
            "AAPL",
            historical_price,
            date(2024, 3, 20),
        )

        assert adjusted_price_after == 200.0

    def test_adjust_price_for_dividend(self):
        """Test price adjustment for dividends."""
        processor = CorporateActionProcessor()

        # Add processed dividend
        dividend = CashDividend(
            action_id="DIV001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            dividend_per_share=2.0,
        )
        processor.processed_actions.append(dividend)

        # Price before dividend should be adjusted down
        historical_price = 150.0
        adjusted_price = processor.adjust_price_for_actions(
            "AAPL",
            historical_price,
            date(2024, 3, 10),
        )

        # Should be reduced by dividend amount
        assert adjusted_price == 148.0

    def test_get_processed_actions(self):
        """Test filtering processed actions."""
        processor = CorporateActionProcessor()

        # Add some processed actions
        action1 = CashDividend(
            action_id="DIV001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            dividend_per_share=0.25,
        )

        action2 = StockSplit(
            action_id="SPLIT001",
            asset_id="GOOGL",
            ex_date=date(2024, 3, 20),
            split_ratio=2.0,
        )

        processor.processed_actions.extend([action1, action2])

        # Filter by asset
        aapl_actions = processor.get_processed_actions(asset_id="AAPL")
        assert len(aapl_actions) == 1
        assert aapl_actions[0] == action1

        # Filter by date range
        march_actions = processor.get_processed_actions(
            start_date=date(2024, 3, 1),
            end_date=date(2024, 3, 31),
        )
        assert len(march_actions) == 2

    def test_no_position_scenarios(self):
        """Test scenarios where no position exists."""
        processor = CorporateActionProcessor()

        dividend = CashDividend(
            action_id="DIV001",
            asset_id="MSFT",  # Not in positions
            ex_date=date(2024, 3, 15),
            dividend_per_share=0.25,
        )

        processor.add_action(dividend)

        positions = {"AAPL": 1000.0}  # No MSFT position
        _, _, cash, notifications = processor.process_actions(
            date(2024, 3, 15),
            positions,
            [],
            10000.0,
        )

        # Cash should not change
        assert cash == 10000.0
        assert "No position" in notifications[0]


class TestCorporateActionDataProvider:
    """Test CorporateActionDataProvider."""

    def test_create_dividend_from_row(self):
        """Test creating dividend from data row."""
        provider = CorporateActionDataProvider()

        # Mock pandas Series
        class MockRow:
            def __init__(self, data):
                self.data = data

            def __getitem__(self, key):
                return self.data.get(key)

            def get(self, key, default=None):
                return self.data.get(key, default)

        row = MockRow(
            {
                "action_id": "DIV001",
                "asset_id": "AAPL",
                "action_type": "DIVIDEND",
                "ex_date": "2024-03-15",
                "dividend_per_share": 0.25,
                "record_date": None,
                "payment_date": None,
            },
        )

        action = provider._create_action_from_row(row)

        assert isinstance(action, CashDividend)
        assert action.action_id == "DIV001"
        assert action.dividend_per_share == 0.25

    def test_get_actions_for_asset(self):
        """Test getting actions for specific asset."""
        provider = CorporateActionDataProvider()

        action1 = CashDividend(
            action_id="DIV001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            dividend_per_share=0.25,
        )

        action2 = StockSplit(
            action_id="SPLIT001",
            asset_id="AAPL",
            ex_date=date(2024, 6, 15),
            split_ratio=2.0,
        )

        action3 = CashDividend(
            action_id="DIV002",
            asset_id="GOOGL",
            ex_date=date(2024, 3, 15),
            dividend_per_share=1.0,
        )

        provider.actions = {
            "DIV001": action1,
            "SPLIT001": action2,
            "DIV002": action3,
        }

        # Get all AAPL actions
        aapl_actions = provider.get_actions_for_asset("AAPL")
        assert len(aapl_actions) == 2
        assert aapl_actions[0] == action1  # Should be sorted by date
        assert aapl_actions[1] == action2

        # Get AAPL actions with date filter
        march_actions = provider.get_actions_for_asset(
            "AAPL",
            start_date=date(2024, 3, 1),
            end_date=date(2024, 3, 31),
        )
        assert len(march_actions) == 1
        assert march_actions[0] == action1


class TestRightsOffering:
    """Test RightsOffering corporate action."""

    def test_creation(self):
        """Test creating rights offering."""
        rights = RightsOffering(
            action_id="RIGHTS001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            subscription_price=100.0,
            rights_ratio=0.1,
            shares_per_right=1.0,
            expiration_date=date(2024, 4, 15),
        )

        assert rights.action_type == "RIGHTS_OFFERING"
        assert rights.subscription_price == 100.0
        assert rights.rights_ratio == 0.1
        assert rights.expiration_date == date(2024, 4, 15)


class TestComplexScenarios:
    """Test complex corporate action scenarios."""

    def test_multiple_actions_same_date(self, sample_positions):
        """Test multiple corporate actions on the same date."""
        processor = CorporateActionProcessor()

        # Dividend and split on same date
        dividend = CashDividend(
            action_id="DIV001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            dividend_per_share=0.25,
        )

        split = StockSplit(
            action_id="SPLIT001",
            asset_id="GOOGL",
            ex_date=date(2024, 3, 15),
            split_ratio=2.0,
        )

        processor.add_action(dividend)
        processor.add_action(split)

        initial_cash = 10000.0
        updated_positions, _, updated_cash, notifications = processor.process_actions(
            date(2024, 3, 15),
            sample_positions,
            [],
            initial_cash,
        )

        # Both actions should be processed
        assert updated_cash == initial_cash + 250.0  # AAPL dividend
        assert updated_positions["GOOGL"] == 1000.0  # GOOGL split
        assert len(notifications) == 2

    def test_sequential_actions(self, sample_positions):
        """Test sequential corporate actions affecting same asset."""
        processor = CorporateActionProcessor()

        # First a dividend, then a split
        dividend = CashDividend(
            action_id="DIV001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 10),
            dividend_per_share=0.50,
        )

        split = StockSplit(
            action_id="SPLIT001",
            asset_id="AAPL",
            ex_date=date(2024, 3, 20),
            split_ratio=2.0,
        )

        processor.add_action(dividend)
        processor.add_action(split)

        # Process both actions
        initial_cash = 10000.0

        # Process dividend first
        updated_positions, _, updated_cash, _ = processor.process_actions(
            date(2024, 3, 15),
            sample_positions,
            [],
            initial_cash,
        )

        assert updated_cash == initial_cash + 500.0  # 1000 shares × $0.50
        assert updated_positions["AAPL"] == 1000.0  # Position unchanged

        # Process split next
        final_positions, _, final_cash, _ = processor.process_actions(
            date(2024, 3, 25),
            updated_positions,
            [],
            updated_cash,
        )

        assert final_cash == updated_cash  # No change from split
        assert final_positions["AAPL"] == 2000.0  # Position doubled
