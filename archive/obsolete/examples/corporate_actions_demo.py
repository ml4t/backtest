"""Demonstration of corporate actions handling in ml4t.backtest.

This example shows how ml4t.backtest handles various corporate actions:
- Cash dividends
- Stock splits
- Mergers and acquisitions
- Spin-offs
- Symbol changes

Corporate actions are critical for realistic backtesting as they significantly
affect returns, positions, and order books.
"""

from datetime import date, datetime

import pandas as pd

from ml4t.backtest.execution.corporate_actions import (
    CashDividend,
    CorporateActionDataProvider,
    CorporateActionProcessor,
    Merger,
    SpinOff,
    StockDividend,
    StockSplit,
    SymbolChange,
)
from ml4t.backtest.execution.order import Order, OrderSide, OrderType


def demo_basic_corporate_actions():
    """Demonstrate basic corporate action processing."""
    print("=" * 80)
    print("BASIC CORPORATE ACTIONS DEMONSTRATION")
    print("=" * 80)

    # Create processor
    processor = CorporateActionProcessor()

    # Initial portfolio
    positions = {
        "AAPL": 1000.0,
        "TSLA": 500.0,
        "NVDA": 200.0,
    }
    cash = 50000.0

    print("\nInitial Portfolio:")
    print(f"Cash: ${cash:,.2f}")
    for asset, qty in positions.items():
        print(f"{asset}: {qty:,.0f} shares")

    # Add various corporate actions
    actions = [
        CashDividend(
            action_id="AAPL_DIV_2024Q1",
            asset_id="AAPL",
            ex_date=date(2024, 3, 15),
            dividend_per_share=0.25,
            record_date=date(2024, 3, 13),
            payment_date=date(2024, 3, 28),
        ),
        StockSplit(
            action_id="TSLA_SPLIT_2024",
            asset_id="TSLA",
            ex_date=date(2024, 4, 1),
            split_ratio=3.0,  # 3:1 split
        ),
        StockDividend(
            action_id="NVDA_STOCK_DIV_2024",
            asset_id="NVDA",
            ex_date=date(2024, 4, 15),
            dividend_ratio=0.05,  # 5% stock dividend
        ),
    ]

    for action in actions:
        processor.add_action(action)
        print(f"\nAdded {action.action_type}: {action.action_id}")

    # Process actions over time
    test_dates = [
        date(2024, 3, 20),  # After AAPL dividend
        date(2024, 4, 5),  # After TSLA split
        date(2024, 4, 20),  # After NVDA stock dividend
    ]

    current_positions = positions.copy()
    current_cash = cash

    for test_date in test_dates:
        print("\n" + "-" * 50)
        print(f"Processing actions as of {test_date}")
        print("-" * 50)

        updated_positions, _, updated_cash, notifications = processor.process_actions(
            test_date,
            current_positions,
            [],
            current_cash,
        )

        for notification in notifications:
            print(f"• {notification}")

        # Show portfolio changes
        if updated_positions != current_positions or updated_cash != current_cash:
            print("\nUpdated Portfolio:")
            print(f"Cash: ${updated_cash:,.2f} (was ${current_cash:,.2f})")
            for asset in set(current_positions.keys()) | set(updated_positions.keys()):
                old_qty = current_positions.get(asset, 0)
                new_qty = updated_positions.get(asset, 0)
                if old_qty != new_qty:
                    print(f"{asset}: {new_qty:,.0f} shares (was {old_qty:,.0f})")

        current_positions = updated_positions
        current_cash = updated_cash


def demo_merger_and_spinoff():
    """Demonstrate merger and spin-off scenarios."""
    print("\n\n" + "=" * 80)
    print("MERGER AND SPIN-OFF DEMONSTRATION")
    print("=" * 80)

    processor = CorporateActionProcessor()

    # Portfolio with companies involved in M&A activity
    positions = {
        "TARGET_CORP": 1000.0,
        "PARENT_CORP": 500.0,
        "OLDNAME_INC": 300.0,
    }
    cash = 75000.0

    print("\nInitial Portfolio:")
    print(f"Cash: ${cash:,.2f}")
    for asset, qty in positions.items():
        print(f"{asset}: {qty:,.0f} shares")

    # M&A scenarios
    merger_actions = [
        # Cash buyout
        Merger(
            action_id="TARGET_BUYOUT_2024",
            asset_id="TARGET_CORP",
            target_asset_id="ACQUIRER_INC",
            ex_date=date(2024, 5, 1),
            cash_consideration=85.0,  # $85 per share
        ),
        # Stock-for-stock merger
        Merger(
            action_id="PARENT_MERGER_2024",
            asset_id="PARENT_CORP",
            target_asset_id="BIG_CORP",
            ex_date=date(2024, 5, 15),
            stock_consideration=0.75,  # 0.75 shares of BIG_CORP per share
        ),
        # Spin-off
        SpinOff(
            action_id="PARENT_SPINOFF_2024",
            asset_id="BIG_CORP",  # Parent company
            new_asset_id="SPINOFF_CO",
            ex_date=date(2024, 6, 1),
            distribution_ratio=0.25,  # 0.25 spinoff shares per parent share
        ),
        # Symbol change
        SymbolChange(
            action_id="SYMBOL_CHANGE_2024",
            asset_id="OLDNAME_INC",
            new_asset_id="NEWNAME_LTD",
            ex_date=date(2024, 6, 10),
        ),
    ]

    for action in merger_actions:
        processor.add_action(action)
        print(f"\nAdded {action.action_type}: {action.action_id}")

    # Process M&A events
    test_dates = [
        date(2024, 5, 5),  # After cash buyout
        date(2024, 5, 20),  # After stock merger
        date(2024, 6, 5),  # After spin-off
        date(2024, 6, 15),  # After symbol change
    ]

    current_positions = positions.copy()
    current_cash = cash

    for test_date in test_dates:
        print("\n" + "-" * 50)
        print(f"Processing M&A actions as of {test_date}")
        print("-" * 50)

        updated_positions, _, updated_cash, notifications = processor.process_actions(
            test_date,
            current_positions,
            [],
            current_cash,
        )

        for notification in notifications:
            print(f"• {notification}")

        # Show detailed portfolio changes
        print("\nPortfolio Summary:")
        print(f"Cash: ${updated_cash:,.2f}")
        for asset, qty in updated_positions.items():
            if qty > 0:
                print(f"{asset}: {qty:,.0f} shares")

        current_positions = updated_positions
        current_cash = updated_cash


def demo_order_adjustments():
    """Demonstrate corporate action adjustments to open orders."""
    print("\n\n" + "=" * 80)
    print("ORDER ADJUSTMENTS DEMONSTRATION")
    print("=" * 80)

    processor = CorporateActionProcessor()

    # Create some open orders
    orders = [
        Order(
            order_id="AAPL_BUY_1",
            asset_id="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=180.0,
            created_time=datetime.now(),
        ),
        Order(
            order_id="AAPL_SELL_1",
            asset_id="AAPL",
            quantity=200,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=160.0,
            created_time=datetime.now(),
        ),
        Order(
            order_id="TSLA_BUY_1",
            asset_id="TSLA",
            quantity=50,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=250.0,
            created_time=datetime.now(),
        ),
    ]

    print("\nInitial Open Orders:")
    for order in orders:
        print(f"• {order.order_id}: {order.side.value} {order.quantity} {order.asset_id}")
        if order.limit_price:
            print(f"  Limit Price: ${order.limit_price:.2f}")
        if order.stop_price:
            print(f"  Stop Price: ${order.stop_price:.2f}")

    # Add stock split
    split = StockSplit(
        action_id="AAPL_SPLIT_4_1",
        asset_id="AAPL",
        ex_date=date(2024, 7, 1),
        split_ratio=4.0,  # 4:1 split
    )

    processor.add_action(split)
    print("\nAdded stock split: AAPL 4:1 split")

    # Process the split
    positions = {"AAPL": 1000.0, "TSLA": 500.0}

    updated_positions, updated_orders, _, notifications = processor.process_actions(
        date(2024, 7, 5),
        positions,
        orders,
        10000.0,
    )

    print("\nAfter Split Processing:")
    for notification in notifications:
        print(f"• {notification}")

    print("\nAdjusted Orders:")
    for order in updated_orders:
        if order.asset_id == "AAPL":
            print(f"• {order.order_id}: {order.side.value} {order.quantity} {order.asset_id}")
            if order.limit_price:
                print(f"  Limit Price: ${order.limit_price:.2f} (adjusted)")
            if order.stop_price:
                print(f"  Stop Price: ${order.stop_price:.2f} (adjusted)")
            if "corporate_action" in order.metadata:
                print(f"  Note: {order.metadata['corporate_action']}")
        else:
            print(
                f"• {order.order_id}: {order.side.value} {order.quantity} {order.asset_id} (unchanged)",
            )


def demo_price_adjustments():
    """Demonstrate historical price adjustments."""
    print("\n\n" + "=" * 80)
    print("HISTORICAL PRICE ADJUSTMENTS DEMONSTRATION")
    print("=" * 80)

    processor = CorporateActionProcessor()

    # Add some historical corporate actions
    historical_actions = [
        CashDividend(
            action_id="AAPL_DIV_Q4_2023",
            asset_id="AAPL",
            ex_date=date(2023, 11, 15),
            dividend_per_share=0.24,
        ),
        StockSplit(
            action_id="AAPL_SPLIT_2_1_2023",
            asset_id="AAPL",
            ex_date=date(2023, 8, 15),
            split_ratio=2.0,
        ),
        CashDividend(
            action_id="AAPL_DIV_Q2_2023",
            asset_id="AAPL",
            ex_date=date(2023, 5, 15),
            dividend_per_share=0.22,
        ),
    ]

    # Mark as processed (simulating they happened in the past)
    processor.processed_actions.extend(historical_actions)

    print("Historical Corporate Actions:")
    for action in historical_actions:
        print(f"• {action.ex_date}: {action.action_type} - {action.action_id}")

    # Test price adjustments for different historical dates
    test_prices = [
        (date(2023, 3, 1), 160.00),  # Before all actions
        (date(2023, 6, 1), 180.00),  # After first dividend
        (date(2023, 9, 1), 185.00),  # After split
        (date(2023, 12, 1), 190.00),  # After second dividend
    ]

    print("\nPrice Adjustments for Continuity:")
    print(f"{'Date':<12} {'Original':<10} {'Adjusted':<10} {'Actions Applied'}")
    print("-" * 60)

    for test_date, original_price in test_prices:
        adjusted_price = processor.adjust_price_for_actions("AAPL", original_price, test_date)

        # Determine which actions affected this price
        actions_applied = []
        for action in historical_actions:
            if action.ex_date > test_date:
                if isinstance(action, StockSplit):
                    actions_applied.append(f"Split {action.split_ratio}:1")
                elif isinstance(action, CashDividend):
                    actions_applied.append(f"Div ${action.dividend_per_share}")

        actions_str = ", ".join(actions_applied) if actions_applied else "None"
        print(f"{test_date} ${original_price:>8.2f} ${adjusted_price:>8.2f}  {actions_str}")

    print("\nNote: Adjusted prices maintain continuity across corporate actions")
    print("This is essential for accurate backtesting performance calculations")


def demo_data_loading():
    """Demonstrate loading corporate actions from data."""
    print("\n\n" + "=" * 80)
    print("CORPORATE ACTIONS DATA LOADING DEMONSTRATION")
    print("=" * 80)

    # Create sample data
    sample_data = [
        {
            "action_id": "AAPL_DIV_2024_Q1",
            "asset_id": "AAPL",
            "action_type": "DIVIDEND",
            "ex_date": "2024-03-15",
            "record_date": "2024-03-13",
            "payment_date": "2024-03-28",
            "dividend_per_share": 0.25,
            "split_ratio": None,
            "target_asset_id": None,
            "cash_consideration": None,
            "stock_consideration": None,
        },
        {
            "action_id": "TSLA_SPLIT_2024",
            "asset_id": "TSLA",
            "action_type": "SPLIT",
            "ex_date": "2024-04-01",
            "record_date": None,
            "payment_date": None,
            "dividend_per_share": None,
            "split_ratio": 3.0,
            "target_asset_id": None,
            "cash_consideration": None,
            "stock_consideration": None,
        },
        {
            "action_id": "NVDA_MERGER_2024",
            "asset_id": "NVDA",
            "action_type": "MERGER",
            "ex_date": "2024-05-01",
            "record_date": None,
            "payment_date": None,
            "dividend_per_share": None,
            "split_ratio": None,
            "target_asset_id": "AMD",
            "cash_consideration": 50.0,
            "stock_consideration": 0.5,
        },
    ]

    # Create temporary CSV for demonstration
    import os
    import tempfile

    df = pd.DataFrame(sample_data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        # Load data
        provider = CorporateActionDataProvider()
        provider.load_from_csv(temp_file)

        print(f"Loaded {len(provider.actions)} corporate actions from CSV:")

        for action_id, action in provider.actions.items():
            print(f"• {action_id}: {action.action_type} for {action.asset_id} on {action.ex_date}")

        # Get actions for specific assets
        print("\nActions for AAPL:")
        aapl_actions = provider.get_actions_for_asset("AAPL")
        for action in aapl_actions:
            print(f"  - {action.action_id} ({action.action_type})")

        print("\nActions for 2024 Q1:")
        q1_actions = provider.get_actions_for_asset(
            "AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
        )
        for action in q1_actions:
            print(f"  - {action.action_id} on {action.ex_date}")

    finally:
        # Cleanup
        os.unlink(temp_file)


def main():
    """Run all corporate actions demonstrations."""
    print("CORPORATE ACTIONS HANDLING IN ML4T.BACKTEST")
    print("Advanced Backtesting Framework with Realistic Corporate Action Processing")
    print()

    demo_basic_corporate_actions()
    demo_merger_and_spinoff()
    demo_order_adjustments()
    demo_price_adjustments()
    demo_data_loading()

    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("ml4t.backtest Corporate Actions Features:")
    print()
    print("📈 Supported Actions:")
    print("   • Cash Dividends - Automatic cash distribution")
    print("   • Stock Splits - Position and order quantity adjustments")
    print("   • Stock Dividends - Additional share distribution")
    print("   • Mergers & Acquisitions - Cash and/or stock consideration")
    print("   • Spin-offs - Distribution of new company shares")
    print("   • Symbol Changes - Asset identifier updates")
    print("   • Rights Offerings - Notification and tracking")
    print()
    print("🔧 Key Features:")
    print("   • Automatic position adjustments")
    print("   • Open order price/quantity adjustments")
    print("   • Historical price continuity adjustments")
    print("   • Point-in-time correctness")
    print("   • Multiple data source support")
    print("   • Comprehensive audit trail")
    print()
    print("💡 Benefits:")
    print("   • Realistic backtesting with proper corporate action handling")
    print("   • Accurate performance attribution")
    print("   • Proper risk management with adjusted positions")
    print("   • Professional-grade simulation quality")
    print()
    print("This comprehensive corporate actions system ensures your")
    print("backtests accurately reflect real-world trading conditions!")


if __name__ == "__main__":
    main()
