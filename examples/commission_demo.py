"""Demonstration of commission models in ml4t.backtest."""

from ml4t.backtest.execution.commission import (
    AssetClassCommission,
    FlatCommission,
    InteractiveBrokersCommission,
    MakerTakerCommission,
    NoCommission,
    PercentageCommission,
    PerShareCommission,
    TieredCommission,
)
from ml4t.backtest.execution.order import Order, OrderSide, OrderType


def main():
    """Demonstrate all commission models."""

    # Create sample orders
    market_order = Order(
        order_id="DEMO001",
        asset_id="AAPL",
        quantity=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    limit_order = Order(
        order_id="DEMO002",
        asset_id="AAPL",
        quantity=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=150.0,
    )

    # Test parameters
    fill_price = 150.0

    print("=" * 60)
    print("Commission Models Demonstration")
    print("=" * 60)
    print(f"Order: {market_order.side.value} {market_order.quantity} shares @ ${fill_price}")
    print(f"Notional Value: ${market_order.quantity * fill_price:,.2f}")
    print("-" * 60)

    # Test all commission models
    models = [
        ("No Commission", NoCommission()),
        ("Flat $1", FlatCommission(1.0)),
        ("Flat $5", FlatCommission(5.0)),
        ("0.1% (10bps)", PercentageCommission(0.001)),
        ("0.2% (20bps)", PercentageCommission(0.002)),
        ("$0.005/share", PerShareCommission(0.005)),
        ("$0.01/share", PerShareCommission(0.01)),
        ("Tiered (default)", TieredCommission()),
        ("Maker-Taker (Market)", MakerTakerCommission()),
        ("Asset Class (Equity)", AssetClassCommission()),
        ("IB Fixed", InteractiveBrokersCommission("fixed")),
        ("IB Tiered", InteractiveBrokersCommission("tiered")),
    ]

    print("\nCommission Comparison:")
    print(f"{'Model':<25} {'Commission':>12} {'Effective Rate':>15}")
    print("-" * 52)

    for name, model in models:
        # Special handling for maker-taker with limit orders
        if isinstance(model, MakerTakerCommission) and "Market" in name:
            commission = model.calculate(market_order, market_order.quantity, fill_price)
        elif isinstance(model, MakerTakerCommission):
            commission = model.calculate(limit_order, limit_order.quantity, fill_price)
        else:
            commission = model.calculate(market_order, market_order.quantity, fill_price)

        notional = market_order.quantity * fill_price
        effective_rate = (commission / notional) * 10000  # basis points

        # Format negative values (rebates) specially
        if commission < 0:
            print(f"{name:<25} ${commission:>11.2f} {effective_rate:>14.2f} bps (rebate)")
        else:
            print(f"{name:<25} ${commission:>11.2f} {effective_rate:>14.2f} bps")

    # Demonstrate maker-taker rebate
    print("\n" + "=" * 60)
    print("Maker-Taker Model Demonstration")
    print("=" * 60)

    maker_taker = MakerTakerCommission()
    market_comm = maker_taker.calculate(market_order, 100, 150.0)
    limit_comm = maker_taker.calculate(limit_order, 100, 150.0)

    print(f"Market Order (Taker): ${market_comm:.2f}")
    print(f"Limit Order (Maker):  ${limit_comm:.2f} (rebate)")

    # Demonstrate tiered commission
    print("\n" + "=" * 60)
    print("Tiered Commission Demonstration")
    print("=" * 60)

    tiered = TieredCommission()
    trade_sizes = [5_000, 20_000, 60_000, 200_000]

    print(f"{'Notional Value':>15} {'Commission':>12} {'Effective Rate':>15}")
    print("-" * 42)

    for notional in trade_sizes:
        quantity = notional / fill_price
        commission = tiered.calculate(market_order, quantity, fill_price)
        effective_rate = (commission / notional) * 10000
        print(f"${notional:>14,.0f} ${commission:>11.2f} {effective_rate:>14.2f} bps")

    # Demonstrate asset class differences
    print("\n" + "=" * 60)
    print("Asset Class Commission Demonstration")
    print("=" * 60)

    asset_model = AssetClassCommission()

    # Different asset classes
    asset_classes = [
        ("Equity", 100, 150.0, "equity"),
        ("Futures", 5, 4000.0, "futures"),  # 5 contracts
        ("Options", 500, 2.5, "options"),  # 5 contracts (100 shares each)
        ("Forex", 10000, 1.2, "forex"),  # 10k units
        ("Crypto", 0.5, 40000.0, "crypto"),  # 0.5 BTC
    ]

    print(f"{'Asset Class':<12} {'Quantity':>10} {'Price':>10} {'Notional':>12} {'Commission':>12}")
    print("-" * 66)

    for asset_class, quantity, price, class_type in asset_classes:
        order = Order(
            order_id=f"AC_{asset_class}",
            asset_id=asset_class,
            quantity=quantity,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        order.metadata["asset_class"] = class_type

        commission = asset_model.calculate(order, quantity, price)
        notional = quantity * price

        if class_type in ["futures", "options"]:
            # These charge per contract, not on notional
            print(
                f"{asset_class:<12} {quantity:>10.1f} ${price:>9.2f} ${notional:>11.2f} ${commission:>11.2f}",
            )
        else:
            print(
                f"{asset_class:<12} {quantity:>10.1f} ${price:>9.2f} ${notional:>11.2f} ${commission:>11.2f}",
            )

    # Interactive Brokers comparison
    print("\n" + "=" * 60)
    print("Interactive Brokers Tier Comparison")
    print("=" * 60)

    ib_fixed = InteractiveBrokersCommission("fixed")
    ib_tiered = InteractiveBrokersCommission("tiered")

    share_quantities = [100, 500, 1000, 5000]

    print(f"{'Shares':>8} {'Fixed Tier':>12} {'Tiered':>12} {'Difference':>12}")
    print("-" * 44)

    for shares in share_quantities:
        fixed_comm = ib_fixed.calculate(market_order, shares, 50.0)
        tiered_comm = ib_tiered.calculate(market_order, shares, 50.0)
        diff = fixed_comm - tiered_comm

        print(f"{shares:>8} ${fixed_comm:>11.2f} ${tiered_comm:>11.2f} ${diff:>11.2f}")

    print("\n" + "=" * 60)
    print("Commission models demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
