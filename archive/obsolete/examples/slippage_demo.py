"""Demonstration of slippage models in ml4t.backtest."""

import sys

sys.path.insert(0, "~/ml4t/ml4t.backtest/src")
sys.path.insert(0, "~/ml4t/qfeatures/src")
sys.path.insert(0, "~/ml4t/qeval/src")

from datetime import datetime

import polars as pl

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType, OrderSide, OrderType
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.execution.order import Order
from ml4t.backtest.execution.slippage import (
    AssetClassSlippage,
    FixedSlippage,
    LinearImpactSlippage,
    NoSlippage,
    PercentageSlippage,
    SquareRootImpactSlippage,
    VolumeShareSlippage,
)


def demonstrate_slippage_models():
    """Demonstrate different slippage models."""

    # Market price for all examples
    market_price = 100.0
    order_quantity = 1000

    print("=" * 60)
    print("SLIPPAGE MODEL COMPARISON")
    print("=" * 60)
    print(f"Market Price: ${market_price:.2f}")
    print(f"Order Size: {order_quantity} shares")
    print(f"Notional Value: ${market_price * order_quantity:,.2f}\n")

    # Test each model
    models = [
        ("No Slippage", NoSlippage()),
        ("Fixed Spread (1¢)", FixedSlippage(spread=0.01)),
        ("Fixed Spread (5¢)", FixedSlippage(spread=0.05)),
        ("Percentage (0.1%)", PercentageSlippage(slippage_pct=0.001)),
        ("Percentage (0.5%)", PercentageSlippage(slippage_pct=0.005)),
        (
            "Linear Impact (Small)",
            LinearImpactSlippage(base_slippage=0.0001, impact_coefficient=0.000001),
        ),
        (
            "Linear Impact (Large)",
            LinearImpactSlippage(base_slippage=0.0001, impact_coefficient=0.00001),
        ),
        (
            "Square Root Impact",
            SquareRootImpactSlippage(temporary_impact=0.1, permanent_impact=0.05),
        ),
        ("Asset Class (Equity)", AssetClassSlippage()),
        ("Asset Class (Crypto)", AssetClassSlippage()),
    ]

    results = []

    for model_name, model in models:
        # Create broker with slippage model
        broker = SimulationBroker(
            initial_cash=1000000.0,
            slippage_model=model,
        )

        # Create buy order
        order = Order(
            asset_id="TEST",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=order_quantity,
        )

        # Special handling for asset class model
        if "Crypto" in model_name:
            order.metadata["asset_class"] = "crypto"
        elif "Equity" in model_name:
            order.metadata["asset_class"] = "equity"

        # Submit order
        broker.submit_order(order)

        # Send market event
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            asset_id="TEST",
            data_type=MarketDataType.BAR,
            close=market_price,
        )

        fills = broker.on_market_event(market_event)

        if fills:
            fill = fills[0]
            fill_price = fill.fill_price
            slippage_cost = fill.slippage
            slippage_pct = ((fill_price - market_price) / market_price) * 100

            results.append(
                {
                    "Model": model_name,
                    "Fill Price": fill_price,
                    "Slippage $": slippage_cost,
                    "Slippage %": slippage_pct,
                    "Total Cost": fill_price * order_quantity,
                },
            )

    # Display results
    df = pl.DataFrame(results)
    print(df.to_pandas().to_string(index=False))

    print("\n" + "=" * 60)
    print("VOLUME-BASED SLIPPAGE DEMONSTRATION")
    print("=" * 60)

    # Demonstrate volume-based slippage
    volume_model = VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)
    volume_model.set_daily_volume(100000)  # 100k daily volume

    order_sizes = [100, 1000, 2500, 5000, 10000]

    print("Daily Volume: 100,000 shares")
    print("Volume Limit: 2.5% per order")
    print("Price Impact Coefficient: 0.1\n")

    volume_results = []

    for size in order_sizes:
        broker = SimulationBroker(
            initial_cash=1000000.0,
            slippage_model=volume_model,
        )

        order = Order(
            asset_id="TEST",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=size,
        )

        broker.submit_order(order)

        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            asset_id="TEST",
            data_type=MarketDataType.BAR,
            close=market_price,
        )

        fills = broker.on_market_event(market_event)

        if fills:
            fill = fills[0]
            volume_pct = (size / 100000) * 100

            volume_results.append(
                {
                    "Order Size": size,
                    "% of Volume": f"{volume_pct:.1f}%",
                    "Fill Price": f"${fill.fill_price:.4f}",
                    "Slippage": f"${fill.slippage:.2f}",
                    "Impact bps": f"{((fill.fill_price - market_price) / market_price) * 10000:.1f}",
                },
            )

    volume_df = pl.DataFrame(volume_results)
    print(volume_df.to_pandas().to_string(index=False))

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("1. NoSlippage: Ideal for testing strategies without market friction")
    print("2. FixedSlippage: Simple spread-based model, good for liquid markets")
    print("3. PercentageSlippage: Scales with price, suitable for diverse price ranges")
    print("4. LinearImpactSlippage: Order size matters linearly")
    print("5. SquareRootImpactSlippage: Realistic for large orders (Almgren-Chriss)")
    print("6. VolumeShareSlippage: Most realistic, considers market liquidity")
    print("7. AssetClassSlippage: Different costs for different asset types")


if __name__ == "__main__":
    demonstrate_slippage_models()
