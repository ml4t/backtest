"""Demonstration of market impact models in QEngine."""

from datetime import datetime, timedelta

import numpy as np

from qengine.execution.market_impact import (
    AlmgrenChrissImpact,
    IntraDayMomentum,
    LinearMarketImpact,
    NoMarketImpact,
    ObizhaevWangImpact,
    PropagatorImpact,
)
from qengine.execution.order import Order, OrderSide, OrderType


def create_order(side: OrderSide, quantity: float, asset_id: str = "AAPL") -> Order:
    """Create a test order."""
    return Order(
        order_id=f"ORDER_{np.random.randint(1000, 9999)}",
        asset_id=asset_id,
        quantity=quantity,
        side=side,
        order_type=OrderType.MARKET,
        created_time=datetime.now(),
    )


def demo_basic_impact_models():
    """Demonstrate basic market impact models."""
    print("=" * 80)
    print("BASIC MARKET IMPACT MODELS")
    print("=" * 80)

    # Common parameters
    fill_price = 100.0
    timestamp = datetime.now()

    # Test orders of different sizes
    order_sizes = [100, 1000, 5000, 10000]

    models = {
        "No Impact": NoMarketImpact(),
        "Linear (Basic)": LinearMarketImpact(
            permanent_impact_factor=0.05,
            temporary_impact_factor=0.2,
            avg_daily_volume=100_000,
        ),
        "Almgren-Chriss": AlmgrenChrissImpact(
            permanent_impact_const=0.01,
            temporary_impact_const=0.1,
            daily_volatility=0.02,
            avg_daily_volume=100_000,
        ),
        "Obizhaeva-Wang": ObizhaevWangImpact(
            price_impact_const=0.1,
            information_share=0.3,
            book_depth=50_000,
        ),
    }

    print(f"{'Model':<18} {'Size':>8} {'Permanent':>12} {'Temporary':>12} {'Total':>12}")
    print("-" * 70)

    for size in order_sizes:
        order = create_order(OrderSide.BUY, size)

        for name, model in models.items():
            permanent, temporary = model.calculate_impact(
                order,
                size,
                fill_price,
                timestamp,
            )
            total = permanent + temporary

            print(f"{name:<18} {size:>8} ${permanent:>11.4f} ${temporary:>11.4f} ${total:>11.4f}")
        print()


def demo_square_root_scaling():
    """Demonstrate square root impact scaling."""
    print("=" * 80)
    print("SQUARE ROOT IMPACT SCALING (Almgren-Chriss)")
    print("=" * 80)

    model = AlmgrenChrissImpact(
        permanent_impact_const=0.02,
        daily_volatility=0.02,
        avg_daily_volume=100_000,
    )

    sizes = np.array([100, 400, 1600, 6400])  # Squares: 1x, 4x, 16x, 64x
    impacts = []

    print(
        f"{'Order Size':>12} {'Volume Fraction':>16} {'Permanent Impact':>18} {'Scaling Factor':>15}",
    )
    print("-" * 63)

    base_impact = None
    for size in sizes:
        order = create_order(OrderSide.BUY, size)
        permanent, _ = model.calculate_impact(order, size, 100.0, datetime.now())

        volume_fraction = size / 100_000
        scaling = permanent / base_impact if base_impact else 1.0

        if base_impact is None:
            base_impact = permanent

        print(f"{size:>12} {volume_fraction:>15.3%} ${permanent:>17.4f} {scaling:>14.2f}x")
        impacts.append(permanent)

    print("\nTheoretical square root scaling factors: 1.0x, 2.0x, 4.0x, 8.0x")
    actual_factors = [impacts[i] / impacts[0] for i in range(len(impacts))]
    print(f"Actual scaling factors: {', '.join(f'{f:.1f}x' for f in actual_factors)}")


def demo_propagator_impact():
    """Demonstrate propagator impact with history."""
    print("=" * 80)
    print("PROPAGATOR IMPACT MODEL (Historical Effects)")
    print("=" * 80)

    model = PropagatorImpact(
        impact_coefficient=0.1,
        propagator_exponent=0.5,
        decay_exponent=0.6,
        avg_daily_volume=100_000,
    )

    base_time = datetime.now()
    trades = []

    print("Executing sequence of trades and observing cumulative impact...")
    print(f"{'Trade #':>8} {'Side':>6} {'Size':>8} {'Instant Impact':>15} {'Total Impact':>15}")
    print("-" * 62)

    # Execute a series of trades
    trade_sequence = [
        (OrderSide.BUY, 1000),
        (OrderSide.BUY, 2000),
        (OrderSide.SELL, 500),
        (OrderSide.BUY, 1500),
        (OrderSide.SELL, 3000),
    ]

    for i, (side, size) in enumerate(trade_sequence):
        timestamp = base_time + timedelta(seconds=i * 30)
        order = create_order(side, size)

        permanent, temporary = model.calculate_impact(order, size, 100.0, timestamp)
        instant_impact = permanent + temporary

        model.update_market_state("AAPL", permanent, temporary, timestamp)
        total_impact = model.get_current_impact("AAPL", timestamp)

        side_str = "BUY" if side == OrderSide.BUY else "SELL"
        print(f"{i + 1:>8} {side_str:>6} {size:>8} ${instant_impact:>14.4f} ${total_impact:>14.4f}")

        trades.append(
            {
                "trade": i + 1,
                "side": side_str,
                "size": size,
                "instant": instant_impact,
                "total": total_impact,
            },
        )

    print(f"\nFinal accumulated impact: ${total_impact:.4f}")
    print("Note: Impact includes both permanent effects and decaying propagation")


def demo_momentum_impact():
    """Demonstrate intraday momentum impact."""
    print("=" * 80)
    print("INTRADAY MOMENTUM IMPACT MODEL")
    print("=" * 80)

    model = IntraDayMomentum(
        base_impact=0.05,
        momentum_factor=0.4,
        momentum_decay=0.1,
        avg_daily_volume=100_000,
    )

    print("Demonstrating momentum buildup and reversal effects...")
    print(f"{'Trade #':>8} {'Side':>6} {'Size':>8} {'Impact':>12} {'Momentum':>12}")
    print("-" * 58)

    # Build momentum with buy orders
    for i in range(3):
        order = create_order(OrderSide.BUY, 1000)
        permanent, temporary = model.calculate_impact(order, 1000, 100.0, datetime.now())
        total_impact = permanent + temporary
        momentum = model.momentum_states.get("AAPL", 0.0)

        print(f"{i + 1:>8} {'BUY':>6} {1000:>8} ${total_impact:>11.4f} {momentum:>11.4f}")

    print("\n... switching to sell orders (against momentum) ...")

    # Switch to sell orders (against momentum)
    for i in range(3):
        order = create_order(OrderSide.SELL, 1000)
        permanent, temporary = model.calculate_impact(order, 1000, 100.0, datetime.now())
        total_impact = permanent + temporary
        momentum = model.momentum_states.get("AAPL", 0.0)

        print(f"{i + 4:>8} {'SELL':>6} {1000:>8} ${total_impact:>11.4f} {momentum:>11.4f}")

    print("\nNotice how:")
    print("- Same-direction trades have increasing impact (momentum buildup)")
    print("- Opposite-direction trades have decreasing impact (momentum reduction)")


def demo_impact_decay():
    """Demonstrate impact decay over time."""
    print("=" * 80)
    print("MARKET IMPACT DECAY OVER TIME")
    print("=" * 80)

    model = LinearMarketImpact(
        permanent_impact_factor=0.1,
        temporary_impact_factor=0.5,
        decay_rate=0.1,  # 10% decay per second
    )

    # Create initial impact
    order = create_order(OrderSide.BUY, 2000)
    base_time = datetime.now()

    permanent, temporary = model.calculate_impact(order, 2000, 100.0, base_time)
    model.update_market_state("AAPL", permanent, temporary, base_time)

    print(f"Initial impact: Permanent=${permanent:.4f}, Temporary=${temporary:.4f}")
    print(f"Total=${permanent + temporary:.4f}")
    print()

    print(f"{'Time (seconds)':>15} {'Permanent':>12} {'Temporary':>12} {'Total':>12}")
    print("-" * 53)

    # Show decay over time
    for seconds in [0, 1, 2, 5, 10, 20]:
        future_time = base_time + timedelta(seconds=seconds)
        current_impact = model.get_current_impact("AAPL", future_time)

        # Calculate components (permanent stays, temporary decays)
        current_temporary = temporary * np.exp(-0.1 * seconds)

        print(
            f"{seconds:>15} ${permanent:>11.4f} ${current_temporary:>11.4f} ${current_impact:>11.4f}",
        )

    print("\nNote: Permanent impact persists, temporary impact decays exponentially")


def demo_volume_impact_comparison():
    """Compare impact scaling across different models."""
    print("=" * 80)
    print("VOLUME IMPACT SCALING COMPARISON")
    print("=" * 80)

    models = {
        "Linear": LinearMarketImpact(
            permanent_impact_factor=0.1,
            temporary_impact_factor=0.3,
        ),
        "Square Root": AlmgrenChrissImpact(
            permanent_impact_const=0.02,
            temporary_impact_const=0.1,
            daily_volatility=0.02,
        ),
        "Propagator": PropagatorImpact(
            impact_coefficient=0.15,
            propagator_exponent=0.5,
        ),
    }

    sizes = [100, 500, 1000, 2000, 5000]

    print(f"{'Size':>8} {'Linear':>12} {'Square Root':>15} {'Propagator':>15}")
    print("-" * 52)

    for size in sizes:
        order = create_order(OrderSide.BUY, size)
        impacts = {}

        for name, model in models.items():
            permanent, temporary = model.calculate_impact(
                order,
                size,
                100.0,
                datetime.now(),
            )
            impacts[name] = permanent + temporary

        print(
            f"{size:>8} ${impacts['Linear']:>11.4f} ${impacts['Square Root']:>14.4f} ${impacts['Propagator']:>14.4f}",
        )

    print("\nScaling characteristics:")
    print("- Linear: Impact ∝ volume")
    print("- Square Root: Impact ∝ √volume")
    print("- Propagator: Impact ∝ volume^0.5 + historical effects")


def main():
    """Run all market impact demonstrations."""
    print("MARKET IMPACT MODELS DEMONSTRATION")
    print("QEngine - Advanced Backtesting Framework")
    print()

    demo_basic_impact_models()
    print("\n")

    demo_square_root_scaling()
    print("\n")

    demo_propagator_impact()
    print("\n")

    demo_momentum_impact()
    print("\n")

    demo_impact_decay()
    print("\n")

    demo_volume_impact_comparison()
    print("\n")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Market impact models implemented:")
    print("1. NoMarketImpact - Zero impact for testing")
    print("2. LinearMarketImpact - Simple linear scaling")
    print("3. AlmgrenChrissImpact - Academic square-root model")
    print("4. PropagatorImpact - Bouchaud et al. propagation model")
    print("5. IntraDayMomentum - Momentum-based impact")
    print("6. ObizhaevWangImpact - Order book dynamics model")
    print()
    print("Key features:")
    print("- Separate permanent and temporary impact components")
    print("- Time-based decay of temporary impact")
    print("- Per-asset impact tracking")
    print("- Realistic scaling laws (linear, square-root, power-law)")
    print("- Historical order propagation effects")
    print("- Momentum and mean-reversion dynamics")
    print()
    print("Use these models to add realistic market microstructure")
    print("effects to your backtesting simulations!")


if __name__ == "__main__":
    main()
