"""Basic multi-asset backtest with equal-weight rebalancing.

This example demonstrates:
- Multi-asset portfolio with 5 stocks
- Monthly rebalancing to equal weights
- Using TargetWeightExecutor for weight-to-order conversion

No external optimizer dependencies required.
"""

import polars as pl

from ml4t.backtest import (
    DataFeed,
    Engine,
    ExecutionMode,
    RebalanceConfig,
    Strategy,
    TargetWeightExecutor,
)


def generate_sample_data(assets: list[str], n_bars: int = 252, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic OHLCV data for multiple assets."""
    from datetime import datetime, timedelta

    import numpy as np

    np.random.seed(seed)

    rows = []
    base_date = datetime(2020, 1, 1)

    for asset in assets:
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, n_bars)
        prices = 100 * np.exp(np.cumsum(returns))

        for i in range(n_bars):
            timestamp = base_date + timedelta(days=i)
            close = prices[i]
            # Simple OHLC approximation
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = (high + low) / 2

            rows.append(
                {
                    "timestamp": timestamp,
                    "asset": asset,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": np.random.randint(100000, 1000000),
                }
            )

    return pl.DataFrame(rows).sort(["timestamp", "asset"])


class EqualWeightStrategy(Strategy):
    """Simple equal-weight rebalancing strategy.

    Rebalances to equal weights every `rebalance_frequency` bars.
    """

    def __init__(self, assets: list[str], rebalance_frequency: int = 21):
        """Initialize strategy.

        Args:
            assets: List of asset symbols to trade.
            rebalance_frequency: Number of bars between rebalances (21 ≈ monthly).
        """
        self.assets = assets
        self.rebalance_frequency = rebalance_frequency
        self.bar_count = 0

        # Configure the rebalancer
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=100,  # Skip trades < $100
                min_weight_change=0.02,  # Skip if weight change < 2%
                allow_fractional=True,  # Allow fractional shares
            )
        )

    def on_data(self, timestamp, data, context, broker):
        """Called on each bar with market data."""
        self.bar_count += 1

        # Only rebalance every N bars
        if self.bar_count % self.rebalance_frequency != 0:
            return

        # Equal weight for all assets
        n_assets = len(self.assets)
        target_weights = dict.fromkeys(self.assets, 1.0 / n_assets)

        # Execute rebalancing
        orders = self.executor.execute(target_weights, data, broker)

        if orders:
            print(f"[{timestamp}] Rebalanced: {len(orders)} orders")


def main():
    """Run the multi-asset backtest."""
    # Define assets
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

    # Generate sample data (1 year)
    print("Generating sample data...")
    df = generate_sample_data(assets, n_bars=252)

    # Create data feed
    feed = DataFeed(prices_df=df)

    # Create strategy
    strategy = EqualWeightStrategy(assets, rebalance_frequency=21)

    # Run backtest
    print("Running backtest...")
    initial_cash = 100_000.0
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=initial_cash,
        execution_mode=ExecutionMode.NEXT_BAR,  # More realistic
    )
    result = engine.run()

    # Print results
    print("\n=== Backtest Results ===")
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"Final Value:     ${result['final_value']:,.2f}")
    print(f"Total Return:    {result['total_return']:.2%}")  # total_return is decimal
    print(f"Sharpe Ratio:    {result['sharpe']:.2f}")
    print(f"Max Drawdown:    {result['max_drawdown_pct']:.2%}")

    # Show final positions
    print("\n=== Final Positions ===")
    for asset, pos in engine.broker.positions.items():
        print(f"  {asset}: {pos.quantity:.2f} shares @ ${pos.entry_price:.2f}")


if __name__ == "__main__":
    main()
