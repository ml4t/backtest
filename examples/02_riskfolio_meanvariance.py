"""Mean-Variance Optimization with Riskfolio-lib.

This example demonstrates:
- Integration with riskfolio-lib for portfolio optimization
- Rolling window mean-variance optimization
- Maximum Sharpe ratio portfolio
- Quarterly rebalancing

Requirements:
    pip install riskfolio-lib
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

# Check if riskfolio is available
try:
    import riskfolio as rp

    RISKFOLIO_AVAILABLE = True
except ImportError:
    RISKFOLIO_AVAILABLE = False
    print("Warning: riskfolio-lib not installed. Run: pip install riskfolio-lib")


def generate_sample_data(assets: list[str], n_bars: int = 504, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic OHLCV data for multiple assets (2 years)."""
    from datetime import datetime, timedelta

    import numpy as np

    np.random.seed(seed)

    rows = []
    base_date = datetime(2020, 1, 1)

    # Different characteristics per asset
    asset_params = {
        "AAPL": (0.0008, 0.018),  # Higher return, lower vol
        "GOOGL": (0.0006, 0.022),
        "MSFT": (0.0007, 0.016),
        "AMZN": (0.0005, 0.025),  # Higher vol
        "META": (0.0004, 0.028),  # Highest vol
        "JPM": (0.0005, 0.020),
        "V": (0.0006, 0.015),  # Low vol
        "JNJ": (0.0003, 0.012),  # Defensive
    }

    for asset in assets:
        mu, sigma = asset_params.get(asset, (0.0005, 0.02))
        returns = np.random.normal(mu, sigma, n_bars)
        prices = 100 * np.exp(np.cumsum(returns))

        for i in range(n_bars):
            timestamp = base_date + timedelta(days=i)
            close = prices[i]
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


class MeanVarianceStrategy(Strategy):
    """Mean-Variance optimization using riskfolio-lib.

    Uses rolling window to estimate expected returns and covariance,
    then optimizes for maximum Sharpe ratio.
    """

    def __init__(
        self,
        assets: list[str],
        lookback: int = 126,  # ~6 months
        rebalance_frequency: int = 63,  # ~quarterly
    ):
        """Initialize strategy.

        Args:
            assets: List of asset symbols.
            lookback: Rolling window for return estimation.
            rebalance_frequency: Bars between rebalances.
        """
        self.assets = assets
        self.lookback = lookback
        self.rebalance_frequency = rebalance_frequency
        self.bar_count = 0

        # Price history buffer
        self.prices: dict[str, list[float]] = {asset: [] for asset in assets}

        # Rebalancer configuration
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=500,
                min_weight_change=0.03,  # 3% threshold
                allow_fractional=True,
                allow_short=False,  # Long-only
                max_single_weight=0.40,  # Max 40% per asset
            )
        )

    def on_data(self, timestamp, data, context, broker):
        """Called on each bar."""
        # Update price history
        for asset in self.assets:
            if asset in data and "close" in data[asset]:
                self.prices[asset].append(data[asset]["close"])

        self.bar_count += 1

        # Wait for enough history
        min_history = min(len(p) for p in self.prices.values())
        if min_history < self.lookback:
            return

        # Only rebalance every N bars
        if self.bar_count % self.rebalance_frequency != 0:
            return

        # Compute optimal weights
        target_weights = self._optimize_portfolio()

        if target_weights:
            orders = self.executor.execute(target_weights, data, broker)
            if orders:
                print(f"[{timestamp}] Rebalanced with {len(orders)} orders")
                for asset, weight in sorted(target_weights.items()):
                    print(f"  {asset}: {weight:.1%}")

    def _optimize_portfolio(self) -> dict[str, float]:
        """Compute optimal weights using riskfolio-lib."""
        import pandas as pd

        if not RISKFOLIO_AVAILABLE:
            # Fallback to equal weight
            return {asset: 1.0 / len(self.assets) for asset in self.assets}

        # Build returns DataFrame
        price_df = pd.DataFrame(
            {asset: prices[-self.lookback :] for asset, prices in self.prices.items()}
        )
        returns_df = price_df.pct_change().dropna()

        # Create portfolio object
        port = rp.Portfolio(returns=returns_df)

        # Estimate expected returns and covariance
        port.assets_stats(method_mu="hist", method_cov="hist")

        # Optimize for maximum Sharpe ratio
        try:
            weights = port.optimization(
                model="Classic",  # Mean-Variance
                rm="MV",  # Variance as risk measure
                obj="Sharpe",  # Maximize Sharpe ratio
                rf=0.02 / 252,  # Daily risk-free rate
                l=0,  # No regularization
                hist=True,
            )

            if weights is None or weights.empty:
                return {}

            # Convert to dict
            return weights["weights"].to_dict()

        except Exception as e:
            print(f"Optimization failed: {e}")
            return {}


def main():
    """Run mean-variance optimization backtest."""
    if not RISKFOLIO_AVAILABLE:
        print("This example requires riskfolio-lib. Install with:")
        print("  pip install riskfolio-lib")
        print("\nRunning with equal weights as fallback...")

    # Define universe
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "JPM", "V", "JNJ"]

    # Generate 2 years of data
    print("Generating sample data...")
    df = generate_sample_data(assets, n_bars=504)

    # Create components
    feed = DataFeed(prices_df=df)
    strategy = MeanVarianceStrategy(
        assets,
        lookback=126,  # 6 months
        rebalance_frequency=63,  # Quarterly
    )

    initial_cash = 100_000.0
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=initial_cash,
        execution_mode=ExecutionMode.NEXT_BAR,
    )

    # Run backtest
    print("Running backtest...")
    result = engine.run()

    # Results
    print("\n=== Backtest Results ===")
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"Final Value:     ${result['final_value']:,.2f}")
    print(f"Total Return:    {result['total_return']:.2%}")
    print(f"Sharpe Ratio:    {result['sharpe']:.2f}")

    print("\n=== Final Positions ===")
    for asset, pos in sorted(engine.broker.positions.items()):
        value = pos.quantity * pos.entry_price
        print(f"  {asset}: {pos.quantity:.2f} shares (${value:,.0f})")


if __name__ == "__main__":
    main()
