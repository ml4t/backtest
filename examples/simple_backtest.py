"""Simple backtest example demonstrating QEngine's basic functionality."""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl
from ml4t_backtest import BacktestEngine

from qengine.data.feed import ParquetDataFeed
from qengine.strategy.base import Strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BuyAndHoldStrategy(Strategy):
    """Simple buy-and-hold strategy for demonstration."""

    def __init__(self, asset_id: str = "SPY", position_size: float = 0.95):
        """Initialize strategy.

        Args:
            asset_id: Asset to trade
            position_size: Fraction of capital to invest
        """
        super().__init__()
        self.asset_id = asset_id
        self.position_size = position_size
        self.position_taken = False

    def on_event(self, event):
        """Handle events (uses fixed event routing)."""
        from qengine.core.event import EventType

        # Only handle market events for our asset
        if event.event_type != EventType.MARKET or event.asset_id != self.asset_id:
            return

        # Take position on first market event
        if not self.position_taken and hasattr(event, "close"):
            # Calculate position size
            portfolio_value = self.portfolio.get_total_value()
            cash_to_invest = portfolio_value * self.position_size
            shares = int(cash_to_invest / event.close)

            if shares > 0:
                # Create market order
                from qengine.core.event import OrderEvent
                from qengine.core.types import OrderSide, OrderType

                order = OrderEvent(
                    timestamp=event.timestamp,
                    order_id=f"ORDER_{event.timestamp}",
                    asset_id=self.asset_id,
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=shares,
                )

                self.event_bus.publish(order)
                self.position_taken = True
                logger.info(f"Submitted BUY order for {shares} shares at {event.close:.2f}")


def create_sample_data(output_path: Path = Path("sample_data.parquet")) -> Path:
    """Create sample market data for testing.

    Args:
        output_path: Path to save the sample data

    Returns:
        Path to the created file
    """
    import numpy as np

    # Generate synthetic price data
    np.random.seed(42)
    n_days = 252  # One trading year

    # Generate timestamps (daily bars)
    timestamps = pl.datetime_range(
        datetime(2023, 1, 1, 9, 30),
        datetime(2023, 12, 31, 16, 0),
        interval="1d",
        time_zone="America/New_York",
        eager=True,
    )[:n_days]

    # Generate price series with random walk
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = initial_price * np.exp(np.cumsum(returns))

    # Add OHLC variation
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    opens = prices * (1 + np.random.normal(0, 0.003, n_days))

    # Generate volume
    avg_volume = 1_000_000
    volumes = np.random.exponential(avg_volume, n_days).astype(int)

    # Create DataFrame
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        },
    )

    # Save to parquet
    df.write_parquet(str(output_path))
    logger.info(f"Created sample data with {len(df)} bars at {output_path}")

    return output_path


def main():
    """Run a simple backtest example."""

    # Create sample data
    data_path = create_sample_data()

    # Create data feed
    data_feed = ParquetDataFeed(path=data_path, asset_id="SPY", timestamp_column="timestamp")

    # Create strategy
    strategy = BuyAndHoldStrategy(asset_id="SPY", position_size=0.95)

    # Create and run backtest engine
    # Note: Execution delay is enabled by default to prevent lookahead bias
    engine = BacktestEngine(data_feed=data_feed, strategy=strategy, initial_capital=100000.0)

    logger.info("Starting backtest...")
    results = engine.run()

    # Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Events Processed: {results['events_processed']:,}")
    print(f"Duration: {results['duration_seconds']:.2f}s")
    print(f"Events/Second: {results['events_per_second']:.0f}")

    # Show metrics if available
    if results.get("metrics"):
        print("\nPerformance Metrics:")
        for key, value in results["metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Clean up
    data_path.unlink()  # Remove sample data file

    return results


if __name__ == "__main__":
    results = main()
