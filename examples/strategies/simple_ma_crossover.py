"""Simple Moving Average Crossover Strategy (Simple Mode).

This example demonstrates:
1. Using on_market_event() callback for single-asset strategies
2. Reading indicators from MarketEvent.indicators dict
3. Using helper methods (buy_percent, close_position)
4. Context-dependent logic (VIX filtering)

Strategy Logic:
- Buy when fast MA crosses above slow MA
- Sell when fast MA crosses below slow MA
- Don't trade during high VIX periods (> 30)
- Position size: 20% of portfolio
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.data.feed import ParquetDataFeed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleMACrossoverStrategy(Strategy):
    """Moving average crossover strategy using Simple Mode (on_market_event).

    This strategy demonstrates the default execution mode where each market
    event is processed independently. Ideal for single-asset strategies.
    """

    def __init__(
        self,
        asset_id: str = "SPY",
        position_pct: float = 0.20,
        max_vix: float = 30.0,
    ):
        """Initialize MA crossover strategy.

        Args:
            asset_id: Asset to trade
            position_pct: Position size as % of portfolio (0.20 = 20%)
            max_vix: Maximum VIX level to allow trading
        """
        super().__init__()
        self.asset_id = asset_id
        self.position_pct = position_pct
        self.max_vix = max_vix

        # Track MA state for crossover detection
        self._prev_fast_ma = None
        self._prev_slow_ma = None

    def on_event(self, event):
        """Required abstract method - delegates to on_market_event."""
        pass

    def on_market_event(self, event, context=None):
        """Process market events with indicators (Simple Mode).

        Args:
            event: MarketEvent with OHLCV data and indicators dict
            context: Optional dict with market-wide indicators (VIX, regime, etc.)
        """
        # Only process our target asset
        if event.asset_id != self.asset_id:
            return

        # Extract indicators from event.signals dict
        # (They'll be in event.indicators once indicator_columns param is added)
        fast_ma = event.signals.get("sma_10", None)
        slow_ma = event.signals.get("sma_50", None)

        # Skip if indicators not available yet (warming up)
        if fast_ma is None or slow_ma is None:
            return

        # Extract context if available
        vix = 0.0
        if context:
            vix = context.get("VIX", 0.0)

        # Risk check: VIX too high?
        if vix > self.max_vix:
            logger.debug(
                f"{event.timestamp}: VIX too high ({vix:.1f} > {self.max_vix}), "
                "skipping trade"
            )
            # Close position if we have one during high volatility
            position = self.get_position(self.asset_id)
            if position != 0:
                logger.info(f"{event.timestamp}: Closing position due to high VIX")
                self.close_position(self.asset_id)
            return

        # Get current position
        position = self.get_position(self.asset_id)

        # Detect crossover if we have previous values
        if self._prev_fast_ma is not None and self._prev_slow_ma is not None:
            # Golden cross: fast MA crosses above slow MA (bullish)
            if self._prev_fast_ma <= self._prev_slow_ma and fast_ma > slow_ma:
                if position == 0:
                    logger.info(
                        f"{event.timestamp}: Golden cross detected - "
                        f"fast_ma={fast_ma:.2f} > slow_ma={slow_ma:.2f}, "
                        f"VIX={vix:.1f}"
                    )
                    self.buy_percent(self.asset_id, self.position_pct, event.close)

            # Death cross: fast MA crosses below slow MA (bearish)
            elif self._prev_fast_ma >= self._prev_slow_ma and fast_ma < slow_ma:
                if position != 0:
                    logger.info(
                        f"{event.timestamp}: Death cross detected - "
                        f"fast_ma={fast_ma:.2f} < slow_ma={slow_ma:.2f}"
                    )
                    self.close_position(self.asset_id)

        # Update state for next event
        self._prev_fast_ma = fast_ma
        self._prev_slow_ma = slow_ma


def create_sample_data_with_indicators(
    output_path: Path = Path("ma_sample_data.parquet"),
) -> tuple[Path, dict]:
    """Create sample market data with MA indicators and VIX context.

    Args:
        output_path: Path to save the sample data

    Returns:
        Tuple of (data_path, context_dict)
    """
    np.random.seed(42)
    n_days = 252  # One trading year

    # Generate timestamps (daily bars)
    timestamps = pl.datetime_range(
        datetime(2023, 1, 3, 9, 30),
        datetime(2023, 12, 31, 16, 0),
        interval="1d",
        time_zone="America/New_York",
        eager=True,
    )[:n_days]

    # Generate price series with trend
    initial_price = 400.0
    prices = [initial_price]

    for i in range(1, n_days):
        # Upward trend with noise
        drift = 0.0005
        volatility = 0.012
        daily_return = np.random.normal(drift, volatility)
        prices.append(prices[-1] * (1 + daily_return))

    prices = np.array(prices)

    # Generate OHLC from close prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.003, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.003, n_days)))
    opens = prices * (1 + np.random.normal(0, 0.002, n_days))
    volumes = np.random.exponential(10_000_000, n_days).astype(int)

    # Calculate moving averages
    sma_10 = []
    sma_50 = []

    for i in range(n_days):
        # SMA-10: Need at least 10 days
        if i >= 9:
            sma_10.append(np.mean(prices[i - 9 : i + 1]))
        else:
            sma_10.append(None)

        # SMA-50: Need at least 50 days
        if i >= 49:
            sma_50.append(np.mean(prices[i - 49 : i + 1]))
        else:
            sma_50.append(None)

    # Generate VIX context (lower values, some spikes)
    vix_values = []
    for i in range(n_days):
        base_vix = 16.0
        # Add occasional spikes
        if i % 50 == 0:
            base_vix += np.random.uniform(10, 20)
        vix = base_vix + np.random.normal(0, 2)
        vix = max(10.0, min(50.0, vix))
        vix_values.append(vix)

    # Create DataFrame with indicators
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
            "sma_10": sma_10,
            "sma_50": sma_50,
        },
    )

    # Create context data
    context_data = {}
    for i, ts in enumerate(timestamps):
        py_ts = ts.to_py() if hasattr(ts, "to_py") else ts
        context_data[py_ts] = {
            "VIX": vix_values[i],
        }

    # Save to parquet
    df.write_parquet(str(output_path))
    logger.info(f"Created sample data with {len(df)} bars at {output_path}")
    logger.info(f"Created context data with {len(context_data)} timestamps")

    return output_path, context_data


def main():
    """Run simple MA crossover strategy example."""

    # Create sample data with indicators
    logger.info("Creating sample data with MA indicators...")
    data_path, context_data = create_sample_data_with_indicators()

    # Create data feed with indicators as signals (for now)
    # TODO: Once indicator_columns parameter is added, change this
    data_feed = ParquetDataFeed(
        path=data_path,
        asset_id="SPY",
        timestamp_column="timestamp",
        signal_columns=["sma_10", "sma_50"],  # Extract indicators
    )

    # Create MA crossover strategy
    strategy = SimpleMACrossoverStrategy(
        asset_id="SPY",
        position_pct=0.20,  # 20% position size
        max_vix=30.0,
    )

    # Verify strategy is in simple mode
    logger.info(f"Strategy execution mode: {strategy.execution_mode}")
    assert strategy.execution_mode == "simple", "Should be in simple mode"

    # Create and run backtest engine
    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        context_data=context_data,
        initial_capital=100000.0,
    )

    logger.info("Starting MA crossover strategy backtest...")
    logger.info("=" * 60)
    results = engine.run()

    # Display results
    print("\n" + "=" * 60)
    print("SIMPLE MA CROSSOVER STRATEGY RESULTS")
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

    # Show strategy-specific info
    print("\nStrategy Configuration:")
    print(f"  Asset: {strategy.asset_id}")
    print(f"  Position Size: {strategy.position_pct:.0%}")
    print(f"  Max VIX: {strategy.max_vix}")
    print(f"  Execution Mode: {strategy.execution_mode}")

    # Clean up
    data_path.unlink()

    return results


if __name__ == "__main__":
    results = main()
