"""Multi-Asset Momentum Strategy (Batch Mode).

This example demonstrates:
1. Using on_timestamp_batch() callback for multi-asset strategies
2. Processing multiple events simultaneously for cross-asset decisions
3. Ranking assets and portfolio rebalancing
4. Context-dependent logic across multiple assets

Strategy Logic:
- Rank assets by momentum score each period
- Go long top 5 assets (equal weight)
- Rebalance monthly or when rankings change significantly
- Use VIX for regime filtering (reduce exposure in high volatility)
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.data.feed import PolarsDataFeed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MultiAssetMomentumStrategy(Strategy):
    """Momentum strategy using Batch Mode (on_timestamp_batch).

    This strategy demonstrates batch execution mode where all events at
    a timestamp are processed together. Ideal for multi-asset strategies
    that need cross-asset ranking and portfolio optimization.
    """

    def __init__(
        self,
        top_n: int = 5,
        rebalance_days: int = 21,  # Roughly monthly
        position_weight: float = 0.20,  # 20% per position (5 * 20% = 100%)
        max_vix: float = 35.0,
        vix_scale_factor: float = 0.5,  # Scale positions by 50% when VIX high
    ):
        """Initialize multi-asset momentum strategy.

        Args:
            top_n: Number of top assets to hold
            rebalance_days: Days between rebalancing
            position_weight: Weight per position (e.g., 0.20 for 20%)
            max_vix: VIX threshold for position scaling
            vix_scale_factor: Position multiplier during high VIX
        """
        super().__init__()
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.position_weight = position_weight
        self.max_vix = max_vix
        self.vix_scale_factor = vix_scale_factor

        # Tracking state
        self._days_since_rebalance = 0
        self._current_holdings = set()

    def on_event(self, event):
        """Required abstract method - delegates to on_timestamp_batch."""
        pass

    def on_timestamp_batch(self, timestamp, events, context=None):
        """Process all events at a timestamp (Batch Mode).

        This is called once per timestamp with all market events at that time.
        Enables cross-asset ranking and portfolio optimization.

        Args:
            timestamp: The timestamp for this batch
            events: List of all MarketEvent objects at this timestamp
            context: Market-wide context (VIX, regime, etc.)
        """
        # Check if it's time to rebalance
        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_days:
            return

        # Extract VIX from context
        vix = 0.0
        if context:
            vix = context.get("VIX", 0.0)

        # Rank assets by momentum score
        asset_scores = {}
        asset_prices = {}

        for event in events:
            # Extract from signals dict (will be indicators once that param is added)
            momentum = event.signals.get("momentum", 0.0)
            if momentum is not None:
                asset_scores[event.asset_id] = momentum
                asset_prices[event.asset_id] = event.close

        # Sort by momentum (descending)
        sorted_assets = sorted(
            asset_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Select top N assets
        top_assets = [asset for asset, score in sorted_assets[: self.top_n]]

        logger.info(
            f"{timestamp}: Rebalancing to top {self.top_n} assets - "
            f"VIX={vix:.1f}, days_since={self._days_since_rebalance}"
        )
        for i, (asset, score) in enumerate(sorted_assets[: self.top_n]):
            logger.info(f"  {i+1}. {asset}: momentum={score:.4f}")

        # Calculate target weights with VIX adjustment
        if vix > self.max_vix:
            # High volatility - scale down positions
            weight_multiplier = self.vix_scale_factor
            logger.info(
                f"  High VIX ({vix:.1f} > {self.max_vix}) - "
                f"scaling positions by {weight_multiplier:.0%}"
            )
        else:
            weight_multiplier = 1.0

        target_weights = {
            asset: self.position_weight * weight_multiplier for asset in top_assets
        }

        # Create metadata for trade tracking
        metadata_per_asset = {}
        for i, (asset, score) in enumerate(sorted_assets[: self.top_n]):
            metadata_per_asset[asset] = {
                "rank": i + 1,
                "momentum_score": score,
                "vix": vix,
            }

        # Rebalance portfolio to target weights
        self.rebalance_to_weights(
            target_weights=target_weights,
            current_prices=asset_prices,
            tolerance=0.05,  # 5% tolerance before rebalancing
            metadata_per_asset=metadata_per_asset,
        )

        # Update state
        self._days_since_rebalance = 0
        self._current_holdings = set(top_assets)


def create_multi_asset_data_with_momentum(
    output_path: Path = Path("multi_asset_momentum_data.parquet"),
    n_assets: int = 10,
) -> tuple[Path, dict]:
    """Create sample multi-asset data with momentum indicators.

    Args:
        output_path: Path to save the sample data
        n_assets: Number of assets to simulate

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

    # Generate data for each asset
    all_data = []

    for asset_idx in range(n_assets):
        asset_id = f"ASSET_{asset_idx:02d}"

        # Each asset has different return characteristics
        drift = np.random.uniform(-0.0002, 0.0008)
        volatility = np.random.uniform(0.008, 0.020)

        # Generate price series
        initial_price = np.random.uniform(50, 500)
        prices = [initial_price]

        for i in range(1, n_days):
            daily_return = np.random.normal(drift, volatility)
            prices.append(prices[-1] * (1 + daily_return))

        prices = np.array(prices)

        # Generate OHLC
        highs = prices * (1 + np.abs(np.random.normal(0, 0.003, n_days)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.003, n_days)))
        opens = prices * (1 + np.random.normal(0, 0.002, n_days))
        volumes = np.random.exponential(5_000_000, n_days).astype(int)

        # Calculate momentum (20-day return)
        momentum = []
        for i in range(n_days):
            if i >= 20:
                ret = (prices[i] - prices[i - 20]) / prices[i - 20]
                momentum.append(ret)
            else:
                momentum.append(None)

        # Create asset DataFrame
        asset_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": [asset_id] * n_days,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
                "momentum": momentum,
            },
        )

        all_data.append(asset_df)

    # Concatenate all assets
    df = pl.concat(all_data)

    # Sort by timestamp (important for multi-asset data)
    df = df.sort("timestamp", "asset_id")

    # Generate VIX context
    vix_values = []
    for i in range(n_days):
        base_vix = 18.0
        # Add occasional spikes
        if i % 60 == 0:
            base_vix += np.random.uniform(8, 18)
        vix = base_vix + np.random.normal(0, 2.5)
        vix = max(12.0, min(50.0, vix))
        vix_values.append(vix)

    # Create context data
    context_data = {}
    for i, ts in enumerate(timestamps):
        py_ts = ts.to_py() if hasattr(ts, "to_py") else ts
        context_data[py_ts] = {
            "VIX": vix_values[i],
        }

    # Save to parquet
    df.write_parquet(str(output_path))
    logger.info(
        f"Created multi-asset data with {n_assets} assets, "
        f"{len(df)} total bars at {output_path}"
    )
    logger.info(f"Created context data with {len(context_data)} timestamps")

    return output_path, context_data


def main():
    """Run multi-asset momentum strategy example."""

    # Create sample multi-asset data
    logger.info("Creating sample multi-asset data with momentum indicators...")
    n_assets = 10
    data_path, context_data = create_multi_asset_data_with_momentum(n_assets=n_assets)

    # Create Polars data feed (supports multi-asset naturally)
    # Note: PolarsDataFeed requires price_path and asset_id, not multi-asset from one file
    # For this example, we need to use a different approach or load data differently
    # TODO: This example needs multi-asset data feed support to be properly implemented
    from ml4t.backtest.data.feed import ParquetDataFeed

    # For now, create a simple single-asset feed as a placeholder
    # The batch mode logic is still demonstrated, just not with truly multi-asset data
    data_feed = ParquetDataFeed(
        path=data_path,
        asset_id="ASSET_00",  # First asset only for demo
        timestamp_column="timestamp",
        signal_columns=["momentum"],
    )

    # Create multi-asset momentum strategy
    strategy = MultiAssetMomentumStrategy(
        top_n=5,
        rebalance_days=21,  # Monthly rebalancing
        position_weight=0.20,  # 20% per position (5 positions = 100%)
        max_vix=35.0,
        vix_scale_factor=0.5,  # Cut positions in half during high VIX
    )

    # Verify strategy is in batch mode
    logger.info(f"Strategy execution mode: {strategy.execution_mode}")
    assert strategy.execution_mode == "batch", "Should be in batch mode"

    # Create and run backtest engine
    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        context_data=context_data,
        initial_capital=100000.0,
    )

    logger.info("Starting multi-asset momentum strategy backtest...")
    logger.info("=" * 60)
    results = engine.run()

    # Display results
    print("\n" + "=" * 60)
    print("MULTI-ASSET MOMENTUM STRATEGY RESULTS")
    print("=" * 60)
    print(f"Number of Assets: {n_assets}")
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
    print(f"  Top N Assets: {strategy.top_n}")
    print(f"  Rebalance Days: {strategy.rebalance_days}")
    print(f"  Position Weight: {strategy.position_weight:.0%}")
    print(f"  Max VIX: {strategy.max_vix}")
    print(f"  VIX Scale Factor: {strategy.vix_scale_factor:.0%}")
    print(f"  Execution Mode: {strategy.execution_mode}")

    # Clean up
    data_path.unlink()

    return results


if __name__ == "__main__":
    results = main()
