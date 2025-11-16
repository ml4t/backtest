"""ML Strategy Example demonstrating signal-based trading with context.

This example shows how to:
1. Create data with ML predictions (signals)
2. Use market-wide context (VIX, regime indicators)
3. Use strategy helper methods for clean code
4. Implement ML-specific position sizing
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


class MLMomentumStrategy(Strategy):
    """ML-powered momentum strategy using predictions and context.

    Strategy Logic:
    - Enter long when ML prediction > entry_threshold and confidence > min_confidence
    - Don't trade during high VIX periods (risk-off)
    - Size positions based on ML confidence (Kelly-like)
    - Exit when unrealized P&L > profit_target or < stop_loss
    """

    def __init__(
        self,
        asset_id: str = "SPY",
        entry_threshold: float = 0.6,
        min_confidence: float = 0.7,
        max_position_pct: float = 0.20,
        profit_target: float = 0.15,  # 15% gain
        stop_loss: float = -0.05,  # 5% loss
        max_vix: float = 30.0,  # Don't trade if VIX > 30
    ):
        """Initialize ML momentum strategy.

        Args:
            asset_id: Asset to trade
            entry_threshold: Minimum ML prediction to enter (0-1)
            min_confidence: Minimum confidence to enter (0-1)
            max_position_pct: Maximum position size as % of portfolio
            profit_target: Take profit at this % gain
            stop_loss: Stop loss at this % loss
            max_vix: Maximum VIX level to allow trading
        """
        super().__init__()
        self.asset_id = asset_id
        self.entry_threshold = entry_threshold
        self.min_confidence = min_confidence
        self.max_position_pct = max_position_pct
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_vix = max_vix

    def on_event(self, event):
        """Required abstract method - delegates to on_market_event."""
        # This method is called by the engine's generic event dispatch
        # We handle everything in on_market_event instead
        pass

    def on_market_event(self, event, context=None):
        """Handle market events with ML signals and context.

        Args:
            event: MarketEvent with OHLCV data and signals dict
            context: Optional dict with market-wide indicators (VIX, regime, etc.)
        """
        # Only process our target asset
        if event.asset_id != self.asset_id:
            return

        # Extract ML signals from event
        ml_prediction = event.signals.get("prediction", 0.0)
        confidence = event.signals.get("confidence", 0.0)

        # Extract context if available
        vix = 0.0
        regime = "unknown"
        if context:
            vix = context.get("VIX", 0.0)
            regime = context.get("regime", "unknown")

        # Get current position
        position = self.get_position(self.asset_id)

        # Exit Logic: Check for profit target or stop loss
        if position != 0:
            pnl_pct = self.get_unrealized_pnl_pct(self.asset_id)

            if pnl_pct and pnl_pct >= self.profit_target:
                logger.info(
                    f"{event.timestamp}: Taking profit at {pnl_pct:.2%} "
                    f"(target: {self.profit_target:.2%})"
                )
                self.close_position(self.asset_id)
                return

            if pnl_pct and pnl_pct <= self.stop_loss:
                logger.info(
                    f"{event.timestamp}: Stop loss triggered at {pnl_pct:.2%} "
                    f"(stop: {self.stop_loss:.2%})"
                )
                self.close_position(self.asset_id)
                return

        # Entry Logic: No position, strong signal, good conditions
        if position == 0:
            # Risk check: VIX too high?
            if vix > self.max_vix:
                logger.debug(
                    f"{event.timestamp}: VIX too high ({vix:.1f} > {self.max_vix}), "
                    "skipping entry"
                )
                return

            # Signal check: Strong enough?
            if ml_prediction > self.entry_threshold and confidence > self.min_confidence:
                logger.info(
                    f"{event.timestamp}: Entry signal - pred={ml_prediction:.2f}, "
                    f"conf={confidence:.2f}, VIX={vix:.1f}, regime={regime}"
                )

                # Size position by confidence (Kelly-like approach)
                self.size_by_confidence(
                    asset_id=self.asset_id,
                    confidence=confidence,
                    max_percent=self.max_position_pct,
                    price=event.close,
                )


def create_ml_data_with_context(
    output_path: Path = Path("ml_sample_data.parquet"),
) -> tuple[Path, dict]:
    """Create sample market data with ML signals and context data.

    This generates:
    - OHLCV price data
    - ML predictions (0-1 probability of 5-day positive return)
    - Confidence scores (0-1)
    - Market context (VIX, regime)

    Args:
        output_path: Path to save the sample data

    Returns:
        Tuple of (data_path, context_dict)
    """
    np.random.seed(42)
    n_days = 252 * 2  # Two trading years

    # Generate timestamps (daily bars)
    timestamps = pl.datetime_range(
        datetime(2022, 1, 3, 9, 30),
        datetime(2023, 12, 31, 16, 0),
        interval="1d",
        time_zone="America/New_York",
        eager=True,
    )[:n_days]

    # Generate price series with momentum and mean reversion
    initial_price = 400.0
    prices = [initial_price]

    # Create regime changes (bull/bear markets)
    regime_changes = [0, 150, 300, 400]  # Days when regime changes
    regimes = []
    current_regime = "bull"
    regime_idx = 0

    for i in range(1, n_days):
        # Check for regime change
        if regime_idx < len(regime_changes) - 1 and i >= regime_changes[regime_idx + 1]:
            regime_idx += 1
            current_regime = "bear" if current_regime == "bull" else "bull"

        regimes.append(current_regime)

        # Different return characteristics by regime
        if current_regime == "bull":
            drift = 0.0008  # Positive drift
            volatility = 0.015  # Lower volatility
        else:
            drift = -0.0003  # Negative drift
            volatility = 0.025  # Higher volatility

        daily_return = np.random.normal(drift, volatility)
        prices.append(prices[-1] * (1 + daily_return))

    regimes.append(current_regime)  # Add last regime
    prices = np.array(prices)

    # Generate OHLC from close prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    opens = prices * (1 + np.random.normal(0, 0.003, n_days))
    volumes = np.random.exponential(10_000_000, n_days).astype(int)

    # Generate ML predictions
    # Prediction quality: Higher in bull markets, noisier in bear markets
    true_5d_returns = []
    for i in range(n_days):
        if i + 5 < n_days:
            ret = (prices[i + 5] - prices[i]) / prices[i]
            true_5d_returns.append(ret)
        else:
            true_5d_returns.append(0.0)

    true_5d_returns = np.array(true_5d_returns)

    # ML predictions with varying accuracy
    predictions = []
    confidences = []

    for i in range(n_days):
        true_ret = true_5d_returns[i]
        regime = regimes[i]

        # Model is better in bull markets (80% accuracy vs 65% in bear)
        if regime == "bull":
            noise = np.random.normal(0, 0.02)
            accuracy = 0.80
        else:
            noise = np.random.normal(0, 0.04)
            accuracy = 0.65

        # Convert return to probability (sigmoid-like transformation)
        pred = 1 / (1 + np.exp(-10 * (true_ret + noise)))
        predictions.append(pred)

        # Confidence correlates with prediction strength
        conf = min(0.95, max(0.5, accuracy * (0.5 + abs(pred - 0.5))))
        confidences.append(conf)

    # Generate VIX (volatility index)
    # Higher in bear markets and during transitions
    vix_values = []
    for i, regime in enumerate(regimes):
        base_vix = 15.0 if regime == "bull" else 28.0

        # Add spikes near regime changes
        near_transition = any(abs(i - change) < 10 for change in regime_changes)
        if near_transition:
            base_vix += np.random.uniform(5, 15)

        vix = base_vix + np.random.normal(0, 3)
        vix = max(10.0, min(60.0, vix))  # Clamp to realistic range
        vix_values.append(vix)

    # Create DataFrame with signals
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
            "prediction": predictions,  # ML signal: probability of up move
            "confidence": confidences,  # ML confidence score
        },
    )

    # Create context data (market-wide indicators)
    context_data = {}
    for i, ts in enumerate(timestamps):
        # Convert Polars datetime to Python datetime if needed
        py_ts = ts.to_py() if hasattr(ts, "to_py") else ts
        context_data[py_ts] = {
            "VIX": vix_values[i],
            "regime": regimes[i],
        }

    # Save to parquet
    df.write_parquet(str(output_path))
    logger.info(f"Created ML sample data with {len(df)} bars at {output_path}")
    logger.info(f"Created context data with {len(context_data)} timestamps")

    return output_path, context_data


def main():
    """Run ML strategy backtest example."""

    # Create sample data with ML signals and context
    logger.info("Creating sample data with ML signals and context...")
    data_path, context_data = create_ml_data_with_context()

    # Create data feed with signal columns
    data_feed = ParquetDataFeed(
        path=data_path,
        asset_id="SPY",
        timestamp_column="timestamp",
        signal_columns=["prediction", "confidence"],  # Extract these as signals
    )

    # Create ML strategy
    strategy = MLMomentumStrategy(
        asset_id="SPY",
        entry_threshold=0.6,
        min_confidence=0.7,
        max_position_pct=0.20,
        profit_target=0.15,
        stop_loss=-0.05,
        max_vix=30.0,
    )

    # Create and run backtest engine with context
    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        context_data=context_data,  # Pass market-wide context
        initial_capital=100000.0,
    )

    logger.info("Starting ML strategy backtest...")
    logger.info("=" * 60)
    results = engine.run()

    # Display results
    print("\n" + "=" * 60)
    print("ML STRATEGY BACKTEST RESULTS")
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
    print(f"  Entry Threshold: {strategy.entry_threshold}")
    print(f"  Min Confidence: {strategy.min_confidence}")
    print(f"  Max Position: {strategy.max_position_pct:.0%}")
    print(f"  Profit Target: {strategy.profit_target:.0%}")
    print(f"  Stop Loss: {strategy.stop_loss:.0%}")
    print(f"  Max VIX: {strategy.max_vix}")

    # Clean up
    data_path.unlink()  # Remove sample data file

    return results


if __name__ == "__main__":
    results = main()
