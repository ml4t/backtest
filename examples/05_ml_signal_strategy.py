"""ML signal-based trading strategy example.

This example demonstrates how to use pre-computed ML signals
with ml4t.backtest. The signals are included in the price DataFrame
and accessed during strategy execution.

Key concepts:
- Passing ML signals through DataFeed
- Accessing signals in strategy.on_data()
- Position sizing based on signal strength
- Risk management with stop-loss
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ml4t.backtest import (
    DataFeed,
    Engine,
    ExecutionMode,
    StopLoss,
    Strategy,
)


def generate_sample_data_with_signals(
    assets: list[str], n_bars: int = 504, seed: int = 42
) -> pl.DataFrame:
    """Generate synthetic OHLCV data with ML signals.

    The signal column represents a pre-computed ML prediction:
    - signal > 0.5: bullish prediction
    - signal < 0.5: bearish prediction
    - signal ~= 0.5: neutral

    In practice, you would compute these signals using your ML model
    on historical features, ensuring no look-ahead bias.
    """
    np.random.seed(seed)

    rows = []
    base_date = datetime(2020, 1, 1)

    for asset in assets:
        # Generate correlated returns and signals
        # (simulating signal predictive power)
        base_signal = np.random.random(n_bars) * 0.4 + 0.3  # 0.3 to 0.7 range

        # Add some predictive power: higher signals slightly predict higher returns
        noise = np.random.normal(0, 0.02, n_bars)
        predicted_component = (base_signal - 0.5) * 0.01  # Small edge
        returns = predicted_component + noise

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
                    # ML signal column - this is what your model would produce
                    "signal": base_signal[i],
                }
            )

    return pl.DataFrame(rows).sort(["timestamp", "asset"])


class MLSignalStrategy(Strategy):
    """Strategy that trades based on ML signal predictions.

    Entry logic:
    - Enter long when signal > 0.6 (bullish)
    - Position size proportional to signal strength

    Exit logic:
    - Exit when signal < 0.4 (bearish) or stop-loss hit
    - 5% stop-loss on all positions
    """

    def __init__(
        self,
        entry_threshold: float = 0.6,
        exit_threshold: float = 0.4,
        max_position_pct: float = 0.20,
    ):
        """Initialize strategy.

        Args:
            entry_threshold: Signal level to enter (default 0.6)
            exit_threshold: Signal level to exit (default 0.4)
            max_position_pct: Maximum position size as % of portfolio (default 20%)
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_position_pct = max_position_pct

    def on_start(self, broker):
        """Set up risk management rules."""
        # Apply 5% stop-loss to all positions
        broker.set_position_rules(StopLoss(pct=0.05))

    def on_data(self, timestamp, data, context, broker):
        """Execute trading logic based on ML signals."""
        portfolio_value = broker.get_account_value()

        for asset, bar in data.items():
            # Get the ML signal from the signals dict
            signals = bar.get("signals", {})
            signal = signals.get("signal", 0.5)
            price = bar.get("close", 0)

            if price <= 0:
                continue

            position = broker.get_position(asset)

            if position is None:
                # No position - check for entry
                if signal > self.entry_threshold:
                    # Scale position size by signal strength
                    # signal=0.6 -> 50% of max, signal=1.0 -> 100% of max
                    signal_strength = (signal - self.entry_threshold) / (1.0 - self.entry_threshold)
                    position_pct = self.max_position_pct * signal_strength

                    # Calculate shares to buy
                    target_value = portfolio_value * position_pct
                    shares = int(target_value / price)

                    if shares > 0:
                        broker.submit_order(asset, shares)

            else:
                # Have position - check for exit
                if signal < self.exit_threshold:
                    broker.close_position(asset)


def main():
    """Run the ML signal strategy backtest."""
    # Define universe
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

    # Generate sample data with ML signals (2 years)
    print("Generating sample data with ML signals...")
    df = generate_sample_data_with_signals(assets, n_bars=504)

    # Show signal statistics
    print("\nSignal statistics:")
    print(df.select(["signal"]).describe())

    # Create data feed with signal column
    # Signals are passed via signals_df with columns: timestamp, asset, signal
    signals_df = df.select(["timestamp", "asset", "signal"])
    prices_df = df.drop("signal")

    feed = DataFeed(
        prices_df=prices_df,
        signals_df=signals_df,
    )

    # Create strategy
    strategy = MLSignalStrategy(
        entry_threshold=0.6,
        exit_threshold=0.4,
        max_position_pct=0.20,
    )

    # Run backtest
    print("\nRunning backtest...")
    initial_cash = 100_000.0
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=initial_cash,
        execution_mode=ExecutionMode.NEXT_BAR,  # More realistic
    )
    result = engine.run()

    # Print results
    print("\n" + "=" * 50)
    print("ML SIGNAL STRATEGY RESULTS")
    print("=" * 50)
    print(f"Initial Capital:  ${initial_cash:,.2f}")
    print(f"Final Value:      ${result['final_value']:,.2f}")
    print(f"Total Return:     {result['total_return']:.2%}")
    print(f"Sharpe Ratio:     {result['sharpe']:.2f}")
    print(f"Max Drawdown:     {result['max_drawdown_pct']:.2%}")
    print(f"Total Trades:     {len(engine.broker.trades)}")

    # Trade analysis
    if engine.broker.trades:
        wins = [t for t in engine.broker.trades if t.pnl > 0]
        win_rate = len(wins) / len(engine.broker.trades)
        print(f"Win Rate:         {win_rate:.1%}")

    # Show final positions
    print("\n=== Final Positions ===")
    for asset, pos in engine.broker.get_positions().items():
        print(f"  {asset}: {pos.quantity:.0f} shares @ ${pos.entry_price:.2f}")


if __name__ == "__main__":
    main()
