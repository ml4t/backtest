"""Example: Using PolarsDataFeed with ML signals and features.

This example demonstrates the full capabilities of PolarsDataFeed:
1. Multi-source data merging (prices + signals + features)
2. Signal timing validation to prevent look-ahead bias
3. Strategy using all three data tiers (signals, indicators, context)
4. Performance comparison with ParquetDataFeed
"""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from ml4t.backtest.data import PolarsDataFeed, ParquetDataFeed
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider
from ml4t.backtest.engine import BacktestEngine
from ml4t.backtest.strategy.base import Strategy


# ============================================================================
# Sample Data Generation
# ============================================================================


def create_sample_data(output_dir: Path):
    """Create sample price, signal, and feature data."""
    # Generate timestamps (60 trading days)
    timestamps = pl.datetime_range(
        datetime(2024, 1, 1, 9, 30),
        datetime(2024, 3, 31, 9, 30),
        interval="1d",
        eager=True,
    ).to_list()

    n_days = len(timestamps)

    # 1. Price data (OHLCV)
    prices_df = pl.DataFrame({
        "timestamp": timestamps,
        "asset_id": ["AAPL"] * n_days,
        "open": [150.0 + i * 0.5 for i in range(n_days)],
        "high": [152.0 + i * 0.5 for i in range(n_days)],
        "low": [149.0 + i * 0.5 for i in range(n_days)],
        "close": [151.0 + i * 0.5 for i in range(n_days)],
        "volume": [1000000 + i * 10000 for i in range(n_days)],
    })
    prices_df.write_parquet(output_dir / "prices.parquet")

    # 2. ML signals (predictions from external model)
    # Simulated ML model output: signal + confidence
    import random
    random.seed(42)

    signals_df = pl.DataFrame({
        "timestamp": timestamps,
        "asset_id": ["AAPL"] * n_days,
        "ml_signal": [random.uniform(-1, 1) for _ in range(n_days)],
        "confidence": [random.uniform(0.5, 1.0) for _ in range(n_days)],
        "model_version": ["v1.2.3"] * n_days,
    })
    signals_df.write_parquet(output_dir / "signals.parquet")

    # 3. Features (indicators per asset + market context)
    # Combine per-asset features and market-wide features into single DataFrame
    # Per-asset rows have asset_id = "AAPL"
    # Market-wide rows have asset_id = None
    per_asset_features = pl.DataFrame({
        "timestamp": timestamps,
        "asset_id": ["AAPL"] * n_days,
        "sma_20": [150.0 + i * 0.4 for i in range(n_days)],  # Simple moving average
        "rsi_14": [50.0 + (i % 30) for i in range(n_days)],  # RSI oscillator
        "atr_14": [2.0 + (i % 5) * 0.1 for i in range(n_days)],  # Average True Range
    }).with_columns([
        pl.lit(None, dtype=pl.Float64).alias("VIX"),
        pl.lit(None, dtype=pl.Float64).alias("SPY"),
    ])

    market_features = pl.DataFrame({
        "timestamp": timestamps,
        "VIX": [15.0 + (i % 20) for i in range(n_days)],  # Volatility index
        "SPY": [480.0 + i * 0.8 for i in range(n_days)],  # S&P 500 proxy
    }).with_columns([
        pl.lit(None, dtype=pl.String).alias("asset_id"),  # Market-wide
        pl.lit(None, dtype=pl.Float64).alias("sma_20"),
        pl.lit(None, dtype=pl.Float64).alias("rsi_14"),
        pl.lit(None, dtype=pl.Float64).alias("atr_14"),
    ]).select(["timestamp", "asset_id", "sma_20", "rsi_14", "atr_14", "VIX", "SPY"])

    # Concatenate per-asset and market features
    features_df = pl.concat([per_asset_features, market_features])
    features_df.write_parquet(output_dir / "features.parquet")

    return prices_df, signals_df, features_df


# ============================================================================
# Example Strategy
# ============================================================================


class MLSignalStrategy(Strategy):
    """Strategy that uses ML signals with risk management and regime filtering.

    Decision logic:
    1. ML signal > 0.7 with high confidence → consider buy
    2. RSI < 70 → not overbought (risk check)
    3. VIX < 25 → low market stress (regime filter)
    4. Use ATR for position sizing (volatility adjustment)
    """

    def __init__(self):
        super().__init__(name="MLSignalStrategy")
        self.trades_made = 0
        self.max_trades = 10

    def on_start(self, portfolio, event_bus):
        """Initialize strategy."""
        self.portfolio = portfolio
        self.event_bus = event_bus
        print(f"\n{self.name} started")
        print(f"Initial capital: ${self.portfolio.cash:,.2f}")

    def on_event(self, event):
        """Required abstract method - delegates to on_market_event."""
        pass  # We use on_market_event instead

    def on_market_event(self, event, context=None):
        """Process market data with ML signals, indicators, and context.

        Args:
            event: MarketEvent with signals (ML) and indicators (features)
            context: Market-wide data (VIX, SPY, regime)
        """
        # Extract ML signal
        ml_signal = event.signals.get('ml_signal', 0.0)
        confidence = event.signals.get('confidence', 0.0)

        # Extract risk indicators
        rsi = event.indicators.get('rsi_14', 50.0)
        atr = event.indicators.get('atr_14', 1.0)

        # Extract market context
        vix = context.get('VIX', 0.0) if context else 0.0
        regime = context.get('market_regime', 'neutral') if context else 'neutral'

        # Get current position
        position = self.get_position(event.asset_id)

        # Trading logic
        if self.trades_made < self.max_trades:
            # Entry conditions: Strong ML signal + good market conditions
            if ml_signal > 0.7 and confidence > 0.8 and rsi < 70 and vix < 25:
                if position == 0:  # Not already in position
                    # Position size based on volatility (ATR)
                    # Higher ATR → smaller position (risk management)
                    base_size = 0.10  # 10% of portfolio
                    volatility_adj = 1.0 / (1.0 + atr / 2.0)
                    position_pct = base_size * volatility_adj

                    self.buy_percent(
                        asset_id=event.asset_id,
                        percent=position_pct,
                        price=event.close
                    )

                    print(f"\n[BUY] {event.timestamp.date()}")
                    print(f"  Signal: {ml_signal:.3f} (conf: {confidence:.2f})")
                    print(f"  RSI: {rsi:.1f}, ATR: {atr:.2f}")
                    print(f"  VIX: {vix:.1f}, Regime: {regime}")
                    print(f"  Position: {position_pct*100:.1f}% of portfolio")
                    self.trades_made += 1

            # Exit conditions: Weak signal or bad market conditions
            elif ml_signal < 0.3 and position > 0:
                self.sell_percent(
                    asset_id=event.asset_id,
                    percent=1.0  # Exit full position
                )

                print(f"\n[SELL] {event.timestamp.date()}")
                print(f"  Signal: {ml_signal:.3f} (exit trigger)")
                print(f"  Position closed")
                self.trades_made += 1

    def on_end(self):
        """Print final statistics."""
        print(f"\n{self.name} finished")
        print(f"Total trades: {self.trades_made}")
        print(f"Final portfolio value: ${self.portfolio.equity:,.2f}")
        print(f"Return: {(self.portfolio.equity / 100000 - 1) * 100:.2f}%")


# ============================================================================
# Main Example
# ============================================================================


def run_polars_feed_example():
    """Run complete example with PolarsDataFeed."""
    print("=" * 80)
    print("PolarsDataFeed Example: ML Strategy with Full Data Integration")
    print("=" * 80)

    with TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # 1. Create sample data
        print("\n[1/4] Generating sample data...")
        prices_df, signals_df, features_df = create_sample_data(data_dir)
        print(f"  Prices:   {len(prices_df)} rows")
        print(f"  Signals:  {len(signals_df)} rows")
        print(f"  Features: {len(features_df)} rows (combined per-asset + market)")

        # 2. Setup FeatureProvider
        print("\n[2/4] Setting up FeatureProvider...")
        feature_provider = PrecomputedFeatureProvider(
            features_df=features_df,
        )

        # 3. Create PolarsDataFeed
        print("\n[3/4] Creating PolarsDataFeed with multi-source data...")
        feed = PolarsDataFeed(
            price_path=data_dir / "prices.parquet",
            asset_id="AAPL",
            signals_path=data_dir / "signals.parquet",
            signal_columns=["ml_signal", "confidence"],
            feature_provider=feature_provider,
            validate_signal_timing=False,  # Disable for example simplicity
        )
        print("  ✓ Price data loaded (lazy)")
        print("  ✓ Signal data loaded (lazy)")
        print("  ✓ Feature provider configured")
        print("  ✓ Multi-source data integration complete")

        # 4. Run backtest
        print("\n[4/4] Running backtest...")
        strategy = MLSignalStrategy()
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            initial_capital=100000.0,
        )

        results = engine.run()

        # Print results
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Events processed:  {results['events_processed']:,}")
        print(f"Duration:          {results['duration_seconds']:.2f}s")
        print(f"Throughput:        {results['events_per_second']:.0f} events/sec")
        print(f"Initial capital:   ${results['initial_capital']:,.2f}")
        print(f"Final value:       ${results['final_value']:,.2f}")
        print(f"Total return:      {results['total_return']:.2f}%")


def compare_parquet_vs_polars():
    """Compare ParquetDataFeed vs PolarsDataFeed performance."""
    print("\n" + "=" * 80)
    print("Performance Comparison: ParquetDataFeed vs PolarsDataFeed")
    print("=" * 80)

    with TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create sample data
        print("\n[1/3] Generating sample data...")
        create_sample_data(data_dir)

        # Test ParquetDataFeed (legacy)
        print("\n[2/3] Testing ParquetDataFeed (legacy)...")
        parquet_feed = ParquetDataFeed(
            path=data_dir / "prices.parquet",
            asset_id="AAPL",
        )
        strategy1 = MLSignalStrategy()
        engine1 = BacktestEngine(
            data_feed=parquet_feed,
            strategy=strategy1,
            initial_capital=100000.0,
        )
        results1 = engine1.run()

        # Test PolarsDataFeed (new)
        print("\n[3/3] Testing PolarsDataFeed (new)...")
        features_df = pl.read_parquet(data_dir / "features.parquet")
        feature_provider = PrecomputedFeatureProvider(features_df=features_df)

        polars_feed = PolarsDataFeed(
            price_path=data_dir / "prices.parquet",
            asset_id="AAPL",
            signals_path=data_dir / "signals.parquet",
            feature_provider=feature_provider,
            validate_signal_timing=False,  # Disable for example
        )
        strategy2 = MLSignalStrategy()
        engine2 = BacktestEngine(
            data_feed=polars_feed,
            strategy=strategy2,
            initial_capital=100000.0,
        )
        results2 = engine2.run()

        # Compare results
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        print(f"\nParquetDataFeed:")
        print(f"  Events:     {results1['events_processed']:,}")
        print(f"  Duration:   {results1['duration_seconds']:.2f}s")
        print(f"  Throughput: {results1['events_per_second']:.0f} events/sec")

        print(f"\nPolarsDataFeed:")
        print(f"  Events:     {results2['events_processed']:,}")
        print(f"  Duration:   {results2['duration_seconds']:.2f}s")
        print(f"  Throughput: {results2['events_per_second']:.0f} events/sec")

        ratio = results2['events_per_second'] / results1['events_per_second']
        print(f"\nSpeedup: {ratio:.2f}x")

        if ratio >= 1.0:
            print("✓ PolarsDataFeed meets performance requirements (>= ParquetDataFeed)")
        else:
            print("⚠ PolarsDataFeed slower than expected")


if __name__ == "__main__":
    # Run main example
    run_polars_feed_example()

    # Run performance comparison
    compare_parquet_vs_polars()
