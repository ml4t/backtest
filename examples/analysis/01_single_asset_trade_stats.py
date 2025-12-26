"""Single Asset Backtest with Trade Statistics Analysis.

This example demonstrates:
- Running a simple momentum strategy on SPY
- Using BacktestAnalyzer for comprehensive trade statistics
- Computing win rate, profit factor, expectancy, and more

Uses real ETF data from ~/ml4t/data/etfs/

Usage:
    uv run python examples/analysis/01_single_asset_trade_stats.py
"""

from datetime import datetime
from pathlib import Path

import polars as pl

from ml4t.backtest import DataFeed, Engine, ExecutionMode, OrderSide, Strategy
from ml4t.backtest.analysis import BacktestAnalyzer

# Path to ETF data - adjust if needed
DATA_DIR = Path.home() / "ml4t" / "data" / "etfs" / "ohlcv_1d"


def load_etf_data(
    ticker: str, start_date: str | None = None, end_date: str | None = None
) -> pl.DataFrame:
    """Load ETF OHLCV data from Hive-partitioned parquet.

    Args:
        ticker: ETF ticker symbol (e.g., 'SPY')
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        DataFrame with timestamp, open, high, low, close, volume columns
    """
    path = DATA_DIR / f"ticker={ticker}" / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")

    df = pl.read_parquet(path)

    # Add asset column for backtest engine
    df = df.with_columns(pl.lit(ticker).alias("asset"))

    # Filter by date if specified
    if start_date:
        df = df.filter(pl.col("timestamp") >= datetime.fromisoformat(start_date))
    if end_date:
        df = df.filter(pl.col("timestamp") <= datetime.fromisoformat(end_date))

    return df.sort("timestamp")


class MomentumStrategy(Strategy):
    """Simple momentum strategy: long when price > SMA, exit when price < SMA.

    This generates discrete entry/exit signals rather than continuous positions,
    making it good for demonstrating trade statistics.
    """

    def __init__(self, sma_period: int = 50, position_size_pct: float = 0.95):
        """Initialize momentum strategy.

        Args:
            sma_period: Period for simple moving average
            position_size_pct: Fraction of equity to invest
        """
        self.sma_period = sma_period
        self.position_size_pct = position_size_pct
        self.price_history: list[float] = []
        self.in_position = False

    def on_data(self, timestamp, data, context, broker):
        """Process new bar and generate signals."""
        # Get current close price
        asset = list(data.keys())[0]  # Single asset
        close = data[asset]["close"]

        # Track price history
        self.price_history.append(close)
        if len(self.price_history) > self.sma_period:
            self.price_history.pop(0)

        # Wait for enough history
        if len(self.price_history) < self.sma_period:
            return

        # Calculate SMA
        sma = sum(self.price_history) / len(self.price_history)

        # Get current position
        position = broker.get_position(asset)
        has_position = position is not None and position.quantity > 0

        # Entry signal: price crosses above SMA
        if close > sma and not has_position:
            # Calculate position size
            equity = broker.get_account_value()
            position_value = equity * self.position_size_pct
            quantity = position_value / close

            broker.submit_order(asset, quantity, OrderSide.BUY)
            self.in_position = True

        # Exit signal: price crosses below SMA
        elif close < sma and has_position:
            broker.submit_order(asset, position.quantity, OrderSide.SELL)
            self.in_position = False


def main():
    """Run momentum strategy on SPY and analyze trades."""
    print("=" * 70)
    print("Single Asset Backtest with Trade Statistics")
    print("=" * 70)

    # Load real SPY data
    print("\nLoading SPY data from ~/ml4t/data/etfs/...")
    df = load_etf_data("SPY", start_date="2015-01-01", end_date="2023-12-31")
    print(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Create data feed
    feed = DataFeed(prices_df=df)

    # Create strategy
    strategy = MomentumStrategy(sma_period=50, position_size_pct=0.95)

    # Run backtest
    print("\nRunning backtest...")
    initial_cash = 100_000.0
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=initial_cash,
        execution_mode=ExecutionMode.NEXT_BAR,  # Realistic execution
    )
    result = engine.run()

    # Basic results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Initial Capital:  ${initial_cash:,.2f}")
    print(f"Final Value:      ${result['final_value']:,.2f}")
    print(f"Total Return:     {result['total_return']:.2%}")
    print(f"Sharpe Ratio:     {result['sharpe']:.2f}")
    print(f"Max Drawdown:     {result['max_drawdown_pct']:.2%}")

    # Analyze trades using the adapter
    print("\n" + "=" * 70)
    print("TRADE ANALYSIS (via BacktestAnalyzer)")
    print("=" * 70)

    analyzer = BacktestAnalyzer(engine)
    stats = analyzer.trade_statistics()

    # Print detailed statistics
    print(stats.summary())

    # Additional analysis
    print("\n" + "-" * 70)
    print("Additional Metrics")
    print("-" * 70)

    # Get trades as DataFrame for custom analysis
    trades_df = analyzer.get_trades_dataframe()
    if len(trades_df) > 0:
        print("\nTrades by direction:")
        direction_stats = trades_df.group_by("direction").agg(
            pl.len().alias("count"),
            pl.col("pnl").sum().alias("total_pnl"),
            pl.col("pnl_percent").mean().alias("avg_return"),
        )
        print(direction_stats)

        print("\nHolding period distribution (bars):")
        bars_held = trades_df["bars_held"]
        print(f"  Min:    {bars_held.min()}")
        print(f"  Median: {bars_held.median():.0f}")
        print(f"  Max:    {bars_held.max()}")

        # Show best and worst trades
        print("\nTop 3 Best Trades:")
        best = trades_df.sort("pnl", descending=True).head(3)
        for row in best.iter_rows(named=True):
            print(
                f"  {row['entry_time'].date()} → {row['exit_time'].date()}: ${row['pnl']:,.2f} ({row['pnl_percent']:.2%})"
            )

        print("\nTop 3 Worst Trades:")
        worst = trades_df.sort("pnl").head(3)
        for row in worst.iter_rows(named=True):
            print(
                f"  {row['entry_time'].date()} → {row['exit_time'].date()}: ${row['pnl']:,.2f} ({row['pnl_percent']:.2%})"
            )

    # Show how to export for diagnostic library
    print("\n" + "=" * 70)
    print("DIAGNOSTIC LIBRARY INTEGRATION")
    print("=" * 70)
    print("""
To use with ml4t.diagnostic for advanced analysis:

    from ml4t.backtest.analysis import BacktestAnalyzer, to_trade_records
    from ml4t.diagnostic.evaluation import TradeAnalysis

    # Get trades in diagnostic format
    analyzer = BacktestAnalyzer(engine)
    trade_records = analyzer.get_trade_records()

    # Analyze with diagnostic library
    diagnostic = TradeAnalysis(trade_records)
    worst_20 = diagnostic.worst_trades(n=20)
    patterns = diagnostic.identify_failure_patterns()
""")

    # Export trades for further analysis
    if len(trades_df) > 0:
        output_path = Path(__file__).parent / "spy_momentum_trades.parquet"
        trades_df.write_parquet(output_path)
        print(f"\nTrades exported to: {output_path}")


if __name__ == "__main__":
    main()
