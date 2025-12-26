"""Multi-Asset Portfolio Backtest with Portfolio Diagnostics.

This example demonstrates:
- Tactical asset allocation across multiple ETFs
- Cross-asset momentum timing
- Portfolio-level trade analysis (by asset, by regime)
- Integration with diagnostic library for performance attribution

Uses real ETF data from ~/ml4t/data/etfs/

Usage:
    uv run python examples/analysis/02_multi_asset_portfolio.py
"""

from datetime import datetime
from pathlib import Path

import polars as pl

from ml4t.backtest import (
    DataFeed,
    Engine,
    ExecutionMode,
    OrderSide,
    Strategy,
)
from ml4t.backtest.analysis import BacktestAnalyzer

# Path to ETF data
DATA_DIR = Path.home() / "ml4t" / "data" / "etfs" / "ohlcv_1d"

# Asset universe for tactical allocation
ASSET_UNIVERSE = ["SPY", "TLT", "GLD", "EEM", "IWM"]


def load_multi_asset_data(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load and combine multiple ETF OHLCV datasets.

    Args:
        tickers: List of ETF ticker symbols
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Combined DataFrame with all assets
    """
    dfs = []
    for ticker in tickers:
        path = DATA_DIR / f"ticker={ticker}" / "data.parquet"
        if not path.exists():
            print(f"Warning: Data not found for {ticker}, skipping")
            continue

        df = pl.read_parquet(path)

        # Normalize column names (some files have 'date' instead of 'timestamp')
        if "date" in df.columns and "timestamp" not in df.columns:
            df = df.rename({"date": "timestamp"})

        # Normalize timestamp to microseconds (some files have ns, others us)
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")).alias("timestamp"))

        df = df.with_columns(pl.lit(ticker).alias("asset"))

        if start_date:
            df = df.filter(pl.col("timestamp") >= datetime.fromisoformat(start_date))
        if end_date:
            df = df.filter(pl.col("timestamp") <= datetime.fromisoformat(end_date))

        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data found for any ticker")

    return pl.concat(dfs).sort(["timestamp", "asset"])


class CrossAssetMomentumStrategy(Strategy):
    """Momentum-based tactical allocation across multiple assets.

    Strategy:
    - Calculate trailing momentum (returns over lookback period) for each asset
    - Go long assets with positive momentum, weighted by relative strength
    - Rebalance monthly
    """

    def __init__(
        self,
        assets: list[str],
        lookback_period: int = 63,  # ~3 months
        rebalance_frequency: int = 21,  # Monthly
        max_positions: int = 3,  # Hold top N assets
        position_size_pct: float = 0.95,
    ):
        """Initialize cross-asset momentum strategy.

        Args:
            assets: List of asset symbols to trade
            lookback_period: Period for momentum calculation
            rebalance_frequency: Bars between rebalances
            max_positions: Maximum number of simultaneous positions
            position_size_pct: Total portfolio allocation
        """
        self.assets = assets
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct

        # Track price history per asset
        self.price_history: dict[str, list[float]] = {a: [] for a in assets}
        self.bar_count = 0

    def _calculate_momentum(self, prices: list[float]) -> float:
        """Calculate momentum as percentage return over lookback."""
        if len(prices) < self.lookback_period:
            return 0.0
        return (prices[-1] / prices[-self.lookback_period] - 1) * 100

    def on_data(self, timestamp, data, context, broker):
        """Process bar and potentially rebalance."""
        self.bar_count += 1

        # Update price history for all assets
        for asset in self.assets:
            if asset in data:
                close = data[asset]["close"]
                self.price_history[asset].append(close)
                # Keep only needed history
                if len(self.price_history[asset]) > self.lookback_period + 10:
                    self.price_history[asset].pop(0)

        # Wait for enough history
        if self.bar_count < self.lookback_period:
            return

        # Only rebalance periodically
        if self.bar_count % self.rebalance_frequency != 0:
            return

        # Calculate momentum for each asset
        momentum = {}
        for asset in self.assets:
            if asset in data and len(self.price_history[asset]) >= self.lookback_period:
                momentum[asset] = self._calculate_momentum(self.price_history[asset])

        # Rank assets by momentum and select top N with positive momentum
        ranked = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        selected = [(a, m) for a, m in ranked if m > 0][: self.max_positions]

        # Calculate target positions
        equity = broker.get_account_value()
        total_allocation = self.position_size_pct
        per_asset_allocation = total_allocation / len(selected) if selected else 0

        # Exit positions not in selected list
        for asset in self.assets:
            pos = broker.get_position(asset)
            if pos and pos.quantity > 0 and asset not in [a for a, _ in selected]:
                broker.submit_order(asset, pos.quantity, OrderSide.SELL)

        # Enter/adjust positions in selected assets
        for asset, _mom in selected:
            if asset not in data:
                continue

            price = data[asset]["close"]
            target_value = equity * per_asset_allocation
            target_qty = target_value / price

            pos = broker.get_position(asset)
            current_qty = pos.quantity if pos else 0

            diff = target_qty - current_qty
            if abs(diff) > 0.01:  # Minimum trade threshold
                if diff > 0:
                    broker.submit_order(asset, diff, OrderSide.BUY)
                else:
                    broker.submit_order(asset, abs(diff), OrderSide.SELL)


def main():
    """Run multi-asset momentum strategy and analyze portfolio."""
    print("=" * 70)
    print("Multi-Asset Portfolio with Diagnostic Analysis")
    print("=" * 70)

    # Load multi-asset data
    print(f"\nLoading data for: {', '.join(ASSET_UNIVERSE)}")
    df = load_multi_asset_data(ASSET_UNIVERSE, start_date="2010-01-01", end_date="2023-12-31")

    # Show data summary
    assets_loaded = df["asset"].unique().to_list()
    print(f"Assets loaded: {assets_loaded}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total bars: {len(df)}")

    # Create data feed and strategy
    feed = DataFeed(prices_df=df)
    strategy = CrossAssetMomentumStrategy(
        assets=assets_loaded,
        lookback_period=63,  # 3-month momentum
        rebalance_frequency=21,  # Monthly rebalance
        max_positions=3,
    )

    # Run backtest
    print("\nRunning backtest...")
    initial_cash = 100_000.0
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=initial_cash,
        execution_mode=ExecutionMode.NEXT_BAR,
    )
    result = engine.run()

    # Basic results
    print("\n" + "=" * 70)
    print("PORTFOLIO PERFORMANCE")
    print("=" * 70)
    print(f"Initial Capital:  ${initial_cash:,.2f}")
    print(f"Final Value:      ${result['final_value']:,.2f}")
    print(f"Total Return:     {result['total_return']:.2%}")
    print(f"Sharpe Ratio:     {result['sharpe']:.2f}")
    print(f"Max Drawdown:     {result['max_drawdown_pct']:.2%}")

    # Analyze with BacktestAnalyzer
    print("\n" + "=" * 70)
    print("PORTFOLIO TRADE ANALYSIS")
    print("=" * 70)

    analyzer = BacktestAnalyzer(engine)
    stats = analyzer.trade_statistics()
    print(stats.summary())

    # Get trades as DataFrame for portfolio analysis
    trades_df = analyzer.get_trades_dataframe()

    if len(trades_df) > 0:
        # Per-asset breakdown
        print("\n" + "-" * 70)
        print("PER-ASSET BREAKDOWN")
        print("-" * 70)

        asset_stats = (
            trades_df.group_by("asset")
            .agg(
                pl.len().alias("trades"),
                (pl.col("pnl") > 0).sum().alias("winners"),
                pl.col("pnl").sum().alias("total_pnl"),
                pl.col("pnl").mean().alias("avg_pnl"),
                pl.col("pnl_percent").mean().alias("avg_return"),
                pl.col("bars_held").mean().alias("avg_holding"),
            )
            .with_columns((pl.col("winners") / pl.col("trades") * 100).alias("win_rate"))
            .sort("total_pnl", descending=True)
        )

        print(asset_stats)

        # Best/worst performers
        print("\n" + "-" * 70)
        print("BEST AND WORST TRADES ACROSS PORTFOLIO")
        print("-" * 70)

        print("\nTop 5 Best Trades:")
        best = trades_df.sort("pnl", descending=True).head(5)
        for row in best.iter_rows(named=True):
            print(
                f"  {row['asset']}: {row['entry_time'].date()} → {row['exit_time'].date()}: "
                f"${row['pnl']:,.2f} ({row['pnl_percent']:.2%})"
            )

        print("\nTop 5 Worst Trades:")
        worst = trades_df.sort("pnl").head(5)
        for row in worst.iter_rows(named=True):
            print(
                f"  {row['asset']}: {row['entry_time'].date()} → {row['exit_time'].date()}: "
                f"${row['pnl']:,.2f} ({row['pnl_percent']:.2%})"
            )

        # Temporal analysis - by year
        print("\n" + "-" * 70)
        print("PERFORMANCE BY YEAR")
        print("-" * 70)

        yearly = (
            trades_df.with_columns(pl.col("exit_time").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("total_pnl"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
                pl.col("pnl_percent").mean().alias("avg_return"),
            )
            .sort("year")
        )

        print(yearly)

        # Correlation analysis (simplified - direction vs asset)
        print("\n" + "-" * 70)
        print("TRADE DIRECTION ANALYSIS")
        print("-" * 70)

        direction_asset = (
            trades_df.group_by(["asset", "direction"])
            .agg(
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("total_pnl"),
            )
            .sort(["asset", "direction"])
        )

        print(direction_asset)

    # Export for diagnostic library
    print("\n" + "=" * 70)
    print("DIAGNOSTIC LIBRARY INTEGRATION")
    print("=" * 70)
    print("""
For advanced portfolio analysis with ml4t.diagnostic:

    from ml4t.backtest.analysis import BacktestAnalyzer
    from ml4t.diagnostic.evaluation import TradeAnalysis

    analyzer = BacktestAnalyzer(engine)

    # Get trade records in diagnostic format
    records = analyzer.get_trade_records()

    # Use TradeAnalysis for deeper insights
    diagnostic = TradeAnalysis(records)

    # Analyze by regime
    high_vol_trades = diagnostic.filter_trades(
        lambda t: t['metadata']['volatility'] > threshold
    )

    # Cluster trade failures
    patterns = diagnostic.cluster_failures(n_clusters=5)

    # Generate actionable hypotheses
    hypotheses = diagnostic.generate_improvement_hypotheses()
""")

    # Export trades
    if len(trades_df) > 0:
        output_path = Path(__file__).parent / "portfolio_trades.parquet"
        trades_df.write_parquet(output_path)
        print(f"\nTrades exported to: {output_path}")

    # Final positions
    print("\n" + "-" * 70)
    print("FINAL POSITIONS")
    print("-" * 70)
    for asset, pos in engine.broker.positions.items():
        if pos.quantity > 0:
            print(f"  {asset}: {pos.quantity:.2f} shares @ ${pos.entry_price:.2f}")


if __name__ == "__main__":
    main()
