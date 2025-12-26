"""Complete Workflow: Strategy vs Benchmark with Statistical Testing.

This example demonstrates a production-ready workflow:
1. Run strategy backtest
2. Run buy-and-hold benchmark
3. Compare performance metrics
4. Statistical significance testing (Sharpe comparison)
5. Full integration with ml4t.diagnostic

Uses real ETF data from ~/ml4t/data/etfs/

Usage:
    uv run python examples/analysis/05_benchmark_comparison.py
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ml4t.backtest import DataFeed, Engine, ExecutionMode, OrderSide, Strategy
from ml4t.backtest.analysis import BacktestAnalyzer

# Path to ETF data
DATA_DIR = Path.home() / "ml4t" / "data" / "etfs" / "ohlcv_1d"


def load_etf_data(
    ticker: str, start_date: str | None = None, end_date: str | None = None
) -> pl.DataFrame:
    """Load ETF OHLCV data."""
    path = DATA_DIR / f"ticker={ticker}" / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")

    df = pl.read_parquet(path)
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename({"date": "timestamp"})
    df = df.with_columns(pl.lit(ticker).alias("asset"))

    if start_date:
        df = df.filter(pl.col("timestamp") >= datetime.fromisoformat(start_date))
    if end_date:
        df = df.filter(pl.col("timestamp") <= datetime.fromisoformat(end_date))

    return df.sort("timestamp")


# ==============================================================================
# Strategy 1: Buy and Hold (Benchmark)
# ==============================================================================


class BuyAndHoldStrategy(Strategy):
    """Simple buy-and-hold benchmark."""

    def __init__(self, asset: str, position_pct: float = 0.99):
        self.asset = asset
        self.position_pct = position_pct
        self.invested = False

    def on_data(self, timestamp, data, context, broker):
        if not self.invested and self.asset in data:
            equity = broker.get_account_value()
            price = data[self.asset]["close"]
            qty = (equity * self.position_pct) / price
            broker.submit_order(self.asset, qty, OrderSide.BUY)
            self.invested = True


# ==============================================================================
# Strategy 2: Trend Following (Active Strategy)
# ==============================================================================


class DualMovingAverageStrategy(Strategy):
    """Classic dual moving average crossover strategy.

    - Long when fast MA > slow MA
    - Exit when fast MA < slow MA
    """

    def __init__(
        self,
        asset: str,
        fast_period: int = 20,
        slow_period: int = 50,
        position_pct: float = 0.95,
    ):
        self.asset = asset
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_pct = position_pct
        self.price_history: list[float] = []

    def on_data(self, timestamp, data, context, broker):
        if self.asset not in data:
            return

        close = data[self.asset]["close"]
        self.price_history.append(close)

        if len(self.price_history) > self.slow_period:
            self.price_history.pop(0)

        if len(self.price_history) < self.slow_period:
            return

        # Calculate MAs
        fast_ma = sum(self.price_history[-self.fast_period :]) / self.fast_period
        slow_ma = sum(self.price_history) / len(self.price_history)

        # Get position
        pos = broker.get_position(self.asset)
        has_position = pos is not None and pos.quantity > 0

        # Trading logic
        if fast_ma > slow_ma and not has_position:
            # Bullish crossover - enter
            equity = broker.get_account_value()
            qty = (equity * self.position_pct) / close
            broker.submit_order(self.asset, qty, OrderSide.BUY)

        elif fast_ma < slow_ma and has_position:
            # Bearish crossover - exit
            broker.submit_order(self.asset, pos.quantity, OrderSide.SELL)


# ==============================================================================
# Statistical Testing Functions
# ==============================================================================


def compute_sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)


def sharpe_ratio_test(
    returns1: np.ndarray,
    returns2: np.ndarray,
    periods_per_year: int = 252,
) -> dict:
    """Test if Sharpe ratios are significantly different.

    Uses the Ledoit-Wolf (2008) methodology for comparing Sharpe ratios.
    Simplified implementation using paired t-test on Sharpe differences.

    Returns:
        Dict with test statistics and p-value
    """
    # Compute Sharpe ratios
    sr1 = compute_sharpe_ratio(returns1, periods_per_year)
    sr2 = compute_sharpe_ratio(returns2, periods_per_year)

    # Paired difference in returns (simplified test)
    # More sophisticated: use HAC standard errors
    n = min(len(returns1), len(returns2))
    diff = returns1[:n] - returns2[:n]

    # Test if mean difference is significant
    t_stat, p_value = stats.ttest_1samp(diff, 0)

    return {
        "sharpe_strategy": sr1,
        "sharpe_benchmark": sr2,
        "sharpe_difference": sr1 - sr2,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
    }


def compute_drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    """Compute drawdown series from equity curve."""
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return drawdowns


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum drawdown."""
    dd = compute_drawdown_series(equity_curve)
    return float(np.min(dd))


def calmar_ratio(
    returns: np.ndarray, equity_curve: np.ndarray, periods_per_year: int = 252
) -> float:
    """Compute Calmar ratio (annualized return / max drawdown)."""
    ann_return = np.mean(returns) * periods_per_year
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return float("inf") if ann_return > 0 else 0.0
    return ann_return / mdd


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Compute Sortino ratio (using downside deviation)."""
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return float("inf") if np.mean(returns) > 0 else 0.0
    downside_std = np.std(negative_returns)
    if downside_std == 0:
        return float("inf") if np.mean(returns) > 0 else 0.0
    return np.mean(returns) / downside_std * np.sqrt(periods_per_year)


# ==============================================================================
# Main Workflow
# ==============================================================================


def main():
    """Run complete backtest workflow with benchmark comparison."""
    print("=" * 70)
    print("Complete Workflow: Strategy vs Benchmark Comparison")
    print("=" * 70)

    # Configuration
    ASSET = "SPY"
    START_DATE = "2010-01-01"
    END_DATE = "2023-12-31"
    INITIAL_CASH = 100_000.0

    # Load data
    print(f"\nLoading {ASSET} data...")
    df = load_etf_data(ASSET, start_date=START_DATE, end_date=END_DATE)
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total bars: {len(df)}")

    # Create price data for feeds
    prices_df = df.select(["timestamp", "asset", "open", "high", "low", "close", "volume"])

    # === Run Benchmark (Buy and Hold) ===
    print("\n" + "-" * 70)
    print("Running BENCHMARK (Buy and Hold)...")
    print("-" * 70)

    benchmark_strategy = BuyAndHoldStrategy(ASSET)
    benchmark_engine = Engine(
        feed=DataFeed(prices_df=prices_df),  # Fresh feed
        strategy=benchmark_strategy,
        initial_cash=INITIAL_CASH,
        execution_mode=ExecutionMode.NEXT_BAR,
    )
    benchmark_result = benchmark_engine.run()

    print(f"Final Value:   ${benchmark_result['final_value']:,.2f}")
    print(f"Total Return:  {benchmark_result['total_return']:.2%}")
    print(f"Sharpe Ratio:  {benchmark_result['sharpe']:.2f}")

    # === Run Strategy (Dual MA) ===
    print("\n" + "-" * 70)
    print("Running STRATEGY (Dual Moving Average)...")
    print("-" * 70)

    strategy = DualMovingAverageStrategy(ASSET, fast_period=20, slow_period=50)
    strategy_engine = Engine(
        feed=DataFeed(prices_df=prices_df),  # Fresh feed
        strategy=strategy,
        initial_cash=INITIAL_CASH,
        execution_mode=ExecutionMode.NEXT_BAR,
    )
    strategy_result = strategy_engine.run()

    print(f"Final Value:   ${strategy_result['final_value']:,.2f}")
    print(f"Total Return:  {strategy_result['total_return']:.2%}")
    print(f"Sharpe Ratio:  {strategy_result['sharpe']:.2f}")

    # === Comparative Analysis ===
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    # Get returns series using BacktestAnalyzer
    benchmark_analyzer = BacktestAnalyzer(benchmark_engine)
    strategy_analyzer = BacktestAnalyzer(strategy_engine)

    benchmark_returns = benchmark_analyzer.get_returns_series().to_numpy()
    strategy_returns = strategy_analyzer.get_returns_series().to_numpy()

    benchmark_equity = np.array(benchmark_analyzer.equity_history)
    strategy_equity = np.array(strategy_analyzer.equity_history)

    # Performance metrics comparison
    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS")
    print("-" * 70)
    print(f"{'Metric':<25} {'Benchmark':>15} {'Strategy':>15} {'Delta':>15}")
    print("-" * 70)

    metrics = [
        ("Total Return", benchmark_result["total_return"], strategy_result["total_return"]),
        ("Sharpe Ratio", benchmark_result["sharpe"], strategy_result["sharpe"]),
        ("Sortino Ratio", sortino_ratio(benchmark_returns), sortino_ratio(strategy_returns)),
        ("Max Drawdown", max_drawdown(benchmark_equity), max_drawdown(strategy_equity)),
        (
            "Calmar Ratio",
            calmar_ratio(benchmark_returns, benchmark_equity),
            calmar_ratio(strategy_returns, strategy_equity),
        ),
    ]

    for name, bench_val, strat_val in metrics:
        delta = strat_val - bench_val
        print(f"{name:<25} {bench_val:>15.2%} {strat_val:>15.2%} {delta:>+15.2%}")

    # Statistical significance test
    print("\n" + "-" * 70)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("-" * 70)

    test_result = sharpe_ratio_test(strategy_returns, benchmark_returns)

    print(f"Strategy Sharpe:   {test_result['sharpe_strategy']:.3f}")
    print(f"Benchmark Sharpe:  {test_result['sharpe_benchmark']:.3f}")
    print(f"Sharpe Difference: {test_result['sharpe_difference']:+.3f}")
    print(f"t-statistic:       {test_result['t_statistic']:.3f}")
    print(f"p-value:           {test_result['p_value']:.4f}")
    print(f"Significant (5%):  {'YES' if test_result['significant_5pct'] else 'NO'}")
    print(f"Significant (1%):  {'YES' if test_result['significant_1pct'] else 'NO'}")

    # Trade statistics (strategy only)
    print("\n" + "-" * 70)
    print("STRATEGY TRADE STATISTICS")
    print("-" * 70)

    # Reuse strategy_analyzer created earlier
    stats = strategy_analyzer.trade_statistics()

    print(f"Total Trades:   {stats.n_trades}")
    print(f"Win Rate:       {stats.win_rate:.2%}")
    print(f"Profit Factor:  {stats.profit_factor:.2f}" if stats.profit_factor else "N/A")
    print(f"Avg Trade P&L:  ${stats.avg_pnl:,.2f}")
    print(f"Expectancy:     ${stats.expectancy:,.2f}")

    # Best/worst trades
    trades_df = strategy_analyzer.get_trades_dataframe()
    if len(trades_df) > 0:
        print("\nBest Trade:")
        best = trades_df.sort("pnl", descending=True).head(1).row(0, named=True)
        print(
            f"  {best['entry_time'].date()} → {best['exit_time'].date()}: "
            f"${best['pnl']:,.2f} ({best['pnl_percent']:.2%})"
        )

        print("\nWorst Trade:")
        worst = trades_df.sort("pnl").head(1).row(0, named=True)
        print(
            f"  {worst['entry_time'].date()} → {worst['exit_time'].date()}: "
            f"${worst['pnl']:,.2f} ({worst['pnl_percent']:.2%})"
        )

    # Rolling performance analysis
    print("\n" + "-" * 70)
    print("ROLLING PERFORMANCE (Annual)")
    print("-" * 70)

    # Just show yearly summary using trades
    if len(trades_df) > 0:
        yearly = (
            trades_df.with_columns(pl.col("exit_time").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("pnl"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
            )
            .sort("year")
        )
        print(yearly)

    # Integration guidance
    print("\n" + "=" * 70)
    print("DIAGNOSTIC LIBRARY INTEGRATION")
    print("=" * 70)
    print("""
For advanced statistical analysis with ml4t.diagnostic:

    from ml4t.diagnostic.evaluation import stats
    from ml4t.diagnostic.evaluation.stats import (
        deflated_sharpe_ratio,
        probabilistic_sharpe_ratio,
        rademacher_anti_serum,
    )

    # Get returns for analysis
    strategy_returns = analyzer.get_returns_series()

    # 1. Deflated Sharpe Ratio (multiple testing correction)
    #    Accounts for number of strategies tested
    dsr = deflated_sharpe_ratio(
        sharpe_observed=strategy_result['sharpe'],
        n_trials=10,  # Number of parameter combinations tried
        returns=strategy_returns
    )

    # 2. Probabilistic Sharpe Ratio
    #    P(true SR > benchmark | observed data)
    psr = probabilistic_sharpe_ratio(
        sharpe_observed=strategy_result['sharpe'],
        sharpe_benchmark=benchmark_result['sharpe'],
        returns=strategy_returns
    )

    # 3. Rademacher Anti-Serum (backtest overfitting detection)
    #    Tests if strategy is robust to random sign flips
    ras = rademacher_anti_serum(
        strategy_returns=strategy_returns,
        n_simulations=1000
    )

    print(f"Deflated Sharpe:     {dsr:.3f}")
    print(f"Prob. Sharpe Ratio:  {psr:.1%}")
    print(f"RAS Score:           {ras:.3f}")
""")

    # Export results
    output_dir = Path(__file__).parent
    if len(trades_df) > 0:
        trades_df.write_parquet(output_dir / "dma_strategy_trades.parquet")
        print(f"\nTrades exported to: {output_dir / 'dma_strategy_trades.parquet'}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Strategy vs Benchmark ({START_DATE} to {END_DATE})

    Benchmark (Buy & Hold):
      - Total Return: {benchmark_result['total_return']:.2%}
      - Sharpe Ratio: {benchmark_result['sharpe']:.2f}

    Strategy (Dual MA 20/50):
      - Total Return: {strategy_result['total_return']:.2%}
      - Sharpe Ratio: {strategy_result['sharpe']:.2f}
      - Win Rate:     {stats.win_rate:.2%}
      - Trades:       {stats.n_trades}

    Statistical Significance:
      - Sharpe Difference: {test_result['sharpe_difference']:+.3f}
      - p-value:           {test_result['p_value']:.4f}
      - Significant (5%):  {'YES' if test_result['significant_5pct'] else 'NO'}

    Verdict: {'Strategy outperforms' if strategy_result['total_return'] > benchmark_result['total_return'] else 'Benchmark outperforms'}
             {'(statistically significant)' if test_result['significant_5pct'] else '(not statistically significant)'}
""")


if __name__ == "__main__":
    main()
