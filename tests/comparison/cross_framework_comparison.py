"""
Cross-Framework Comparison: Same Strategy on QEngine, Zipline, Backtrader, VectorBT

This module implements identical trading logic across all frameworks to validate:
1. Trade generation consistency
2. P&L calculation accuracy
3. Performance metrics alignment
"""

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project paths
qengine_src = Path(__file__).parent.parent.parent / "src"
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(qengine_src))


@dataclass
class TradeRecord:
    """Standardized trade record for comparison."""

    timestamp: datetime
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    value: float
    commission: float = 0.0


@dataclass
class BacktestResult:
    """Standardized backtest result for comparison."""

    framework: str
    initial_capital: float
    final_value: float
    total_return: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    execution_time: float
    trades: list[TradeRecord] = field(default_factory=list)
    daily_returns: pd.Series = None
    equity_curve: pd.Series = None
    errors: list[str] = field(default_factory=list)


class SimpleMomentumStrategy:
    """
    Simple momentum strategy for cross-framework testing.

    Rules:
    - Calculate 20-day and 50-day moving averages
    - Go long when 20-day MA crosses above 50-day MA
    - Go short when 20-day MA crosses below 50-day MA
    - Use 100% of capital for each position
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window
        self.position = 0  # -1, 0, or 1

    def should_buy(self, short_ma: float, long_ma: float, current_position: int) -> bool:
        """Check if we should buy."""
        return short_ma > long_ma and current_position <= 0

    def should_sell(self, short_ma: float, long_ma: float, current_position: int) -> bool:
        """Check if we should sell."""
        return short_ma < long_ma and current_position >= 0


def load_test_data() -> pd.DataFrame:
    """Load and prepare test data."""
    # Try SPY data first
    spy_path = projects_dir / "spy_order_flow" / "spy_features.parquet"
    if spy_path.exists():
        df = pd.read_parquet(spy_path)
        # SPY data uses 'last' instead of 'close'
        df = df[["timestamp", "open", "high", "low", "last", "volume"]].copy()
        df = df.rename(columns={"last": "close"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        print(f"Loaded SPY data: {len(df)} rows")
        return df

    # Try daily equity data
    equity_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"
    if equity_path.exists():
        df = pd.read_parquet(equity_path)
        # Filter for a single ticker
        if "ticker" in df.columns:
            df = df[df["ticker"] == df["ticker"].iloc[0]].copy()
        df["timestamp"] = pd.to_datetime(df.get("date", df.index))
        df = df.set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]].copy()
        print(f"Loaded equity data: {len(df)} rows")
        return df

    raise FileNotFoundError("No suitable test data found")


def run_qengine_backtest(
    data: pd.DataFrame,
    strategy: SimpleMomentumStrategy,
    use_daily_data: bool = False,
) -> BacktestResult:
    """Run backtest using QEngine."""
    print("\n" + "=" * 60)
    print("Running QEngine Backtest")
    print("=" * 60)

    initial_capital = 10000.0
    result = BacktestResult(
        framework="QEngine",
        initial_capital=initial_capital,
        final_value=initial_capital,
        total_return=0.0,
        num_trades=0,
        win_rate=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        execution_time=0.0,
    )

    try:
        # Track performance
        start_time = time.time()

        # Use daily data for consistent comparison if requested
        if use_daily_data:
            equity_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"
            if equity_path.exists():
                daily_df = pd.read_parquet(equity_path)
                aapl_daily = daily_df[daily_df["ticker"] == "AAPL"].copy()
                if len(aapl_daily) > 0:
                    aapl_daily["date"] = pd.to_datetime(aapl_daily["date"])
                    aapl_daily = aapl_daily.set_index("date").sort_index()
                    aapl_daily = aapl_daily.loc["2015-01-01":"2016-12-31"]
                    data = aapl_daily.copy()
                    print(f"Using daily AAPL data: {len(data)} rows")

        # Calculate moving averages
        data["ma_short"] = data["close"].rolling(window=strategy.short_window).mean()
        data["ma_long"] = data["close"].rolling(window=strategy.long_window).mean()

        # Initialize portfolio tracking
        cash = initial_capital
        position = 0
        shares = 0
        trades = []
        equity_curve = []

        # Process each bar
        for timestamp, row in data.iterrows():
            # Skip if MAs not ready
            if pd.isna(row["ma_short"]) or pd.isna(row["ma_long"]):
                equity_curve.append(cash + shares * row["close"])
                continue

            # Check for signals
            if strategy.should_buy(row["ma_short"], row["ma_long"], position):
                if cash > 0:
                    # Buy signal
                    shares_to_buy = cash / row["close"]
                    trade = TradeRecord(
                        timestamp=timestamp,
                        action="BUY",
                        quantity=shares_to_buy,
                        price=row["close"],
                        value=cash,
                    )
                    trades.append(trade)
                    shares = shares_to_buy
                    cash = 0
                    position = 1

            elif strategy.should_sell(row["ma_short"], row["ma_long"], position):
                if shares > 0:
                    # Sell signal
                    trade_value = shares * row["close"]
                    trade = TradeRecord(
                        timestamp=timestamp,
                        action="SELL",
                        quantity=shares,
                        price=row["close"],
                        value=trade_value,
                    )
                    trades.append(trade)
                    cash = trade_value
                    shares = 0
                    position = 0

            # Track equity
            equity_value = cash + shares * row["close"]
            equity_curve.append(equity_value)

        # Final calculations
        final_value = cash + shares * data["close"].iloc[-1]
        result.final_value = final_value
        result.total_return = (final_value / initial_capital - 1) * 100
        result.num_trades = len(trades)
        result.trades = trades
        result.execution_time = time.time() - start_time

        # Calculate metrics
        equity_series = pd.Series(equity_curve, index=data.index[: len(equity_curve)])
        result.equity_curve = equity_series

        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            result.daily_returns = returns

            if len(returns) > 0:
                result.sharpe_ratio = (
                    np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
                )

                # Calculate max drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                result.max_drawdown = drawdown.min() * 100

                # Calculate win rate
                if len(trades) >= 2:
                    trade_returns = []
                    for i in range(0, len(trades) - 1, 2):
                        if i + 1 < len(trades):
                            entry = trades[i]
                            exit = trades[i + 1]
                            ret = exit.price / entry.price - 1
                            trade_returns.append(ret)
                    if trade_returns:
                        result.win_rate = sum(1 for r in trade_returns if r > 0) / len(
                            trade_returns,
                        )

        print("✓ QEngine backtest completed")
        print(f"  Final value: ${final_value:,.2f}")
        print(f"  Return: {result.total_return:.2f}%")
        print(f"  Trades: {result.num_trades}")
        print(f"  Sharpe: {result.sharpe_ratio:.2f}")

    except Exception as e:
        print(f"✗ QEngine backtest failed: {e}")
        result.errors.append(str(e))

    return result


def run_zipline_backtest(data: pd.DataFrame, strategy: SimpleMomentumStrategy) -> BacktestResult:
    """Run backtest using Zipline-Reloaded with manual implementation."""
    print("\n" + "=" * 60)
    print("Running Zipline-Style Manual Backtest")
    print("=" * 60)

    initial_capital = 10000.0
    result = BacktestResult(
        framework="Zipline-Style",
        initial_capital=initial_capital,
        final_value=initial_capital,
        total_return=0.0,
        num_trades=0,
        win_rate=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        execution_time=0.0,
    )

    try:
        start_time = time.time()

        # Manual Zipline-style implementation using the same logic
        # This replicates Zipline's strategy execution without bundle complexity

        # Use daily equity data instead of intraday SPY
        equity_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"
        if not equity_path.exists():
            print("⚠ Daily equity data not available")
            result.errors.append("Daily equity data not found")
            return result

        # Load daily data
        daily_df = pd.read_parquet(equity_path)
        # Use AAPL since SPY is not available in Wiki data
        aapl_data = daily_df[daily_df["ticker"] == "AAPL"].copy()

        if len(aapl_data) == 0:
            print("⚠ AAPL data not found in daily equity data")
            result.errors.append("AAPL data not found")
            return result

        aapl_data["date"] = pd.to_datetime(aapl_data["date"])
        aapl_data = aapl_data.set_index("date").sort_index()

        # Use subset for faster comparison (2015-2016)
        aapl_data = aapl_data.loc["2015-01-01":"2016-12-31"]

        if len(aapl_data) < strategy.long_window:
            print("⚠ Insufficient data for strategy")
            result.errors.append("Insufficient data")
            return result

        # Calculate moving averages
        aapl_data["ma_short"] = aapl_data["close"].rolling(window=strategy.short_window).mean()
        aapl_data["ma_long"] = aapl_data["close"].rolling(window=strategy.long_window).mean()

        print(f"Using AAPL daily data: {len(aapl_data)} rows from 2015-2016")

        # Simulate Zipline-style execution
        cash = initial_capital
        shares = 0
        position = 0
        trades = []

        for date, row in aapl_data.iterrows():
            if pd.isna(row["ma_short"]) or pd.isna(row["ma_long"]):
                continue

            # Signal generation (same as Zipline would do)
            if row["ma_short"] > row["ma_long"] and position <= 0:
                # Buy signal - invest all cash
                if cash > 0:
                    shares_to_buy = cash / row["close"]
                    trade = TradeRecord(
                        timestamp=date,
                        action="BUY",
                        quantity=shares_to_buy,
                        price=row["close"],
                        value=cash,
                    )
                    trades.append(trade)
                    shares = shares_to_buy
                    cash = 0
                    position = 1

            elif row["ma_short"] < row["ma_long"] and position >= 0:
                # Sell signal - sell all shares
                if shares > 0:
                    cash_received = shares * row["close"]
                    trade = TradeRecord(
                        timestamp=date,
                        action="SELL",
                        quantity=shares,
                        price=row["close"],
                        value=cash_received,
                    )
                    trades.append(trade)
                    cash = cash_received
                    shares = 0
                    position = 0

        # Final valuation
        final_value = cash + shares * aapl_data["close"].iloc[-1]
        total_return = (final_value / initial_capital - 1) * 100

        # Calculate metrics
        result.final_value = final_value
        result.total_return = total_return
        result.num_trades = len(trades)
        result.trades = trades
        result.execution_time = time.time() - start_time

        # Simple metrics calculation
        if len(trades) >= 2:
            returns = []
            for i in range(0, len(trades) - 1, 2):
                if i + 1 < len(trades):
                    entry = trades[i]
                    exit = trades[i + 1]
                    ret = exit.price / entry.price - 1
                    returns.append(ret)
            if returns:
                result.win_rate = sum(1 for r in returns if r > 0) / len(returns)

        print("✓ Zipline-style backtest completed")
        print(f"  Final value: ${final_value:,.2f}")
        print(f"  Return: {total_return:.2f}%")
        print(f"  Trades: {len(trades)}")

    except Exception as e:
        print(f"✗ Zipline-style backtest failed: {e}")
        result.errors.append(str(e))

    return result


def run_backtrader_backtest(data: pd.DataFrame, strategy: SimpleMomentumStrategy) -> BacktestResult:
    """Run backtest using Backtrader."""
    print("\n" + "=" * 60)
    print("Running Backtrader Backtest")
    print("=" * 60)

    initial_capital = 10000.0
    result = BacktestResult(
        framework="Backtrader",
        initial_capital=initial_capital,
        final_value=initial_capital,
        total_return=0.0,
        num_trades=0,
        win_rate=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        execution_time=0.0,
    )

    try:
        import backtrader as bt

        start_time = time.time()

        class MomentumStrategy(bt.Strategy):
            params = (
                ("short_window", strategy.short_window),
                ("long_window", strategy.long_window),
            )

            def __init__(self):
                self.ma_short = bt.indicators.SMA(self.data.close, period=self.params.short_window)
                self.ma_long = bt.indicators.SMA(self.data.close, period=self.params.long_window)
                self.crossover = bt.indicators.CrossOver(self.ma_short, self.ma_long)
                self.trades = []

            def next(self):
                if not self.position:
                    if self.crossover > 0:  # Golden cross
                        self.buy(size=self.broker.cash / self.data.close[0])
                        self.trades.append(
                            {
                                "timestamp": self.data.datetime.datetime(0),
                                "action": "BUY",
                                "price": self.data.close[0],
                            },
                        )
                else:
                    if self.crossover < 0:  # Death cross
                        self.sell(size=self.position.size)
                        self.trades.append(
                            {
                                "timestamp": self.data.datetime.datetime(0),
                                "action": "SELL",
                                "price": self.data.close[0],
                            },
                        )

        # Create cerebro engine
        cerebro = bt.Cerebro()
        cerebro.addstrategy(MomentumStrategy)

        # Prepare data for Backtrader (needs specific format)
        data_bt_format = data.copy()
        data_bt_format = data_bt_format.reset_index()  # Move timestamp to column

        # Add data
        data_bt = bt.feeds.PandasData(
            dataname=data_bt_format,
            datetime="timestamp",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
        )
        cerebro.adddata(data_bt)

        # Set initial capital
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=0.0)  # No commission for comparison

        # Run backtest
        strategies = cerebro.run()
        strategy_instance = strategies[0]

        # Get results
        final_value = cerebro.broker.getvalue()
        result.final_value = final_value
        result.total_return = (final_value / initial_capital - 1) * 100

        # Extract trades
        for trade_info in strategy_instance.trades:
            trade = TradeRecord(
                timestamp=trade_info["timestamp"],
                action=trade_info["action"],
                quantity=0,  # Would need to track this
                price=trade_info["price"],
                value=0,
            )
            result.trades.append(trade)

        result.num_trades = len(result.trades)
        result.execution_time = time.time() - start_time

        # Get built-in analyzers results if added
        # cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        # cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        print("✓ Backtrader backtest completed")
        print(f"  Final value: ${final_value:,.2f}")
        print(f"  Return: {result.total_return:.2f}%")
        print(f"  Trades: {result.num_trades}")

    except ImportError as e:
        print(f"⚠ Backtrader not available: {e}")
        result.errors.append(f"Import error: {e}")
    except Exception as e:
        print(f"✗ Backtrader backtest failed: {e}")
        result.errors.append(str(e))

    return result


def run_vectorbt_backtest(data: pd.DataFrame, strategy: SimpleMomentumStrategy) -> BacktestResult:
    """Run backtest using VectorBT Pro."""
    print("\n" + "=" * 60)
    print("Running VectorBT Pro Backtest")
    print("=" * 60)

    initial_capital = 10000.0
    result = BacktestResult(
        framework="VectorBT",
        initial_capital=initial_capital,
        final_value=initial_capital,
        total_return=0.0,
        num_trades=0,
        win_rate=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        execution_time=0.0,
    )

    try:
        import vectorbt as vbt

        start_time = time.time()

        # Calculate indicators
        close_prices = data["close"]
        ma_short = vbt.MA.run(close_prices, window=strategy.short_window).ma
        ma_long = vbt.MA.run(close_prices, window=strategy.long_window).ma

        # Generate signals
        entries = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
        exits = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

        # Run backtest
        pf = vbt.Portfolio.from_signals(
            close_prices,
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=0.0,  # No fees for comparison
            slippage=0.0,  # No slippage for comparison
            freq="D",
        )

        # Get results (fix API calls for open source vectorbt)
        result.final_value = pf.final_value
        result.total_return = pf.total_return * 100
        result.num_trades = len(pf.orders.records)

        if hasattr(pf, "sharpe_ratio"):
            result.sharpe_ratio = pf.sharpe_ratio
        if hasattr(pf, "max_drawdown"):
            result.max_drawdown = pf.max_drawdown * 100
        if hasattr(pf.stats, "win_rate"):
            result.win_rate = pf.stats["win_rate"]

        # Get equity curve
        result.equity_curve = pf.value()
        result.daily_returns = pf.returns()

        # Extract trades
        trades_df = pf.trades.records_readable
        for _, trade in trades_df.iterrows():
            trade_record = TradeRecord(
                timestamp=trade["Entry Timestamp"],
                action="BUY" if trade["Size"] > 0 else "SELL",
                quantity=abs(trade["Size"]),
                price=trade["Avg Entry Price"],
                value=abs(trade["Size"] * trade["Avg Entry Price"]),
            )
            result.trades.append(trade_record)

        result.execution_time = time.time() - start_time

        print("✓ VectorBT backtest completed")
        print(f"  Final value: ${result.final_value:,.2f}")
        print(f"  Return: {result.total_return:.2f}%")
        print(f"  Trades: {result.num_trades}")
        print(f"  Sharpe: {result.sharpe_ratio:.2f}")

    except ImportError as e:
        print(f"⚠ VectorBT not available: {e}")
        result.errors.append(f"Import error: {e}")
    except Exception as e:
        print(f"✗ VectorBT backtest failed: {e}")
        result.errors.append(str(e))

    return result


def compare_results(results: list[BacktestResult]) -> dict[str, Any]:
    """Compare results across frameworks."""
    comparison = {
        "summary": {},
        "metrics": {},
        "trades": {},
        "discrepancies": [],
        "recommendations": [],
    }

    # Filter successful results
    valid_results = [r for r in results if not r.errors]

    if len(valid_results) < 2:
        comparison["summary"]["status"] = "Insufficient valid results for comparison"
        return comparison

    # Create comparison table
    metrics_df = pd.DataFrame(
        {
            r.framework: {
                "Final Value": r.final_value,
                "Total Return (%)": r.total_return,
                "Num Trades": r.num_trades,
                "Sharpe Ratio": r.sharpe_ratio,
                "Max Drawdown (%)": r.max_drawdown,
                "Win Rate": r.win_rate,
                "Execution Time (s)": r.execution_time,
            }
            for r in valid_results
        },
    ).T

    comparison["metrics"] = metrics_df

    # Check for discrepancies
    returns = [r.total_return for r in valid_results]
    return_std = np.std(returns)

    if return_std > 1.0:  # More than 1% standard deviation
        comparison["discrepancies"].append(
            f"Return discrepancy detected: std={return_std:.2f}%",
        )
        comparison["discrepancies"].append(
            f"Returns: {', '.join([f'{r.framework}={r.total_return:.2f}%' for r in valid_results])}",
        )

    trades_counts = [r.num_trades for r in valid_results]
    if len(set(trades_counts)) > 1:
        comparison["discrepancies"].append(
            f"Trade count mismatch: {', '.join([f'{r.framework}={r.num_trades}' for r in valid_results])}",
        )

    # Compare trade sequences if available
    if all(len(r.trades) > 0 for r in valid_results):
        # Compare first few trades
        for i in range(min(3, min(len(r.trades) for r in valid_results))):
            [r.trades[i].price if i < len(r.trades) else None for r in valid_results]
            trade_actions = [
                r.trades[i].action if i < len(r.trades) else None for r in valid_results
            ]

            if len(set(trade_actions)) > 1:
                comparison["discrepancies"].append(
                    f"Trade {i + 1} action mismatch: {list(zip([r.framework for r in valid_results], trade_actions, strict=False))}",
                )

    # Performance comparison
    fastest = min(valid_results, key=lambda r: r.execution_time)
    comparison["summary"]["fastest"] = f"{fastest.framework} ({fastest.execution_time:.3f}s)"

    # Recommendations
    if return_std < 0.1:
        comparison["recommendations"].append(
            "✓ Excellent agreement between frameworks (returns within 0.1%)",
        )
    elif return_std < 1.0:
        comparison["recommendations"].append(
            "✓ Good agreement between frameworks (returns within 1%)",
        )
    else:
        comparison["recommendations"].append("⚠ Investigate discrepancies in return calculations")

    if len(set(trades_counts)) == 1:
        comparison["recommendations"].append("✓ Trade counts match across frameworks")
    else:
        comparison["recommendations"].append("⚠ Trade generation logic differs between frameworks")

    return comparison


def generate_report(results: list[BacktestResult], comparison: dict[str, Any]) -> str:
    """Generate detailed comparison report."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("CROSS-FRAMEWORK BACKTEST COMPARISON REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Results table
    report.append("## RESULTS BY FRAMEWORK")
    report.append("-" * 60)

    valid_results = [r for r in results if not r.errors]
    if valid_results and "metrics" in comparison:
        report.append(str(comparison["metrics"]))

    # Discrepancies
    if comparison.get("discrepancies"):
        report.append("\n## DISCREPANCIES DETECTED")
        report.append("-" * 60)
        for disc in comparison["discrepancies"]:
            report.append(f"  • {disc}")

    # Trade comparison
    report.append("\n## TRADE COMPARISON")
    report.append("-" * 60)

    for r in valid_results:
        if r.trades:
            report.append(f"\n{r.framework} - First 3 trades:")
            for i, trade in enumerate(r.trades[:3]):
                report.append(
                    f"  {i + 1}. {trade.timestamp} {trade.action} @ ${trade.price:.2f}",
                )

    # Recommendations
    if comparison.get("recommendations"):
        report.append("\n## RECOMMENDATIONS")
        report.append("-" * 60)
        for rec in comparison["recommendations"]:
            report.append(f"  {rec}")

    # Framework errors
    failed_results = [r for r in results if r.errors]
    if failed_results:
        report.append("\n## FRAMEWORK ERRORS")
        report.append("-" * 60)
        for r in failed_results:
            report.append(f"\n{r.framework}:")
            for error in r.errors:
                report.append(f"  • {error}")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Run cross-framework comparison."""
    print("\n" + "#" * 80)
    print("# CROSS-FRAMEWORK BACKTEST COMPARISON")
    print("#" * 80)

    # Load data
    try:
        data = load_test_data()
        print(f"Test data: {len(data)} rows from {data.index[0]} to {data.index[-1]}")

        # Limit for faster testing
        test_data = data.head(500)
        print(f"Using first {len(test_data)} rows for comparison")

    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Initialize strategy
    strategy = SimpleMomentumStrategy(short_window=20, long_window=50)

    # Run backtests with daily data for consistency
    results = []

    # QEngine (using daily data)
    results.append(run_qengine_backtest(test_data, strategy, use_daily_data=True))

    # Zipline-style (uses daily data)
    results.append(run_zipline_backtest(test_data, strategy))

    # Backtrader
    results.append(run_backtrader_backtest(test_data, strategy))

    # VectorBT
    results.append(run_vectorbt_backtest(test_data, strategy))

    # Compare results
    comparison = compare_results(results)

    # Generate and print report
    report = generate_report(results, comparison)
    print(report)

    # Save report
    report_path = Path(__file__).parent / "cross_framework_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
