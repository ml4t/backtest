"""
Strategy Validation Runner

This script validates that different backtesting frameworks produce identical results
when given the SAME entry/exit signals.

The validation process:
1. Strategy generates deterministic signals from OHLCV data
2. Each framework executes those exact signals
3. We compare: trades executed, PnL, and performance metrics
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Add project paths
ml4t.backtest_src = Path(__file__).parent.parent.parent / "src"
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(ml4t.backtest_src))
sys.path.insert(0, str(Path(__file__).parent))

from strategies.base_strategy import (
    BollingerBandBreakout,
    DualMovingAverageCrossover,
    RSIMeanReversionStrategy,
)


@dataclass
class BacktestResult:
    """Standardized backtest result for comparison."""

    framework: str
    strategy: str
    initial_capital: float
    final_value: float
    total_return: float
    num_trades: int
    trades: list[dict] = field(default_factory=list)
    execution_time: float = 0.0
    errors: list[str] = field(default_factory=list)


def load_test_data() -> pd.DataFrame:
    """Load test data for validation."""
    wiki_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"

    if wiki_path.exists():
        df = pd.read_parquet(wiki_path)
        # Use AAPL for consistency
        aapl = df[df["ticker"] == "AAPL"].copy()
        aapl["date"] = pd.to_datetime(aapl["date"])
        aapl = aapl.set_index("date").sort_index()

        # Use 2 years of data for more trades
        test_data = aapl.loc["2014-01-01":"2015-12-31"].copy()
        print(
            f"Loaded AAPL data: {len(test_data)} rows from {test_data.index[0]} to {test_data.index[-1]}",
        )
        return test_data
    raise FileNotFoundError(f"Wiki data not found at {wiki_path}")


def execute_with_ml4t.backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float,
) -> BacktestResult:
    """Execute signals using ml4t.backtest (manual implementation)."""
    import time

    start_time = time.time()

    result = BacktestResult(
        framework="ml4t.backtest",
        strategy="PreCalculated",
        initial_capital=initial_capital,
        final_value=initial_capital,
        total_return=0.0,
        num_trades=0,
    )

    try:
        cash = initial_capital
        shares = 0.0
        trades = []

        for date, row in signals.iterrows():
            price = row["close"]

            if row["entry"] and cash > 0:
                # Buy signal - use all cash
                shares = cash / price
                trades.append(
                    {
                        "date": date,
                        "action": "BUY",
                        "price": price,
                        "shares": shares,
                        "value": cash,
                    },
                )
                cash = 0.0

            elif row["exit"] and shares > 0:
                # Sell signal - liquidate position
                cash = shares * price
                trades.append(
                    {
                        "date": date,
                        "action": "SELL",
                        "price": price,
                        "shares": shares,
                        "value": cash,
                    },
                )
                shares = 0.0

        # Final value
        final_price = signals["close"].iloc[-1]
        result.final_value = cash + shares * final_price
        result.total_return = (result.final_value / initial_capital - 1) * 100
        result.num_trades = len(trades)
        result.trades = trades
        result.execution_time = time.time() - start_time

    except Exception as e:
        result.errors.append(str(e))

    return result


def execute_with_vectorbt(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float,
) -> BacktestResult:
    """Execute signals using VectorBT."""
    import time

    start_time = time.time()

    result = BacktestResult(
        framework="VectorBT",
        strategy="PreCalculated",
        initial_capital=initial_capital,
        final_value=initial_capital,
        total_return=0.0,
        num_trades=0,
    )

    try:
        import vectorbt as vbt

        # Use the pre-calculated signals directly
        pf = vbt.Portfolio.from_signals(
            signals["close"],
            entries=signals["entry"],
            exits=signals["exit"],
            init_cash=initial_capital,
            fees=0.0,
            slippage=0.0,
            freq="D",
        )

        result.final_value = float(pf.final_value())
        result.total_return = float(pf.total_return()) * 100

        if hasattr(pf.orders, "records"):
            result.num_trades = len(pf.orders.records)

        result.execution_time = time.time() - start_time

    except ImportError:
        result.errors.append("VectorBT not available")
    except Exception as e:
        result.errors.append(str(e))

    return result


def execute_with_zipline(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float,
) -> BacktestResult:
    """Execute signals using Zipline-Reloaded."""
    import time

    start_time = time.time()

    result = BacktestResult(
        framework="Zipline",
        strategy="PreCalculated",
        initial_capital=initial_capital,
        final_value=initial_capital,
        total_return=0.0,
        num_trades=0,
    )

    try:
        # Check if Zipline is available
        from zipline import run_algorithm
        from zipline.api import (
            date_rules,
            get_datetime,
            order_target_percent,
            record,
            schedule_function,
            set_commission,
            symbol,
            time_rules,
        )
        from zipline.finance import commission

        # Store signals globally for the algorithm
        global zipline_signal_dict
        zipline_signal_dict = {}
        for idx, row in signals.iterrows():
            if row["entry"]:
                zipline_signal_dict[idx.date() if hasattr(idx, "date") else idx] = "entry"
            elif row["exit"]:
                zipline_signal_dict[idx.date() if hasattr(idx, "date") else idx] = "exit"

        # Track trades
        global zipline_trades
        zipline_trades = []

        def initialize(context):
            """Initialize Zipline algorithm."""
            context.asset = symbol("AAPL")
            context.position = False
            context.trade_count = 0

            # No commission for fair comparison
            set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))

            # Schedule trading logic
            schedule_function(trade_logic, date_rules.every_day(), time_rules.market_open())

        def trade_logic(context, data):
            """Execute pre-calculated signals."""
            current_date = get_datetime().date()

            if current_date in zipline_signal_dict:
                signal = zipline_signal_dict[current_date]
                current_price = data.current(context.asset, "price")

                if signal == "entry" and not context.position:
                    # Buy with all available cash
                    order_target_percent(context.asset, 1.0)
                    context.position = True
                    context.trade_count += 1
                    zipline_trades.append(
                        {"date": get_datetime(), "action": "BUY", "price": current_price},
                    )

                elif signal == "exit" and context.position:
                    # Sell entire position
                    order_target_percent(context.asset, 0.0)
                    context.position = False
                    context.trade_count += 1
                    zipline_trades.append(
                        {"date": get_datetime(), "action": "SELL", "price": current_price},
                    )

                # Record for analysis
                record(price=current_price)

        def analyze(context, perf):
            """Extract results after backtest."""

        # Run the algorithm
        start_date = signals.index[0]
        end_date = signals.index[-1]

        perf = run_algorithm(
            start=start_date,
            end=end_date,
            initialize=initialize,
            capital_base=initial_capital,
            bundle="quandl",  # Using quandl bundle which has AAPL
        )

        # Extract results
        result.final_value = float(perf["portfolio_value"].iloc[-1])
        result.total_return = (result.final_value / initial_capital - 1) * 100
        result.num_trades = len(zipline_trades)
        result.execution_time = time.time() - start_time

    except ImportError:
        result.errors.append("Zipline not available")
    except Exception as e:
        result.errors.append(str(e))

    return result


def execute_with_backtrader(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float,
) -> BacktestResult:
    """Execute signals using Backtrader."""
    import time

    start_time = time.time()

    result = BacktestResult(
        framework="Backtrader",
        strategy="PreCalculated",
        initial_capital=initial_capital,
        final_value=initial_capital,
        total_return=0.0,
        num_trades=0,
    )

    try:
        import backtrader as bt

        # Store signals globally for the strategy to access
        global bt_signal_dict
        bt_signal_dict = {}
        for idx, row in signals.iterrows():
            if row["entry"]:
                bt_signal_dict[idx.date() if hasattr(idx, "date") else idx] = "entry"
            elif row["exit"]:
                bt_signal_dict[idx.date() if hasattr(idx, "date") else idx] = "exit"

        class SignalStrategy(bt.Strategy):
            """Execute pre-calculated signals."""

            def __init__(self):
                self.trade_count = 0

            def next(self):
                current_date = self.data.datetime.date(0)

                # Get signal for current date
                if current_date in bt_signal_dict:
                    signal = bt_signal_dict[current_date]

                    if signal == "entry" and not self.position:
                        # Buy with all available cash
                        size = self.broker.getcash() / self.data.close[0]
                        self.buy(size=size)
                        self.trade_count += 1

                    elif signal == "exit" and self.position:
                        # Sell entire position
                        self.close()
                        self.trade_count += 1

        # Create cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SignalStrategy)

        # Prepare data
        bt_data = signals.copy()
        bt_data = bt_data.reset_index()

        # Add data feed
        feed = bt.feeds.PandasData(
            dataname=bt_data,
            datetime="date",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )
        cerebro.adddata(feed)

        # Set initial capital
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=0.0)

        # Run backtest
        strategies = cerebro.run()
        strategy = strategies[0]

        # Get results
        result.final_value = cerebro.broker.getvalue()
        result.total_return = (result.final_value / initial_capital - 1) * 100
        result.num_trades = strategy.trade_count
        result.execution_time = time.time() - start_time

    except ImportError:
        result.errors.append("Backtrader not available")
    except Exception as e:
        result.errors.append(str(e))

    return result


def validate_strategy(
    strategy,
    data: pd.DataFrame,
    initial_capital: float = 10000,
) -> dict[str, Any]:
    """
    Validate a strategy across multiple frameworks.

    Process:
    1. Generate signals once using the strategy
    2. Execute those exact signals on each framework
    3. Compare results
    """
    print(f"\n{'=' * 80}")
    print(f"VALIDATING STRATEGY: {strategy.get_name()}")
    print(f"{'=' * 80}")
    print(f"Parameters: {strategy.get_parameters()}")

    # Step 1: Generate signals (deterministic)
    print("\n1. Generating signals...")
    signals = strategy.calculate_signals(data)

    num_entries = signals["entry"].sum()
    num_exits = signals["exit"].sum()
    print(f"   Generated {num_entries} entry signals and {num_exits} exit signals")

    if num_entries == 0:
        print("   ⚠ Warning: No entry signals generated!")
        return {"error": "No signals generated"}

    # Show first few signals for debugging
    entry_dates = signals[signals["entry"]].index[:3]
    exit_dates = signals[signals["exit"]].index[:3]
    print(f"   First entries: {[d.strftime('%Y-%m-%d') for d in entry_dates]}")
    print(f"   First exits: {[d.strftime('%Y-%m-%d') for d in exit_dates]}")

    # Step 2: Execute signals on each framework
    print("\n2. Executing signals on each framework...")

    results = []

    # ml4t.backtest
    print("   Running ml4t.backtest...")
    qe_result = execute_with_ml4t.backtest(data, signals, initial_capital)
    results.append(qe_result)
    print(f"      Final value: ${qe_result.final_value:,.2f} | Trades: {qe_result.num_trades}")

    # VectorBT
    print("   Running VectorBT...")
    vbt_result = execute_with_vectorbt(data, signals, initial_capital)
    results.append(vbt_result)
    if not vbt_result.errors:
        print(
            f"      Final value: ${vbt_result.final_value:,.2f} | Trades: {vbt_result.num_trades}",
        )
    else:
        print(f"      Error: {vbt_result.errors}")

    # Zipline
    print("   Running Zipline...")
    zl_result = execute_with_zipline(data, signals, initial_capital)
    results.append(zl_result)
    if not zl_result.errors:
        print(f"      Final value: ${zl_result.final_value:,.2f} | Trades: {zl_result.num_trades}")
    else:
        print(f"      Error: {zl_result.errors}")

    # Backtrader
    print("   Running Backtrader...")
    bt_result = execute_with_backtrader(data, signals, initial_capital)
    results.append(bt_result)
    if not bt_result.errors:
        print(f"      Final value: ${bt_result.final_value:,.2f} | Trades: {bt_result.num_trades}")
    else:
        print(f"      Error: {bt_result.errors}")

    # Step 3: Compare results
    print("\n3. Comparing results...")

    successful_results = [r for r in results if not r.errors]

    if len(successful_results) < 2:
        print("   ⚠ Not enough successful results for comparison")
        return {"error": "Insufficient results"}

    # Check agreement
    final_values = [r.final_value for r in successful_results]
    returns = [r.total_return for r in successful_results]
    trade_counts = [r.num_trades for r in successful_results]

    value_spread = max(final_values) - min(final_values)
    return_spread = max(returns) - min(returns)

    print("\n   COMPARISON SUMMARY:")
    print(f"   {'Framework':<15} {'Final Value':<15} {'Return (%)':<12} {'Trades':<10}")
    print(f"   {'-' * 52}")

    for r in successful_results:
        print(
            f"   {r.framework:<15} ${r.final_value:<14,.2f} {r.total_return:<11.2f} {r.num_trades:<10}",
        )

    print(f"\n   Value spread: ${value_spread:.2f}")
    print(f"   Return spread: {return_spread:.2f}%")

    # Check if results match
    value_match = value_spread < 1.0  # Within $1
    return_match = return_spread < 0.01  # Within 0.01%
    trade_match = len(set(trade_counts)) == 1  # Exact match

    if value_match and return_match and trade_match:
        print("\n   ✅ PERFECT AGREEMENT! All frameworks produce identical results")
    else:
        print("\n   ⚠ DISCREPANCY DETECTED:")
        if not value_match:
            print(f"      - Final values differ by ${value_spread:.2f}")
        if not return_match:
            print(f"      - Returns differ by {return_spread:.2f}%")
        if not trade_match:
            print(f"      - Trade counts don't match: {trade_counts}")

    return {
        "strategy": strategy.get_name(),
        "signals": {"entries": num_entries, "exits": num_exits},
        "results": successful_results,
        "agreement": {
            "value_match": value_match,
            "return_match": return_match,
            "trade_match": trade_match,
        },
    }


def main():
    """Run comprehensive strategy validation."""
    print("CROSS-FRAMEWORK STRATEGY VALIDATION")
    print("=" * 80)
    print("Testing: Identical signals should produce identical results")
    print()

    # Load data
    data = load_test_data()

    # Test strategies
    strategies = [
        DualMovingAverageCrossover(fast_period=20, slow_period=50),
        RSIMeanReversionStrategy(rsi_period=14, oversold_level=30, overbought_level=70),
        BollingerBandBreakout(bb_period=20, bb_std=2.0, volume_multiplier=1.5),
    ]

    all_results = []

    for strategy in strategies:
        result = validate_strategy(strategy, data, initial_capital=10000)
        all_results.append(result)

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'=' * 80}")

    for result in all_results:
        if "error" not in result:
            agreement = result["agreement"]
            status = "✅ PASS" if all(agreement.values()) else "⚠ FAIL"
            print(f"{result['strategy']:<30} {status}")
            print(
                f"  Signals: {result['signals']['entries']} entries, {result['signals']['exits']} exits",
            )
            print(
                f"  Agreement: Value={agreement['value_match']}, Return={agreement['return_match']}, Trades={agreement['trade_match']}",
            )

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
