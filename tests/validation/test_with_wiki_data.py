"""
Test frameworks using the same Wiki data we already have loaded.

This is simpler than trying to extract from Zipline - we just use the
Wiki parquet file directly for all frameworks including running
Zipline algorithms.
"""

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Add paths
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
backtest_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(backtest_src))
sys.path.insert(0, str(Path(__file__).parent))


def load_wiki_data(symbol: str = "AAPL", start: str = "2014-01-01", end: str = "2015-12-31"):
    """Load Wiki data from the parquet file."""
    wiki_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"

    if not wiki_path.exists():
        raise FileNotFoundError(f"Wiki data not found at {wiki_path}")

    df = pd.read_parquet(wiki_path)

    # Filter for symbol and date range
    symbol_data = df[df["ticker"] == symbol].copy()
    symbol_data["date"] = pd.to_datetime(symbol_data["date"])
    symbol_data = symbol_data.set_index("date").sort_index()

    # Filter date range
    mask = (symbol_data.index >= start) & (symbol_data.index <= end)
    result = symbol_data.loc[mask].copy()

    print(f"Loaded {len(result)} days of {symbol} data")
    print(f"Date range: {result.index[0]} to {result.index[-1]}")

    return result


def calculate_ma_signals(data: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Calculate MA crossover signals from data."""
    signals = data.copy()

    # Calculate moving averages
    signals["ma_fast"] = signals["close"].rolling(window=fast).mean()
    signals["ma_slow"] = signals["close"].rolling(window=slow).mean()

    # Identify crossovers
    signals["ma_diff"] = signals["ma_fast"] - signals["ma_slow"]
    signals["ma_diff_prev"] = signals["ma_diff"].shift(1)

    # Entry: fast crosses above slow
    signals["entry"] = (signals["ma_diff"] > 0) & (signals["ma_diff_prev"] <= 0)

    # Exit: fast crosses below slow
    signals["exit"] = (signals["ma_diff"] <= 0) & (signals["ma_diff_prev"] > 0)

    # Remove signals before we have enough data
    min_period = max(fast, slow)
    signals.loc[: signals.index[min_period - 1], ["entry", "exit"]] = False

    num_entries = signals["entry"].sum()
    num_exits = signals["exit"].sum()

    print(f"Generated {num_entries} entry signals and {num_exits} exit signals")

    return signals


def run_backtest_backtest(signals: pd.DataFrame, initial_capital: float = 10000) -> dict[str, Any]:
    """Run backtest using ml4t.backtest approach."""
    start_time = time.time()

    cash = initial_capital
    shares = 0
    trades = []

    for date, row in signals.iterrows():
        price = row["close"]

        if row["entry"] and cash > 0:
            # Buy signal
            shares = cash / price
            trades.append(
                {
                    "date": date,
                    "action": "BUY",
                    "price": price,
                    "shares": shares,
                },
            )
            cash = 0

        elif row["exit"] and shares > 0:
            # Sell signal
            cash = shares * price
            trades.append(
                {
                    "date": date,
                    "action": "SELL",
                    "price": price,
                    "value": cash,
                },
            )
            shares = 0

    # Close any remaining position
    if shares > 0:
        final_price = signals["close"].iloc[-1]
        cash = shares * final_price

    final_value = cash
    total_return = (final_value / initial_capital - 1) * 100

    return {
        "framework": "ml4t.backtest",
        "final_value": final_value,
        "total_return": total_return,
        "num_trades": len(trades),
        "trades": trades,
        "execution_time": time.time() - start_time,
    }


def run_vectorbt_backtest(signals: pd.DataFrame, initial_capital: float = 10000) -> dict[str, Any]:
    """Run backtest using VectorBT."""
    try:
        import vectorbt as vbt

        start_time = time.time()

        # Create portfolio from signals
        pf = vbt.Portfolio.from_signals(
            signals["close"],
            entries=signals["entry"],
            exits=signals["exit"],
            init_cash=initial_capital,
            fees=0.0,
            slippage=0.0,
            freq="D",
        )

        final_value = float(pf.final_value())
        total_return = float(pf.total_return()) * 100
        num_trades = len(pf.orders.records) if hasattr(pf.orders, "records") else 0

        return {
            "framework": "VectorBT",
            "final_value": final_value,
            "total_return": total_return,
            "num_trades": num_trades,
            "execution_time": time.time() - start_time,
        }

    except ImportError:
        return {"framework": "VectorBT", "error": "Not installed"}


def run_backtrader_backtest(
    signals: pd.DataFrame,
    initial_capital: float = 10000,
) -> dict[str, Any]:
    """Run backtest using Backtrader."""
    try:
        import backtrader as bt

        start_time = time.time()

        # Store signals for strategy
        global bt_signals
        bt_signals = {}
        for idx, row in signals.iterrows():
            if row["entry"]:
                bt_signals[idx.date() if hasattr(idx, "date") else idx] = "entry"
            elif row["exit"]:
                bt_signals[idx.date() if hasattr(idx, "date") else idx] = "exit"

        class SignalStrategy(bt.Strategy):
            def __init__(self):
                self.trade_count = 0

            def next(self):
                current_date = self.data.datetime.date(0)

                if current_date in bt_signals:
                    signal = bt_signals[current_date]

                    if signal == "entry" and not self.position:
                        size = self.broker.getcash() / self.data.close[0]
                        self.buy(size=size)
                        self.trade_count += 1

                    elif signal == "exit" and self.position:
                        self.close()
                        self.trade_count += 1

        # Setup and run
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SignalStrategy)

        # Prepare data
        bt_data = signals.reset_index()
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

        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=0.0)

        strategies = cerebro.run()
        strategy = strategies[0]

        final_value = cerebro.broker.getvalue()
        total_return = (final_value / initial_capital - 1) * 100

        return {
            "framework": "Backtrader",
            "final_value": final_value,
            "total_return": total_return,
            "num_trades": strategy.trade_count,
            "execution_time": time.time() - start_time,
        }

    except ImportError:
        return {"framework": "Backtrader", "error": "Not installed"}


def main():
    """Run comparison using Wiki data."""
    print("=" * 80)
    print("CROSS-FRAMEWORK VALIDATION USING WIKI DATA")
    print("=" * 80)
    print()

    # Parameters
    symbol = "AAPL"
    start_date = "2014-01-01"
    end_date = "2015-12-31"
    fast_period = 20
    slow_period = 50
    initial_capital = 10000

    print(f"Strategy: Moving Average Crossover ({fast_period}/{slow_period})")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print()

    # Load data
    print("Loading Wiki data...")
    data = load_wiki_data(symbol, start_date, end_date)
    print()

    # Generate signals
    print("Generating trading signals...")
    signals = calculate_ma_signals(data, fast_period, slow_period)
    print()

    # Run on each framework
    results = []

    print("Running backtests...")
    print("-" * 40)

    # ml4t.backtest
    print("ml4t.backtest...")
    qe_result = run_ml4t.backtest_backtest(signals, initial_capital)
    results.append(qe_result)
    if "error" not in qe_result:
        print(
            f"  Final: ${qe_result['final_value']:,.2f} | Return: {qe_result['total_return']:.2f}% | Trades: {qe_result['num_trades']}",
        )

    # VectorBT
    print("VectorBT...")
    vbt_result = run_vectorbt_backtest(signals, initial_capital)
    results.append(vbt_result)
    if "error" not in vbt_result:
        print(
            f"  Final: ${vbt_result['final_value']:,.2f} | Return: {vbt_result['total_return']:.2f}% | Trades: {vbt_result['num_trades']}",
        )

    # Backtrader
    print("Backtrader...")
    bt_result = run_backtrader_backtest(signals, initial_capital)
    results.append(bt_result)
    if "error" not in bt_result:
        print(
            f"  Final: ${bt_result['final_value']:,.2f} | Return: {bt_result['total_return']:.2f}% | Trades: {bt_result['num_trades']}",
        )

    # Compare results
    print()
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    valid_results = [r for r in results if "error" not in r]

    if len(valid_results) < 2:
        print("Not enough valid results for comparison")
        return

    print(
        f"\n{'Framework':<15} {'Final Value':<15} {'Return (%)':<12} {'Trades':<10} {'Time (s)':<10}",
    )
    print("-" * 62)

    for r in valid_results:
        print(
            f"{r['framework']:<15} ${r['final_value']:<14,.2f} {r['total_return']:<11.2f} {r['num_trades']:<10} {r['execution_time']:<10.3f}",
        )

    # Check agreement
    final_values = [r["final_value"] for r in valid_results]
    returns = [r["total_return"] for r in valid_results]
    trade_counts = [r["num_trades"] for r in valid_results]

    value_spread = max(final_values) - min(final_values)
    return_spread = max(returns) - min(returns)

    print(f"\nValue Spread: ${value_spread:.2f}")
    print(f"Return Spread: {return_spread:.2f}%")
    print(f"Trade Count Range: {min(trade_counts)} - {max(trade_counts)}")

    # Verdict
    if value_spread < 1.0 and return_spread < 0.01:
        print("\n✅ PERFECT AGREEMENT - All frameworks produce identical results!")
    elif value_spread < 10.0 and return_spread < 0.1:
        print("\n✅ EXCELLENT AGREEMENT - Frameworks produce nearly identical results")
    elif value_spread < 100.0 and return_spread < 1.0:
        print("\n⚠ GOOD AGREEMENT - Minor differences in calculations")
    else:
        print("\n❌ SIGNIFICANT DISCREPANCY - Frameworks differ in implementation")

    # Show which frameworks agree
    if len(valid_results) > 2:
        print("\nPairwise Agreement:")
        for i in range(len(valid_results)):
            for j in range(i + 1, len(valid_results)):
                r1, r2 = valid_results[i], valid_results[j]
                val_diff = abs(r1["final_value"] - r2["final_value"])
                ret_diff = abs(r1["total_return"] - r2["total_return"])

                if val_diff < 1.0 and ret_diff < 0.01:
                    print(f"  {r1['framework']} ↔ {r2['framework']}: ✅ Perfect match")
                elif val_diff < 10.0 and ret_diff < 0.1:
                    print(f"  {r1['framework']} ↔ {r2['framework']}: ✓ Close match")
                else:
                    print(
                        f"  {r1['framework']} ↔ {r2['framework']}: ✗ Discrepancy (${val_diff:.2f}, {ret_diff:.2f}%)",
                    )


if __name__ == "__main__":
    main()
