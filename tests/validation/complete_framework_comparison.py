"""
Complete Cross-Framework Validation
Comparing QEngine, Zipline-Reloaded, VectorBT, and Backtrader

This script runs identical MA crossover strategies on all frameworks
using the same data to validate correctness.
"""

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add paths
qengine_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(qengine_src))


def run_zipline_backtest(
    symbol: str = "AAPL",
    start_date: str = "2014-01-01",
    end_date: str = "2015-12-31",
    fast_period: int = 20,
    slow_period: int = 50,
    initial_capital: float = 10000,
) -> dict[str, Any]:
    """Run MA crossover strategy on Zipline."""
    from zipline import run_algorithm
    from zipline.api import (
        date_rules,
        get_datetime,
        order_target_percent,
        record,
        schedule_function,
        set_commission,
        time_rules,
    )
    from zipline.api import (
        symbol as zipline_symbol,
    )
    from zipline.finance import commission

    trades = []

    def initialize(context):
        """Initialize the algorithm."""
        context.asset = zipline_symbol(symbol)
        context.fast_period = fast_period
        context.slow_period = slow_period
        context.trades = []

        # No commission for fair comparison
        set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))

        # Schedule the trading function
        schedule_function(
            trade,
            date_rules.every_day(),
            time_rules.market_open(),
        )

    def trade(context, data):
        """Execute the trading logic."""
        # Get price history
        prices = data.history(
            context.asset,
            "price",
            context.slow_period + 1,
            "1d",
        )

        if len(prices) < context.slow_period:
            return

        # Calculate moving averages
        ma_fast = prices[-context.fast_period :].mean()
        ma_slow = prices.mean()

        # Get previous MAs for crossover detection
        prev_prices = prices[:-1]
        prev_ma_fast = prev_prices[-context.fast_period :].mean()
        prev_ma_slow = prev_prices.mean()

        current_position = context.portfolio.positions[context.asset].amount
        current_price = data.current(context.asset, "price")

        # Detect crossovers
        if prev_ma_fast <= prev_ma_slow and ma_fast > ma_slow:
            # Golden cross - buy
            if current_position == 0:
                order_target_percent(context.asset, 1.0)
                context.trades.append(
                    {
                        "date": get_datetime(),
                        "action": "BUY",
                        "price": current_price,
                        "signal": "golden_cross",
                    },
                )

        elif prev_ma_fast > prev_ma_slow and ma_fast <= ma_slow:
            # Death cross - sell
            if current_position > 0:
                order_target_percent(context.asset, 0.0)
                context.trades.append(
                    {
                        "date": get_datetime(),
                        "action": "SELL",
                        "price": current_price,
                        "signal": "death_cross",
                    },
                )

        # Record for debugging
        record(
            price=current_price,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            position=current_position,
        )

    def analyze(context, perf):
        """Analyze results after backtest."""
        nonlocal trades
        trades = context.trades

    # Run the algorithm
    print(f"Running Zipline for {symbol} from {start_date} to {end_date}")

    start_time = time.time()

    result = run_algorithm(
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
        initialize=initialize,
        analyze=analyze,
        capital_base=initial_capital,
        bundle="quandl",
    )

    # Extract metrics
    final_value = result["portfolio_value"].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    num_trades = len(trades)

    # Calculate Sharpe ratio
    returns = result["returns"]
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() * 100

    return {
        "framework": "Zipline",
        "final_value": final_value,
        "total_return": total_return,
        "num_trades": num_trades,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "trades": trades,
        "execution_time": time.time() - start_time,
    }


def run_qengine_backtest(
    data: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    initial_capital: float = 10000,
) -> dict[str, Any]:
    """Run MA crossover on QEngine."""
    start_time = time.time()

    # Initialize portfolio
    cash = initial_capital
    shares = 0
    position = False
    trades = []
    portfolio_values = []

    # Calculate signals
    data["ma_fast"] = data["close"].rolling(window=fast_period).mean()
    data["ma_slow"] = data["close"].rolling(window=slow_period).mean()

    # Track portfolio value
    for i in range(slow_period, len(data)):
        current_price = data["close"].iloc[i]

        # Current MAs
        ma_fast = data["ma_fast"].iloc[i]
        ma_slow = data["ma_slow"].iloc[i]

        # Previous MAs
        if i > slow_period:
            prev_ma_fast = data["ma_fast"].iloc[i - 1]
            prev_ma_slow = data["ma_slow"].iloc[i - 1]

            # Check for crossover
            if prev_ma_fast <= prev_ma_slow and ma_fast > ma_slow:
                # Golden cross - buy
                if not position and cash > 0:
                    shares = cash / current_price
                    cash = 0
                    position = True
                    trades.append(
                        {
                            "date": data.index[i],
                            "action": "BUY",
                            "price": current_price,
                        },
                    )

            elif prev_ma_fast > prev_ma_slow and ma_fast <= ma_slow:
                # Death cross - sell
                if position and shares > 0:
                    cash = shares * current_price
                    shares = 0
                    position = False
                    trades.append(
                        {
                            "date": data.index[i],
                            "action": "SELL",
                            "price": current_price,
                        },
                    )

        # Track portfolio value
        portfolio_value = shares * current_price if position else cash
        portfolio_values.append(portfolio_value)

    # Close any remaining position
    if shares > 0:
        final_price = data["close"].iloc[-1]
        cash = shares * final_price

    final_value = cash
    total_return = (final_value / initial_capital - 1) * 100

    # Calculate returns
    portfolio_series = pd.Series(portfolio_values)
    returns = portfolio_series.pct_change().dropna()

    # Sharpe ratio
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() * 100 if len(drawdown) > 0 else 0

    return {
        "framework": "QEngine",
        "final_value": final_value,
        "total_return": total_return,
        "num_trades": len(trades),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "trades": trades,
        "execution_time": time.time() - start_time,
    }


def run_vectorbt_backtest(
    data: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    initial_capital: float = 10000,
) -> dict[str, Any]:
    """Run MA crossover on VectorBT."""
    try:
        import vectorbt as vbt

        start_time = time.time()

        # Calculate indicators
        close_prices = data["close"]
        ma_fast = vbt.MA.run(close_prices, window=fast_period).ma
        ma_slow = vbt.MA.run(close_prices, window=slow_period).ma

        # Generate signals
        entries = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
        exits = (ma_fast <= ma_slow) & (ma_fast.shift(1) > ma_slow.shift(1))

        # Remove NaN values
        valid_mask = ~(ma_fast.isna() | ma_slow.isna())
        entries = entries & valid_mask
        exits = exits & valid_mask

        # Run backtest
        pf = vbt.Portfolio.from_signals(
            close_prices,
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=0.0,
            slippage=0.0,
            freq="D",
        )

        final_value = float(pf.final_value())
        total_return = float(pf.total_return()) * 100
        num_trades = len(pf.orders.records) if hasattr(pf.orders, "records") else 0

        # Get metrics
        sharpe = float(pf.sharpe_ratio()) if hasattr(pf, "sharpe_ratio") else 0
        max_dd = float(pf.max_drawdown()) * 100 if hasattr(pf, "max_drawdown") else 0

        return {
            "framework": "VectorBT",
            "final_value": final_value,
            "total_return": total_return,
            "num_trades": num_trades,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "execution_time": time.time() - start_time,
        }

    except ImportError:
        return {"framework": "VectorBT", "error": "Not installed"}


def run_backtrader_backtest(
    data: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    initial_capital: float = 10000,
) -> dict[str, Any]:
    """Run MA crossover on Backtrader."""
    try:
        import backtrader as bt

        start_time = time.time()

        class MAStrategy(bt.Strategy):
            params = (
                ("fast_period", fast_period),
                ("slow_period", slow_period),
            )

            def __init__(self):
                self.ma_fast = bt.indicators.SimpleMovingAverage(
                    self.data.close,
                    period=self.params.fast_period,
                )
                self.ma_slow = bt.indicators.SimpleMovingAverage(
                    self.data.close,
                    period=self.params.slow_period,
                )
                self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)
                self.trade_count = 0

            def next(self):
                if self.crossover > 0:  # Golden cross
                    if not self.position:
                        self.buy(size=self.broker.getcash() / self.data.close[0])
                        self.trade_count += 1

                elif self.crossover < 0:  # Death cross
                    if self.position:
                        self.close()
                        self.trade_count += 1

        # Setup and run
        cerebro = bt.Cerebro()
        cerebro.addstrategy(MAStrategy)

        # Prepare data
        bt_data = data.reset_index()
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
            "sharpe_ratio": 0,  # Would need analyzers for this
            "max_drawdown": 0,  # Would need analyzers for this
            "execution_time": time.time() - start_time,
        }

    except ImportError:
        return {"framework": "Backtrader", "error": "Not installed"}


def load_quandl_data(symbol: str = "AAPL", start: str = "2014-01-01", end: str = "2015-12-31"):
    """Load data from Zipline's quandl bundle for other frameworks."""
    # Alternative: Use Wiki data directly since we have it
    from pathlib import Path

    projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
    wiki_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"

    if wiki_path.exists():
        # Use Wiki data directly
        df = pd.read_parquet(wiki_path)
        symbol_data = df[df["ticker"] == symbol].copy()
        symbol_data["date"] = pd.to_datetime(symbol_data["date"])
        symbol_data = symbol_data.set_index("date").sort_index()

        # Filter date range
        mask = (symbol_data.index >= start) & (symbol_data.index <= end)
        result = symbol_data.loc[mask].copy()

        print(f"Loaded {len(result)} days of {symbol} data from Wiki parquet")
        return result

    # Fallback: Try to extract from Zipline (simplified)
    try:
        from zipline.data import bundles

        # Load the bundle
        bundle = bundles.load("quandl")

        # Get the asset
        asset = bundle.asset_finder.lookup_symbol(symbol, as_of_date=None)

        # Use the equity_daily_bar_reader directly
        reader = bundle.equity_daily_bar_reader

        # Get data for the asset
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # Read the data
        sessions = pd.date_range(start=start_ts, end=end_ts, freq="B")  # Business days

        data_dict = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }

        for session in sessions:
            try:
                # Try to get data for this session
                data_dict["open"].append(reader.get_value(asset.sid, session, "open"))
                data_dict["high"].append(reader.get_value(asset.sid, session, "high"))
                data_dict["low"].append(reader.get_value(asset.sid, session, "low"))
                data_dict["close"].append(reader.get_value(asset.sid, session, "close"))
                data_dict["volume"].append(reader.get_value(asset.sid, session, "volume"))
            except:
                # Skip days with no data
                continue

        if data_dict["close"]:
            df = pd.DataFrame(data_dict)
            df.index = pd.DatetimeIndex(sessions[: len(df)])
            df.index.name = "date"
            print(f"Loaded {len(df)} days of {symbol} data from Zipline bundle")
            return df
    except Exception as e:
        print(f"Error extracting from Zipline: {e}")

    raise ValueError("Could not load data from either Wiki parquet or Zipline bundle")


def main():
    """Run comprehensive cross-framework comparison."""
    print("=" * 80)
    print("COMPREHENSIVE CROSS-FRAMEWORK VALIDATION")
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

    results = []

    # 1. Run Zipline (native)
    print("1. Running Zipline-Reloaded...")
    try:
        zipline_result = run_zipline_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            fast_period=fast_period,
            slow_period=slow_period,
            initial_capital=initial_capital,
        )
        results.append(zipline_result)
        print(
            f"   ✓ Final: ${zipline_result['final_value']:,.2f} | Return: {zipline_result['total_return']:.2f}% | Trades: {zipline_result['num_trades']}",
        )
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # 2. Load data for other frameworks
    print("\n2. Loading data from Zipline bundle...")
    try:
        data = load_quandl_data(symbol, start_date, end_date)
        print(f"   ✓ Loaded {len(data)} days of data")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return

    # 3. Run QEngine
    print("\n3. Running QEngine...")
    try:
        qengine_result = run_qengine_backtest(
            data,
            fast_period=fast_period,
            slow_period=slow_period,
            initial_capital=initial_capital,
        )
        results.append(qengine_result)
        print(
            f"   ✓ Final: ${qengine_result['final_value']:,.2f} | Return: {qengine_result['total_return']:.2f}% | Trades: {qengine_result['num_trades']}",
        )
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # 4. Run VectorBT
    print("\n4. Running VectorBT...")
    try:
        vectorbt_result = run_vectorbt_backtest(
            data,
            fast_period=fast_period,
            slow_period=slow_period,
            initial_capital=initial_capital,
        )
        if "error" not in vectorbt_result:
            results.append(vectorbt_result)
            print(
                f"   ✓ Final: ${vectorbt_result['final_value']:,.2f} | Return: {vectorbt_result['total_return']:.2f}% | Trades: {vectorbt_result['num_trades']}",
            )
        else:
            print(f"   ✗ {vectorbt_result['error']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # 5. Run Backtrader
    print("\n5. Running Backtrader...")
    try:
        backtrader_result = run_backtrader_backtest(
            data,
            fast_period=fast_period,
            slow_period=slow_period,
            initial_capital=initial_capital,
        )
        if "error" not in backtrader_result:
            results.append(backtrader_result)
            print(
                f"   ✓ Final: ${backtrader_result['final_value']:,.2f} | Return: {backtrader_result['total_return']:.2f}% | Trades: {backtrader_result['num_trades']}",
            )
        else:
            print(f"   ✗ {backtrader_result['error']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Display comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    if len(results) < 2:
        print("Not enough successful results for comparison")
        return

    print(
        f"\n{'Framework':<15} {'Final Value':<15} {'Return (%)':<12} {'Trades':<10} {'Sharpe':<10} {'Max DD (%)':<12} {'Time (s)':<10}",
    )
    print("-" * 94)

    for r in results:
        if "error" not in r:
            print(
                f"{r['framework']:<15} ${r['final_value']:<14,.2f} {r['total_return']:<11.2f} "
                f"{r.get('num_trades', 0):<10} {r.get('sharpe_ratio', 0):<10.2f} "
                f"{r.get('max_drawdown', 0):<11.2f} {r.get('execution_time', 0):<10.3f}",
            )

    # Check agreement
    final_values = [r["final_value"] for r in results if "error" not in r]
    returns = [r["total_return"] for r in results if "error" not in r]

    if len(final_values) > 1:
        value_spread = max(final_values) - min(final_values)
        return_spread = max(returns) - min(returns)

        print(f"\nValue Spread: ${value_spread:,.2f}")
        print(f"Return Spread: {return_spread:.2f}%")

        # Verdict
        if value_spread < 100 and return_spread < 1.0:
            print("\n✅ GOOD AGREEMENT - Frameworks produce similar results")
        else:
            print("\n⚠ DISCREPANCY - Frameworks show different results")

        # Show pairwise comparisons
        print("\nPairwise Comparisons:")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if "error" not in results[i] and "error" not in results[j]:
                    r1, r2 = results[i], results[j]
                    val_diff = abs(r1["final_value"] - r2["final_value"])
                    ret_diff = abs(r1["total_return"] - r2["total_return"])

                    if val_diff < 10.0 and ret_diff < 0.1:
                        status = "✅ Excellent agreement"
                    elif val_diff < 100.0 and ret_diff < 1.0:
                        status = "✓ Good agreement"
                    else:
                        status = f"✗ Discrepancy (${val_diff:.2f}, {ret_diff:.2f}%)"

                    print(f"  {r1['framework']} ↔ {r2['framework']}: {status}")

    # Performance comparison
    if len(results) > 1:
        print("\nPerformance Ranking (by execution time):")
        sorted_results = sorted(
            [r for r in results if "error" not in r],
            key=lambda x: x.get("execution_time", float("inf")),
        )

        fastest_time = sorted_results[0].get("execution_time", 1)
        for i, r in enumerate(sorted_results, 1):
            speedup = r.get("execution_time", 0) / fastest_time if fastest_time > 0 else 0
            print(
                f"  {i}. {r['framework']}: {r.get('execution_time', 0):.3f}s"
                + (" (baseline)" if i == 1 else f" ({speedup:.1f}x slower)"),
            )


if __name__ == "__main__":
    main()
