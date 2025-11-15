"""
Zipline Wiki Data Comparison

This module runs the same strategies on Zipline using Wiki data,
then runs them on other frameworks using the same Wiki data extracted from Zipline.
This ensures we're comparing apples to apples.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add paths
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
ml4t.backtest_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(ml4t.backtest_src))
sys.path.insert(0, str(Path(__file__).parent))


def run_zipline_ma_crossover(
    symbol: str = "AAPL",
    start_date: str = "2014-01-01",
    end_date: str = "2015-12-31",
    fast_period: int = 20,
    slow_period: int = 50,
    initial_capital: float = 10000,
) -> dict[str, Any]:
    """
    Run MA crossover strategy on Zipline using Wiki data.
    """
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

    # Track trades
    trades = []

    def initialize(context):
        """Initialize the algorithm."""
        context.asset = zipline_symbol(symbol)
        context.fast_period = fast_period
        context.slow_period = slow_period

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
                trades.append(
                    {
                        "date": get_datetime(),
                        "action": "BUY",
                        "price": current_price,
                        "type": "golden_cross",
                    },
                )

        elif prev_ma_fast > prev_ma_slow and ma_fast <= ma_slow:
            # Death cross - sell
            if current_position > 0:
                order_target_percent(context.asset, 0.0)
                trades.append(
                    {
                        "date": get_datetime(),
                        "action": "SELL",
                        "price": current_price,
                        "type": "death_cross",
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
        """Analyze the results."""

    # Run the algorithm
    print(f"Running Zipline MA Crossover for {symbol} from {start_date} to {end_date}")
    result = run_algorithm(
        start=pd.Timestamp(start_date).tz_localize("UTC"),
        end=pd.Timestamp(end_date).tz_localize("UTC"),
        initialize=initialize,
        capital_base=initial_capital,
        bundle="quandl",
    )

    # Extract metrics
    final_value = result["portfolio_value"].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    num_trades = len(trades)

    # Get the actual price data used
    price_data = result[["price"]].copy()
    price_data.index = price_data.index.normalize()  # Remove time component

    return {
        "framework": "Zipline",
        "strategy": "MA_Crossover",
        "symbol": symbol,
        "final_value": final_value,
        "total_return": total_return,
        "num_trades": num_trades,
        "trades": trades,
        "price_data": price_data,
        "full_results": result,
    }


def extract_wiki_data_from_zipline(
    symbol: str = "AAPL",
    start_date: str = "2014-01-01",
    end_date: str = "2015-12-31",
) -> pd.DataFrame:
    """
    Extract Wiki data from Zipline bundle for use in other frameworks.
    """
    import pandas as pd
    from zipline import get_calendar
    from zipline.data import bundles
    from zipline.data.data_portal import DataPortal

    # Load the bundle
    bundle = bundles.load("quandl")

    # Get the asset
    asset = bundle.asset_finder.lookup_symbol(symbol, as_of_date=None)

    # Create data portal
    trading_calendar = get_calendar("NYSE")

    data_portal = DataPortal(
        bundle.asset_finder,
        trading_calendar=trading_calendar,
        first_trading_day=bundle.equity_daily_bar_reader.first_trading_day,
        equity_daily_reader=bundle.equity_daily_bar_reader,
        adjustment_reader=bundle.adjustment_reader,
    )

    # Get the data
    start = pd.Timestamp(start_date, tz="utc")
    end = pd.Timestamp(end_date, tz="utc")

    # Get all trading days in range
    sessions = trading_calendar.sessions_in_range(start, end)

    # Extract OHLCV data
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    for session in sessions:
        try:
            opens.append(
                data_portal.get_spot_value(
                    asset,
                    "open",
                    session,
                    "daily",
                ),
            )
            highs.append(
                data_portal.get_spot_value(
                    asset,
                    "high",
                    session,
                    "daily",
                ),
            )
            lows.append(
                data_portal.get_spot_value(
                    asset,
                    "low",
                    session,
                    "daily",
                ),
            )
            closes.append(
                data_portal.get_spot_value(
                    asset,
                    "close",
                    session,
                    "daily",
                ),
            )
            volumes.append(
                data_portal.get_spot_value(
                    asset,
                    "volume",
                    session,
                    "daily",
                ),
            )
        except:
            # Handle missing data
            opens.append(np.nan)
            highs.append(np.nan)
            lows.append(np.nan)
            closes.append(np.nan)
            volumes.append(np.nan)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=sessions,
    )

    # Remove any NaN rows
    df = df.dropna()

    print(f"Extracted {len(df)} days of {symbol} data from Zipline bundle")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


def run_backtest_on_wiki_data(
    data: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    initial_capital: float = 10000,
) -> dict[str, Any]:
    """
    Run MA crossover on ml4t.backtest using Wiki data.
    """
    import time

    start_time = time.time()

    # Initialize portfolio
    cash = initial_capital
    shares = 0
    position = False
    trades = []

    # Need enough data for slow MA
    for i in range(slow_period, len(data)):
        # Calculate MAs
        ma_fast = data["close"].iloc[i - fast_period : i].mean()
        ma_slow = data["close"].iloc[i - slow_period : i].mean()

        # Previous MAs
        prev_ma_fast = data["close"].iloc[i - fast_period - 1 : i - 1].mean()
        prev_ma_slow = data["close"].iloc[i - slow_period - 1 : i - 1].mean()

        current_price = data["close"].iloc[i]
        current_date = data.index[i]

        # Check for crossover
        if prev_ma_fast <= prev_ma_slow and ma_fast > ma_slow:
            # Golden cross - buy
            if not position and cash > 0:
                shares = cash / current_price
                cash = 0
                position = True
                trades.append(
                    {
                        "date": current_date,
                        "action": "BUY",
                        "price": current_price,
                        "shares": shares,
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
                        "date": current_date,
                        "action": "SELL",
                        "price": current_price,
                        "cash": cash,
                    },
                )

    # Close any remaining position
    if shares > 0:
        final_price = data["close"].iloc[-1]
        cash = shares * final_price

    final_value = cash
    total_return = (final_value / initial_capital - 1) * 100

    return {
        "framework": "ml4t.backtest",
        "strategy": "MA_Crossover",
        "final_value": final_value,
        "total_return": total_return,
        "num_trades": len(trades),
        "trades": trades,
        "execution_time": time.time() - start_time,
    }


def run_vectorbt_on_wiki_data(
    data: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    initial_capital: float = 10000,
) -> dict[str, Any]:
    """
    Run MA crossover on VectorBT using Wiki data.
    """
    try:
        import time

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

        return {
            "framework": "VectorBT",
            "strategy": "MA_Crossover",
            "final_value": final_value,
            "total_return": total_return,
            "num_trades": num_trades,
            "execution_time": time.time() - start_time,
        }

    except ImportError:
        return {
            "framework": "VectorBT",
            "error": "VectorBT not installed",
        }


def compare_frameworks_on_wiki_data():
    """
    Main comparison function using Wiki data.
    """
    print("=" * 80)
    print("CROSS-FRAMEWORK COMPARISON USING ZIPLINE WIKI DATA")
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

    # 1. Run on Zipline (native Wiki data)
    print("1. Running on Zipline (native Wiki data)...")
    try:
        zipline_result = run_zipline_ma_crossover(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            fast_period=fast_period,
            slow_period=slow_period,
            initial_capital=initial_capital,
        )
        results.append(zipline_result)
        print(f"   ✓ Final Value: ${zipline_result['final_value']:,.2f}")
        print(f"   ✓ Return: {zipline_result['total_return']:.2f}%")
        print(f"   ✓ Trades: {zipline_result['num_trades']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        zipline_result = None

    # 2. Extract Wiki data from Zipline
    print("\n2. Extracting Wiki data from Zipline bundle...")
    try:
        wiki_data = extract_wiki_data_from_zipline(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        print(f"   ✓ Extracted {len(wiki_data)} days of data")
    except Exception as e:
        print(f"   ✗ Error extracting data: {e}")
        return None

    # 3. Run on ml4t.backtest
    print("\n3. Running on ml4t.backtest (using extracted Wiki data)...")
    try:
        ml4t.backtest_result = run_ml4t.backtest_on_wiki_data(
            wiki_data,
            fast_period=fast_period,
            slow_period=slow_period,
            initial_capital=initial_capital,
        )
        results.append(ml4t.backtest_result)
        print(f"   ✓ Final Value: ${ml4t.backtest_result['final_value']:,.2f}")
        print(f"   ✓ Return: {ml4t.backtest_result['total_return']:.2f}%")
        print(f"   ✓ Trades: {ml4t.backtest_result['num_trades']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # 4. Run on VectorBT
    print("\n4. Running on VectorBT (using extracted Wiki data)...")
    try:
        vectorbt_result = run_vectorbt_on_wiki_data(
            wiki_data,
            fast_period=fast_period,
            slow_period=slow_period,
            initial_capital=initial_capital,
        )
        if "error" not in vectorbt_result:
            results.append(vectorbt_result)
            print(f"   ✓ Final Value: ${vectorbt_result['final_value']:,.2f}")
            print(f"   ✓ Return: {vectorbt_result['total_return']:.2f}%")
            print(f"   ✓ Trades: {vectorbt_result['num_trades']}")
        else:
            print(f"   ✗ {vectorbt_result['error']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # 5. Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    if len(results) < 2:
        print("Not enough successful results for comparison")
        return None

    print(f"\n{'Framework':<15} {'Final Value':<15} {'Return (%)':<12} {'Trades':<10}")
    print("-" * 52)

    for r in results:
        if "error" not in r:
            print(
                f"{r['framework']:<15} ${r['final_value']:<14,.2f} {r['total_return']:<11.2f} {r['num_trades']:<10}",
            )

    # Check agreement
    final_values = [r["final_value"] for r in results if "error" not in r]
    returns = [r["total_return"] for r in results if "error" not in r]
    [r["num_trades"] for r in results if "error" not in r]

    if len(final_values) > 1:
        value_spread = max(final_values) - min(final_values)
        return_spread = max(returns) - min(returns)

        print(f"\nValue Spread: ${value_spread:,.2f}")
        print(f"Return Spread: {return_spread:.2f}%")

        # Check if results are close
        if value_spread < 100 and return_spread < 1.0:
            print("\n✅ GOOD AGREEMENT - Frameworks produce similar results")
        else:
            print("\n⚠ DISCREPANCY - Frameworks show different results")
            print("This may be due to differences in data handling or calculation methods")

    return results


if __name__ == "__main__":
    compare_frameworks_on_wiki_data()
