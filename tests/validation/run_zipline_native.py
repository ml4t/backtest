"""
Run a native Zipline strategy using the quandl bundle.
This runs Zipline the way it's meant to be used.
"""

import pandas as pd
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


def initialize(context):
    """Initialize the algorithm."""
    context.asset = symbol("AAPL")
    context.fast_period = 20
    context.slow_period = 50
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
            print(f"{get_datetime().date()}: BUY at ${current_price:.2f} (Golden Cross)")

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
            print(f"{get_datetime().date()}: SELL at ${current_price:.2f} (Death Cross)")

    # Record for debugging
    record(
        price=current_price,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        position=current_position,
    )


def analyze(context, perf):
    """Analyze the results."""
    print("\n" + "=" * 60)
    print("ZIPLINE BACKTEST RESULTS")
    print("=" * 60)

    initial_capital = 10000
    final_value = perf["portfolio_value"].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {len(context.trades)}")

    if context.trades:
        print("\nTrade History:")
        for trade in context.trades:
            print(f"  {trade['date'].date()}: {trade['action']} at ${trade['price']:.2f}")

    # Calculate some metrics
    returns = perf["returns"]
    if len(returns) > 0:
        sharpe = (returns.mean() / returns.std()) * (252**0.5) if returns.std() > 0 else 0
        print(f"\nSharpe Ratio: {sharpe:.2f}")

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100
        print(f"Max Drawdown: {max_dd:.2f}%")


def main():
    """Run the Zipline algorithm."""
    print("Running Zipline MA Crossover Strategy")
    print("Symbol: AAPL")
    print("Period: 2014-01-01 to 2015-12-31")
    print("Strategy: MA(20/50) Crossover")
    print()

    # Run the algorithm
    result = run_algorithm(
        start=pd.Timestamp("2014-01-01"),
        end=pd.Timestamp("2015-12-31"),
        initialize=initialize,
        analyze=analyze,
        capital_base=10000,
        bundle="quandl",
    )

    return result


if __name__ == "__main__":
    main()
