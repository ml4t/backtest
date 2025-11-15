"""
Test high-frequency trading strategies for cross-framework validation.
"""

import sys
from pathlib import Path

import pandas as pd

# Add paths
backtest_src = Path(__file__).parent.parent.parent / "src"
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(backtest_src))
sys.path.insert(0, str(Path(__file__).parent))

from strategies.high_frequency import (
    MicroReversalStrategy,
    ScalpingStrategy,
    VolatilityBreakoutScalper,
)


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


def validate_strategy(strategy, data: pd.DataFrame):
    """Validate a single strategy and show signal generation (helper function, not a pytest test)."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {strategy.get_name()}")
    print(f"{'=' * 80}")
    print(f"Parameters: {strategy.get_parameters()}")

    # Generate signals
    signals = strategy.calculate_signals(data)

    # Count signals
    num_entries = signals["entry"].sum()
    num_exits = signals["exit"].sum()

    print("\nSignal Generation:")
    print(f"  Entry signals: {num_entries}")
    print(f"  Exit signals: {num_exits}")
    print(f"  Average trades per year: {num_entries / 2:.1f}")

    # Show some signal dates
    if num_entries > 0:
        entry_dates = signals[signals["entry"]].index[:5]
        print(f"  First entries: {[d.strftime('%Y-%m-%d') for d in entry_dates]}")

    if num_exits > 0:
        exit_dates = signals[signals["exit"]].index[:5]
        print(f"  First exits: {[d.strftime('%Y-%m-%d') for d in exit_dates]}")

    # Calculate basic metrics
    if num_entries > 0:
        # Simulate basic P&L
        cash = 10000
        shares = 0
        trades = []

        for date, row in signals.iterrows():
            price = row["close"]

            if row["entry"] and cash > 0:
                shares = cash / price
                cash = 0
                trades.append(("BUY", date, price))

            elif row["exit"] and shares > 0:
                cash = shares * price
                trades.append(("SELL", date, price))
                shares = 0

        # Final value
        if shares > 0:
            cash = shares * signals["close"].iloc[-1]

        final_return = (cash / 10000 - 1) * 100
        print("\nSimulated Performance:")
        print(f"  Final value: ${cash:,.2f}")
        print(f"  Total return: {final_return:.2f}%")
        print(f"  Total round trips: {num_entries}")

    return signals


def main():
    """Test all high-frequency strategies."""
    print("HIGH-FREQUENCY TRADING STRATEGY TESTING")
    print("=" * 80)
    print("Goal: Generate 100+ trades per year on daily data")

    # Load data
    data = load_test_data()

    # Test strategies with different parameters for high frequency
    strategies = [
        # Very aggressive scalping
        ScalpingStrategy(
            fast_ema=3,
            slow_ema=8,
            momentum_period=5,
            profit_target=0.005,
            stop_loss=0.003,
            max_holding_days=3,
        ),
        # Ultra-aggressive scalping
        ScalpingStrategy(
            fast_ema=2,
            slow_ema=5,
            momentum_period=3,
            profit_target=0.003,
            stop_loss=0.002,
            max_holding_days=2,
        ),
        # Micro reversals
        MicroReversalStrategy(
            ma_period=5,
            entry_threshold=0.01,
            rsi_period=5,
            rsi_oversold=35,
            rsi_overbought=65,
            profit_target=0.003,
            max_holding_days=2,
        ),
        # Very aggressive micro reversals
        MicroReversalStrategy(
            ma_period=3,
            entry_threshold=0.008,
            rsi_period=3,
            rsi_oversold=40,
            rsi_overbought=60,
            profit_target=0.002,
            max_holding_days=1,
        ),
        # Volatility breakout scalper
        VolatilityBreakoutScalper(
            bb_period=10,
            bb_std=1.5,
            volume_threshold=1.2,
            profit_target=0.004,
            stop_loss=0.002,
            max_holding_days=2,
        ),
        # Ultra-aggressive volatility scalper
        VolatilityBreakoutScalper(
            bb_period=7,
            bb_std=1.2,
            volume_threshold=1.1,
            profit_target=0.003,
            stop_loss=0.0015,
            max_holding_days=1,
        ),
    ]

    # Track which strategies meet the 100+ trades goal
    high_frequency_strategies = []

    for strategy in strategies:
        signals = test_strategy(strategy, data)
        num_trades = signals["entry"].sum()

        if num_trades >= 50:  # 50 trades per year = 100 over 2 years
            high_frequency_strategies.append(
                {
                    "name": strategy.get_name(),
                    "params": strategy.get_parameters(),
                    "trades": num_trades,
                    "trades_per_year": num_trades / 2,
                },
            )

    # Summary
    print(f"\n{'=' * 80}")
    print("HIGH-FREQUENCY STRATEGY SUMMARY")
    print(f"{'=' * 80}")

    if high_frequency_strategies:
        print("Strategies achieving 50+ trades per year:")
        for strat in high_frequency_strategies:
            print(f"  - {strat['name']}: {strat['trades_per_year']:.1f} trades/year")
    else:
        print("No strategies achieved the 50+ trades per year target.")
        print("Consider adjusting parameters for more aggressive signal generation.")

    # Find the best high-frequency strategy
    if high_frequency_strategies:
        best = max(high_frequency_strategies, key=lambda x: x["trades"])
        print("\nBest high-frequency strategy:")
        print(f"  {best['name']} with {best['trades_per_year']:.1f} trades per year")
        print(f"  Parameters: {best['params']}")


if __name__ == "__main__":
    main()
