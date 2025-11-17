"""
Generate rebalancing signal datasets for portfolio rotation validation.

Creates pre-computed signals for "always hold top N" rebalancing strategies
to test framework handling of frequent position changes and portfolio management.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import numpy as np

# Import stock download from the multi-asset generator
from generate_multi_asset import (
    download_stock_data,
    get_sp500_top_symbols,
    save_multi_asset_signal_set,
)

# Directories
SIGNAL_DIR = Path(__file__).parent


def calculate_momentum_rank(
    data_dict: dict[str, pd.DataFrame],
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Calculate momentum-based ranking for all assets.

    Returns a DataFrame where each row is a date and each column is an asset,
    with values representing the rank (1 = highest momentum).

    Args:
        data_dict: Dictionary mapping symbol -> OHLCV DataFrame
        lookback: Number of days for momentum calculation

    Returns:
        DataFrame with rankings (columns=symbols, index=dates)
    """
    # Align all price series to common index
    all_prices = {}
    for symbol, data in data_dict.items():
        all_prices[symbol] = data["close"]

    # Create aligned DataFrame
    prices_df = pd.DataFrame(all_prices)

    # Calculate returns over lookback period
    returns = prices_df.pct_change(lookback)

    # Rank assets (1 = highest return, N = lowest return)
    # Use min ranking method to handle ties
    ranks = returns.rank(axis=1, ascending=False, method="min")

    return ranks


def calculate_volatility_rank(
    data_dict: dict[str, pd.DataFrame],
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Calculate volatility-based ranking (inverse - prefer low volatility).

    Args:
        data_dict: Dictionary mapping symbol -> OHLCV DataFrame
        lookback: Number of days for volatility calculation

    Returns:
        DataFrame with rankings (1 = lowest volatility)
    """
    # Align all price series
    all_prices = {}
    for symbol, data in data_dict.items():
        all_prices[symbol] = data["close"]

    prices_df = pd.DataFrame(all_prices)

    # Calculate rolling volatility
    returns = prices_df.pct_change()
    volatility = returns.rolling(window=lookback, min_periods=lookback).std()

    # Rank assets (1 = lowest volatility, N = highest volatility)
    ranks = volatility.rank(axis=1, ascending=True, method="min")

    return ranks


def generate_rebalancing_signals(
    data_dict: dict[str, pd.DataFrame],
    hold_count: int = 10,
    rebalance_freq: Literal["daily", "weekly", "monthly"] = "weekly",
    ranking_method: Literal["momentum", "volatility", "random"] = "momentum",
    ranking_lookback: int = 20,
    seed: int | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Generate rebalancing signals for "always hold top N" strategy.

    Strategy logic:
    1. Rank all assets by chosen metric (momentum, volatility, random)
    2. Hold top N assets
    3. Rebalance at specified frequency
    4. Generate entry/exit signals for position changes

    Args:
        data_dict: Dictionary mapping symbol -> OHLCV DataFrame
        hold_count: Number of positions to hold simultaneously
        rebalance_freq: Rebalancing frequency ('daily', 'weekly', 'monthly')
        ranking_method: How to rank assets ('momentum', 'volatility', 'random')
        ranking_lookback: Lookback period for ranking calculations
        seed: Random seed (for random ranking method)

    Returns:
        Dictionary mapping symbol -> {'data': df, 'signals': signals_df}
    """
    print(f"\n🔄 Generating rebalancing signals for {len(data_dict)} assets...")
    print(f"   Strategy: Always hold top {hold_count} by {ranking_method}")
    print(f"   Rebalance: {rebalance_freq}")
    print(f"   Ranking lookback: {ranking_lookback} days")

    # Calculate rankings based on method
    if ranking_method == "momentum":
        ranks = calculate_momentum_rank(data_dict, lookback=ranking_lookback)
    elif ranking_method == "volatility":
        ranks = calculate_volatility_rank(data_dict, lookback=ranking_lookback)
    elif ranking_method == "random":
        # Random ranking
        rng = np.random.default_rng(seed)
        all_prices = {sym: data["close"] for sym, data in data_dict.items()}
        prices_df = pd.DataFrame(all_prices)
        ranks = pd.DataFrame(
            rng.random(size=prices_df.shape),
            index=prices_df.index,
            columns=prices_df.columns,
        ).rank(axis=1, ascending=False, method="min")
    else:
        raise ValueError(f"Unknown ranking method: {ranking_method}")

    # Determine rebalancing dates
    rebalance_dates = _get_rebalance_dates(ranks.index, rebalance_freq)
    print(f"   Rebalance dates: {len(rebalance_dates)} total")

    # Generate entry/exit signals for each asset
    result = {}

    for symbol in data_dict.keys():
        data = data_dict[symbol]
        asset_ranks = ranks[symbol]

        # Initialize signal arrays
        entry = pd.Series(False, index=data.index)
        exit_signal = pd.Series(False, index=data.index)

        # Track if we're currently holding this asset
        holding = False

        for date in rebalance_dates:
            if date not in asset_ranks.index:
                continue  # Skip if no rank available

            # Check if this asset should be in top N
            rank = asset_ranks[date]
            should_hold = rank <= hold_count and not pd.isna(rank)

            if should_hold and not holding:
                # Enter position
                entry[date] = True
                holding = True
            elif not should_hold and holding:
                # Exit position
                exit_signal[date] = True
                holding = False

        signals = pd.DataFrame({
            "entry": entry,
            "exit": exit_signal,
        })

        result[symbol] = {
            "data": data,
            "signals": signals,
        }

        num_entries = signals["entry"].sum()
        num_exits = signals["exit"].sum()
        print(f"  ✓ {symbol}: {num_entries} entries, {num_exits} exits")

    return result


def _get_rebalance_dates(
    all_dates: pd.DatetimeIndex,
    frequency: Literal["daily", "weekly", "monthly"],
) -> list[pd.Timestamp]:
    """
    Get list of rebalancing dates based on frequency.

    Args:
        all_dates: All available dates in the dataset
        frequency: Rebalancing frequency

    Returns:
        List of rebalancing dates
    """
    if frequency == "daily":
        # Rebalance every day
        return list(all_dates)
    elif frequency == "weekly":
        # Rebalance on Mondays (weekday=0)
        # If Monday missing, use first available day of week
        weekly_dates = []
        for date in all_dates:
            if date.weekday() == 0:  # Monday
                weekly_dates.append(date)
        return weekly_dates if weekly_dates else list(all_dates[::5])  # Fallback: every 5th day
    elif frequency == "monthly":
        # Rebalance on first trading day of month
        monthly_dates = []
        current_month = None
        for date in all_dates:
            month_key = (date.year, date.month)
            if month_key != current_month:
                monthly_dates.append(date)
                current_month = month_key
        return monthly_dates
    else:
        raise ValueError(f"Unknown frequency: {frequency}")


def generate_rebalancing_signal_dataset(
    universe_size: int = 10,
    hold_count: int = 5,
    rebalance_freq: Literal["daily", "weekly", "monthly"] = "weekly",
    ranking_method: Literal["momentum", "volatility", "random"] = "momentum",
    ranking_lookback: int = 20,
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    seed: int = 42,
) -> Path:
    """
    Generate rebalancing signal dataset.

    Args:
        universe_size: Number of stocks in universe (10, 50, or 100)
        hold_count: Number of positions to hold simultaneously
        rebalance_freq: Rebalancing frequency
        ranking_method: How to rank assets
        ranking_lookback: Lookback period for ranking
        start_date: Start date for historical data
        end_date: End date (defaults to today)
        seed: Random seed (for random ranking)

    Returns:
        Path to saved signal dataset
    """
    print("=" * 80)
    print(f"Generating Rebalancing Signal Dataset ({universe_size} stocks)")
    print("=" * 80)

    # Validation
    if hold_count >= universe_size:
        raise ValueError(f"hold_count ({hold_count}) must be < universe_size ({universe_size})")

    # Get symbols
    symbols = get_sp500_top_symbols(universe_size)
    print(f"\n📋 Symbols: {len(symbols)} stocks")

    # Download data
    data_dict = download_stock_data(symbols, start_date, end_date)

    # Generate rebalancing signals
    multi_asset_signals = generate_rebalancing_signals(
        data_dict,
        hold_count=hold_count,
        rebalance_freq=rebalance_freq,
        ranking_method=ranking_method,
        ranking_lookback=ranking_lookback,
        seed=seed,
    )

    # Prepare metadata
    metadata = {
        "signal_type": "rebalancing",
        "parameters": {
            "hold_count": hold_count,
            "rebalance_freq": rebalance_freq,
            "ranking_method": ranking_method,
            "ranking_lookback": ranking_lookback,
            "seed": seed if ranking_method == "random" else None,
        },
        "universe": f"sp500_top{universe_size}",
        "start_date": start_date,
        "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
    }

    # Save with descriptive filename
    output_name = f"sp500_top{universe_size}_rebal_{ranking_method}_top{hold_count}_{rebalance_freq}"

    output_path = save_multi_asset_signal_set(
        output_name, multi_asset_signals, metadata
    )

    print(f"\n✅ Rebalancing signal dataset generation complete!")
    return output_path


if __name__ == "__main__":
    import sys

    # Default parameters
    universe_size = 10
    hold_count = 5
    rebalance_freq = "weekly"
    ranking_method = "momentum"

    # Parse command-line arguments
    if len(sys.argv) > 1:
        try:
            universe_size = int(sys.argv[1])
            if universe_size not in [10, 50, 100]:
                raise ValueError("Universe size must be 10, 50, or 100")
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: python generate_rebalancing.py [universe] [hold_count] [freq] [method]")
            print("  universe: 10, 50, or 100 (default: 10)")
            print("  hold_count: number of positions to hold (default: 5)")
            print("  freq: daily, weekly, or monthly (default: weekly)")
            print("  method: momentum, volatility, or random (default: momentum)")
            print("\nExamples:")
            print("  python generate_rebalancing.py 50 10 weekly momentum")
            print("  python generate_rebalancing.py 100 20 monthly volatility")
            sys.exit(1)

    if len(sys.argv) > 2:
        hold_count = int(sys.argv[2])

    if len(sys.argv) > 3:
        rebalance_freq = sys.argv[3]
        if rebalance_freq not in ["daily", "weekly", "monthly"]:
            print(f"Error: freq must be daily, weekly, or monthly")
            sys.exit(1)

    if len(sys.argv) > 4:
        ranking_method = sys.argv[4]
        if ranking_method not in ["momentum", "volatility", "random"]:
            print(f"Error: method must be momentum, volatility, or random")
            sys.exit(1)

    # Validate hold_count
    if hold_count >= universe_size:
        print(f"Error: hold_count ({hold_count}) must be < universe_size ({universe_size})")
        sys.exit(1)

    # Generate dataset
    generate_rebalancing_signal_dataset(
        universe_size=universe_size,
        hold_count=hold_count,
        rebalance_freq=rebalance_freq,
        ranking_method=ranking_method,
        ranking_lookback=20,
        start_date="2020-01-01",
        seed=42,
    )
