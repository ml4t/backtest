"""
Generate random signal datasets for stress testing and edge case validation.

Creates pre-computed random signals with configurable entry/exit frequency
to test framework robustness under various trading patterns.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

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


def generate_random_signals(
    prices: pd.Series,
    entry_prob: float = 0.05,
    exit_prob: float = 0.05,
    min_holding_bars: int = 5,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate random entry/exit signals with configurable probabilities.

    Key features:
    - Random entries with specified probability per bar
    - Random exits only when holding a position
    - Minimum holding period to avoid whipsaw
    - Reproducible with seed

    Args:
        prices: Price series (used only for index/length)
        entry_prob: Probability of entry signal per bar (0.0-1.0)
        exit_prob: Probability of exit signal per bar when in position (0.0-1.0)
        min_holding_bars: Minimum bars to hold before allowing exit
        seed: Random seed for reproducibility

    Returns:
        DataFrame with boolean columns ['entry', 'exit']
    """
    rng = np.random.default_rng(seed)
    n = len(prices)

    # Initialize signal arrays
    entry = np.zeros(n, dtype=bool)
    exit_signal = np.zeros(n, dtype=bool)

    # Track position state
    in_position = False
    bars_held = 0

    for i in range(n):
        if not in_position:
            # Not in position - check for random entry
            if rng.random() < entry_prob:
                entry[i] = True
                in_position = True
                bars_held = 0
        else:
            # In position - check for random exit (respect min holding period)
            bars_held += 1
            if bars_held >= min_holding_bars and rng.random() < exit_prob:
                exit_signal[i] = True
                in_position = False
                bars_held = 0

    return pd.DataFrame(
        {"entry": entry, "exit": exit_signal},
        index=prices.index,
    )


def generate_multi_asset_random_signals(
    data_dict: dict[str, pd.DataFrame],
    entry_prob: float = 0.05,
    exit_prob: float = 0.05,
    min_holding_bars: int = 5,
    seed: int | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Generate random signals for multiple assets.

    Args:
        data_dict: Dictionary mapping symbol -> OHLCV DataFrame
        entry_prob: Probability of entry per bar
        exit_prob: Probability of exit per bar when in position
        min_holding_bars: Minimum holding period
        seed: Random seed (base seed, each asset gets seed+offset)

    Returns:
        Dictionary mapping symbol -> {'data': df, 'signals': signals_df}
    """
    print(f"\n🎲 Generating random signals for {len(data_dict)} assets...")
    print(f"   Entry probability: {entry_prob:.1%} per bar")
    print(f"   Exit probability: {exit_prob:.1%} per bar (when in position)")
    print(f"   Min holding period: {min_holding_bars} bars")

    result = {}

    for idx, (symbol, data) in enumerate(data_dict.items()):
        # Use different seed for each asset to avoid identical signals
        asset_seed = (seed + idx) if seed is not None else None

        # Generate random signals
        signals = generate_random_signals(
            data["close"],
            entry_prob=entry_prob,
            exit_prob=exit_prob,
            min_holding_bars=min_holding_bars,
            seed=asset_seed,
        )

        result[symbol] = {
            "data": data,
            "signals": signals,
        }

        num_entries = signals["entry"].sum()
        num_exits = signals["exit"].sum()
        print(f"  ✓ {symbol}: {num_entries} entries, {num_exits} exits")

    return result


def generate_random_signal_dataset(
    universe_size: int = 10,
    entry_prob: float = 0.05,
    exit_prob: float = 0.05,
    min_holding_bars: int = 5,
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    seed: int = 42,
) -> Path:
    """
    Generate random signal dataset for specified universe size.

    Args:
        universe_size: Number of stocks (10, 50, or 100)
        entry_prob: Entry probability per bar
        exit_prob: Exit probability per bar when in position
        min_holding_bars: Minimum holding period
        start_date: Start date for historical data
        end_date: End date (defaults to today)
        seed: Random seed for reproducibility

    Returns:
        Path to saved signal dataset
    """
    print("=" * 80)
    print(f"Generating Random Signal Dataset ({universe_size} stocks)")
    print("=" * 80)

    # Get symbols
    symbols = get_sp500_top_symbols(universe_size)
    print(f"\n📋 Symbols: {len(symbols)} stocks")

    # Download data
    data_dict = download_stock_data(symbols, start_date, end_date)

    # Generate random signals
    multi_asset_signals = generate_multi_asset_random_signals(
        data_dict,
        entry_prob=entry_prob,
        exit_prob=exit_prob,
        min_holding_bars=min_holding_bars,
        seed=seed,
    )

    # Prepare metadata
    metadata = {
        "signal_type": "random",
        "parameters": {
            "entry_prob": entry_prob,
            "exit_prob": exit_prob,
            "min_holding_bars": min_holding_bars,
            "seed": seed,
        },
        "universe": f"sp500_top{universe_size}",
        "start_date": start_date,
        "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
    }

    # Save with descriptive filename
    freq_pct = int(entry_prob * 100)
    output_name = f"sp500_top{universe_size}_random_{freq_pct}pct"

    output_path = save_multi_asset_signal_set(
        output_name, multi_asset_signals, metadata
    )

    print(f"\n✅ Random signal dataset generation complete!")
    return output_path


if __name__ == "__main__":
    import sys

    # Default parameters
    universe_size = 10
    entry_prob = 0.05  # 5% chance per bar
    exit_prob = 0.05   # 5% chance per bar when in position

    # Parse command-line arguments
    if len(sys.argv) > 1:
        try:
            universe_size = int(sys.argv[1])
            if universe_size not in [10, 50, 100]:
                raise ValueError("Universe size must be 10, 50, or 100")
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: python generate_random.py [universe_size] [entry_prob] [exit_prob]")
            print("  universe_size: 10, 50, or 100 (default: 10)")
            print("  entry_prob: 0.0-1.0 (default: 0.05 = 5%)")
            print("  exit_prob: 0.0-1.0 (default: 0.05 = 5%)")
            sys.exit(1)

    if len(sys.argv) > 2:
        entry_prob = float(sys.argv[2])

    if len(sys.argv) > 3:
        exit_prob = float(sys.argv[3])

    # Generate dataset
    generate_random_signal_dataset(
        universe_size=universe_size,
        entry_prob=entry_prob,
        exit_prob=exit_prob,
        min_holding_bars=5,
        start_date="2020-01-01",
        seed=42,  # Reproducible
    )
