"""
Generate multi-asset signal datasets for large-scale validation.

Creates pre-computed signals for 10, 50, 100, and 1000-stock universes
to validate ml4t.backtest against VectorBT/Backtrader at production scale.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

# Directories
SIGNAL_DIR = Path(__file__).parent


def get_sp500_top_symbols(n: int = 10) -> list[str]:
    """
    Get top N SP500 stocks by market cap.

    For simplicity, using a hardcoded list of large-cap stocks.
    In production, you'd query actual market cap data.

    Args:
        n: Number of stocks to return

    Returns:
        List of ticker symbols
    """
    # Top SP500 stocks by market cap (as of 2024)
    top_stocks = [
        # Mega caps (>$1T)
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet
        "AMZN",  # Amazon
        "NVDA",  # NVIDIA
        "META",  # Meta
        "TSLA",  # Tesla
        "LLY",   # Eli Lilly
        "V",     # Visa
        "UNH",   # UnitedHealth

        # Large caps ($500B-$1T)
        "JNJ",   # Johnson & Johnson
        "WMT",   # Walmart
        "XOM",   # Exxon Mobil
        "JPM",   # JPMorgan Chase
        "PG",    # Procter & Gamble
        "MA",    # Mastercard
        "HD",    # Home Depot
        "CVX",   # Chevron
        "ABBV",  # AbbVie
        "MRK",   # Merck

        # More large caps
        "KO",    # Coca-Cola
        "PEP",   # PepsiCo
        "COST",  # Costco
        "AVGO",  # Broadcom
        "TMO",   # Thermo Fisher
        "ORCL",  # Oracle
        "CSCO",  # Cisco
        "ACN",   # Accenture
        "NKE",   # Nike
        "ABT",   # Abbott

        # Mid-large caps
        "TXN",   # Texas Instruments
        "CRM",   # Salesforce
        "DHR",   # Danaher
        "AMD",   # AMD
        "QCOM",  # Qualcomm
        "INTC",  # Intel
        "UNP",   # Union Pacific
        "NEE",   # NextEra Energy
        "PM",    # Philip Morris
        "UPS",   # UPS

        # Additional for 50-stock dataset
        "RTX",   # Raytheon
        "HON",   # Honeywell
        "LOW",   # Lowe's
        "IBM",   # IBM
        "BA",    # Boeing
        "GE",    # General Electric
        "CAT",   # Caterpillar
        "MDT",   # Medtronic
        "BMY",   # Bristol Myers Squibb
        "SPGI",  # S&P Global

        # Additional for 100-stock dataset (50-100)
        "LIN",   # Linde
        "AMGN",  # Amgen
        "PLD",   # Prologis
        "ISRG",  # Intuitive Surgical
        "BLK",   # BlackRock
        "TJX",   # TJ Maxx
        "SYK",   # Stryker
        "BKNG",  # Booking Holdings
        "GILD",  # Gilead Sciences
        "VRTX",  # Vertex Pharmaceuticals
        "ADP",   # Automatic Data Processing
        "SCHW",  # Charles Schwab
        "ADI",   # Analog Devices
        "MDLZ",  # Mondelez
        "CB",    # Chubb
        "REGN",  # Regeneron
        "MMC",   # Marsh & McLennan
        "DUK",   # Duke Energy
        "SO",    # Southern Company
        "CI",    # Cigna
        "MO",    # Altria
        "SHW",   # Sherwin-Williams
        "ZTS",   # Zoetis
        "PGR",   # Progressive
        "BDX",   # Becton Dickinson
        "CME",   # CME Group
        "LRCX",  # Lam Research
        "USB",   # U.S. Bancorp
        "PNC",   # PNC Financial
        "TGT",   # Target
        "AON",   # Aon
        "ITW",   # Illinois Tool Works
        "CL",    # Colgate-Palmolive
        "EOG",   # EOG Resources
        "APD",   # Air Products
        "BSX",   # Boston Scientific
        "CSX",   # CSX Corporation
        "MMM",   # 3M
        "GD",    # General Dynamics
        "FCX",   # Freeport-McMoRan
        "SLB",   # Schlumberger
        "EMR",   # Emerson Electric
        "NOC",   # Northrop Grumman
        "NSC",   # Norfolk Southern
        "AXP",   # American Express
        "FI",    # Fiserv
        "ICE",   # Intercontinental Exchange
        "HCA",   # HCA Healthcare
        "WM",    # Waste Management
    ]

    return top_stocks[:n]


def download_stock_data(
    symbols: list[str],
    start_date: str = "2020-01-01",
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download stock data using yfinance.

    Args:
        symbols: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        Dictionary mapping symbol -> OHLCV DataFrame
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance required for stock data download. "
            "Install with: pip install yfinance"
        )

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"📊 Downloading {len(symbols)} stocks from {start_date} to {end_date}...")

    data_dict = {}
    failed = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                print(f"  ⚠️  {symbol}: No data available")
                failed.append(symbol)
                continue

            # Standardize column names (lowercase)
            df.columns = [col.lower() for col in df.columns]

            # Keep only OHLCV
            required_cols = ["open", "high", "low", "close", "volume"]
            df = df[required_cols]

            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Remove timezone info for consistency
            df.index = df.index.tz_localize(None)

            data_dict[symbol] = df
            print(f"  ✓ {symbol}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

        except Exception as e:
            print(f"  ✗ {symbol}: Failed - {e}")
            failed.append(symbol)

    if failed:
        print(f"\n⚠️  Failed to download: {', '.join(failed)}")

    print(f"\n✅ Successfully downloaded {len(data_dict)}/{len(symbols)} stocks")

    return data_dict


def generate_sma_crossover_signals(
    prices: pd.Series, fast: int = 10, slow: int = 20
) -> pd.DataFrame:
    """
    Generate SMA crossover signals (entry/exit booleans).

    Same logic as single-asset version in generate.py.

    CRITICAL: Strips leading EXIT signals to ensure first signal is always ENTRY.
    This prevents frameworks from diverging due to undefined initial position state.

    Args:
        prices: Price series (typically 'close')
        fast: Fast SMA period
        slow: Slow SMA period

    Returns:
        DataFrame with boolean columns ['entry', 'exit']
    """
    # Calculate SMAs
    sma_fast = prices.rolling(window=fast, min_periods=fast).mean()
    sma_slow = prices.rolling(window=slow, min_periods=slow).mean()

    # Detect crossovers
    # Entry: fast crosses ABOVE slow
    entry = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))

    # Exit: fast crosses BELOW slow
    exit_signal = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

    # Create signals DataFrame
    signals = pd.DataFrame({"entry": entry.fillna(False), "exit": exit_signal.fillna(False)})

    # CRITICAL FIX: Strip leading EXIT signals
    # Problem: If first signal is EXIT, frameworks have no position to exit
    # Solution: Find first ENTRY and clear all signals before it
    if signals['entry'].any():
        first_entry_idx = signals['entry'].idxmax()
        # Clear all signals before first entry
        signals.loc[:first_entry_idx, ['entry', 'exit']] = False
        # Set the first entry back to True
        signals.loc[first_entry_idx, 'entry'] = True

    return signals


def generate_multi_asset_signals(
    data_dict: dict[str, pd.DataFrame],
    signal_func: callable = generate_sma_crossover_signals,
    **signal_params,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Generate signals for multiple assets.

    Args:
        data_dict: Dictionary mapping symbol -> OHLCV DataFrame
        signal_func: Function to generate signals (default: SMA crossover)
        **signal_params: Parameters passed to signal_func

    Returns:
        Dictionary mapping symbol -> {'data': df, 'signals': signals_df}
    """
    print(f"\n📈 Generating signals for {len(data_dict)} assets...")

    result = {}

    for symbol, data in data_dict.items():
        # Generate signals using close prices
        signals = signal_func(data["close"], **signal_params)

        result[symbol] = {
            "data": data,
            "signals": signals,
        }

        num_entries = signals["entry"].sum()
        num_exits = signals["exit"].sum()
        print(f"  ✓ {symbol}: {num_entries} entries, {num_exits} exits")

    return result


def save_multi_asset_signal_set(
    name: str,
    multi_asset_data: dict[str, dict[str, pd.DataFrame]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Save multi-asset signal set to pickle.

    Format:
    {
        'assets': {
            'AAPL': {'data': df, 'signals': signals_df},
            'MSFT': {'data': df, 'signals': signals_df},
            ...
        },
        'metadata': {
            'signal_type': 'sma_crossover',
            'parameters': {'fast': 10, 'slow': 20},
            'num_assets': 10,
            'generated_at': '2025-11-16T16:00:00',
            ...
        }
    }

    Args:
        name: Signal set name (e.g., 'sp500_top10_sma_crossover')
        multi_asset_data: Dictionary of asset data + signals
        metadata: Optional metadata

    Returns:
        Path to saved pickle file
    """
    output_file = SIGNAL_DIR / f"{name}.pkl"

    # Create signal set
    signal_set = {
        "assets": multi_asset_data,
        "metadata": metadata or {},
    }

    # Add auto-generated metadata
    if "generated_at" not in signal_set["metadata"]:
        signal_set["metadata"]["generated_at"] = datetime.now().isoformat()

    signal_set["metadata"]["num_assets"] = len(multi_asset_data)
    signal_set["metadata"]["symbols"] = sorted(multi_asset_data.keys())

    # Save to pickle
    with open(output_file, "wb") as f:
        pickle.dump(signal_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"\n💾 Saved to: {output_file}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Assets: {len(multi_asset_data)}")

    return output_file


def generate_sp500_top10_dataset(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    fast: int = 10,
    slow: int = 20,
) -> Path:
    """
    Generate 10-stock signal dataset (SP500 top 10 by market cap).

    Args:
        start_date: Start date for historical data
        end_date: End date (defaults to today)
        fast: Fast SMA period
        slow: Slow SMA period

    Returns:
        Path to saved signal dataset
    """
    print("=" * 80)
    print("Generating SP500 Top 10 Signal Dataset")
    print("=" * 80)

    # Get symbols
    symbols = get_sp500_top_symbols(10)
    print(f"\n📋 Symbols: {', '.join(symbols)}")

    # Download data
    data_dict = download_stock_data(symbols, start_date, end_date)

    # Generate signals
    multi_asset_signals = generate_multi_asset_signals(
        data_dict, signal_func=generate_sma_crossover_signals, fast=fast, slow=slow
    )

    # Prepare metadata
    metadata = {
        "signal_type": "sma_crossover",
        "parameters": {"fast": fast, "slow": slow},
        "universe": "sp500_top10",
        "start_date": start_date,
        "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
    }

    # Save
    output_path = save_multi_asset_signal_set(
        "sp500_top10_sma_crossover", multi_asset_signals, metadata
    )

    print("\n✅ SP500 Top 10 dataset generation complete!")
    return output_path


def generate_sp500_top50_dataset(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    fast: int = 10,
    slow: int = 20,
) -> Path:
    """Generate 50-stock signal dataset (SP500 top 50)."""
    print("=" * 80)
    print("Generating SP500 Top 50 Signal Dataset")
    print("=" * 80)

    symbols = get_sp500_top_symbols(50)
    print(f"\n📋 Symbols ({len(symbols)} total)")

    data_dict = download_stock_data(symbols, start_date, end_date)
    multi_asset_signals = generate_multi_asset_signals(
        data_dict, signal_func=generate_sma_crossover_signals, fast=fast, slow=slow
    )

    metadata = {
        "signal_type": "sma_crossover",
        "parameters": {"fast": fast, "slow": slow},
        "universe": "sp500_top50",
        "start_date": start_date,
        "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
    }

    output_path = save_multi_asset_signal_set(
        "sp500_top50_sma_crossover", multi_asset_signals, metadata
    )

    print("\n✅ SP500 Top 50 dataset generation complete!")
    return output_path


def generate_sp500_top100_dataset(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    fast: int = 10,
    slow: int = 20,
) -> Path:
    """Generate 100-stock signal dataset (SP500 top 100)."""
    print("=" * 80)
    print("Generating SP500 Top 100 Signal Dataset")
    print("=" * 80)

    symbols = get_sp500_top_symbols(100)
    print(f"\n📋 Symbols ({len(symbols)} total)")

    data_dict = download_stock_data(symbols, start_date, end_date)
    multi_asset_signals = generate_multi_asset_signals(
        data_dict, signal_func=generate_sma_crossover_signals, fast=fast, slow=slow
    )

    metadata = {
        "signal_type": "sma_crossover",
        "parameters": {"fast": fast, "slow": slow},
        "universe": "sp500_top100",
        "start_date": start_date,
        "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
    }

    output_path = save_multi_asset_signal_set(
        "sp500_top100_sma_crossover", multi_asset_signals, metadata
    )

    print("\n✅ SP500 Top 100 dataset generation complete!")
    return output_path


if __name__ == "__main__":
    import sys

    # Allow command-line selection of dataset size
    if len(sys.argv) > 1:
        dataset_size = sys.argv[1]
        if dataset_size == "10":
            generate_sp500_top10_dataset(start_date="2020-01-01")
        elif dataset_size == "50":
            generate_sp500_top50_dataset(start_date="2020-01-01")
        elif dataset_size == "100":
            generate_sp500_top100_dataset(start_date="2020-01-01")
        else:
            print(f"Unknown dataset size: {dataset_size}")
            print("Usage: python generate_multi_asset.py [10|50|100]")
            sys.exit(1)
    else:
        # Default: generate 10-stock dataset
        generate_sp500_top10_dataset(start_date="2020-01-01")
