"""Generate pre-calculated signals for framework validation.

Signals are computed ONCE, independently, and saved to disk.
All frameworks load these SAME signals for validation.

Signal Format:
    {
        'data': pd.DataFrame(index=DatetimeIndex, columns=['open','high','low','close','volume']),
        'signals': pd.DataFrame(index=DatetimeIndex, columns=['entry','exit']),  # Boolean
        'metadata': {
            'signal_type': str,
            'parameters': dict,
            'generated_at': str,
            'asset': str
        }
    }
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

# Directories
SIGNAL_DIR = Path(__file__).parent
# We're in: /home/stefan/ml4t/software/backtest/tests/validation/signals/
# Crypto data is in: /home/stefan/ml4t/software/projects/crypto_futures/data/
# Go up to /home/stefan/ml4t/software/backtest then ../projects
CRYPTO_DATA_DIR = (SIGNAL_DIR.parent.parent.parent.parent / "projects" / "crypto_futures" / "data")


def load_crypto_data(asset: str = "BTC", resample_to_daily: bool = True) -> pd.DataFrame:
    """Load crypto data from ../../projects/crypto_futures/data/.

    Args:
        asset: Crypto asset symbol (BTC, ETH, SOL)
        resample_to_daily: If True, resample hourly data to daily

    Returns:
        DataFrame with OHLCV data (index=DatetimeIndex)

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    # Try to find BTC data file
    potential_files = [
        CRYPTO_DATA_DIR / f"{asset.lower()}_corrected_backtest.parquet",
        CRYPTO_DATA_DIR / f"{asset.lower()}_backtest_results.parquet",
        CRYPTO_DATA_DIR / f"{asset}_daily.parquet",
        CRYPTO_DATA_DIR / f"{asset.lower()}_daily.parquet",
    ]

    for file_path in potential_files:
        if file_path.exists():
            # Load with Polars then convert to Pandas
            df_pl = pl.read_parquet(file_path)

            # Convert to Pandas
            df = df_pl.to_pandas()

            # Try to identify timestamp column
            timestamp_cols = [
                col
                for col in df.columns
                if "time" in col.lower()
                or "date" in col.lower()
                or col.lower() == "timestamp"
                or col.lower() == "hour"
            ]

            if timestamp_cols:
                df = df.set_index(timestamp_cols[0])

            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Standardize column names (lowercase)
            df.columns = [col.lower() for col in df.columns]

            # Verify required columns
            required_cols = {"open", "high", "low", "close", "volume"}
            available_cols = set(df.columns)

            if not required_cols.issubset(available_cols):
                missing = required_cols - available_cols
                raise ValueError(
                    f"Data file {file_path} missing required columns: {missing}. "
                    f"Available: {available_cols}"
                )

            # Extract just OHLCV
            df = df[["open", "high", "low", "close", "volume"]]

            # Resample to daily if requested
            if resample_to_daily:
                df = df.resample("D").agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                # Drop any NaN rows (from resampling)
                df = df.dropna()

            return df

    raise FileNotFoundError(
        f"Could not find {asset} data in {CRYPTO_DATA_DIR}. Tried: {[f.name for f in potential_files]}"
    )


def generate_sma_crossover(
    prices: pd.Series, fast: int = 10, slow: int = 20
) -> pd.DataFrame:
    """Generate SMA crossover signals (entry/exit booleans).

    Args:
        prices: Price series (typically 'close')
        fast: Fast SMA period (default: 10)
        slow: Slow SMA period (default: 20)

    Returns:
        DataFrame with boolean columns ['entry', 'exit']
            - entry: True when fast SMA crosses above slow SMA
            - exit: True when fast SMA crosses below slow SMA

    Example:
        >>> prices = df['close']
        >>> signals = generate_sma_crossover(prices, fast=10, slow=20)
        >>> signals.head()
                        entry  exit
        2024-01-01      False  False
        2024-01-02       True  False  # Fast crosses above slow
        2024-01-03      False  False
        2024-01-04      False   True  # Fast crosses below slow
    """
    # Calculate SMAs
    sma_fast = prices.rolling(window=fast, min_periods=fast).mean()
    sma_slow = prices.rolling(window=slow, min_periods=slow).mean()

    # Detect crossovers
    # Entry: fast crosses ABOVE slow (fast > slow AND previous fast <= previous slow)
    entry = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))

    # Exit: fast crosses BELOW slow (fast < slow AND previous fast >= previous slow)
    exit_signal = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_signal.fillna(False)})


def save_signal_set(
    name: str,
    data: pd.DataFrame,
    signals: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save data + signals as pickle for framework validation.

    Args:
        name: Signal set name (e.g., 'btc_sma_crossover_daily')
        data: OHLCV DataFrame
        signals: Boolean entry/exit DataFrame
        metadata: Optional metadata dict

    Returns:
        Path to saved pickle file

    Example:
        >>> data = load_crypto_data('BTC')
        >>> signals = generate_sma_crossover(data['close'], fast=10, slow=20)
        >>> metadata = {'signal_type': 'sma_crossover', 'parameters': {'fast': 10, 'slow': 20}}
        >>> path = save_signal_set('btc_sma_crossover_daily', data, signals, metadata)
    """
    output_file = SIGNAL_DIR / f"{name}.pkl"

    # Validate signal format
    if not isinstance(signals.index, pd.DatetimeIndex):
        raise ValueError("Signals must have DatetimeIndex")

    if not all(col in signals.columns for col in ["entry", "exit"]):
        raise ValueError("Signals must have 'entry' and 'exit' columns")

    if signals.index.name != data.index.name:
        raise ValueError("Signal and data indexes must have same name")

    # Validate all timestamps in signals exist in data
    missing_timestamps = signals.index.difference(data.index)
    if len(missing_timestamps) > 0:
        raise ValueError(
            f"Signal timestamps not in data: {missing_timestamps[:5]} "
            f"(showing first 5 of {len(missing_timestamps)})"
        )

    # Create signal set
    signal_set = {
        "data": data,
        "signals": signals,
        "metadata": metadata or {},
    }

    # Add generation timestamp if not in metadata
    if "generated_at" not in signal_set["metadata"]:
        signal_set["metadata"]["generated_at"] = datetime.now().isoformat()

    # Save to pickle
    with open(output_file, "wb") as f:
        pickle.dump(signal_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    return output_file


def load_signal_set(name: str) -> dict[str, Any]:
    """Load signal set from pickle file.

    Args:
        name: Signal set name (e.g., 'btc_sma_crossover_daily')

    Returns:
        Dictionary with:
            - data: OHLCV DataFrame
            - signals: Boolean entry/exit DataFrame
            - metadata: Signal metadata dict

    Example:
        >>> signal_set = load_signal_set('btc_sma_crossover_daily')
        >>> data = signal_set['data']
        >>> signals = signal_set['signals']
        >>> print(signal_set['metadata'])
    """
    signal_file = SIGNAL_DIR / f"{name}.pkl"

    if not signal_file.exists():
        raise FileNotFoundError(
            f"Signal set '{name}' not found. Expected file: {signal_file}. "
            f"Available signals: {list_available_signals()}"
        )

    with open(signal_file, "rb") as f:
        signal_set = pickle.load(f)

    # Validate loaded format
    required_keys = {"data", "signals", "metadata"}
    if not required_keys.issubset(signal_set.keys()):
        raise ValueError(f"Invalid signal set format. Required keys: {required_keys}")

    return signal_set


def list_available_signals() -> list[str]:
    """List all available signal sets.

    Returns:
        List of signal set names (without .pkl extension)
    """
    return [f.stem for f in SIGNAL_DIR.glob("*.pkl")]


def generate_btc_sma_crossover() -> Path:
    """Convenience function to generate BTC SMA(10,20) crossover signals.

    Returns:
        Path to saved signal file

    Example:
        >>> path = generate_btc_sma_crossover()
        >>> print(f"Signals saved to: {path}")
    """
    print("Loading BTC data...")
    data = load_crypto_data("BTC")

    print(f"Loaded {len(data)} BTC daily bars from {data.index[0]} to {data.index[-1]}")

    print("Generating SMA(10,20) crossover signals...")
    signals = generate_sma_crossover(data["close"], fast=10, slow=20)

    entry_count = signals["entry"].sum()
    exit_count = signals["exit"].sum()
    print(f"Generated {entry_count} entry signals and {exit_count} exit signals")

    metadata = {
        "asset": "BTC",
        "signal_type": "sma_crossover",
        "parameters": {"fast": 10, "slow": 20},
        "data_source": "projects/crypto_futures/data/",
        "num_bars": len(data),
        "date_range": f"{data.index[0]} to {data.index[-1]}",
        "num_entries": int(entry_count),
        "num_exits": int(exit_count),
    }

    print("Saving signal set...")
    output_path = save_signal_set("btc_sma_crossover_daily", data, signals, metadata)

    print(f"✓ Saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate BTC SMA crossover signals for validation
    generate_btc_sma_crossover()
