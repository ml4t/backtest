"""Generate synthetic OHLCV data for validation tests and load real data."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_ohlcv(
    n_bars: int = 1000,
    symbol: str = "BTC",
    start_date: str = "2021-01-04 00:00:00",
    freq: str = "1min",
    base_price: float = 35000.0,
    volatility: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data.

    Properties:
    - Close of bar N equals Open of bar N+1 (continuous)
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - Realistic price movements with controlled volatility
    - Fixed seed for reproducibility

    Args:
        n_bars: Number of bars to generate
        symbol: Asset symbol
        start_date: Starting timestamp
        freq: Bar frequency (e.g., '1min', '1h', '1D')
        base_price: Starting price
        volatility: Price volatility (std dev of returns)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume]
    """
    np.random.seed(seed)

    # Generate timestamps
    start = pd.Timestamp(start_date, tz='UTC')
    timestamps = pd.date_range(start=start, periods=n_bars, freq=freq)

    # Generate returns with controlled volatility
    returns = np.random.normal(0, volatility, n_bars)

    # Generate close prices from returns
    close_prices = np.zeros(n_bars)
    close_prices[0] = base_price

    for i in range(1, n_bars):
        close_prices[i] = close_prices[i-1] * (1 + returns[i])

    # Generate open prices (close of previous bar)
    open_prices = np.zeros(n_bars)
    open_prices[0] = base_price
    open_prices[1:] = close_prices[:-1]

    # Generate high/low within OHLC constraints
    high_prices = np.zeros(n_bars)
    low_prices = np.zeros(n_bars)

    for i in range(n_bars):
        # High is max of open/close + small random amount
        bar_max = max(open_prices[i], close_prices[i])
        high_prices[i] = bar_max * (1 + abs(np.random.normal(0, volatility/2)))

        # Low is min of open/close - small random amount
        bar_min = min(open_prices[i], close_prices[i])
        low_prices[i] = bar_min * (1 - abs(np.random.normal(0, volatility/2)))

    # Generate volume (random but realistic)
    volumes = np.random.uniform(100, 1000, n_bars)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes,
        'symbol': symbol,
    })

    # Set timestamp as index
    df = df.set_index('timestamp')

    return df


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV data for common issues.

    Checks:
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - No NaN values
    - Prices > 0
    - Close[i-1] == Open[i] (continuity)

    Args:
        df: OHLCV DataFrame

    Returns:
        True if valid, raises AssertionError otherwise
    """
    # Check for NaN
    assert not df.isnull().any().any(), "OHLCV contains NaN values"

    # Check positive prices
    assert (df['open'] > 0).all(), "Open prices must be positive"
    assert (df['high'] > 0).all(), "High prices must be positive"
    assert (df['low'] > 0).all(), "Low prices must be positive"
    assert (df['close'] > 0).all(), "Close prices must be positive"

    # Check OHLC relationships
    assert (df['high'] >= df['open']).all(), "High must be >= Open"
    assert (df['high'] >= df['close']).all(), "High must be >= Close"
    assert (df['low'] <= df['open']).all(), "Low must be <= Open"
    assert (df['low'] <= df['close']).all(), "Low must be <= Close"

    # Check continuity (close[i-1] == open[i])
    if len(df) > 1:
        close_shifted = df['close'].shift(1).iloc[1:]
        open_current = df['open'].iloc[1:]
        assert np.allclose(close_shifted.values, open_current.values, rtol=1e-9), \
            "Close of bar N must equal Open of bar N+1"

    return True


def load_real_crypto_data(
    symbol: str = "BTC",
    data_type: str = "spot",
    start_date: str | None = None,
    end_date: str | None = None,
    n_bars: int | None = None,
) -> pd.DataFrame:
    """
    Load real cryptocurrency data from the projects directory.

    Args:
        symbol: Asset symbol (BTC, ETH, etc.)
        data_type: 'spot' or 'futures'
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        n_bars: Optional number of bars to load (from start_date or beginning)

    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume, symbol]
        Index is timestamp (datetime)
    """
    # Find the data file
    base_path = Path(__file__).parent.parent.parent.parent.parent
    data_path = base_path / "projects" / "crypto_futures" / "data" / data_type / f"{symbol}_{data_type}.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading real {symbol} {data_type} data from {data_path.name}...")

    # Load with polars (fast) then convert to pandas
    try:
        import polars as pl
        df_pl = pl.read_parquet(data_path)
        df = df_pl.to_pandas()
    except ImportError:
        # Fallback to pandas if polars not available
        df = pd.read_parquet(data_path)

    # Standardize column names
    df = df.rename(columns={'timestamp': 'timestamp'})

    # Filter by date range if specified
    if start_date is not None:
        start = pd.Timestamp(start_date, tz='UTC')
        df = df[df['timestamp'] >= start]

    if end_date is not None:
        end = pd.Timestamp(end_date, tz='UTC')
        df = df[df['timestamp'] <= end]

    # Limit to n_bars if specified
    if n_bars is not None:
        df = df.head(n_bars)

    # Set timestamp as index
    df = df.set_index('timestamp')

    # Keep only OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume', 'symbol']]

    print(f"   ✅ Loaded {len(df):,} bars")
    print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
    print(f"   💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


if __name__ == "__main__":
    # Test data generation
    print("Generating synthetic OHLCV data...")
    df = generate_ohlcv(n_bars=100)

    print(f"\nGenerated {len(df)} bars")
    print(f"\nFirst 5 bars:")
    print(df.head())

    print(f"\nPrice range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"Validating OHLCV constraints...")

    try:
        validate_ohlcv(df)
        print("✅ All OHLCV constraints satisfied")
    except AssertionError as e:
        print(f"❌ Validation failed: {e}")
