"""Market data fixtures for validation testing.

Provides utilities to load real market data from various sources for
use in validation scenarios.
"""

from pathlib import Path
from datetime import datetime, date
import polars as pl
import pandas as pd
from typing import Optional


# Data source paths
PROJECTS_DIR = Path(__file__).parents[4] / 'projects'
WIKI_PRICES_PATH = PROJECTS_DIR / 'daily_us_equities' / 'wiki_prices.parquet'


def load_wiki_prices(use_adjusted: bool = False) -> pl.DataFrame:
    """
    Load the complete Quandl Wiki prices dataset.

    Args:
        use_adjusted: If True, use adjusted OHLCV prices (default: False)

    Returns:
        DataFrame with columns: ticker, date, open, high, low, close, volume
    """
    if not WIKI_PRICES_PATH.exists():
        raise FileNotFoundError(
            f"Wiki prices not found at {WIKI_PRICES_PATH}\n"
            f"Please ensure the data exists in ~/ml4t/software/projects/daily_us_equities/"
        )

    df = pl.read_parquet(WIKI_PRICES_PATH)

    # Select appropriate price columns
    if use_adjusted:
        price_cols = {
            'ticker': 'ticker',
            'date': 'date',
            'adj_open': 'open',
            'adj_high': 'high',
            'adj_low': 'low',
            'adj_close': 'close',
            'adj_volume': 'volume',
        }
    else:
        price_cols = {
            'ticker': 'ticker',
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
        }

    return df.select(list(price_cols.keys())).rename(price_cols)


def get_ticker_data(
    ticker: str,
    start_date: Optional[datetime | date | str] = None,
    end_date: Optional[datetime | date | str] = None,
    use_adjusted: bool = False,
) -> pl.DataFrame:
    """
    Get OHLCV data for a specific ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date for filtering (inclusive)
        end_date: End date for filtering (inclusive)
        use_adjusted: If True, use adjusted prices

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume

    Example:
        >>> df = get_ticker_data('AAPL', '2020-01-01', '2020-12-31')
        >>> print(df.head())
    """
    # Load full dataset
    df = load_wiki_prices(use_adjusted=use_adjusted)

    # Filter to ticker
    df = df.filter(pl.col('ticker') == ticker)

    # Apply date filters
    if start_date:
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date).date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()
        df = df.filter(pl.col('date') >= start_date)

    if end_date:
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date).date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()
        df = df.filter(pl.col('date') <= end_date)

    # Rename columns to standard format
    df = df.rename({'ticker': 'symbol', 'date': 'timestamp'})

    # Ensure timestamp is datetime (not just date)
    df = df.with_columns(
        pl.col('timestamp').cast(pl.Datetime(time_zone='UTC'))
    )

    # Sort by timestamp
    df = df.sort('timestamp')

    return df


def prepare_zipline_bundle_data(
    tickers: list[str],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    output_dir: Path,
    use_adjusted: bool = True,
) -> dict:
    """
    Prepare market data in Zipline bundle format.

    Creates an HDF5 store with:
    - Individual ticker data (stored by SID)
    - Equity metadata
    - Splits and dividends (from wiki_prices)

    Args:
        tickers: List of ticker symbols to include
        start_date: Start date for data range
        end_date: End date for data range
        output_dir: Directory to save bundle data
        use_adjusted: Whether to use adjusted prices

    Returns:
        dict with metadata about created bundle
    """
    import h5py

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_file = output_dir / 'validation_bundle.h5'

    # Load full dataset with all columns for splits/dividends
    df_full = pl.read_parquet(WIKI_PRICES_PATH)

    # Get just OHLCV for price data
    df = load_wiki_prices(use_adjusted=False)

    # Convert dates
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date).date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()

    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date).date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()

    # Filter to date range
    df = df.filter(
        (pl.col('date') >= start_date) & (pl.col('date') <= end_date)
    )

    # Create metadata DataFrame
    metadata = []

    # Write to HDF5 using pandas (h5py doesn't work well with polars)
    with pd.HDFStore(bundle_file, 'w') as store:
        for sid, ticker in enumerate(tickers):
            ticker_df = df.filter(pl.col('ticker') == ticker)

            if len(ticker_df) == 0:
                print(f"Warning: No data found for {ticker}")
                continue

            # Convert to pandas for HDF5 storage
            ticker_pd = ticker_df.to_pandas()
            ticker_pd = ticker_pd.set_index('date')

            # Remove ticker column (redundant in per-ticker storage)
            ticker_pd = ticker_pd.drop(columns=['ticker'])

            # Ensure proper timezone (UTC) for timestamps
            ticker_pd.index = pd.DatetimeIndex(ticker_pd.index).tz_localize('UTC')

            # Store ticker data by SID
            store.put(str(sid), ticker_pd, format='table')

            # Add to metadata
            metadata.append({
                'sid': sid,
                'symbol': ticker,
                'asset_name': ticker,
                'start_date': ticker_pd.index[0],
                'end_date': ticker_pd.index[-1],
                'first_traded': ticker_pd.index[0],
                'auto_close_date': ticker_pd.index[-1] + pd.Timedelta(days=1),
                'exchange': 'NYSE',
            })

        # Store metadata
        metadata_df = pd.DataFrame(metadata)
        store.put('equities', metadata_df, format='table')

        # Extract splits and dividends from full dataset
        splits_df = df_full.filter(
            (pl.col('ticker').is_in(tickers)) & (pl.col('split_ratio') != 1.0)
        ).select(['ticker', 'date', 'split_ratio']).to_pandas()

        if len(splits_df) > 0:
            # Map ticker to sid
            ticker_to_sid = {row['symbol']: row['sid'] for row in metadata}
            splits_df['sid'] = splits_df['ticker'].map(ticker_to_sid)
            splits_df = splits_df.rename(columns={
                'date': 'effective_date',
                'split_ratio': 'ratio'
            })
            splits_df = splits_df[['sid', 'effective_date', 'ratio']]
            splits_df['effective_date'] = pd.to_datetime(splits_df['effective_date']).dt.tz_localize('UTC')
            store.put('splits', splits_df, format='table')

        # Dividends
        divs_df = df_full.filter(
            (pl.col('ticker').is_in(tickers)) & (pl.col('ex-dividend') > 0)
        ).select(['ticker', 'date', 'ex-dividend']).to_pandas()

        if len(divs_df) > 0:
            ticker_to_sid = {row['symbol']: row['sid'] for row in metadata}
            divs_df['sid'] = divs_df['ticker'].map(ticker_to_sid)
            divs_df = divs_df.rename(columns={
                'date': 'ex_date',
                'ex-dividend': 'amount'
            })
            divs_df = divs_df[['sid', 'ex_date', 'amount']]
            divs_df['ex_date'] = pd.to_datetime(divs_df['ex_date']).dt.tz_localize('UTC')

            # Add required columns for Zipline
            divs_df['record_date'] = divs_df['ex_date']
            divs_df['declared_date'] = divs_df['ex_date'] - pd.Timedelta(days=30)
            divs_df['pay_date'] = divs_df['ex_date'] + pd.Timedelta(days=14)

            store.put('dividends', divs_df, format='table')

    return {
        'bundle_file': bundle_file,
        'tickers': tickers,
        'num_tickers': len(metadata),
        'start_date': start_date,
        'end_date': end_date,
        'metadata': metadata,
    }
