"""Data loader for ../projects/ datasets."""
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl


class DataLoader:
    """Load and standardize data from projects directory.

    Supports multiple formats and provides conversion to platform-specific formats.
    """

    def __init__(self, projects_dir: str = None):
        if projects_dir is None:
            # Auto-detect: look for projects directory relative to this file
            current_file = Path(__file__).resolve()
            # We're in backtest/tests/validation/data/loader.py
            # Projects is at software/projects (go up 4 levels to software/, then to projects/)
            backtest_dir = current_file.parent.parent.parent.parent  # software/backtest/
            projects_dir = backtest_dir.parent / "projects"  # software/projects/

        self.projects_dir = Path(projects_dir)

    def load_daily_equities(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | None = None,
    ) -> pl.DataFrame:
        """Load daily US equities data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols (None = all)

        Returns:
            Polars DataFrame with columns:
            - timestamp, symbol, open, high, low, close, volume
        """
        path = self.projects_dir / "daily_us_equities" / "equity_prices_enhanced_1962_2025.parquet"

        if not path.exists():
            # Fallback to wiki_prices
            path = self.projects_dir / "daily_us_equities" / "wiki_prices.parquet"

        if not path.exists():
            raise FileNotFoundError(f"No daily equities data found at {self.projects_dir}/daily_us_equities/")

        # Load data
        df = pl.read_parquet(path)

        # Standardize column names (handle different schemas)
        column_map = {
            'date': 'timestamp',
            'ticker': 'symbol',
            'adj_close': 'close',  # Prefer adjusted close if available
        }

        # Rename columns if they exist
        for old_col, new_col in column_map.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename({old_col: new_col})

        # Ensure timestamp is datetime
        if df.schema['timestamp'] != pl.Datetime:
            df = df.with_columns([
                pl.col('timestamp').str.to_datetime()
            ])

        # Filter by date range
        if start_date:
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date
            df = df.filter(pl.col('timestamp') >= start_dt)
        if end_date:
            from datetime import datetime
            end_dt = datetime.fromisoformat(end_date) if isinstance(end_date, str) else end_date
            df = df.filter(pl.col('timestamp') <= end_dt)

        # Filter by symbols
        if symbols:
            df = df.filter(pl.col('symbol').is_in(symbols))

        # Ensure required columns exist
        required = ['timestamp', 'symbol', 'open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Select and order columns
        select_cols = required + (['volume'] if 'volume' in df.columns else [])
        df = df.select(select_cols).sort(['symbol', 'timestamp'])

        return df

    def load_nasdaq100_minute(
        self,
        year: int = 2021,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | None = None,
    ) -> pl.DataFrame:
        """Load NASDAQ-100 minute bar data.

        Args:
            year: Year to load (2021 or 2022)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols (None = all)

        Returns:
            Polars DataFrame with minute-level OHLCV data
        """
        path = self.projects_dir / "nasdaq100_minute_bars" / f"{year}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"No minute data found at {path}")

        df = pl.read_parquet(path)

        # Standardize columns (similar to daily)
        if 'date' in df.columns:
            df = df.rename({'date': 'timestamp'})
        if 'ticker' in df.columns:
            df = df.rename({'ticker': 'symbol'})

        # Ensure timestamp is datetime
        if df.schema['timestamp'] != pl.Datetime:
            df = df.with_columns([
                pl.col('timestamp').str.to_datetime()
            ])

        # Filter by date range
        if start_date:
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date
            df = df.filter(pl.col('timestamp') >= start_dt)
        if end_date:
            from datetime import datetime
            end_dt = datetime.fromisoformat(end_date) if isinstance(end_date, str) else end_date
            df = df.filter(pl.col('timestamp') <= end_dt)

        # Filter by symbols
        if symbols:
            df = df.filter(pl.col('symbol').is_in(symbols))

        return df.sort(['symbol', 'timestamp'])

    def to_pandas(self, df: pl.DataFrame) -> pd.DataFrame:
        """Convert Polars DataFrame to Pandas with proper index."""
        pdf = df.to_pandas()

        # Set multi-index for zipline compatibility
        if 'symbol' in pdf.columns and 'timestamp' in pdf.columns:
            pdf = pdf.set_index(['timestamp', 'symbol'])

        return pdf

    def to_zipline_format(self, df: pl.DataFrame) -> pd.DataFrame:
        """Convert to zipline-reloaded format.

        Zipline expects:
        - DatetimeIndex
        - Columns: open, high, low, close, volume (if available)
        - One symbol per DataFrame OR multi-index (timestamp, symbol)
        """
        pdf = self.to_pandas(df)

        # Zipline prefers lower case column names
        pdf.columns = pdf.columns.str.lower()

        return pdf

    def to_backtrader_format(self, df: pl.DataFrame, symbol: str) -> pd.DataFrame:
        """Convert to backtrader format for a single symbol.

        Backtrader expects:
        - DatetimeIndex
        - Columns: open, high, low, close, volume, openinterest (optional)
        - Single symbol per DataFrame
        """
        # Filter to single symbol
        symbol_df = df.filter(pl.col('symbol') == symbol)

        pdf = symbol_df.to_pandas()
        pdf = pdf.set_index('timestamp')
        pdf = pdf.drop(columns=['symbol'], errors='ignore')

        # Add openinterest column if not present (required by some bt feeds)
        if 'openinterest' not in pdf.columns:
            pdf['openinterest'] = 0.0

        return pdf

    def split_by_symbol(self, df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """Split multi-symbol DataFrame into dict of single-symbol DataFrames."""
        symbols = df['symbol'].unique().to_list()
        return {
            symbol: df.filter(pl.col('symbol') == symbol)
            for symbol in symbols
        }


# Convenience function
def load_test_data(
    dataset: Literal['daily_equities', 'nasdaq_minute'] = 'daily_equities',
    **kwargs
) -> pl.DataFrame:
    """Quick data loader for testing.

    Examples:
        >>> data = load_test_data('daily_equities', symbols=['AAPL'], start_date='2020-01-01')
        >>> data = load_test_data('nasdaq_minute', year=2021, symbols=['AAPL', 'GOOGL'])
    """
    loader = DataLoader()

    if dataset == 'daily_equities':
        return loader.load_daily_equities(**kwargs)
    elif dataset == 'nasdaq_minute':
        return loader.load_nasdaq100_minute(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
