"""
Custom Zipline bundle for cross-framework validation test data.

Supports both single-asset (AAPL) and multi-asset (STOCK00-STOCK24) synthetic data.
"""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from zipline.data.bundles import register


# Global variable to store multi-asset test data
# This will be set by the test fixture before bundle ingestion
_MULTI_ASSET_TEST_DATA = None


def set_multi_asset_data(data_dict):
    """Set multi-asset test data for bundle ingestion.

    Args:
        data_dict: Dictionary mapping symbol -> OHLCV DataFrame
    """
    global _MULTI_ASSET_TEST_DATA
    _MULTI_ASSET_TEST_DATA = data_dict


def load_test_data():
    """Load test data - either AAPL or multi-asset synthetic data."""
    global _MULTI_ASSET_TEST_DATA

    # If multi-asset data is available, use it
    if _MULTI_ASSET_TEST_DATA is not None:
        return _MULTI_ASSET_TEST_DATA

    # Otherwise, load AAPL data from pickle file
    signal_file = Path(__file__).parent.parent / "signals" / "sp500_top10_sma_crossover.pkl"
    with open(signal_file, 'rb') as f:
        signal_set = pickle.load(f)

    asset_data = signal_set['assets']['AAPL']
    df = asset_data['data'].copy()

    # Ensure column names are lowercase (Zipline requirement)
    df.columns = df.columns.str.lower()

    # Ensure timezone-aware UTC index
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    return {'AAPL': df}


def test_data_bundle(interval='1d'):
    """
    Create bundle ingest function for test data.

    Parameters
    ----------
    interval : str
        '1d' for daily data (default), '1m' for minute data

    Returns
    -------
    callable
        Ingest function for zipline.data.bundles.register()
    """

    def ingest(environ,
               asset_db_writer,
               minute_bar_writer,
               daily_bar_writer,
               adjustment_writer,
               calendar,
               start_session,
               end_session,
               cache,
               show_progress,
               output_dir):

        # Load test data (returns dict: symbol -> DataFrame)
        data_dict = load_test_data()

        # Build metadata for all symbols
        metadata_rows = []
        for sid, (symbol, df) in enumerate(sorted(data_dict.items())):
            # Ensure column names are lowercase
            df.columns = df.columns.str.lower()

            # Ensure timezone-aware UTC index
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            metadata_rows.append({
                'symbol': symbol,
                'asset_name': symbol,  # Use symbol as name for synthetic data
                'start_date': df.index[0],
                'end_date': df.index[-1],
                'first_traded': df.index[0],
                'auto_close_date': df.index[-1] + pd.Timedelta(days=1),
                'exchange': 'NASDAQ',
            })

        metadata = pd.DataFrame(metadata_rows, index=range(len(metadata_rows)))

        # Generator for bar writer (yields (sid, dataframe) tuples)
        def data_generator():
            for sid, (symbol, df) in enumerate(sorted(data_dict.items())):
                # Ensure lowercase columns and UTC timezone
                df.columns = df.columns.str.lower()
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                else:
                    df.index = df.index.tz_convert('UTC')
                yield (sid, df)

        # Write OHLCV bars
        if interval == '1m':
            minute_bar_writer.write(data_generator(), show_progress=show_progress)
        else:
            daily_bar_writer.write(data_generator(), show_progress=show_progress)

        # Write asset metadata
        asset_db_writer.write(equities=metadata)

        # Write adjustments (empty for test data - no splits/dividends)
        adjustment_writer.write()

    return ingest


# Register the bundle
register(
    'test_data',
    test_data_bundle(interval='1d'),
    calendar_name='NYSE',  # Use NYSE calendar
)
