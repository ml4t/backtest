"""
Custom Zipline bundle for cross-framework validation test data.

Ingests the AAPL test data from sp500_top10_sma_crossover.pkl
to enable Zipline validation against ml4t.backtest, Backtrader, and VectorBT.
"""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from zipline.data.bundles import register


def load_test_data():
    """Load AAPL data from test signals pickle file."""
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
    elif df.index.tz != pd.UTC:
        df.index = df.index.tz_convert('UTC')

    return df


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

        # Load test data
        df = load_test_data()

        # Define asset metadata (index must be the sid)
        metadata = pd.DataFrame({
            'symbol': ['AAPL'],
            'asset_name': ['Apple Inc.'],
            'start_date': [df.index[0]],
            'end_date': [df.index[-1]],
            'first_traded': [df.index[0]],
            'auto_close_date': [df.index[-1] + pd.Timedelta(days=1)],
            'exchange': ['NASDAQ'],
        }, index=[0])  # sid=0 for AAPL

        # Generator for bar writer (yields (sid, dataframe) tuples)
        def data_generator():
            sid = 0  # AAPL gets sid=0 (matches Zipline's symbol() lookup)
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
