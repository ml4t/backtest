#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zipline bundle ingest function for validation testing.

Based on the AlgoSeek pattern from ML4T, adapted for validation bundle data.
"""

from pathlib import Path
from os import getenv
import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

# Get bundle data path
BUNDLE_DATA_DIR = Path(__file__).parent
BUNDLE_DATA_FILE = BUNDLE_DATA_DIR / 'validation_bundle.h5'


def load_equities():
    """Load equity metadata from bundle data."""
    return pd.read_hdf(BUNDLE_DATA_FILE, 'equities')


def ticker_generator():
    """
    Lazily return (sid, symbol, asset_name) tuples.
    """
    equities = load_equities()
    return ((row['sid'], row['symbol'], row['asset_name'])
            for _, row in equities.iterrows())


def data_generator():
    """
    Generate ticker data with metadata.

    Yields:
        (sid, df), symbol, asset_name, start_date, end_date,
        first_traded, auto_close_date, exchange
    """
    equities = load_equities()

    for _, row in equities.iterrows():
        sid = row['sid']
        symbol = row['symbol']
        asset_name = row['asset_name']

        # Load ticker data
        df = pd.read_hdf(BUNDLE_DATA_FILE, str(sid))

        # Data should already be in UTC from preparation
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')

        start_date = df.index[0]
        end_date = df.index[-1]
        first_traded = start_date.date()
        auto_close_date = end_date + pd.Timedelta(days=1)
        exchange = row.get('exchange', 'NYSE')

        yield (sid, df), symbol, asset_name, start_date, end_date, \
              first_traded, auto_close_date, exchange


def metadata_frame():
    """Create empty metadata DataFrame with proper structure."""
    equities = load_equities()

    dtype = [
        ('symbol', 'object'),
        ('asset_name', 'object'),
        ('start_date', 'datetime64[ns]'),
        ('end_date', 'datetime64[ns]'),
        ('first_traded', 'datetime64[ns]'),
        ('auto_close_date', 'datetime64[ns]'),
        ('exchange', 'object'),
    ]

    return pd.DataFrame(np.empty(len(equities), dtype=dtype))


def validation_to_bundle(interval='1d'):
    """
    Create bundle ingest function for validation testing.

    Args:
        interval: Data frequency ('1d' for daily)

    Returns:
        ingest function compatible with Zipline's bundle API
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
        """
        Ingest function called by Zipline.

        This function is called when running: zipline ingest -b validation
        """
        print(f"Ingesting validation bundle from {BUNDLE_DATA_FILE}")

        # Check bundle data exists
        if not BUNDLE_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Bundle data not found at {BUNDLE_DATA_FILE}\n"
                f"Run prepare_zipline_bundle_data() first to create bundle data."
            )

        metadata = metadata_frame()

        # Daily data generator
        def daily_data_generator():
            for (sid, df), *meta in data_generator():
                # Update metadata
                symbol, asset_name, start_date, end_date, \
                    first_traded, auto_close_date, exchange = meta

                # Reindex to fill missing sessions
                # Zipline expects all calendar days between start and end
                all_sessions = calendar.sessions_in_range(start_session, end_session)

                # Ensure timezone compatibility
                if hasattr(all_sessions, 'tz_localize') and all_sessions.tz is None:
                    all_sessions = all_sessions.tz_localize('UTC')
                elif hasattr(all_sessions, 'tz_convert'):
                    all_sessions = all_sessions.tz_convert('UTC')

                df = df.reindex(all_sessions, method='ffill')

                # Update dates after reindexing
                start_date = df.index[0]
                end_date = df.index[-1]
                first_traded = start_date.date() if hasattr(start_date, 'date') else start_date
                auto_close_date = end_date + pd.Timedelta(days=1)

                metadata.iloc[sid] = (
                    symbol, asset_name, start_date, end_date,
                    first_traded, auto_close_date, exchange
                )

                # Yield data for writing
                yield sid, df

        # Write daily bars
        print(f"Writing daily bars...")
        daily_bar_writer.write(daily_data_generator(), show_progress=show_progress)

        # Write metadata
        print(f"Writing asset metadata...")
        metadata.dropna(inplace=True)
        asset_db_writer.write(equities=metadata)

        # Write adjustments (splits and dividends)
        print(f"Writing adjustments...")

        # Splits
        try:
            splits = pd.read_hdf(BUNDLE_DATA_FILE, 'splits')
            if len(splits) > 0:
                adjustment_writer.write(splits=splits)
                print(f"  ✅ Wrote {len(splits)} split adjustments")
        except KeyError:
            print("  ⚠️  No splits data found")

        # Dividends
        try:
            dividends = pd.read_hdf(BUNDLE_DATA_FILE, 'dividends')
            if len(dividends) > 0:
                adjustment_writer.write(dividends=dividends)
                print(f"  ✅ Wrote {len(dividends)} dividend adjustments")
        except KeyError:
            print("  ⚠️  No dividends data found")

        print(f"✅ Bundle ingestion complete!")

    return ingest
