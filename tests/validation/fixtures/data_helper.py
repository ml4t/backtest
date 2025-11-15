"""
Test data helper with intelligent fallback.

Provides reproducible test data with multiple fallback strategies:
1. qdata integration (if available) - tests ml4t.backtest + qdata together
2. yfinance directly - standalone mode
3. Cached fixtures - offline mode
4. Synthetic GBM - pure unit tests

Design:
- Smart fallback ensures tests work in any environment
- Cache successful downloads for reproducibility
- Synthetic data for deterministic unit tests
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent
CACHE_DIR = FIXTURES_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


class TestDataProvider:
    """Smart test data provider with fallback strategies."""

    def __init__(self, cache_enabled: bool = True, verbose: bool = False):
        """
        Initialize test data provider.

        Args:
            cache_enabled: Cache downloaded data for offline use
            verbose: Print data source used
        """
        self.cache_enabled = cache_enabled
        self.verbose = verbose

    def get_equity_data(
        self,
        symbol: str = "AAPL",
        start: str = "2015-01-01",
        end: str = "2016-12-31",
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """
        Get equity OHLCV data with smart fallback.

        Strategy:
        1. Try qdata provider (integration test)
        2. Try yfinance directly
        3. Try cached data
        4. Generate synthetic data

        Args:
            symbol: Ticker symbol
            start: Start date YYYY-MM-DD
            end: End date YYYY-MM-DD
            frequency: Data frequency (daily, hourly, etc.)

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        # Generate cache key
        cache_key = self._get_cache_key(symbol, start, end, frequency)
        cache_file = CACHE_DIR / f"{cache_key}.parquet"

        # Try cache first (fastest)
        if self.cache_enabled and cache_file.exists():
            if self.verbose:
                print(f"📦 Using cached data: {cache_file.name}")
            return pd.read_parquet(cache_file)

        # Try qdata (integration test)
        df = self._try_qdata(symbol, start, end, frequency)
        if df is not None:
            if self.verbose:
                print(f"🔗 Using qdata provider (integration test)")
            if self.cache_enabled:
                df.to_parquet(cache_file)
            return df

        # Try yfinance directly (standalone)
        df = self._try_yfinance(symbol, start, end, frequency)
        if df is not None:
            if self.verbose:
                print(f"📡 Using yfinance directly")
            if self.cache_enabled:
                df.to_parquet(cache_file)
            return df

        # Generate synthetic data (offline/deterministic)
        if self.verbose:
            print(f"🎲 Using synthetic GBM data (offline mode)")
        return self._generate_synthetic(symbol, start, end, frequency)

    def _try_qdata(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pd.DataFrame | None:
        """Try to fetch data using qdata provider."""
        try:
            # Check if qdata is available
            import sys
            from pathlib import Path

            # Try to import qdata from ../data/src
            qdata_path = Path(__file__).parent.parent.parent.parent / "data" / "src"
            if qdata_path.exists():
                sys.path.insert(0, str(qdata_path))

            from ml4t.data.providers.yahoo import YahooFinanceProvider

            provider = YahooFinanceProvider(enable_progress=False)
            df_polars = provider.fetch_ohlcv(
                symbol=symbol,
                start=start,
                end=end,
                frequency=frequency,
            )

            # Convert Polars to Pandas
            df = df_polars.to_pandas()

            # Standardize column names
            df = df.rename(
                columns={
                    "timestamp": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )

            return df

        except (ImportError, Exception):
            # qdata not available or error - fall back
            return None

    def _try_yfinance(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pd.DataFrame | None:
        """Try to fetch data using yfinance directly."""
        try:
            import yfinance as yf

            # Map frequency to yfinance interval
            interval_map = {
                "daily": "1d",
                "1day": "1d",
                "hourly": "1h",
                "1hour": "1h",
                "minute": "1m",
                "1minute": "1m",
            }
            interval = interval_map.get(frequency.lower(), "1d")

            # Download data
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                return None

            # Handle multi-level columns (Price/Close, Ticker/SPY)
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten: take first level only (Price -> Close, High, Low, etc.)
                df.columns = df.columns.get_level_values(0)

            # Reset index to get date column
            df = df.reset_index()

            # Standardize column names (yfinance uses capitalized)
            df.columns = df.columns.str.lower()

            # Rename date/datetime column to timestamp
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            elif "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})

            return df

        except Exception as e:
            # yfinance failed - fall back
            if self.verbose:
                print(f"yfinance failed: {e}")
            return None

    def _generate_synthetic(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data using Geometric Brownian Motion.

        Deterministic (seeded) for reproducible tests.

        Args:
            symbol: Used as seed for deterministic generation
            start: Start date
            end: End date
            frequency: Data frequency

        Returns:
            DataFrame with realistic-looking OHLCV data
        """
        # Calculate number of bars
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        if frequency.lower() in ["daily", "1day"]:
            n_bars = (end_dt - start_dt).days
            freq_str = "D"
        elif frequency.lower() in ["hourly", "1hour"]:
            n_bars = (end_dt - start_dt).days * 24
            freq_str = "h"
        else:
            n_bars = 500  # Default
            freq_str = "D"

        # Create deterministic seed from symbol
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % (2**32)
        np.random.seed(seed)

        # GBM parameters
        S0 = 100.0  # Initial price
        mu = 0.0005  # Drift (daily return)
        sigma = 0.02  # Volatility (daily)
        dt = 1.0  # Time step (1 day)

        # Generate price series
        returns = np.random.normal(mu, sigma, n_bars)
        prices = S0 * np.exp(np.cumsum(returns))

        # Generate OHLC from close prices
        # Add realistic intraday variation
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))
        open_prices = prices * (1 + np.random.normal(0, 0.005, n_bars))

        # Ensure OHLC relationships hold
        high = np.maximum(high, np.maximum(open_prices, prices))
        low = np.minimum(low, np.minimum(open_prices, prices))

        # Generate volume (lognormal distribution)
        volume = np.random.lognormal(14, 1, n_bars).astype(int)

        # Create timestamps
        timestamps = pd.date_range(start=start_dt, periods=n_bars, freq=freq_str)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_prices,
                "high": high,
                "low": low,
                "close": prices,
                "volume": volume,
            }
        )

        return df

    def _get_cache_key(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> str:
        """Generate cache key for data."""
        key_str = f"{symbol}_{start}_{end}_{frequency}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]


# Convenience function for tests
def get_test_data(
    symbol: str = "AAPL",
    start: str = "2015-01-01",
    end: str = "2016-12-31",
    frequency: str = "daily",
    cache: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Get test data with smart fallback.

    This is the main function tests should use.

    Args:
        symbol: Ticker symbol
        start: Start date YYYY-MM-DD
        end: End date YYYY-MM-DD
        frequency: Data frequency
        cache: Enable caching
        verbose: Print data source

    Returns:
        DataFrame with OHLCV data
    """
    provider = TestDataProvider(cache_enabled=cache, verbose=verbose)
    return provider.get_equity_data(symbol, start, end, frequency)
