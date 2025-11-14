"""
Universal data loader for cross-framework validation.

Loads data from ~/ml4t/projects/ and converts to framework-specific formats.
"""

from pathlib import Path
from typing import Literal

import pandas as pd


class UniversalDataLoader:
    """Load and convert data for cross-framework validation."""

    def __init__(self, data_root: str | Path = "~/ml4t/software/projects/"):
        """
        Initialize data loader.

        Args:
            data_root: Root directory containing data projects
        """
        self.data_root = Path(data_root).expanduser()

        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")

    def load_daily_equities(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        source: Literal["wiki", "enhanced"] = "wiki",
    ) -> pd.DataFrame:
        """
        Load daily US equity data.

        Args:
            tickers: List of tickers to load (None = all)
            start_date: Start date 'YYYY-MM-DD' (None = from beginning)
            end_date: End date 'YYYY-MM-DD' (None = to end)
            source: Data source ('wiki' or 'enhanced')

        Returns:
            DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
        """
        if source == "wiki":
            path = self.data_root / "daily_us_equities" / "wiki_prices.parquet"
        else:
            path = self.data_root / "daily_us_equities" / "equity_prices_enhanced_1962_2025.parquet"

        df = pd.read_parquet(path)

        # Filter tickers
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]

        # Filter dates
        if start_date is not None:
            df = df[df["date"] >= start_date]
        if end_date is not None:
            df = df[df["date"] <= end_date]

        # Convert to standard format
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def load_minute_bars(
        self,
        year: int,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Load NASDAQ100 minute bar data.

        Args:
            year: Year to load (2021 or 2022)
            tickers: List of security IDs to load (None = all)
            start_date: Start date 'YYYY-MM-DD' (None = from beginning)
            end_date: End date 'YYYY-MM-DD' (None = to end)

        Returns:
            DataFrame with DatetimeIndex and minute-level OHLCV data
        """
        path = self.data_root / "nasdaq100_minute_bars" / f"{year}.parquet"

        df = pd.read_parquet(path)

        # Reset MultiIndex to columns
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        # Filter tickers (sec_id in this dataset)
        if tickers is not None:
            df = df[df["sec_id"].isin(tickers)]

        # Use bar_start as timestamp
        df["timestamp"] = pd.to_datetime(df["bar_start"])

        # Filter dates
        if start_date is not None:
            df = df[df["timestamp"] >= start_date]
        if end_date is not None:
            df = df[df["timestamp"] <= end_date]

        # Extract OHLCV from available columns
        # This dataset has bid/ask/trade prices - use trade prices for OHLC
        ohlcv_df = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "ticker": df["sec_id"],
                "open": df["first_trade_price"],
                "high": df["high_trade_price"],
                "low": df["low_trade_price"],
                "close": df["last_trade_price"],
                "volume": df["volume"],
            }
        )

        return ohlcv_df

    def load_crypto(
        self,
        symbol: str = "BTC",
        data_type: Literal["futures", "spot"] = "futures",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Load cryptocurrency data.

        Args:
            symbol: Crypto symbol (BTC or ETH)
            data_type: 'futures' or 'spot'
            start_date: Start date 'YYYY-MM-DD HH:MM:SS' (None = from beginning)
            end_date: End date 'YYYY-MM-DD HH:MM:SS' (None = to end)

        Returns:
            DataFrame with DatetimeIndex and minute-level OHLCV data
        """
        path = self.data_root / "crypto_futures" / "data" / data_type / f"{symbol}.parquet"

        df = pd.read_parquet(path)

        # Rename date_time to timestamp
        df = df.rename(columns={"date_time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter dates
        if start_date is not None:
            df = df[df["timestamp"] >= start_date]
        if end_date is not None:
            df = df[df["timestamp"] <= end_date]

        # Add ticker column
        df["ticker"] = symbol

        return df

    def load_order_flow(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Load SPY order flow microstructure data.

        Args:
            start_date: Start date 'YYYY-MM-DD' (None = from beginning)
            end_date: End date 'YYYY-MM-DD' (None = to end)

        Returns:
            DataFrame with order flow features
        """
        path = self.data_root / "spy_order_flow" / "spy_features.parquet"

        df = pd.read_parquet(path)

        # Assume first column is timestamp (inspect actual schema if needed)
        if "timestamp" not in df.columns and "date" not in df.columns:
            # Try to infer timestamp column
            time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
            if time_cols:
                df = df.rename(columns={time_cols[0]: "timestamp"})

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter dates
            if start_date is not None:
                df = df[df["timestamp"] >= start_date]
            if end_date is not None:
                df = df[df["timestamp"] <= end_date]

        return df

    # ============================================================================
    # Framework-Specific Format Converters
    # ============================================================================

    def to_framework_format(
        self,
        df: pd.DataFrame,
        framework: Literal["vectorbt", "zipline", "backtrader", "qengine"],
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """
        Convert data to framework-specific format.

        Args:
            df: DataFrame with standard format (ticker, timestamp, OHLCV columns)
            framework: Target framework
            symbol: Symbol name (for single-asset strategies)

        Returns:
            DataFrame in framework-specific format
        """
        if framework == "vectorbt":
            return self._to_vectorbt_format(df, symbol)
        elif framework == "zipline":
            return self._to_zipline_format(df, symbol)
        elif framework == "backtrader":
            return self._to_backtrader_format(df, symbol)
        elif framework == "qengine":
            return self._to_qengine_format(df, symbol)
        else:
            raise ValueError(f"Unknown framework: {framework}")

    def _to_vectorbt_format(self, df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
        """
        Convert to VectorBT Pro format.

        VectorBT expects:
        - DatetimeIndex
        - Columns: open, high, low, close, volume
        - Single symbol per DataFrame
        """
        result = df.copy()

        # Filter to single symbol if specified
        if symbol is not None and "ticker" in result.columns:
            result = result[result["ticker"] == symbol].copy()

        # Set DatetimeIndex
        if "timestamp" in result.columns:
            result = result.set_index("timestamp")

        # Keep only OHLCV columns
        keep_cols = ["open", "high", "low", "close", "volume"]
        result = result[[col for col in keep_cols if col in result.columns]]

        # Sort by timestamp
        result = result.sort_index()

        return result

    def _to_zipline_format(self, df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
        """
        Convert to Zipline-Reloaded format.

        Zipline expects:
        - DatetimeIndex with UTC timezone
        - Columns: open, high, low, close, volume
        - Single symbol per DataFrame
        """
        result = self._to_vectorbt_format(df, symbol)

        # Ensure UTC timezone
        if result.index.tz is None:
            result.index = result.index.tz_localize("UTC")
        else:
            result.index = result.index.tz_convert("UTC")

        return result

    def _to_backtrader_format(self, df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
        """
        Convert to Backtrader format.

        Backtrader expects:
        - DatetimeIndex
        - Columns: open, high, low, close, volume
        - Lowercase column names
        """
        result = self._to_vectorbt_format(df, symbol)

        # Ensure lowercase column names
        result.columns = [col.lower() for col in result.columns]

        return result

    def _to_qengine_format(self, df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
        """
        Convert to QEngine format.

        QEngine uses Polars internally, but for validation we'll use pandas.
        Format:
        - DatetimeIndex
        - Columns: open, high, low, close, volume
        - ticker column for multi-asset support
        """
        result = df.copy()

        # Set DatetimeIndex
        if "timestamp" in result.columns:
            result = result.set_index("timestamp")

        # Keep ticker and OHLCV columns
        keep_cols = ["ticker", "open", "high", "low", "close", "volume"]
        result = result[[col for col in keep_cols if col in result.columns]]

        # Filter to single symbol if specified
        if symbol is not None and "ticker" in result.columns:
            result = result[result["ticker"] == symbol].copy()

        # Sort by timestamp
        result = result.sort_index()

        return result

    # ============================================================================
    # Convenience Methods
    # ============================================================================

    def load_simple_equity_data(
        self,
        ticker: str = "AAPL",
        start_date: str = "2020-01-01",
        end_date: str = "2021-12-31",
        framework: str = "vectorbt",
    ) -> pd.DataFrame:
        """
        Load simple single-equity dataset for hello-world tests.

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            framework: Target framework format

        Returns:
            DataFrame ready for framework backtest
        """
        df = self.load_daily_equities(
            tickers=[ticker], start_date=start_date, end_date=end_date, source="wiki"
        )

        return self.to_framework_format(df, framework, symbol=ticker)

    def load_crypto_simple(
        self,
        symbol: str = "BTC",
        start_date: str = "2020-01-01",
        end_date: str = "2021-12-31",
        framework: str = "vectorbt",
    ) -> pd.DataFrame:
        """
        Load simple crypto dataset for testing.

        Args:
            symbol: Crypto symbol (BTC or ETH)
            start_date: Start date
            end_date: End date
            framework: Target framework format

        Returns:
            DataFrame ready for framework backtest
        """
        df = self.load_crypto(symbol=symbol, data_type="futures", start_date=start_date, end_date=end_date)

        return self.to_framework_format(df, framework, symbol=symbol)
