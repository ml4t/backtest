"""Data feed interfaces and implementations for ml4t.backtest."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from ml4t.backtest.core.event import Event, MarketEvent, SignalEvent
from ml4t.backtest.core.types import AssetId, MarketDataType


class DataFeed(ABC):
    """Abstract base class for all data feeds."""

    @abstractmethod
    def get_next_event(self) -> Event | None:
        """
        Get the next event from this data feed.

        Returns:
            Next event or None if no more data
        """

    @abstractmethod
    def peek_next_timestamp(self) -> datetime | None:
        """
        Peek at the timestamp of the next event without consuming it.

        Returns:
            Timestamp of next event or None if no more data
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the data feed to the beginning."""

    @abstractmethod
    def seek(self, timestamp: datetime) -> None:
        """
        Seek to a specific timestamp.

        Args:
            timestamp: Target timestamp to seek to
        """

    @property
    @abstractmethod
    def is_exhausted(self) -> bool:
        """Check if the data feed has no more events."""


class SignalSource(ABC):
    """Abstract base class for ML signal sources."""

    @abstractmethod
    def get_next_signal(self) -> SignalEvent | None:
        """
        Get the next signal from this source.

        Returns:
            Next signal event or None if no more signals
        """

    @abstractmethod
    def peek_next_timestamp(self) -> datetime | None:
        """
        Peek at the timestamp of the next signal.

        Returns:
            Timestamp of next signal or None
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the signal source."""


class CSVDataFeed(DataFeed):
    """Data feed that reads from CSV files.

    Supports embedded ML signals for multi-signal strategies.
    """

    def __init__(
        self,
        path: Path,
        asset_id: AssetId,
        data_type: MarketDataType = MarketDataType.BAR,
        timestamp_column: str = "timestamp",
        signal_columns: list[str] | None = None,
        parse_dates: bool = True,
        **csv_kwargs,
    ):
        """
        Initialize CSV data feed.

        Args:
            path: Path to CSV file
            asset_id: Asset identifier
            data_type: Type of market data
            timestamp_column: Name of timestamp column
            signal_columns: Optional list of columns containing ML signals
                           (e.g., ['ml_pred_5d', 'confidence', 'ml_exit_pred'])
            parse_dates: Whether to parse dates automatically
            **csv_kwargs: Additional arguments for Polars read_csv
        """
        self.path = Path(path)
        self.asset_id = asset_id
        self.data_type = data_type
        self.timestamp_column = timestamp_column
        self.signal_columns = signal_columns or []

        # Read CSV with Polars
        if parse_dates:
            csv_kwargs["try_parse_dates"] = True

        self.df = pl.read_csv(str(self.path), **csv_kwargs).sort(timestamp_column)

        self.current_index = 0
        self.max_index = len(self.df) - 1

    def get_next_event(self) -> MarketEvent | None:
        """Get the next market event."""
        if self.is_exhausted:
            return None

        row = self.df.row(self.current_index, named=True)
        self.current_index += 1

        return self._create_market_event(row)

    def _create_market_event(self, row: dict[str, Any]) -> MarketEvent:
        """Create a MarketEvent from a data row."""
        timestamp = row[self.timestamp_column]

        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(str(timestamp))

        # Extract signals if signal_columns specified
        signals = {}
        for col in self.signal_columns:
            if col in row and row[col] is not None:
                signals[col] = float(row[col])

        return MarketEvent(
            timestamp=timestamp,
            asset_id=self.asset_id,
            data_type=self.data_type,
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=row.get("volume"),
            price=row.get("price", row.get("close")),
            signals=signals,
        )

    def peek_next_timestamp(self) -> datetime | None:
        """Peek at the next timestamp."""
        if self.is_exhausted:
            return None

        timestamp = self.df[self.timestamp_column][self.current_index]
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(str(timestamp))

        return timestamp

    def reset(self) -> None:
        """Reset to the beginning."""
        self.current_index = 0

    def seek(self, timestamp: datetime) -> None:
        """Seek to a specific timestamp."""
        mask = self.df[self.timestamp_column] >= timestamp
        indices = mask.arg_true()

        if len(indices) > 0:
            self.current_index = indices[0]
        else:
            self.current_index = self.max_index + 1

    @property
    def is_exhausted(self) -> bool:
        """Check if all data has been consumed."""
        return self.current_index > self.max_index


class ParquetSignalSource(SignalSource):
    """Signal source that reads ML signals from Parquet files."""

    def __init__(
        self,
        path: Path,
        model_id: str,
        timestamp_column: str = "timestamp",
        asset_column: str = "asset_id",
        signal_column: str = "signal",
        confidence_column: str | None = "confidence",
        ts_event_column: str | None = "ts_event",
        ts_arrival_column: str | None = "ts_arrival",
    ):
        """
        Initialize Parquet signal source.

        Args:
            path: Path to Parquet file with signals
            model_id: Identifier for the ML model
            timestamp_column: Column with signal timestamp
            asset_column: Column with asset identifiers
            signal_column: Column with signal values
            confidence_column: Optional column with confidence scores
            ts_event_column: Optional column with event generation time
            ts_arrival_column: Optional column with signal arrival time
        """
        self.path = Path(path)
        self.model_id = model_id
        self.timestamp_column = timestamp_column
        self.asset_column = asset_column
        self.signal_column = signal_column
        self.confidence_column = confidence_column
        self.ts_event_column = ts_event_column
        self.ts_arrival_column = ts_arrival_column

        # Load signals with Polars
        self.df = pl.scan_parquet(str(self.path)).sort(timestamp_column).collect()

        self.current_index = 0
        self.max_index = len(self.df) - 1

    def get_next_signal(self) -> SignalEvent | None:
        """Get the next signal event."""
        if self.current_index > self.max_index:
            return None

        row = self.df.row(self.current_index, named=True)
        self.current_index += 1

        timestamp = row[self.timestamp_column]
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(str(timestamp))

        return SignalEvent(
            timestamp=timestamp,
            asset_id=AssetId(row[self.asset_column]),
            signal_value=float(row[self.signal_column]),
            model_id=self.model_id,
            confidence=float(row[self.confidence_column])
            if self.confidence_column and self.confidence_column in row
            else None,
            ts_event=row.get(self.ts_event_column) if self.ts_event_column else None,
            ts_arrival=row.get(self.ts_arrival_column) if self.ts_arrival_column else timestamp,
        )

    def peek_next_timestamp(self) -> datetime | None:
        """Peek at the next signal timestamp."""
        if self.current_index > self.max_index:
            return None

        timestamp = self.df[self.timestamp_column][self.current_index]
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(str(timestamp))

        return timestamp

    def reset(self) -> None:
        """Reset to the beginning."""
        self.current_index = 0
