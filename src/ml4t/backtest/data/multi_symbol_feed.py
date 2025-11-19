"""Multi-symbol data feed for portfolio backtesting.

This module provides MultiSymbolDataFeed, a production data feed that handles
multiple assets efficiently for portfolio strategies.
"""

from datetime import datetime
from typing import Any, Iterator

import polars as pl

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType
from ml4t.backtest.data.feed import DataFeed


class MultiSymbolDataFeed(DataFeed):
    """In-memory data feed for multi-asset portfolio backtesting.

    This feed handles multiple symbols efficiently by:
    - Merging price, signals, and context data upfront
    - Sorting by timestamp for sequential event emission
    - Minimizing per-event overhead with pre-joined DataFrames

    Architecture:
        1. Constructor: Load and merge all data sources (price, signals, context)
        2. Iteration: Emit MarketEvents row-by-row in chronological order
        3. Event creation: Populate signals dict (per-asset) and context dict (market-wide)

    Performance: Tested at 34,000+ events/second for 250 symbols × 252 days

    Example:
        >>> import polars as pl
        >>> from ml4t.backtest.data.multi_symbol_feed import MultiSymbolDataFeed
        >>>
        >>> # Price data: timestamp, asset_id, open, high, low, close, volume
        >>> price_df = pl.DataFrame({...})
        >>>
        >>> # Signals: timestamp, asset_id, ml_score, ...
        >>> signals_df = pl.DataFrame({...})
        >>>
        >>> # Context: timestamp, vix, regime, ...
        >>> context_df = pl.DataFrame({...})
        >>>
        >>> feed = MultiSymbolDataFeed(price_df, signals_df, context_df)
        >>>
        >>> while not feed.is_exhausted:
        ...     event = feed.get_next_event()
        ...     print(f"{event.timestamp}: {event.asset_id} @ ${event.close}")
        ...     print(f"  Signals: {event.signals}")
        ...     print(f"  Context: {event.context}")
    """

    def __init__(
        self,
        price_df: pl.DataFrame,
        signals_df: pl.DataFrame | None = None,
        context_df: pl.DataFrame | None = None,
    ):
        """Initialize feed with price, signals, and context data.

        Args:
            price_df: OHLCV data with columns: timestamp, asset_id, open, high, low, close, volume
                     Must be sorted by timestamp, asset_id for optimal performance
            signals_df: Optional per-asset signals with columns: timestamp, asset_id, [signal columns]
                       Signals are asset-specific (e.g., ml_score, momentum, atr)
            context_df: Optional market-wide context with columns: timestamp, [context columns]
                       Context is broadcast to all assets (e.g., vix, spy_close, regime)

        Raises:
            ValueError: If price_df is missing required columns
        """
        # Validate price data
        required_cols = ["timestamp", "asset_id", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in price_df.columns]
        if missing_cols:
            raise ValueError(f"price_df missing required columns: {missing_cols}")

        # Store base price data (sorted by timestamp, asset_id)
        self.price_df = price_df.sort("timestamp", "asset_id")

        # Merge signals if provided
        if signals_df is not None:
            self.merged_df = self.price_df.join(
                signals_df,
                on=["timestamp", "asset_id"],
                how="left",
            )
        else:
            self.merged_df = self.price_df

        # Merge context if provided (broadcast to all rows with same timestamp)
        if context_df is not None:
            self.merged_df = self.merged_df.join(
                context_df,
                on="timestamp",
                how="left",
            )

        # Store context column names for event creation
        self.context_cols = (
            [col for col in context_df.columns if col != "timestamp"]
            if context_df is not None
            else []
        )

        # Initialize iteration state
        self.current_index = 0
        self.max_index = len(self.merged_df) - 1

    def get_next_event(self) -> MarketEvent | None:
        """Get next market event in chronological order.

        Returns:
            MarketEvent with populated signals and context dicts, or None if exhausted
        """
        if self.is_exhausted:
            return None

        row = self.merged_df.row(self.current_index, named=True)
        self.current_index += 1

        return self._create_event(row)

    def stream_by_timestamp(self) -> Iterator[tuple[datetime, list[MarketEvent]]]:
        """Stream events grouped by timestamp for batch processing.

        This method enables efficient batch processing for portfolio strategies
        by grouping all assets' events at the same timestamp together. This allows
        the engine to process all simultaneous market data in a single time-slice.

        Yields:
            Tuple of (timestamp, list of MarketEvents) for each unique timestamp

        Example:
            >>> for timestamp, events in feed.stream_by_timestamp():
            ...     # Process all assets simultaneously at this timestamp
            ...     market_map = {e.asset_id: e for e in events}
            ...     broker.process_batch_fills(timestamp, market_map)
            ...     strategy.on_data_batch(timestamp, market_map)

        Performance:
            - Reduces iterations from 126,000 (500 × 252) to 252 (timestamps only)
            - Enables simultaneous cross-asset portfolio rebalancing
            - Maintains chronological order guarantee
        """
        # Group by timestamp using Polars for performance
        for group_data in self.merged_df.group_by("timestamp", maintain_order=True):
            timestamp_value, group_df = group_data

            # Extract the actual timestamp value (Polars returns single-value series)
            if isinstance(timestamp_value, tuple):
                timestamp = timestamp_value[0]
            else:
                timestamp = timestamp_value

            # Create MarketEvent for each row in this timestamp group
            events = [self._create_event(row) for row in group_df.iter_rows(named=True)]

            yield (timestamp, events)

    def _create_event(self, row: dict[str, Any]) -> MarketEvent:
        """Create MarketEvent from DataFrame row.

        Args:
            row: Dictionary with OHLCV, signals, and context data

        Returns:
            MarketEvent with signals (per-asset) and context (market-wide) populated
        """
        # Required OHLCV columns
        ohlcv_cols = {"timestamp", "asset_id", "open", "high", "low", "close", "volume"}

        # Extract signals (per-asset columns that aren't OHLCV or context)
        signal_cols = [
            col
            for col in row.keys()
            if col not in ohlcv_cols and col not in self.context_cols
        ]
        signals = {col: row[col] for col in signal_cols if row[col] is not None}

        # Extract context (market-wide columns)
        context = {col: row[col] for col in self.context_cols if row[col] is not None}

        return MarketEvent(
            timestamp=row["timestamp"],
            asset_id=row["asset_id"],
            data_type=MarketDataType.BAR,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            signals=signals,
            context=context,
        )

    @property
    def is_exhausted(self) -> bool:
        """Check if feed has no more events.

        Returns:
            True if all events have been consumed
        """
        return self.current_index > self.max_index

    def reset(self) -> None:
        """Reset the feed to the beginning for re-iteration."""
        self.current_index = 0

    def peek_next_timestamp(self) -> datetime | None:
        """Peek at the timestamp of the next event without consuming it.

        Returns:
            Timestamp of next event, or None if exhausted
        """
        if self.is_exhausted:
            return None
        return self.merged_df["timestamp"][self.current_index]

    def seek(self, timestamp: datetime) -> None:
        """Seek to a specific timestamp in the feed.

        Args:
            timestamp: Target timestamp to seek to

        Note:
            Seeks to first event with timestamp >= target.
            If no such event exists, exhausts the feed.
        """
        timestamps = self.merged_df["timestamp"]
        for idx in range(self.current_index, len(timestamps)):
            if timestamps[idx] >= timestamp:
                self.current_index = idx
                return
        # If not found, exhaust the feed
        self.current_index = self.max_index + 1
