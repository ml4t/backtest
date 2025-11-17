"""Data validation module for ml4t.backtest.

This module provides validation functions to ensure data quality and correctness:
- Signal timing validation (prevent look-ahead bias)
- Data completeness checks
- Price sanity validation
- OHLC consistency checks

The validation framework is critical for backtest correctness - invalid data
can lead to misleading results and incorrect strategy evaluation.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import polars as pl

from ml4t.backtest.core.types import AssetId


class SignalTimingMode(Enum):
    """Signal timing validation modes.

    Controls when signals are allowed to be used relative to when they appear
    in the data:

    - STRICT: Signal must appear at same timestamp as price bar (same-bar execution)
    - NEXT_BAR: Signal used on next bar after it appears (1-bar lag, most common)
    - CUSTOM: Signal used N bars after it appears (configurable lag)

    Example:
        >>> # Signal appears at 10:00, price bar at 10:00
        >>> # STRICT: Signal used for 10:00 bar decision
        >>> # NEXT_BAR: Signal used for 10:01 bar decision
        >>> # CUSTOM(lag=2): Signal used for 10:02 bar decision
    """

    STRICT = "strict"  # Same bar execution
    NEXT_BAR = "next_bar"  # 1 bar lag (most realistic)
    CUSTOM = "custom"  # N bar lag (configurable)


class SignalTimingViolation(Exception):
    """Exception raised when signal timing validation fails.

    Attributes:
        asset_id: Asset where violation occurred
        signal_timestamp: When signal was generated
        use_timestamp: When signal was first used
        lag: Time difference (negative = look-ahead)
        message: Detailed error message
    """

    def __init__(
        self,
        asset_id: AssetId,
        signal_timestamp: datetime,
        use_timestamp: datetime,
        lag: timedelta,
        message: str,
    ):
        self.asset_id = asset_id
        self.signal_timestamp = signal_timestamp
        self.use_timestamp = use_timestamp
        self.lag = lag
        super().__init__(message)


def validate_signal_timing(
    signals_df: pl.DataFrame,
    prices_df: pl.DataFrame,
    mode: SignalTimingMode = SignalTimingMode.NEXT_BAR,
    custom_lag_bars: int = 1,
    timestamp_column: str = "timestamp",
    asset_column: str = "asset_id",
    fail_on_violation: bool = True,
) -> tuple[bool, list[dict[str, Any]]]:
    """Validate signal timing to prevent look-ahead bias.

    Ensures signals are not used before they were generated, which would
    constitute look-ahead bias and invalidate backtest results.

    Args:
        signals_df: DataFrame with ML signals (must have timestamp, asset_id)
        prices_df: DataFrame with price data (must have timestamp, asset_id)
        mode: Timing validation mode (STRICT, NEXT_BAR, or CUSTOM)
        custom_lag_bars: Number of bars lag for CUSTOM mode (default: 1)
        timestamp_column: Name of timestamp column (default: "timestamp")
        asset_column: Name of asset ID column (default: "asset_id")
        fail_on_violation: If True, raise exception on violation; if False,
                          return violations as warnings (default: True)

    Returns:
        Tuple of (is_valid, violations_list)
        - is_valid: True if no violations found
        - violations_list: List of dict with violation details

    Raises:
        SignalTimingViolation: If fail_on_violation=True and timing violation detected

    Example:
        >>> signals = pl.DataFrame({
        ...     "timestamp": [ts1, ts2],
        ...     "asset_id": ["AAPL", "AAPL"],
        ...     "signal": [1.0, -1.0]
        ... })
        >>> prices = pl.DataFrame({
        ...     "timestamp": [ts1, ts2, ts3],
        ...     "asset_id": ["AAPL", "AAPL", "AAPL"],
        ...     "close": [100, 101, 102]
        ... })
        >>> is_valid, violations = validate_signal_timing(
        ...     signals, prices, mode=SignalTimingMode.NEXT_BAR
        ... )

    Notes:
        - STRICT mode: Signal timestamp must match price timestamp exactly
        - NEXT_BAR mode: Signal can be used starting from next price bar
        - CUSTOM mode: Signal can be used after custom_lag_bars price bars
        - Violations with negative lag indicate look-ahead bias (CRITICAL)
    """
    violations = []

    # For each unique asset, check signal timing
    assets = signals_df[asset_column].unique().to_list()

    for asset_id in assets:
        # Get signals and prices for this asset
        asset_signals = signals_df.filter(pl.col(asset_column) == asset_id).sort(
            timestamp_column
        )
        asset_prices = prices_df.filter(pl.col(asset_column) == asset_id).sort(
            timestamp_column
        )

        # Get sorted timestamps
        signal_timestamps = asset_signals[timestamp_column].to_list()
        price_timestamps = asset_prices[timestamp_column].to_list()

        # For each signal, check if it's being used at valid timestamps
        for signal_ts in signal_timestamps:
            # Determine which price bars could use this signal without look-ahead bias
            if mode == SignalTimingMode.STRICT:
                # Signal can only be used at its own timestamp
                valid_price_timestamps = [signal_ts]

            elif mode == SignalTimingMode.NEXT_BAR:
                # Signal can be used starting from next bar after it appears
                valid_price_timestamps = [ts for ts in price_timestamps if ts > signal_ts]

            elif mode == SignalTimingMode.CUSTOM:
                # Signal can be used starting N bars after it appears
                later_prices = [ts for ts in price_timestamps if ts > signal_ts]
                if len(later_prices) >= custom_lag_bars:
                    # Can be used from Nth bar onward
                    valid_price_timestamps = later_prices[custom_lag_bars - 1 :]
                else:
                    # Not enough bars after signal
                    valid_price_timestamps = []
            else:
                raise ValueError(f"Unknown timing mode: {mode}")

            if not valid_price_timestamps:
                # Signal appears after all price data or not enough bars - not a violation
                continue

            # Check if there are any price timestamps BEFORE the signal that could
            # potentially use this signal (which would be look-ahead bias)
            #
            # Exception: In STRICT mode, if signal_ts matches a price timestamp exactly,
            # this is valid (signal can be used at its own timestamp)
            if mode == SignalTimingMode.STRICT and signal_ts in price_timestamps:
                # Signal appears at exactly the same timestamp as a price bar
                # This is valid in STRICT mode (signal used at its own timestamp)
                continue

            price_timestamps_before_signal = [
                ts for ts in price_timestamps if ts < signal_ts
            ]

            if price_timestamps_before_signal:
                # There are price bars before the signal timestamp
                # In STRICT mode, signal at 11:00 cannot be used for 10:00 bar
                # This is look-ahead bias

                earliest_invalid_use = price_timestamps_before_signal[-1]  # Last bar before signal
                lag = signal_ts - earliest_invalid_use

                violation = {
                    "asset_id": asset_id,
                    "signal_timestamp": signal_ts,
                    "invalid_use_timestamp": earliest_invalid_use,
                    "lag": lag,
                    "lag_seconds": lag.total_seconds(),
                    "mode": mode.value,
                    "severity": "CRITICAL",
                    "message": f"Look-ahead bias detected for {asset_id}: "
                    f"signal at {signal_ts} would be used at {earliest_invalid_use} "
                    f"before it was available (lag: {lag})",
                }

                violations.append(violation)

                if fail_on_violation:
                    raise SignalTimingViolation(
                        asset_id=asset_id,
                        signal_timestamp=signal_ts,
                        use_timestamp=earliest_invalid_use,
                        lag=lag,
                        message=violation["message"],
                    )

    is_valid = len(violations) == 0
    return is_valid, violations


def validate_no_duplicate_timestamps(
    df: pl.DataFrame,
    asset_column: str = "asset_id",
    timestamp_column: str = "timestamp",
) -> tuple[bool, list[dict[str, Any]]]:
    """Validate no duplicate timestamps for same asset.

    Args:
        df: DataFrame to validate
        asset_column: Name of asset ID column
        timestamp_column: Name of timestamp column

    Returns:
        Tuple of (is_valid, duplicates_list)
    """
    duplicates = (
        df.group_by([asset_column, timestamp_column])
        .agg(pl.count().alias("count"))
        .filter(pl.col("count") > 1)
    )

    duplicate_records = []
    if duplicates.height > 0:
        for row in duplicates.to_dicts():
            duplicate_records.append(
                {
                    "asset_id": row[asset_column],
                    "timestamp": row[timestamp_column],
                    "count": row["count"],
                    "message": f"Duplicate timestamp for {row[asset_column]} "
                    f"at {row[timestamp_column]} ({row['count']} occurrences)",
                }
            )

    is_valid = len(duplicate_records) == 0
    return is_valid, duplicate_records


def validate_ohlc_consistency(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> tuple[bool, list[dict[str, Any]]]:
    """Validate OHLC price relationships are consistent.

    Checks:
    - high >= max(open, close, low)
    - low <= min(open, close, high)
    - All prices > 0

    Args:
        df: DataFrame with OHLC data
        open_col: Name of open price column
        high_col: Name of high price column
        low_col: Name of low price column
        close_col: Name of close price column

    Returns:
        Tuple of (is_valid, violations_list)
    """
    violations = []

    # Check high >= max(open, close, low)
    invalid_high = df.filter(
        (pl.col(high_col) < pl.col(open_col))
        | (pl.col(high_col) < pl.col(close_col))
        | (pl.col(high_col) < pl.col(low_col))
    )

    # Check low <= min(open, close, high)
    invalid_low = df.filter(
        (pl.col(low_col) > pl.col(open_col))
        | (pl.col(low_col) > pl.col(close_col))
        | (pl.col(low_col) > pl.col(high_col))
    )

    # Check all prices > 0 (check each price column separately)
    for col in [open_col, high_col, low_col, close_col]:
        invalid_prices_col = df.filter(pl.col(col) <= 0)
        for row in invalid_prices_col.to_dicts():
            violations.append(
                {
                    "type": "non_positive_price",
                    "column": col,
                    "row": row,
                    "message": f"Non-positive price in {col}: {row[col]} at "
                    f"{row.get('timestamp', 'unknown time')}",
                }
            )

    # Collect violations
    for row in invalid_high.to_dicts():
        violations.append(
            {
                "type": "invalid_high",
                "row": row,
                "message": f"High price {row[high_col]} is less than "
                f"open/close/low at {row.get('timestamp', 'unknown time')}",
            }
        )

    for row in invalid_low.to_dicts():
        violations.append(
            {
                "type": "invalid_low",
                "row": row,
                "message": f"Low price {row[low_col]} is greater than "
                f"open/close/high at {row.get('timestamp', 'unknown time')}",
            }
        )

    is_valid = len(violations) == 0
    return is_valid, violations


def validate_missing_data(
    df: pl.DataFrame, required_columns: list[str]
) -> tuple[bool, list[dict[str, Any]]]:
    """Validate no missing data in required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must not have nulls

    Returns:
        Tuple of (is_valid, missing_data_list)
    """
    missing_data = []

    for col in required_columns:
        if col not in df.columns:
            missing_data.append(
                {
                    "column": col,
                    "type": "missing_column",
                    "message": f"Required column '{col}' not found in DataFrame",
                }
            )
            continue

        null_count = df[col].null_count()
        if null_count > 0:
            missing_data.append(
                {
                    "column": col,
                    "type": "null_values",
                    "count": null_count,
                    "message": f"Column '{col}' has {null_count} null values",
                }
            )

    is_valid = len(missing_data) == 0
    return is_valid, missing_data
