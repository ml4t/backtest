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


def validate_volume_sanity(
    df: pl.DataFrame,
    volume_col: str = "volume",
    asset_column: str = "asset_id",
    timestamp_column: str = "timestamp",
    max_outlier_std: float = 10.0,
) -> tuple[bool, list[dict[str, Any]]]:
    """Validate volume data sanity.

    Checks:
    - Volume >= 0 (cannot be negative)
    - Extreme outliers detected (values > max_outlier_std standard deviations from mean)

    Args:
        df: DataFrame with volume data
        volume_col: Name of volume column
        asset_column: Name of asset ID column (for per-asset outlier detection)
        timestamp_column: Name of timestamp column (for error reporting)
        max_outlier_std: Maximum standard deviations from mean before flagging as outlier

    Returns:
        Tuple of (is_valid, violations_list)

    Example:
        >>> df = pl.DataFrame({
        ...     "timestamp": [ts1, ts2, ts3],
        ...     "asset_id": ["AAPL", "AAPL", "AAPL"],
        ...     "volume": [1_000_000, 2_000_000, -100]  # Negative volume is invalid
        ... })
        >>> is_valid, violations = validate_volume_sanity(df)
        >>> assert not is_valid
        >>> assert "negative_volume" in violations[0]["type"]
    """
    violations = []

    # Check for negative volumes (critical error)
    negative_volumes = df.filter(pl.col(volume_col) < 0)
    for row in negative_volumes.to_dicts():
        violations.append(
            {
                "type": "negative_volume",
                "severity": "CRITICAL",
                "asset_id": row.get(asset_column),
                "timestamp": row.get(timestamp_column),
                "volume": row[volume_col],
                "message": f"Negative volume {row[volume_col]} for "
                f"{row.get(asset_column, 'unknown')} at "
                f"{row.get(timestamp_column, 'unknown time')}",
            }
        )

    # Check for extreme outliers (warning, not critical)
    # Group by asset and calculate mean + std for each asset
    if asset_column in df.columns and df.height > 0:
        # Calculate per-asset statistics
        asset_stats = df.group_by(asset_column).agg(
            [
                pl.col(volume_col).mean().alias("mean_volume"),
                pl.col(volume_col).std().alias("std_volume"),
            ]
        )

        # Join stats back to original data
        df_with_stats = df.join(asset_stats, on=asset_column, how="left")

        # Find outliers (volume > mean + max_outlier_std * std)
        outliers = df_with_stats.filter(
            (pl.col(volume_col) > pl.col("mean_volume") + max_outlier_std * pl.col("std_volume"))
            & (pl.col("std_volume").is_not_null())  # Exclude cases where std is null
        )

        for row in outliers.to_dicts():
            violations.append(
                {
                    "type": "volume_outlier",
                    "severity": "WARNING",
                    "asset_id": row.get(asset_column),
                    "timestamp": row.get(timestamp_column),
                    "volume": row[volume_col],
                    "mean_volume": row.get("mean_volume"),
                    "std_volume": row.get("std_volume"),
                    "std_deviations": (row[volume_col] - row.get("mean_volume", 0))
                    / max(row.get("std_volume", 1), 1e-10),
                    "message": f"Volume outlier for {row.get(asset_column, 'unknown')}: "
                    f"{row[volume_col]:,.0f} (mean: {row.get('mean_volume', 0):,.0f}, "
                    f"std: {row.get('std_volume', 0):,.0f}) at "
                    f"{row.get(timestamp_column, 'unknown time')}",
                }
            )

    is_valid = len(violations) == 0
    return is_valid, violations


def validate_time_series_gaps(
    df: pl.DataFrame,
    asset_column: str = "asset_id",
    timestamp_column: str = "timestamp",
    expected_frequency: str | None = None,
    max_gap_multiplier: float = 3.0,
) -> tuple[bool, list[dict[str, Any]]]:
    """Detect gaps in time series (missing bars).

    Identifies periods where data is missing by detecting gaps larger than expected
    frequency. Useful for detecting data quality issues like missing trading days.

    Args:
        df: DataFrame with time series data
        asset_column: Name of asset ID column
        timestamp_column: Name of timestamp column
        expected_frequency: Expected frequency (e.g., "1d" for daily, "1h" for hourly)
                          If None, infers from median gap
        max_gap_multiplier: Flag gaps larger than max_gap_multiplier × expected_frequency

    Returns:
        Tuple of (is_valid, gaps_list)

    Example:
        >>> df = pl.DataFrame({
        ...     "timestamp": [
        ...         datetime(2024, 1, 1),
        ...         datetime(2024, 1, 2),
        ...         datetime(2024, 1, 5),  # Missing 1/3 and 1/4
        ...     ],
        ...     "asset_id": ["AAPL", "AAPL", "AAPL"],
        ...     "close": [100, 101, 105]
        ... })
        >>> is_valid, gaps = validate_time_series_gaps(df, expected_frequency="1d")
        >>> assert not is_valid  # Gap detected between 1/2 and 1/5
    """
    gaps = []

    if df.height == 0:
        return True, gaps

    # Process each asset separately
    assets = df[asset_column].unique().to_list()

    for asset_id in assets:
        # Get data for this asset, sorted by timestamp
        asset_df = df.filter(pl.col(asset_column) == asset_id).sort(timestamp_column)

        if asset_df.height < 2:
            # Need at least 2 rows to detect gaps
            continue

        # Calculate time differences between consecutive rows
        asset_df = asset_df.with_columns(
            [
                pl.col(timestamp_column).diff().alias("time_diff"),
            ]
        )

        # Determine expected frequency
        if expected_frequency is None:
            # Infer from median gap (more robust than mean)
            median_gap = asset_df["time_diff"].drop_nulls().median()
            if median_gap is None:
                continue
            expected_gap = median_gap
        else:
            # Parse expected frequency (simple implementation for common cases)
            expected_gap = _parse_frequency_to_timedelta(expected_frequency)

        # Find gaps larger than max_gap_multiplier × expected
        max_allowed_gap = expected_gap * max_gap_multiplier

        large_gaps = asset_df.filter(
            (pl.col("time_diff").is_not_null())
            & (pl.col("time_diff") > max_allowed_gap)
        )

        for row in large_gaps.to_dicts():
            time_diff = row["time_diff"]
            gaps.append(
                {
                    "type": "time_series_gap",
                    "severity": "WARNING",
                    "asset_id": asset_id,
                    "timestamp_after_gap": row[timestamp_column],
                    "gap_duration": time_diff,
                    "gap_seconds": time_diff.total_seconds() if hasattr(time_diff, "total_seconds") else None,
                    "expected_gap_seconds": expected_gap.total_seconds() if hasattr(expected_gap, "total_seconds") else None,
                    "message": f"Time series gap for {asset_id}: {time_diff} gap detected "
                    f"(expected: {expected_gap}) ending at {row[timestamp_column]}",
                }
            )

    is_valid = len(gaps) == 0
    return is_valid, gaps


def _parse_frequency_to_timedelta(freq_str: str) -> timedelta:
    """Parse frequency string to timedelta.

    Args:
        freq_str: Frequency string (e.g., "1d", "1h", "5m", "1w")

    Returns:
        timedelta object

    Example:
        >>> _parse_frequency_to_timedelta("1d")
        timedelta(days=1)
        >>> _parse_frequency_to_timedelta("5m")
        timedelta(minutes=5)
    """
    import re

    # Parse format like "1d", "5m", "1h", "1w"
    match = re.match(r"(\d+)([smhdw])", freq_str.lower())
    if not match:
        raise ValueError(f"Cannot parse frequency string: {freq_str}")

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "s":
        return timedelta(seconds=value)
    elif unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    elif unit == "w":
        return timedelta(weeks=value)
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def validate_price_sanity(
    df: pl.DataFrame,
    price_columns: list[str] | None = None,
    asset_column: str = "asset_id",
    timestamp_column: str = "timestamp",
    max_daily_change: float = 0.50,  # 50% daily change is extreme
    min_price: float = 0.01,  # Prices below $0.01 are suspicious
    max_price: float = 1_000_000.0,  # Prices above $1M are suspicious
) -> tuple[bool, list[dict[str, Any]]]:
    """Validate price data sanity (extreme movements, outliers).

    Checks:
    - Prices within reasonable range [min_price, max_price]
    - No extreme percentage changes (> max_daily_change)

    Args:
        df: DataFrame with price data
        price_columns: List of price column names to check (default: ["open", "high", "low", "close"])
        asset_column: Name of asset ID column
        timestamp_column: Name of timestamp column
        max_daily_change: Maximum allowed daily percentage change (default: 0.50 = 50%)
        min_price: Minimum valid price (default: 0.01)
        max_price: Maximum valid price (default: 1,000,000)

    Returns:
        Tuple of (is_valid, violations_list)

    Example:
        >>> df = pl.DataFrame({
        ...     "timestamp": [ts1, ts2],
        ...     "asset_id": ["AAPL", "AAPL"],
        ...     "close": [100.0, 300.0]  # 200% increase in one bar
        ... })
        >>> is_valid, violations = validate_price_sanity(df, max_daily_change=0.20)
        >>> assert not is_valid  # 200% change exceeds 20% threshold
    """
    violations = []

    if price_columns is None:
        price_columns = ["open", "high", "low", "close"]

    # Filter to only columns that exist in the DataFrame
    price_columns = [col for col in price_columns if col in df.columns]

    if not price_columns:
        # No price columns to validate
        return True, violations

    # Check for prices outside valid range
    for col in price_columns:
        # Prices below minimum
        too_low = df.filter(pl.col(col) < min_price)
        for row in too_low.to_dicts():
            violations.append(
                {
                    "type": "price_too_low",
                    "severity": "WARNING",
                    "asset_id": row.get(asset_column),
                    "timestamp": row.get(timestamp_column),
                    "column": col,
                    "price": row[col],
                    "min_price": min_price,
                    "message": f"Price too low in {col}: {row[col]} < {min_price} "
                    f"for {row.get(asset_column, 'unknown')} at "
                    f"{row.get(timestamp_column, 'unknown time')}",
                }
            )

        # Prices above maximum
        too_high = df.filter(pl.col(col) > max_price)
        for row in too_high.to_dicts():
            violations.append(
                {
                    "type": "price_too_high",
                    "severity": "WARNING",
                    "asset_id": row.get(asset_column),
                    "timestamp": row.get(timestamp_column),
                    "column": col,
                    "price": row[col],
                    "max_price": max_price,
                    "message": f"Price too high in {col}: {row[col]} > {max_price} "
                    f"for {row.get(asset_column, 'unknown')} at "
                    f"{row.get(timestamp_column, 'unknown time')}",
                }
            )

    # Check for extreme percentage changes (per asset)
    # Use close price for percentage change calculation
    if "close" in price_columns:
        assets = df[asset_column].unique().to_list()

        for asset_id in assets:
            asset_df = df.filter(pl.col(asset_column) == asset_id).sort(timestamp_column)

            if asset_df.height < 2:
                continue

            # Calculate percentage change
            asset_df = asset_df.with_columns(
                [
                    (pl.col("close").pct_change().abs()).alias("pct_change"),
                ]
            )

            # Find extreme changes
            extreme_changes = asset_df.filter(
                (pl.col("pct_change").is_not_null())
                & (pl.col("pct_change") > max_daily_change)
            )

            for row in extreme_changes.to_dicts():
                violations.append(
                    {
                        "type": "extreme_price_change",
                        "severity": "WARNING",
                        "asset_id": asset_id,
                        "timestamp": row[timestamp_column],
                        "pct_change": row["pct_change"],
                        "max_daily_change": max_daily_change,
                        "close_price": row["close"],
                        "message": f"Extreme price change for {asset_id}: "
                        f"{row['pct_change']:.2%} change (max allowed: {max_daily_change:.2%}) "
                        f"at {row[timestamp_column]}",
                    }
                )

    is_valid = len(violations) == 0
    return is_valid, violations


def validate_comprehensive(
    df: pl.DataFrame,
    validate_duplicates: bool = True,
    validate_ohlc: bool = True,
    validate_missing: bool = True,
    validate_volume: bool = True,
    validate_price: bool = True,
    validate_gaps: bool = True,
    required_columns: list[str] | None = None,
    asset_column: str = "asset_id",
    timestamp_column: str = "timestamp",
    volume_col: str = "volume",
    expected_frequency: str | None = None,
) -> tuple[bool, dict[str, list[dict[str, Any]]]]:
    """Run all validation checks on a DataFrame.

    Master validation function that orchestrates all individual validation checks
    and returns a comprehensive report.

    Args:
        df: DataFrame to validate
        validate_duplicates: Check for duplicate timestamps (default: True)
        validate_ohlc: Check OHLC consistency (default: True)
        validate_missing: Check for missing data (default: True)
        validate_volume: Check volume sanity (default: True)
        validate_price: Check price sanity (default: True)
        validate_gaps: Check for time series gaps (default: True)
        required_columns: List of required columns (default: ["timestamp", "asset_id", "open", "high", "low", "close"])
        asset_column: Name of asset ID column
        timestamp_column: Name of timestamp column
        volume_col: Name of volume column
        expected_frequency: Expected time series frequency (e.g., "1d")

    Returns:
        Tuple of (is_valid, violations_by_category)
        - is_valid: True if all checks pass
        - violations_by_category: Dict mapping check name to list of violations

    Example:
        >>> df = pl.DataFrame({...})  # Your price data
        >>> is_valid, violations = validate_comprehensive(df)
        >>> if not is_valid:
        ...     for check_name, check_violations in violations.items():
        ...         print(f"{check_name}: {len(check_violations)} violations")
        ...         for v in check_violations:
        ...             print(f"  - {v['message']}")

    Notes:
        - All checks use Polars native operations for performance
        - Validation runs on ALL rows, not samples
        - Performance target: < 1 second for 250 symbols × 1 year (252 bars)
    """
    all_violations: dict[str, list[dict[str, Any]]] = {}

    if required_columns is None:
        required_columns = [timestamp_column, asset_column, "open", "high", "low", "close"]

    # 1. Check for missing required columns and null values
    if validate_missing:
        is_valid, violations = validate_missing_data(df, required_columns)
        if not is_valid:
            all_violations["missing_data"] = violations

    # 2. Check for duplicate timestamps
    if validate_duplicates:
        is_valid, violations = validate_no_duplicate_timestamps(
            df, asset_column=asset_column, timestamp_column=timestamp_column
        )
        if not is_valid:
            all_violations["duplicates"] = violations

    # 3. Check OHLC consistency
    if validate_ohlc and all(col in df.columns for col in ["open", "high", "low", "close"]):
        is_valid, violations = validate_ohlc_consistency(df)
        if not is_valid:
            all_violations["ohlc_consistency"] = violations

    # 4. Check volume sanity
    if validate_volume and volume_col in df.columns:
        is_valid, violations = validate_volume_sanity(
            df,
            volume_col=volume_col,
            asset_column=asset_column,
            timestamp_column=timestamp_column,
        )
        if not is_valid:
            all_violations["volume_sanity"] = violations

    # 5. Check price sanity
    if validate_price:
        is_valid, violations = validate_price_sanity(
            df,
            asset_column=asset_column,
            timestamp_column=timestamp_column,
        )
        if not is_valid:
            all_violations["price_sanity"] = violations

    # 6. Check for time series gaps
    if validate_gaps:
        is_valid, violations = validate_time_series_gaps(
            df,
            asset_column=asset_column,
            timestamp_column=timestamp_column,
            expected_frequency=expected_frequency,
        )
        if not is_valid:
            all_violations["time_series_gaps"] = violations

    # Overall validation result
    is_valid = len(all_violations) == 0

    return is_valid, all_violations
