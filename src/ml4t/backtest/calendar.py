"""Exchange calendar integration using pandas_market_calendars.

This module provides a Polars-native interface to exchange calendars,
following the recommendation from the calendar_libraries.md analysis.

Key features:
- Trading schedule retrieval as Polars DataFrames
- Support for all major exchanges (NYSE, CME, LSE, etc.)
- Product-specific calendars for futures (CME_Equity, CME_Bond, etc.)
- Intraday break support (CME maintenance breaks)
- Trading day validation and date range generation

Example usage:
    from ml4t.backtest.calendar import (
        get_calendar,
        get_schedule,
        get_trading_days,
        is_trading_day,
    )

    # Get NYSE trading schedule for 2024
    schedule = get_schedule("XNYS", date(2024, 1, 1), date(2024, 12, 31))

    # Get trading days only
    trading_days = get_trading_days("XNYS", date(2024, 1, 1), date(2024, 12, 31))

    # Check if a specific date is a trading day
    if is_trading_day("XNYS", date(2024, 7, 4)):
        print("Market is open")
"""

from datetime import date, datetime
from functools import lru_cache

import pandas as pd
import polars as pl

# Import pandas_market_calendars with lazy loading to avoid import overhead
_mcal = None


def _get_mcal():
    """Lazy load pandas_market_calendars."""
    global _mcal
    if _mcal is None:
        import pandas_market_calendars as mcal

        _mcal = mcal
    return _mcal


# Common calendar aliases for convenience
CALENDAR_ALIASES = {
    # US Equities
    "NYSE": "XNYS",
    "NASDAQ": "NASDAQ",  # pandas_market_calendars uses "NASDAQ" directly
    "AMEX": "NYSE",  # AMEX follows NYSE calendar
    # US Futures (product-specific via pandas_market_calendars)
    "CME": "CME_Equity",  # Default to equity futures
    "CME_EQUITY": "CME_Equity",
    "CME_BOND": "CME_Bond",
    "CME_AGRICULTURE": "CME_Agriculture",
    "CBOT": "CME_Bond",  # CBOT is now CME
    "NYMEX": "CME_Equity",  # Energy futures
    "COMEX": "CME_Equity",  # Metals
    "ICE": "IEPA",
    # Europe
    "LSE": "XLON",
    "EUREX": "EUREX",
    "XETRA": "XFRA",
    # Asia-Pacific
    "TSE": "JPX",  # Tokyo - uses JPX in pandas_market_calendars
    "HKEX": "XHKG",
    "SSE": "XSHG",  # Shanghai
    "SZSE": "XSHE",  # Shenzhen
    "ASX": "XASX",
    "NSE": "XNSE",  # India
    # Americas
    "TSX": "XTSE",  # Toronto
    "BMV": "XMEX",  # Mexico
    "B3": "BVMF",  # Brazil
    # Crypto (24/7)
    "CRYPTO": "24/7",
}


@lru_cache(maxsize=32)
def get_calendar(calendar_id: str):
    """Get a market calendar instance.

    Args:
        calendar_id: Exchange MIC code (e.g., 'XNYS') or alias (e.g., 'NYSE').
            See CALENDAR_ALIASES for common aliases.

    Returns:
        pandas_market_calendars calendar instance

    Examples:
        >>> cal = get_calendar("NYSE")  # NYSE via alias
        >>> cal = get_calendar("XNYS")  # NYSE via MIC code
        >>> cal = get_calendar("CME_Equity")  # CME equity futures
    """
    mcal = _get_mcal()

    # Resolve alias to MIC code
    resolved_id = CALENDAR_ALIASES.get(calendar_id.upper(), calendar_id)

    return mcal.get_calendar(resolved_id)


def get_schedule(
    calendar_id: str,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    *,
    include_breaks: bool = False,
    include_extended_hours: bool = False,
) -> pl.DataFrame:
    """Get trading schedule as a Polars DataFrame.

    This is the primary function for retrieving exchange schedules.
    Uses the efficient pandas -> pyarrow -> polars conversion path.

    Args:
        calendar_id: Exchange MIC code or alias (e.g., 'NYSE', 'XNYS', 'CME_Equity')
        start_date: Start date of the schedule range
        end_date: End date of the schedule range
        include_breaks: If True, include break_start/break_end columns for
            exchanges with intraday breaks (e.g., CME maintenance break)
        include_extended_hours: If True, include pre_market/post_market columns

    Returns:
        Polars DataFrame with columns:
        - session_date: Date of the trading session
        - market_open: UTC datetime when market opens
        - market_close: UTC datetime when market closes
        - timezone: Exchange timezone (e.g., 'America/New_York')
        - (optional) break_start, break_end: Intraday break times
        - (optional) pre_market, post_market: Extended hours

    Examples:
        >>> schedule = get_schedule("NYSE", date(2024, 1, 1), date(2024, 12, 31))
        >>> schedule.head()
        shape: (5, 4)
        ┌──────────────┬─────────────────────────┬─────────────────────────┬──────────────────┐
        │ session_date ┆ market_open             ┆ market_close            ┆ timezone         │
        │ ---          ┆ ---                     ┆ ---                     ┆ ---              │
        │ date         ┆ datetime[μs, UTC]       ┆ datetime[μs, UTC]       ┆ str              │
        ╞══════════════╪═════════════════════════╪═════════════════════════╪══════════════════╡
        │ 2024-01-02   ┆ 2024-01-02 14:30:00 UTC ┆ 2024-01-02 21:00:00 UTC ┆ America/New_York │
        │ 2024-01-03   ┆ 2024-01-03 14:30:00 UTC ┆ 2024-01-03 21:00:00 UTC ┆ America/New_York │
        │ ...          ┆ ...                     ┆ ...                     ┆ ...              │
        └──────────────┴─────────────────────────┴─────────────────────────┴──────────────────┘
    """
    calendar = get_calendar(calendar_id)

    # Generate schedule using pandas_market_calendars
    schedule_pd = calendar.schedule(start_date=start_date, end_date=end_date)

    if schedule_pd.empty:
        # Return empty DataFrame with correct schema
        return pl.DataFrame(
            schema={
                "session_date": pl.Date,
                "market_open": pl.Datetime("us", "UTC"),
                "market_close": pl.Datetime("us", "UTC"),
                "timezone": pl.Utf8,
            }
        )

    # Convert to Polars via pyarrow (efficient zero-copy path)
    schedule_pl = pl.from_pandas(schedule_pd.reset_index())

    # Rename columns to standardized names
    rename_map = {
        "index": "session_date",
        "market_open": "market_open",
        "market_close": "market_close",
    }

    # Handle optional columns
    if include_breaks and "break_start" in schedule_pl.columns:
        rename_map["break_start"] = "break_start"
        rename_map["break_end"] = "break_end"

    if include_extended_hours:
        if "pre" in schedule_pl.columns:
            rename_map["pre"] = "pre_market"
        if "post" in schedule_pl.columns:
            rename_map["post"] = "post_market"

    # Apply renames for columns that exist
    existing_renames = {k: v for k, v in rename_map.items() if k in schedule_pl.columns}
    schedule_pl = schedule_pl.rename(existing_renames)

    # Select and structure output columns
    output_columns = ["session_date", "market_open", "market_close"]

    if include_breaks and "break_start" in schedule_pl.columns:
        output_columns.extend(["break_start", "break_end"])

    if include_extended_hours:
        if "pre_market" in schedule_pl.columns:
            output_columns.append("pre_market")
        if "post_market" in schedule_pl.columns:
            output_columns.append("post_market")

    # Get timezone string
    tz_key = str(calendar.tz) if hasattr(calendar, "tz") else "UTC"

    # Build final DataFrame
    result = schedule_pl.select(
        [col for col in output_columns if col in schedule_pl.columns]
    ).with_columns(
        pl.col("session_date").cast(pl.Date),
        pl.lit(tz_key).alias("timezone"),
    )

    return result


def get_trading_days(
    calendar_id: str,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> pl.Series:
    """Get list of trading days as a Polars Series.

    This is useful for filtering data to only trading days or
    generating date ranges for backtesting.

    Args:
        calendar_id: Exchange MIC code or alias
        start_date: Start date of the range
        end_date: End date of the range

    Returns:
        Polars Series of dates (pl.Date dtype)

    Examples:
        >>> trading_days = get_trading_days("NYSE", date(2024, 1, 1), date(2024, 1, 31))
        >>> len(trading_days)
        21  # NYSE had 21 trading days in Jan 2024
    """
    calendar = get_calendar(calendar_id)
    valid_days = calendar.valid_days(start_date=start_date, end_date=end_date)

    # Convert pandas DatetimeIndex to Polars Series
    return pl.Series("trading_day", valid_days.date)


def is_trading_day(calendar_id: str, check_date: date | datetime | str) -> bool:
    """Check if a specific date is a trading day.

    Args:
        calendar_id: Exchange MIC code or alias
        check_date: Date to check

    Returns:
        True if the date is a trading day, False otherwise

    Examples:
        >>> is_trading_day("NYSE", date(2024, 7, 4))  # Independence Day
        False
        >>> is_trading_day("NYSE", date(2024, 7, 5))  # Regular Friday
        True
    """
    calendar = get_calendar(calendar_id)

    # Convert to pandas Timestamp for comparison
    if isinstance(check_date, str | date | datetime):
        check_date = pd.Timestamp(check_date)

    valid_days = calendar.valid_days(start_date=check_date, end_date=check_date)
    return len(valid_days) > 0


def is_market_open(
    calendar_id: str,
    check_datetime: datetime,
) -> bool:
    """Check if the market is open at a specific datetime.

    This accounts for:
    - Regular trading hours
    - Intraday breaks (e.g., CME maintenance break 4-5 PM CT)
    - Early closes

    Args:
        calendar_id: Exchange MIC code or alias
        check_datetime: Datetime to check (should be timezone-aware or UTC)

    Returns:
        True if market is open, False otherwise

    Examples:
        >>> from datetime import datetime, timezone
        >>> dt = datetime(2024, 7, 5, 15, 30, tzinfo=timezone.utc)  # 11:30 AM ET
        >>> is_market_open("NYSE", dt)
        True
    """
    calendar = get_calendar(calendar_id)

    # Convert to pandas Timestamp
    ts = pd.Timestamp(check_datetime)

    # Get schedule for that day
    schedule = calendar.schedule(start_date=ts.date(), end_date=ts.date())

    if schedule.empty:
        return False

    market_open = schedule.iloc[0]["market_open"]
    market_close = schedule.iloc[0]["market_close"]

    # Check if within regular hours
    if not (market_open <= ts <= market_close):
        return False

    # Check for intraday breaks if available
    if "break_start" in schedule.columns and pd.notna(schedule.iloc[0]["break_start"]):
        break_start = schedule.iloc[0]["break_start"]
        break_end = schedule.iloc[0]["break_end"]
        if break_start <= ts <= break_end:
            return False

    return True


def next_trading_day(
    calendar_id: str,
    from_date: date | datetime | str,
    n: int = 1,
) -> date:
    """Get the next N trading day(s) after a given date.

    Args:
        calendar_id: Exchange MIC code or alias
        from_date: Starting date
        n: Number of trading days to advance (default 1)

    Returns:
        The nth trading day after from_date

    Examples:
        >>> next_trading_day("NYSE", date(2024, 7, 3))  # Wed before July 4th
        date(2024, 7, 5)  # Skips Thursday holiday
    """
    calendar = get_calendar(calendar_id)

    # Convert to pandas Timestamp
    if isinstance(from_date, str):
        from_date = pd.Timestamp(from_date).date()
    elif isinstance(from_date, datetime):
        from_date = from_date.date()

    # Get enough future days to find n trading days
    # Worst case: 2 weeks of holidays, so look 30 days ahead per n
    end_date = pd.Timestamp(from_date) + pd.Timedelta(days=max(30, n * 5))

    valid_days = calendar.valid_days(start_date=from_date, end_date=end_date)

    # Filter to days strictly after from_date
    future_days = [d.date() for d in valid_days if d.date() > from_date]

    if len(future_days) < n:
        raise ValueError(f"Could not find {n} trading days after {from_date}")

    return future_days[n - 1]


def previous_trading_day(
    calendar_id: str,
    from_date: date | datetime | str,
    n: int = 1,
) -> date:
    """Get the previous N trading day(s) before a given date.

    Args:
        calendar_id: Exchange MIC code or alias
        from_date: Starting date
        n: Number of trading days to go back (default 1)

    Returns:
        The nth trading day before from_date

    Examples:
        >>> previous_trading_day("NYSE", date(2024, 7, 5))  # Fri after July 4th
        date(2024, 7, 3)  # Skips Thursday holiday
    """
    calendar = get_calendar(calendar_id)

    # Convert to pandas Timestamp
    if isinstance(from_date, str):
        from_date = pd.Timestamp(from_date).date()
    elif isinstance(from_date, datetime):
        from_date = from_date.date()

    # Look back enough days
    start_date = pd.Timestamp(from_date) - pd.Timedelta(days=max(30, n * 5))

    valid_days = calendar.valid_days(start_date=start_date, end_date=from_date)

    # Filter to days strictly before from_date
    past_days = [d.date() for d in valid_days if d.date() < from_date]

    if len(past_days) < n:
        raise ValueError(f"Could not find {n} trading days before {from_date}")

    return past_days[-n]


def list_calendars() -> list[str]:
    """List all available calendar identifiers.

    Returns:
        List of available calendar MIC codes

    Examples:
        >>> calendars = list_calendars()
        >>> "XNYS" in calendars
        True
    """
    mcal = _get_mcal()
    return mcal.get_calendar_names()


def get_holidays(
    calendar_id: str,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> pl.DataFrame:
    """Get holidays for an exchange within a date range.

    Args:
        calendar_id: Exchange MIC code or alias
        start_date: Start date of the range
        end_date: End date of the range

    Returns:
        Polars DataFrame with columns:
        - date: Holiday date
        - name: Holiday name (if available)

    Examples:
        >>> holidays = get_holidays("NYSE", date(2024, 1, 1), date(2024, 12, 31))
        >>> holidays
        shape: (9, 2)
        ┌────────────┬────────────────────┐
        │ date       ┆ name               │
        │ ---        ┆ ---                │
        │ date       ┆ str                │
        ╞════════════╪════════════════════╡
        │ 2024-01-01 ┆ New Year's Day     │
        │ 2024-01-15 ┆ MLK Day            │
        │ ...        ┆ ...                │
        └────────────┴────────────────────┘
    """
    calendar = get_calendar(calendar_id)

    # Get all dates in range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Generate all calendar days in range
    all_days = pd.date_range(start=start, end=end, freq="D")

    # Get trading days
    trading_days = calendar.valid_days(start_date=start, end_date=end)

    # Holidays are weekdays that are not trading days
    # Note: day_of_week is the modern pandas attribute (dayofweek deprecated)
    weekdays = all_days[all_days.day_of_week < 5]  # ty: ignore[unresolved-attribute]  # Mon-Fri
    holidays = weekdays.difference(trading_days)

    # Build DataFrame
    return pl.DataFrame(
        {
            "date": [h.date() for h in holidays],
            "name": [None] * len(holidays),  # Names not easily available
        }
    )


def filter_to_trading_days(
    df: pl.DataFrame,
    calendar_id: str,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Filter a DataFrame to only include rows on trading days.

    Use this for DAILY bars where you only need to filter by date.
    For intraday data, use `filter_to_trading_hours()` instead.

    Args:
        df: Polars DataFrame with a timestamp column
        calendar_id: Exchange MIC code or alias
        timestamp_col: Name of the timestamp column

    Returns:
        DataFrame filtered to trading days only

    Examples:
        >>> # Filter daily price data to NYSE trading days
        >>> daily_prices = filter_to_trading_days(daily_prices, "NYSE")
    """
    # Get date range from data
    dates = df.select(pl.col(timestamp_col).cast(pl.Date).alias("__date"))
    min_date = dates.min().item()
    max_date = dates.max().item()

    # Get trading days as a Polars Series with proper Date dtype
    calendar = get_calendar(calendar_id)
    valid_days = calendar.valid_days(start_date=min_date, end_date=max_date)

    # Convert to Polars Series with Date dtype
    trading_days = pl.Series("trading_day", [d.date() for d in valid_days], dtype=pl.Date)

    # Filter to trading days
    return df.filter(pl.col(timestamp_col).cast(pl.Date).is_in(trading_days))


def filter_to_trading_sessions(
    df: pl.DataFrame,
    calendar_id: str,
    timestamp_col: str = "timestamp",
    *,
    naive_tz: str = "UTC",
    include_breaks: bool = False,
) -> pl.DataFrame:
    """Filter a DataFrame to only include rows during trading sessions.

    Use this for INTRADAY data (minute bars, tick data, trade bars).
    Filters out:
    - Non-trading days (weekends, holidays)
    - Pre-market hours
    - Post-market hours
    - Intraday breaks (optional, e.g., CME maintenance break)

    Works with any irregular timestamp data - doesn't require fixed frequency.
    Uses efficient Polars join_asof for interval matching.

    Args:
        df: Polars DataFrame with a timestamp column
        calendar_id: Exchange MIC code or alias
        timestamp_col: Name of the timestamp column
        naive_tz: Timezone to assume for naive datetimes (default: "UTC").
            Set this to match your data source (e.g., "America/New_York" for
            US equity data that's already in ET).
        include_breaks: If False (default), also filter out intraday breaks

    Returns:
        DataFrame filtered to trading sessions only

    Examples:
        >>> # Filter minute bars to NYSE trading sessions (data in UTC)
        >>> minute_bars = filter_to_trading_sessions(minute_bars, "NYSE")

        >>> # Data with naive timestamps that are actually in ET
        >>> bars = filter_to_trading_sessions(bars, "NYSE", naive_tz="America/New_York")

        >>> # Keep data during intraday breaks
        >>> data = filter_to_trading_sessions(data, "CME_Equity", include_breaks=True)
    """
    if df.is_empty():
        return df

    # Get date range from data - handle both tz-aware and naive
    ts_col = df[timestamp_col]
    ts_dtype = ts_col.dtype

    # Coerce to UTC for consistent comparison
    if isinstance(ts_dtype, pl.Datetime):
        if ts_dtype.time_zone is None:
            # Naive datetime - use specified timezone, then convert to UTC
            df = df.with_columns(
                pl.col(timestamp_col)
                .dt.replace_time_zone(naive_tz)
                .dt.convert_time_zone("UTC")
                .alias(timestamp_col)
            )
        elif ts_dtype.time_zone != "UTC":
            # Convert to UTC
            df = df.with_columns(
                pl.col(timestamp_col).dt.convert_time_zone("UTC").alias(timestamp_col)
            )

    # Get date range
    min_date = df.select(pl.col(timestamp_col).dt.date().min()).item()
    max_date = df.select(pl.col(timestamp_col).dt.date().max()).item()

    # Get schedule for the date range
    calendar = get_calendar(calendar_id)
    schedule_pd = calendar.schedule(start_date=min_date, end_date=max_date)

    if schedule_pd.empty:
        return df.clear()  # No trading days in range

    # Build sessions list, handling breaks
    sessions = []
    for _, row in schedule_pd.iterrows():
        market_open = row["market_open"]
        market_close = row["market_close"]

        if (
            not include_breaks
            and "break_start" in schedule_pd.columns
            and pd.notna(row.get("break_start"))
        ):
            # Split into pre-break and post-break sessions
            break_start = row["break_start"]
            break_end = row["break_end"]
            sessions.append((market_open, break_start))
            sessions.append((break_end, market_close))
        else:
            sessions.append((market_open, market_close))

    # Create sessions DataFrame with proper UTC dtype
    sessions_df = (
        pl.DataFrame(
            {
                "session_open": [s[0].to_pydatetime() for s in sessions],
                "session_close": [s[1].to_pydatetime() for s in sessions],
            }
        )
        .with_columns(
            pl.col("session_open").cast(pl.Datetime("us", "UTC")),
            pl.col("session_close").cast(pl.Datetime("us", "UTC")),
        )
        .sort("session_open")
    )

    # Use join_asof to find the session that starts at or before each timestamp
    # Then filter to rows where timestamp <= session_close
    result = (
        df.lazy()
        .sort(timestamp_col)
        .join_asof(
            sessions_df.lazy(),
            left_on=timestamp_col,
            right_on="session_open",
            strategy="backward",  # Find session that starts at or before timestamp
        )
        .filter(
            # Timestamp must be within the matched session
            pl.col(timestamp_col) <= pl.col("session_close")
        )
        .drop(["session_open", "session_close"])
        .collect()
    )

    return result


def generate_trading_minutes(
    calendar_id: str,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    *,
    freq: str = "1m",
    include_close: bool = True,
) -> pl.Series:
    """Generate a series of trading minute timestamps.

    Useful for creating a time index for minute-frequency backtests
    or for resampling irregular data to regular minute bars.

    Args:
        calendar_id: Exchange MIC code or alias
        start_date: Start date
        end_date: End date
        freq: Frequency string ('1m', '5m', '15m', '30m', '1h')
        include_close: If True, include the market close timestamp

    Returns:
        Polars Series of datetime timestamps

    Examples:
        >>> # Generate 1-minute timestamps for NYSE
        >>> minutes = generate_trading_minutes("NYSE", date(2024, 1, 2), date(2024, 1, 2))
        >>> len(minutes)
        391  # 9:30 AM to 4:00 PM = 6.5 hours = 390 minutes + close

        >>> # Generate 5-minute bars
        >>> bars_5m = generate_trading_minutes("NYSE", date(2024, 1, 2), date(2024, 1, 5), freq="5m")
    """
    # Parse frequency
    freq_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
    if freq not in freq_map:
        raise ValueError(f"Unsupported frequency: {freq}. Use one of {list(freq_map.keys())}")
    freq_minutes = freq_map[freq]

    # Get schedule
    calendar = get_calendar(calendar_id)
    schedule_pd = calendar.schedule(start_date=start_date, end_date=end_date)

    if schedule_pd.empty:
        return pl.Series("timestamp", [], dtype=pl.Datetime("us", "UTC"))

    # Generate timestamps for each session
    all_timestamps = []

    for _, row in schedule_pd.iterrows():
        market_open = row["market_open"]
        market_close = row["market_close"]

        # Generate minute timestamps
        current = market_open
        while current < market_close:
            all_timestamps.append(current)
            current = current + pd.Timedelta(minutes=freq_minutes)

        # Optionally include close
        if include_close and (not all_timestamps or all_timestamps[-1] != market_close):
            all_timestamps.append(market_close)

    return pl.Series("timestamp", all_timestamps)


def get_early_closes(
    calendar_id: str,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> pl.DataFrame:
    """Get early close days for an exchange within a date range.

    Args:
        calendar_id: Exchange MIC code or alias
        start_date: Start date of the range
        end_date: End date of the range

    Returns:
        Polars DataFrame with columns:
        - date: Early close date
        - close_time: Early close time (local)

    Examples:
        >>> early_closes = get_early_closes("NYSE", date(2024, 1, 1), date(2024, 12, 31))
        >>> early_closes
        shape: (4, 2)
        ┌────────────┬────────────┐
        │ date       ┆ close_time │
        │ ---        ┆ ---        │
        │ date       ┆ time       │
        ╞════════════╪════════════╡
        │ 2024-07-03 ┆ 13:00:00   │  # Day before July 4th
        │ 2024-11-29 ┆ 13:00:00   │  # Day after Thanksgiving
        │ 2024-12-24 ┆ 13:00:00   │  # Christmas Eve
        └────────────┴────────────┘
    """
    # Get full schedule
    schedule = get_schedule(calendar_id, start_date, end_date)

    if schedule.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "close_time": pl.Time})

    calendar = get_calendar(calendar_id)

    # Standard close time for this exchange
    regular_close = calendar.close_time

    # Find days where close is earlier than regular
    # This requires converting UTC close to local time
    tz = str(calendar.tz)

    early_closes = schedule.with_columns(
        pl.col("market_close").dt.convert_time_zone(tz).dt.time().alias("close_time_local")
    ).filter(pl.col("close_time_local") < pl.lit(regular_close))

    return early_closes.select(
        pl.col("session_date").alias("date"),
        pl.col("close_time_local").alias("close_time"),
    )
