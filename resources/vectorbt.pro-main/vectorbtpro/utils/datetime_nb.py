# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled utilities for efficient date and time computations."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils.datetime_ import DTCNT
from vectorbtpro.utils.formatting import prettify_doc

__all__ = []

__pdoc__ = {}

us_ns = 1000
"""Constant representing the number of nanoseconds in a microsecond."""

ms_ns = us_ns * 1000
"""Constant representing the number of nanoseconds in a millisecond."""

s_ns = ms_ns * 1000
"""Constant representing the number of nanoseconds in a second."""

m_ns = s_ns * 60
"""Constant representing the number of nanoseconds in a minute."""

h_ns = m_ns * 60
"""Constant representing the number of nanoseconds in an hour."""

d_ns = h_ns * 24
"""Constant representing the number of nanoseconds in a day."""

w_ns = d_ns * 7
"""Constant representing the number of nanoseconds in a week."""

y_ns = (d_ns * 438291) // 1200
"""Constant representing the number of nanoseconds in a year."""

q_ns = y_ns // 4
"""Constant representing the number of nanoseconds in a quarter."""

mo_ns = q_ns // 3
"""Constant representing the number of nanoseconds in a month."""

semi_mo_ns = mo_ns // 2
"""Constant representing the number of nanoseconds in a semi-month."""

ns_td = np.timedelta64(1, "ns")
"""Timedelta representing one nanosecond."""

us_td = np.timedelta64(us_ns, "ns")
"""Timedelta representing one microsecond."""

ms_td = np.timedelta64(ms_ns, "ns")
"""Timedelta representing one millisecond."""

s_td = np.timedelta64(s_ns, "ns")
"""Timedelta representing one second."""

m_td = np.timedelta64(m_ns, "ns")
"""Timedelta representing one minute."""

h_td = np.timedelta64(h_ns, "ns")
"""Timedelta representing one hour."""

d_td = np.timedelta64(d_ns, "ns")
"""Timedelta representing one day."""

w_td = np.timedelta64(w_ns, "ns")
"""Timedelta representing one week."""

semi_mo_td = np.timedelta64(semi_mo_ns, "ns")
"""Timedelta representing one semi-month."""

mo_td = np.timedelta64(mo_ns, "ns")
"""Timedelta representing one month."""

q_td = np.timedelta64(q_ns, "ns")
"""Timedelta representing one quarter."""

y_td = np.timedelta64(y_ns, "ns")
"""Timedelta representing one year."""

unix_epoch_dt = np.datetime64(0, "ns")
"""Datetime representing the Unix epoch."""


@register_jitted(cache=True)
def second_remainder_nb(ts: int) -> int:
    """Return the nanosecond remainder of a timestamp after removing complete seconds.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Nanosecond remainder after full seconds.
    """
    return ts % 1000000000


@register_jitted(cache=True)
def nanosecond_nb(ts: int) -> int:
    """Return the nanosecond component of a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Nanosecond component.
    """
    return ts % 1000


@register_jitted(cache=True)
def microseconds_nb(ts: int) -> int:
    """Return the total microseconds from a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Total microseconds.
    """
    return ts // us_ns


@register_jitted(cache=True)
def microsecond_nb(ts: int) -> int:
    """Return the microsecond component within the current millisecond of a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Microsecond component within the current millisecond.
    """
    return microseconds_nb(ts) % (ms_ns // us_ns)


@register_jitted(cache=True)
def milliseconds_nb(ts: int) -> int:
    """Return the total milliseconds from a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Total milliseconds.
    """
    return ts // ms_ns


@register_jitted(cache=True)
def millisecond_nb(ts: int) -> int:
    """Return the millisecond component within the current second of a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Millisecond component within the current second.
    """
    return milliseconds_nb(ts) % (s_ns // ms_ns)


@register_jitted(cache=True)
def seconds_nb(ts: int) -> int:
    """Return the total seconds from a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Total seconds.
    """
    return ts // s_ns


@register_jitted(cache=True)
def second_nb(ts: int) -> int:
    """Return the second component within the current minute of a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Second component within the current minute.
    """
    return seconds_nb(ts) % (m_ns // s_ns)


@register_jitted(cache=True)
def minutes_nb(ts: int) -> int:
    """Return the total minutes from a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Total minutes.
    """
    return ts // m_ns


@register_jitted(cache=True)
def minute_nb(ts: int) -> int:
    """Return the minute component within the current hour of a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Minute component within the current hour.
    """
    return minutes_nb(ts) % (h_ns // m_ns)


@register_jitted(cache=True)
def hours_nb(ts: int) -> int:
    """Return the total hours from a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Total hours.
    """
    return ts // h_ns


@register_jitted(cache=True)
def hour_nb(ts: int) -> int:
    """Return the hour component within the current day of a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Hour component within the current day.
    """
    return hours_nb(ts) % (d_ns // h_ns)


@register_jitted(cache=True)
def days_nb(ts: int) -> int:
    """Return the total days from a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Total days.
    """
    return ts // d_ns


@register_jitted(cache=True)
def to_civil_nb(ts: int) -> tp.Tuple[int, int, int]:
    """Return the civil date corresponding to a timestamp as a tuple (year, month, day).

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        Tuple[int, int, int]: Tuple containing the year, month, and day.
    """
    z = days_nb(ts)
    z += 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return y + (m <= 2), m, d


@register_jitted(cache=True)
def from_civil_nb(y: int, m: int, d: int) -> int:
    """Return the timestamp corresponding to a given civil date.

    Args:
        y (int): Year.
        m (int): Month.
        d (int): Day.

    Returns:
        int: Timestamp in nanoseconds.
    """
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m - 3 if m > 2 else m + 9) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return d_ns * days


@register_jitted(cache=True)
def matches_date_nb(ts: int, y: int, m: int, d: int) -> bool:
    """Return whether the timestamp corresponds to the given civil date.

    Args:
        ts (int): Timestamp in nanoseconds.
        y (int): Year.
        m (int): Month.
        d (int): Day.

    Returns:
        int: True if the timestamp matches the date, False otherwise.
    """
    midnight_ts1 = midnight_nb(ts)
    midnight_ts2 = from_civil_nb(y, m, d)
    return midnight_ts1 == midnight_ts2


@register_jitted(cache=True)
def day_nb(ts: int) -> int:
    """Return the day component from the civil date of a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Day of the month.
    """
    y, m, d = to_civil_nb(ts)
    return d


@register_jitted(cache=True)
def midnight_nb(ts: int) -> int:
    """Return the timestamp corresponding to midnight of the day for a given timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Timestamp at midnight.
    """
    return ts - ts % d_ns


@register_jitted(cache=True)
def day_changed_nb(ts1: int, ts2: int) -> bool:
    """Return whether two timestamps occur on different days.

    Args:
        ts1 (int): First timestamp in nanoseconds.
        ts2 (int): Second timestamp in nanoseconds.

    Returns:
        bool: True if the timestamps are on different days, False otherwise.
    """
    return midnight_nb(ts1) != midnight_nb(ts2)


@register_jitted(cache=True)
def weekday_from_days_nb(days: int, zero_start: bool = True) -> int:
    """Return the weekday from the total number of days.

    Args:
        days (int): Total number of days.
        zero_start (bool): Use 0-based weekday indexing if True (0 = Monday, 6 = Sunday);
            otherwise use 1-based indexing (1 = Monday, 7 = Sunday).

    Returns:
        int: Weekday index.
    """
    c_weekday = (days + 4) % 7 if days >= -4 else (days + 5) % 7 + 6
    if c_weekday == 0:
        c_weekday = 7
    if zero_start:
        c_weekday = c_weekday - 1
    return c_weekday


@register_jitted(cache=True)
def weekday_nb(ts: int, zero_start: bool = True) -> int:
    """Return the weekday corresponding to a timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.
        zero_start (bool): Use 0-based weekday indexing if True (0 = Monday, 6 = Sunday);
            otherwise use 1-based indexing (1 = Monday, 7 = Sunday).

    Returns:
        int: Weekday index.
    """
    return weekday_from_days_nb(days_nb(ts), zero_start=zero_start)


@register_jitted(cache=True)
def weekday_diff_nb(weekday1: int, weekday2: int, zero_start: bool = True) -> int:
    """Calculate the forward difference in days from one weekday index to another.

    Args:
        weekday1 (int): Target weekday index.
        weekday2 (int): Starting weekday index.
        zero_start (bool): Use 0-based weekday indexing if True (0 = Monday, 6 = Sunday);
            otherwise use 1-based indexing (1 = Monday, 7 = Sunday).

    Returns:
        int: Number of days from `weekday2` to the next occurrence of `weekday1`.
    """
    if zero_start:
        if weekday1 > 6 or weekday1 < 0:
            raise ValueError("Weekday must be in [0, 6]")
        if weekday2 > 6 or weekday2 < 0:
            raise ValueError("Weekday must be in [0, 6]")
    else:
        if weekday1 > 7 or weekday1 < 1:
            raise ValueError("Weekday must be in [1, 7]")
        if weekday2 > 7 or weekday2 < 1:
            raise ValueError("Weekday must be in [1, 7]")
    weekday_diff = weekday1 - weekday2
    if weekday_diff <= 0:
        weekday_diff += 7
    return weekday_diff


@register_jitted(cache=True)
def past_weekday_nb(ts: int, weekday: int, zero_start: bool = True) -> int:
    """Determine the timestamp of the past occurrence of a specified weekday relative to a given timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.
        weekday (int): Target weekday index.
        zero_start (bool): Use 0-based weekday indexing if True (0 = Monday, 6 = Sunday);
            otherwise use 1-based indexing (1 = Monday, 7 = Sunday).

    Returns:
        int: Timestamp corresponding to the past occurrence at midnight of the specified weekday.
    """
    this_weekday = weekday_nb(ts, zero_start=zero_start)
    weekday_diff = weekday_diff_nb(this_weekday, weekday, zero_start=zero_start)
    return midnight_nb(ts) - weekday_diff * d_ns


@register_jitted(cache=True)
def future_weekday_nb(ts: int, weekday: int, zero_start: bool = True) -> int:
    """Determine the timestamp of the future occurrence of a specified weekday relative to a given timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.
        weekday (int): Target weekday index.
        zero_start (bool): Use 0-based weekday indexing if True (0 = Monday, 6 = Sunday);
            otherwise use 1-based indexing (1 = Monday, 7 = Sunday).

    Returns:
        int: Timestamp corresponding to the future occurrence at midnight of the specified weekday.
    """
    this_weekday = weekday_nb(ts, zero_start=zero_start)
    weekday_diff = weekday_diff_nb(weekday, this_weekday, zero_start=zero_start)
    return midnight_nb(ts) + weekday_diff * d_ns


@register_jitted(cache=True)
def day_of_year_nb(ts: int) -> int:
    """Calculate the day of the year from a given timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Day of the year, starting at 1.
    """
    y, m, d = to_civil_nb(ts)
    y_ts = from_civil_nb(y, 1, 1)
    return (ts - y_ts) // d_ns + 1


@register_jitted(cache=True)
def week_nb(ts: int) -> int:
    """Calculate the week number of the year based on the day of the year.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Week number of the year.
    """
    return day_of_year_nb(ts) // 7


@register_jitted(cache=True)
def month_nb(ts: int) -> int:
    """Determine the month from a given timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Month number.
    """
    y, m, d = to_civil_nb(ts)
    return m


@register_jitted(cache=True)
def year_nb(ts: int) -> int:
    """Determine the year from a given timestamp.

    Args:
        ts (int): Timestamp in nanoseconds.

    Returns:
        int: Year.
    """
    y, m, d = to_civil_nb(ts)
    return y


@register_jitted(cache=True)
def is_leap_year_nb(y: int) -> bool:
    """Determine if a given year is a leap year.

    Args:
        y (int): Year to evaluate.

    Returns:
        int: True if the year is a leap year, False otherwise.
    """
    return (y % 4 == 0) and (y % 100 != 0 or y % 400 == 0)


@register_jitted(cache=True)
def last_day_of_month_nb(y: int, m: int) -> int:
    """Determine the last day for a given month and year.

    Args:
        y (int): Year.
        m (int): Month.

    Returns:
        int: Last day of the month.
    """
    if m == 1:
        return 31
    if m == 2:
        if is_leap_year_nb(y):
            return 29
        return 28
    if m == 3:
        return 31
    if m == 4:
        return 30
    if m == 5:
        return 31
    if m == 6:
        return 30
    if m == 7:
        return 31
    if m == 8:
        return 31
    if m == 9:
        return 30
    if m == 10:
        return 31
    if m == 11:
        return 30
    return 31


@register_jitted(cache=True)
def matches_dtc_nb(dtc: DTCNT, other_dtc: DTCNT) -> bool:
    """Check if the specified datetime components in `dtc` match those in `other_dtc`.

    Components with a value of -1 are considered unspecified and are ignored in the comparison.

    Args:
        dtc (DTCNT): Named tuple with datetime components, where missing values are indicated by -1.
        other_dtc (DTCNT): Datetime components to match against.

    Returns:
        bool: True if the components match, False otherwise.
    """
    if dtc.year != -1 and other_dtc.year != -1 and dtc.year != other_dtc.year:
        return False
    if dtc.month != -1 and other_dtc.month != -1 and dtc.month != other_dtc.month:
        return False
    if dtc.day != -1 and other_dtc.day != -1 and dtc.day != other_dtc.day:
        return False
    if dtc.weekday != -1 and other_dtc.weekday != -1 and dtc.weekday != other_dtc.weekday:
        return False
    if dtc.hour != -1 and other_dtc.hour != -1 and dtc.hour != other_dtc.hour:
        return False
    if dtc.minute != -1 and other_dtc.minute != -1 and dtc.minute != other_dtc.minute:
        return False
    if dtc.second != -1 and other_dtc.second != -1 and dtc.second != other_dtc.second:
        return False
    if (
        dtc.nanosecond != -1
        and other_dtc.nanosecond != -1
        and dtc.nanosecond != other_dtc.nanosecond
    ):
        return False
    return True


@register_jitted(cache=True)
def index_matches_dtc_nb(index: tp.Array1d, other_dtc: DTCNT) -> tp.Array1d:
    """Apply `matches_dtc_nb` to each element in the index array and return a boolean mask.

    Args:
        index (Array1d): Array of timestamps to compare.
        other_dtc (DTCNT): Datetime components to match against.

    Returns:
        Array1d: Boolean mask indicating which elements match.
    """
    out = np.empty_like(index, dtype=np.bool_)
    for i in range(len(index)):
        ns = index[i]
        dtc = DTCNT(
            year=year_nb(ns),
            month=month_nb(ns),
            day=day_nb(ns),
            weekday=weekday_nb(ns),
            hour=hour_nb(ns),
            minute=minute_nb(ns),
            second=second_nb(ns),
            nanosecond=second_remainder_nb(ns),
        )
        out[i] = matches_dtc_nb(dtc, other_dtc)
    return out


class DTCST(tp.NamedTuple):
    SU: int = -3
    EU: int = -2
    U: int = -1
    O: int = 0
    I: int = 1


DTCS = DTCST()
"""_"""

__pdoc__["DTCS"] = f"""Datetime component status enumeration.

Status codes returned by `within_fixed_dtc_nb` and `within_periodic_dtc_nb`.

```python
{prettify_doc(DTCS)}
```

Fields:
    SU: Start matched; remaining components are unspecified.
    EU: End matched; remaining components are unspecified.
    U: Unknown status.
    O: Outside.
    I: Inside.
"""


@register_jitted(cache=True)
def within_fixed_dtc_nb(
    c: int,
    start_c: int = -1,
    end_c: int = -1,
    prev_status: int = DTCS.U,
    closed_start: bool = True,
    closed_end: bool = False,
    is_last: bool = False,
) -> int:
    """Return status indicating whether a datetime component is within a fixed range.

    Args:
        c (int): Datetime component value.
        start_c (int): Start boundary of the fixed range.
        end_c (int): End boundary of the fixed range.
        prev_status (int): Previous status code.

            See `DTCS`.
        closed_start (bool): Whether the start boundary is inclusive.
        closed_end (bool): Whether the end boundary is inclusive.
        is_last (bool): Flag specifying if this is the final evaluation.

    Returns:
        int: Status code from `DTCS` reflecting the component's position.
    """
    if prev_status == DTCS.U:
        _start_c = start_c
        _end_c = end_c
    elif prev_status == DTCS.SU:
        _start_c = start_c
        _end_c = -1
    elif prev_status == DTCS.EU:
        _start_c = -1
        _end_c = end_c
    else:
        raise ValueError("Invalid previous DTC status")

    if _start_c == -1:
        a = 0
    else:
        a = _start_c
    if _end_c == -1:
        b = 0
    else:
        b = _end_c

    if _start_c == -1 and _end_c == -1:
        return DTCS.U
    if _start_c != -1 and _end_c == -1:
        if c < a:
            return DTCS.O
        if c == a:
            if closed_start:
                if is_last:
                    return DTCS.I
            else:
                if is_last:
                    return DTCS.O
            return DTCS.SU
        if c > a:
            return DTCS.I
    if _start_c == -1 and _end_c != -1:
        if c < b:
            return DTCS.I
        if c == b:
            if closed_end:
                if is_last:
                    return DTCS.I
            else:
                if is_last:
                    return DTCS.O
            return DTCS.EU
        if c > b:
            return DTCS.O
    if _start_c != -1 and _end_c != -1:
        if c < a or c > b:
            return DTCS.O
        if c == a and c == b:
            if closed_start and closed_end:
                if is_last:
                    return DTCS.I
            else:
                if is_last:
                    return DTCS.O
            return DTCS.U
        if c == a:
            if closed_start:
                if is_last:
                    return DTCS.I
            else:
                if is_last:
                    return DTCS.O
            return DTCS.SU
        if c == b:
            if closed_end:
                if is_last:
                    return DTCS.I
            else:
                if is_last:
                    return DTCS.O
            return DTCS.EU
        if c > a and c < b:
            return DTCS.I


@register_jitted(cache=True)
def within_periodic_dtc_nb(
    c: int,
    start_c: int = -1,
    end_c: int = -1,
    prev_status: int = DTCS.U,
    closed_start: bool = True,
    closed_end: bool = False,
    overflow_later: bool = False,
    is_last: bool = False,
) -> int:
    """Return status indicating whether a datetime component is within a periodic range.

    Args:
        c (int): Datetime component value.
        start_c (int): Start boundary of the periodic range.
        end_c (int): End boundary of the periodic range.
        prev_status (int): Previous status code.

            See `DTCS`.
        closed_start (bool): Whether the start boundary is inclusive.
        closed_end (bool): Whether the end boundary is inclusive.
        overflow_later (bool): Flag to handle overflow when the range wraps around.
        is_last (bool): Flag specifying if this is the final evaluation.

    Returns:
        int: Status code from `DTCS` reflecting the component's position.
    """
    if prev_status == DTCS.U:
        _start_c = start_c
        _end_c = end_c
    elif prev_status == DTCS.SU:
        _start_c = start_c
        _end_c = -1
    elif prev_status == DTCS.EU:
        _start_c = -1
        _end_c = end_c
    else:
        raise ValueError("Invalid previous DTC status")

    if _start_c == -1:
        a = 0
    else:
        a = _start_c
    if _end_c == -1:
        b = 0
    else:
        b = _end_c

    if _start_c != -1 and _end_c != -1 and a == b:
        if overflow_later:
            return DTCS.U
    if _start_c != -1 and _end_c != -1 and a > b:
        status_after_start = within_fixed_dtc_nb(
            c,
            start_c=_start_c,
            end_c=-1,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            is_last=is_last,
        )
        status_before_end = within_fixed_dtc_nb(
            c,
            start_c=-1,
            end_c=_end_c,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            is_last=is_last,
        )
        if status_after_start == DTCS.O and status_before_end == DTCS.O:
            return DTCS.O
        if status_after_start == DTCS.I or status_before_end == DTCS.I:
            return DTCS.I
        if status_after_start == DTCS.SU:
            return DTCS.SU
        if status_before_end == DTCS.EU:
            return DTCS.EU
        return DTCS.U

    return within_fixed_dtc_nb(
        c,
        start_c=_start_c,
        end_c=_end_c,
        prev_status=prev_status,
        closed_start=closed_start,
        closed_end=closed_end,
        is_last=is_last,
    )


@register_jitted(cache=True)
def must_resolve_dtc_nb(
    c: int = -1,
    start_c: int = -1,
    end_c: int = -1,
) -> bool:
    """Return whether the datetime component must be resolved.

    Args:
        c (int): Datetime component value.
        start_c (int): Start boundary for resolution.
        end_c (int): End boundary for resolution.

    Returns:
        bool: True if the component must be resolved, False otherwise.
    """
    if c == -1:
        return False
    if start_c == -1 and end_c == -1:
        return False
    return True


@register_jitted(cache=True)
def start_dtc_lt_nb(
    c: int = -1,
    start_c: int = -1,
    end_c: int = -1,
) -> bool:
    """Return whether the start boundary is less than the end boundary.

    Args:
        c (int): Datetime component value.
        start_c (int): Start boundary to compare.
        end_c (int): End boundary to compare.

    Returns:
        bool: True if the start boundary is less than the end boundary, False otherwise.
    """
    if c == -1:
        return False
    if start_c == -1:
        return False
    if end_c == -1:
        return False
    return start_c < end_c


@register_jitted(cache=True)
def start_dtc_eq_nb(
    c: int = -1,
    start_c: int = -1,
    end_c: int = -1,
) -> bool:
    """Return whether the start boundary is equal to the end boundary.

    Args:
        c (int): Datetime component value.
        start_c (int): Start boundary to compare.
        end_c (int): End boundary to compare.

    Returns:
        bool: True if the start boundary is equal to the end boundary, False otherwise.
    """
    if c == -1:
        return False
    if start_c == -1:
        return False
    if end_c == -1:
        return False
    return start_c == end_c


@register_jitted(cache=True)
def start_dtc_gt_nb(
    c: int = -1,
    start_c: int = -1,
    end_c: int = -1,
) -> bool:
    """Return whether the start boundary is greater than the end boundary.

    Args:
        c (int): Datetime component value.
        start_c (int): Start boundary to compare.
        end_c (int): End boundary to compare.

    Returns:
        bool: True if the start boundary is greater than the end boundary, False otherwise.
    """
    if c == -1:
        return False
    if start_c == -1:
        return False
    if end_c == -1:
        return False
    return start_c > end_c


@register_jitted(cache=True)
def within_dtc_range_nb(
    dtc: DTCNT,
    start_dtc: DTCNT,
    end_dtc: DTCNT,
    closed_start: bool = True,
    closed_end: bool = False,
) -> bool:
    """Determine if the given datetime components fall within a specified range.

    Args:
        dtc (DTCNT): Named tuple with datetime components, where missing values are indicated by -1.
        start_dtc (DTCNT): Datetime component object representing the start boundary.
        end_dtc (DTCNT): Datetime component object representing the end boundary.
        closed_start (bool): Whether the start boundary is inclusive.
        closed_end (bool): Whether the end boundary is inclusive.

    Returns:
        bool: True if the datetime components fall within the specified range, False otherwise.
    """
    last = -1
    overflow_possible = True
    first_overflow = -1
    if must_resolve_dtc_nb(c=dtc.year, start_c=start_dtc.year, end_c=end_dtc.year):
        last = 0
        overflow_possible = False
    if must_resolve_dtc_nb(c=dtc.month, start_c=start_dtc.month, end_c=end_dtc.month):
        last = 1
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.month, start_c=start_dtc.month, end_c=end_dtc.month):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.month, start_c=start_dtc.month, end_c=end_dtc.month):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.day, start_c=start_dtc.day, end_c=end_dtc.day):
        last = 2
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.day, start_c=start_dtc.day, end_c=end_dtc.day):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.day, start_c=start_dtc.day, end_c=end_dtc.day):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.weekday, start_c=start_dtc.weekday, end_c=end_dtc.weekday):
        last = 3
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.weekday, start_c=start_dtc.weekday, end_c=end_dtc.weekday):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.weekday, start_c=start_dtc.weekday, end_c=end_dtc.weekday):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.hour, start_c=start_dtc.hour, end_c=end_dtc.hour):
        last = 4
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.hour, start_c=start_dtc.hour, end_c=end_dtc.hour):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.hour, start_c=start_dtc.hour, end_c=end_dtc.hour):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.minute, start_c=start_dtc.minute, end_c=end_dtc.minute):
        last = 5
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.minute, start_c=start_dtc.minute, end_c=end_dtc.minute):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.minute, start_c=start_dtc.minute, end_c=end_dtc.minute):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.second, start_c=start_dtc.second, end_c=end_dtc.second):
        last = 6
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.second, start_c=start_dtc.second, end_c=end_dtc.second):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.second, start_c=start_dtc.second, end_c=end_dtc.second):
                first_overflow = last
    if must_resolve_dtc_nb(
        c=dtc.nanosecond, start_c=start_dtc.nanosecond, end_c=end_dtc.nanosecond
    ):
        last = 7
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(
                c=dtc.nanosecond, start_c=start_dtc.nanosecond, end_c=end_dtc.nanosecond
            ):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(
                c=dtc.nanosecond, start_c=start_dtc.nanosecond, end_c=end_dtc.nanosecond
            ):
                first_overflow = last
    if last == -1:
        return True

    prev_status = DTCS.U
    if dtc.year != -1:
        prev_status = within_fixed_dtc_nb(
            dtc.year,
            start_c=start_dtc.year,
            end_c=end_dtc.year,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            is_last=last == 0,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.month != -1:
        prev_status = within_periodic_dtc_nb(
            dtc.month,
            start_c=start_dtc.month,
            end_c=end_dtc.month,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 1,
            is_last=last == 1,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.day != -1:
        prev_status = within_periodic_dtc_nb(
            dtc.day,
            start_c=start_dtc.day,
            end_c=end_dtc.day,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 2,
            is_last=last == 2,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.weekday != -1:
        prev_status = within_periodic_dtc_nb(
            dtc.weekday,
            start_c=start_dtc.weekday,
            end_c=end_dtc.weekday,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 3,
            is_last=last == 3,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.hour != -1:
        prev_status = within_periodic_dtc_nb(
            dtc.hour,
            start_c=start_dtc.hour,
            end_c=end_dtc.hour,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 4,
            is_last=last == 4,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.minute != -1:
        prev_status = within_periodic_dtc_nb(
            dtc.minute,
            start_c=start_dtc.minute,
            end_c=end_dtc.minute,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 5,
            is_last=last == 5,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.second != -1:
        prev_status = within_periodic_dtc_nb(
            dtc.second,
            start_c=start_dtc.second,
            end_c=end_dtc.second,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 6,
            is_last=last == 6,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.nanosecond != -1:
        prev_status = within_periodic_dtc_nb(
            dtc.nanosecond,
            start_c=start_dtc.nanosecond,
            end_c=end_dtc.nanosecond,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 7,
            is_last=last == 7,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True

    return True


@register_jitted(cache=True)
def index_within_dtc_range_nb(
    index: tp.Array1d,
    start_dtc: DTCNT,
    end_dtc: DTCNT,
    closed_start: bool = True,
    closed_end: bool = False,
) -> tp.Array1d:
    """Return a boolean mask indicating whether each element in `index` falls within the date-time
    range defined by `start_dtc` and `end_dtc`.

    Args:
        index (Array1d): One-dimensional array of date-time elements.
        start_dtc (DTCNT): Datetime component object representing the start boundary.
        end_dtc (DTCNT): Datetime component object representing the end boundary.
        closed_start (bool): Whether the start boundary is inclusive.
        closed_end (bool): Whether the end boundary is inclusive.

    Returns:
        Array1d: Boolean mask with True for elements within the range and False otherwise.
    """
    out = np.empty_like(index, dtype=np.bool_)
    for i in range(len(index)):
        ns = index[i]
        dtc = DTCNT(
            year=year_nb(ns),
            month=month_nb(ns),
            day=day_nb(ns),
            weekday=weekday_nb(ns),
            hour=hour_nb(ns),
            minute=minute_nb(ns),
            second=second_nb(ns),
            nanosecond=second_remainder_nb(ns),
        )
        out[i] = within_dtc_range_nb(
            dtc,
            start_dtc,
            end_dtc,
            closed_start=closed_start,
            closed_end=closed_end,
        )
    return out
