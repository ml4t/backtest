"""Tests for rebalance schedule resolution."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from ml4t.backtest.execution import (
    RebalanceCadence,
    RebalanceSchedule,
    resolve_rebalance_timestamps,
)
from ml4t.specs.market_data import FeedSpec


def _make_weekday_series(start: str, end: str) -> pl.Series:
    dates = pl.date_range(
        datetime.strptime(start, "%Y-%m-%d"),
        datetime.strptime(end, "%Y-%m-%d"),
        interval="1d",
        eager=True,
    )
    return (
        pl.DataFrame({"timestamp": dates})
        .filter(pl.col("timestamp").dt.weekday() <= 5)
        .get_column("timestamp")
    )


class TestResolveRebalanceTimestamps:
    def test_every_bar_returns_all_timestamps(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-10")

        result = resolve_rebalance_timestamps(timestamps, RebalanceSchedule.every_bar())

        expected = [datetime.combine(ts, datetime.min.time()) for ts in timestamps.to_list()]
        assert result.to_list() == expected

    def test_explicit_timestamps_intersects_available_bars(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-10")
        selected = [timestamps[1], timestamps[3], datetime(2024, 1, 31)]

        result = resolve_rebalance_timestamps(
            timestamps, RebalanceSchedule.explicit_timestamps(selected)
        )

        expected = [
            datetime.combine(timestamps[1], datetime.min.time()),
            datetime.combine(timestamps[3], datetime.min.time()),
        ]
        assert result.to_list() == expected

    def test_fixed_n_sessions_thins_session_closes(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-10")

        result = resolve_rebalance_timestamps(timestamps, RebalanceSchedule.fixed_n_sessions(2))

        expected = [datetime.combine(ts, datetime.min.time()) for ts in timestamps.to_list()[::2]]
        assert result.to_list() == expected

    def test_weekly_uses_last_available_session_in_week(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-31")

        result = resolve_rebalance_timestamps(timestamps, RebalanceSchedule.weekly())

        assert all(ts.weekday() == 4 for ts in result.to_list()[:-1])

    def test_month_end_uses_last_available_session_in_month(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-03-31")

        result = resolve_rebalance_timestamps(timestamps, RebalanceSchedule.month_end())

        resolved = result.to_list()
        assert len(resolved) == 3
        assert resolved[0].month == 1 and resolved[0].day == 31
        assert resolved[1].month == 2 and resolved[1].day == 29
        assert resolved[2].month == 3 and resolved[2].day == 29

    def test_cme_session_grouping_uses_session_boundaries(self) -> None:
        timestamps = [
            datetime(2024, 1, 7, 18, 0),
            datetime(2024, 1, 8, 10, 0),
            datetime(2024, 1, 8, 18, 0),
            datetime(2024, 1, 9, 10, 0),
        ]

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceCadence.EVERY_SESSION,
            calendar="CME_Equity",
            timezone="America/Chicago",
        )

        assert result.to_list() == [datetime(2024, 1, 8, 10, 0), datetime(2024, 1, 9, 10, 0)]

    def test_explicit_timezone_is_not_overridden_by_calendar(self) -> None:
        timestamps = [
            datetime(2024, 1, 8, 15, 0, tzinfo=UTC),
            datetime(2024, 1, 8, 16, 0, tzinfo=UTC),
        ]

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceCadence.EVERY_SESSION,
            calendar="NYSE",
            timezone="America/Chicago",
        )

        assert result.to_list() == [timestamps[0], timestamps[1]]

    def test_weekly_daily_session_labels_use_labeled_dates_not_prior_sessions(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-12")

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceSchedule.weekly(),
            feed_spec=FeedSpec(
                calendar="NYSE",
                data_frequency="daily",
                timestamp_semantics="session_label",
            ),
        )

        assert result.to_list() == [datetime(2024, 1, 5), datetime(2024, 1, 12)]

    def test_month_end_daily_session_labels_do_not_roll_first_day_into_prior_month(self) -> None:
        timestamps = pl.Series(
            "timestamp",
            [
                datetime(2024, 1, 30),
                datetime(2024, 1, 31),
                datetime(2024, 2, 1),
                datetime(2024, 2, 29),
            ],
        )

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceSchedule.month_end(),
            feed_spec=FeedSpec(
                calendar="NYSE",
                data_frequency="daily",
                timestamp_semantics="session_label",
            ),
        )

        assert result.to_list() == [datetime(2024, 1, 31), datetime(2024, 2, 29)]

    def test_daily_midnight_bars_fallback_to_session_labels_without_explicit_semantics(
        self,
    ) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-12")

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceSchedule.weekly(),
            data_frequency="daily",
            calendar="NYSE",
        )

        assert result.to_list() == [datetime(2024, 1, 5), datetime(2024, 1, 12)]
