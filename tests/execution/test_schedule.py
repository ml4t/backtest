"""Tests for rebalance schedule resolution."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from ml4t.specs.market_data import FeedSpec

from ml4t.backtest.execution import (
    RebalanceCadence,
    RebalanceConfig,
    RebalanceSchedule,
    TargetWeightExecutor,
    is_rebalance_timestamp,
    resolve_rebalance_timestamps,
)
from ml4t.backtest.execution import schedule as schedule_module


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

    def test_weekly_uses_scheduled_week_end(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-31")

        result = resolve_rebalance_timestamps(timestamps, RebalanceSchedule.weekly())

        assert result.to_list() == [
            datetime(2024, 1, 5),
            datetime(2024, 1, 12),
            datetime(2024, 1, 19),
            datetime(2024, 1, 26),
        ]

    def test_month_end_uses_scheduled_month_end(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-03-31")

        result = resolve_rebalance_timestamps(timestamps, RebalanceSchedule.month_end())

        resolved = result.to_list()
        assert len(resolved) == 3
        assert resolved[0].month == 1 and resolved[0].day == 31
        assert resolved[1].month == 2 and resolved[1].day == 29
        assert resolved[2].month == 3 and resolved[2].day == 29

    def test_incomplete_month_does_not_move_rebalance_to_last_available_bar(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-30")

        result = resolve_rebalance_timestamps(timestamps, RebalanceSchedule.month_end())

        assert result.to_list() == []

    def test_calendar_week_end_uses_last_exchange_session_before_holiday(self) -> None:
        timestamps = _make_weekday_series("2024-03-25", "2024-03-28")

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceSchedule.weekly(),
            calendar="NYSE",
            data_frequency="daily",
        )

        assert result.to_list() == [datetime(2024, 3, 28)]

    def test_cme_session_grouping_uses_session_boundaries(self) -> None:
        timestamps = [
            datetime(2024, 1, 7, 18, 0),
            datetime(2024, 1, 8, 16, 0),
            datetime(2024, 1, 8, 18, 0),
            datetime(2024, 1, 9, 16, 0),
        ]

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceCadence.EVERY_SESSION,
            calendar="CME_Equity",
            timezone="America/Chicago",
        )

        assert result.to_list() == [datetime(2024, 1, 8, 16, 0), datetime(2024, 1, 9, 16, 0)]

    def test_naive_intraday_timestamps_default_to_utc(self) -> None:
        timestamps = [datetime(2024, 1, 7, 23), datetime(2024, 1, 8, 22)]

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceCadence.EVERY_SESSION,
            calendar="CME_Equity",
            session_start_time="17:00",
            data_frequency="1m",
            timestamp_semantics="bar_close",
        )

        assert result.to_list() == [datetime(2024, 1, 8, 22)]

    def test_intraday_batch_schedule_requires_the_exchange_close(self) -> None:
        timestamps = [
            datetime(2024, 1, 8, 15, 0, tzinfo=UTC),
            datetime(2024, 1, 8, 21, 0, tzinfo=UTC),
        ]

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceCadence.EVERY_SESSION,
            calendar="NYSE",
            timezone="UTC",
            data_frequency="1m",
            timestamp_semantics="bar_close",
        )

        assert result.to_list() == [timestamps[1]]

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


class TestCausalScheduleEvaluation:
    def test_bar_session_explicit_and_fixed_cadences_use_current_state(self) -> None:
        timestamp = datetime(2024, 1, 5)

        assert is_rebalance_timestamp(timestamp, RebalanceSchedule.every_bar(), session_index=1)
        assert is_rebalance_timestamp(
            timestamp,
            RebalanceSchedule.every_session(),
            session_index=1,
            data_frequency="daily",
        )
        assert is_rebalance_timestamp(
            timestamp,
            RebalanceSchedule.explicit_timestamps([timestamp]),
            session_index=1,
        )
        assert not is_rebalance_timestamp(
            timestamp,
            RebalanceSchedule.explicit_timestamps([datetime(2024, 1, 8)]),
            session_index=1,
        )
        assert is_rebalance_timestamp(
            timestamp,
            RebalanceSchedule.fixed_n_sessions(3),
            session_index=1,
            data_frequency="daily",
        )
        assert not is_rebalance_timestamp(
            timestamp,
            RebalanceSchedule.fixed_n_sessions(3),
            session_index=2,
            data_frequency="daily",
        )
        assert is_rebalance_timestamp(
            timestamp,
            RebalanceSchedule.fixed_n_sessions(3),
            session_index=4,
            data_frequency="daily",
        )

    def test_weekly_calendar_metadata_handles_holiday_friday(self) -> None:
        assert is_rebalance_timestamp(
            datetime(2024, 3, 28),
            RebalanceSchedule.weekly(),
            session_index=4,
            calendar="NYSE",
            data_frequency="daily",
        )
        assert not is_rebalance_timestamp(
            datetime(2024, 3, 27),
            RebalanceSchedule.weekly(),
            session_index=3,
            calendar="NYSE",
            data_frequency="daily",
        )
        assert is_rebalance_timestamp(
            datetime(2024, 1, 5),
            RebalanceSchedule.weekly(),
            session_index=5,
            data_frequency="daily",
        )
        assert not is_rebalance_timestamp(
            datetime(2024, 1, 4),
            RebalanceSchedule.weekly(),
            session_index=4,
            data_frequency="daily",
        )

    def test_month_end_without_calendar_uses_last_weekday(self) -> None:
        assert is_rebalance_timestamp(
            datetime(2024, 3, 29),
            RebalanceSchedule.month_end(),
            session_index=20,
            data_frequency="daily",
        )
        assert not is_rebalance_timestamp(
            datetime(2024, 3, 28),
            RebalanceSchedule.month_end(),
            session_index=19,
            data_frequency="daily",
        )

    def test_intraday_session_cadences_fire_only_at_calendar_close(self) -> None:
        before_close = datetime(2024, 1, 5, 20, 59, tzinfo=UTC)
        at_close = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)

        for schedule in (RebalanceSchedule.every_session(), RebalanceSchedule.weekly()):
            assert not is_rebalance_timestamp(
                before_close,
                schedule,
                session_index=5,
                calendar="NYSE",
                timezone="UTC",
                data_frequency="1m",
            )
            assert is_rebalance_timestamp(
                at_close,
                schedule,
                session_index=5,
                calendar="NYSE",
                timezone="UTC",
                data_frequency="1m",
            )

    def test_intraday_session_cadence_requires_boundary_metadata(self) -> None:
        with pytest.raises(ValueError, match="intraday session schedules require"):
            is_rebalance_timestamp(
                datetime(2024, 1, 5, 15, 0),
                RebalanceSchedule.every_session(),
                session_index=1,
                data_frequency="1m",
            )

    def test_unspecified_metadata_preserves_daily_non_midnight_labels(self) -> None:
        timestamp = datetime(2024, 1, 31, 16, 0)

        with pytest.warns(UserWarning, match="treated as daily session closes") as captured:
            assert is_rebalance_timestamp(
                timestamp,
                RebalanceSchedule.month_end(),
                session_index=22,
                calendar="NYSE",
            )
            assert is_rebalance_timestamp(
                timestamp,
                RebalanceSchedule.every_session(),
                session_index=22,
            )
        assert Path(captured[0].filename).name == "test_schedule.py"

    def test_midnight_is_not_a_daily_label_with_explicit_bar_semantics(self) -> None:
        assert not is_rebalance_timestamp(
            datetime(2024, 1, 31),
            RebalanceSchedule.every_session(),
            session_index=22,
            calendar="NYSE",
            timestamp_semantics="bar_close",
        )

    def test_intraday_midnight_is_not_treated_as_session_close(self) -> None:
        assert not is_rebalance_timestamp(
            datetime(2024, 1, 5, tzinfo=UTC),
            RebalanceSchedule.every_session(),
            session_index=5,
            calendar="CME_Equity",
            timezone="UTC",
            data_frequency="1m",
            timestamp_semantics="event_time",
        )

    def test_event_time_semantics_are_sufficient_to_reject_midnight_as_close(self) -> None:
        assert not is_rebalance_timestamp(
            datetime(2024, 1, 5, tzinfo=UTC),
            RebalanceSchedule.every_session(),
            session_index=5,
            calendar="CME_Equity",
            timezone="UTC",
            timestamp_semantics="event_time",
        )

    def test_batch_and_incremental_fixed_session_schedules_agree(self) -> None:
        timestamps = [
            datetime(2024, 1, 7, 23, 0),
            datetime(2024, 1, 8, 22, 0),
            datetime(2024, 1, 8, 23, 0),
            datetime(2024, 1, 9, 22, 0),
            datetime(2024, 1, 9, 23, 0),
            datetime(2024, 1, 10, 22, 0),
        ]
        schedule = RebalanceSchedule.fixed_n_sessions(2)

        batch = resolve_rebalance_timestamps(
            timestamps,
            schedule,
            calendar="CME_Equity",
            timezone="UTC",
            session_start_time="17:00",
            data_frequency="1m",
            timestamp_semantics="bar_close",
        ).to_list()
        executor = TargetWeightExecutor(
            RebalanceConfig(
                schedule=schedule,
                calendar="CME_Equity",
                timezone="UTC",
                session_start_time="17:00",
                data_frequency="1m",
                timestamp_semantics="bar_close",
            )
        )
        incremental = [
            timestamp for timestamp in timestamps if executor.should_rebalance(timestamp)
        ]

        assert (
            batch
            == incremental
            == [
                datetime(2024, 1, 8, 22, 0),
                datetime(2024, 1, 10, 22, 0),
            ]
        )

    def test_intraday_calendar_closes_are_cached_per_event_date(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0
        get_schedule = schedule_module.get_schedule

        def counted_get_schedule(*args, **kwargs):
            nonlocal calls
            calls += 1
            return get_schedule(*args, **kwargs)

        schedule_module._calendar_closes.cache_clear()
        monkeypatch.setattr(schedule_module, "get_schedule", counted_get_schedule)
        for minute in (58, 59):
            is_rebalance_timestamp(
                datetime(2024, 1, 5, 20, minute, tzinfo=UTC),
                RebalanceSchedule.weekly(),
                session_index=5,
                calendar="NYSE",
                timezone="UTC",
                data_frequency="1m",
            )

        assert calls == 1
