"""Tests for rebalance schedule resolution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from ml4t.specs.market_data import FeedSpec

from ml4t.backtest import calendar as calendar_module
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

    def test_explicit_constructor_normalizes_and_deduplicates_timestamps(self) -> None:
        first = datetime(2024, 1, 1)
        second = datetime(2024, 1, 2)

        schedule = RebalanceSchedule(
            cadence=RebalanceCadence.EXPLICIT_TIMESTAMPS,
            timestamps=(second, first, second),
        )

        assert schedule.timestamps == (first, second)
        assert is_rebalance_timestamp(first, schedule, session_index=1)

    def test_explicit_timestamp_matches_the_same_instant_across_timezones(self) -> None:
        observed = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
        schedule = RebalanceSchedule.explicit_timestamps([datetime(2024, 1, 2, 16, 0)])

        result = resolve_rebalance_timestamps(
            [observed],
            schedule,
            timezone="America/New_York",
        )

        assert result.to_list() == [observed]
        assert is_rebalance_timestamp(
            observed,
            schedule,
            session_index=1,
            timezone="America/New_York",
        )

    def test_one_schedule_caches_each_naive_timestamp_timezone_independently(self) -> None:
        schedule = RebalanceSchedule.explicit_timestamps([datetime(2024, 1, 2, 16, 0)])
        event = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)

        assert is_rebalance_timestamp(
            event,
            schedule,
            session_index=1,
            timezone="America/New_York",
        )
        assert not is_rebalance_timestamp(
            event,
            schedule,
            session_index=1,
            timezone="UTC",
        )
        assert is_rebalance_timestamp(
            datetime(2024, 1, 2, 16, 0, tzinfo=UTC),
            schedule,
            session_index=1,
            timezone="UTC",
        )
        assert set(schedule._instant_sets_by_timezone) == {"America/New_York", "UTC"}

    def test_explicit_timestamp_rejects_an_unmatched_event_on_an_observed_date(self) -> None:
        schedule = RebalanceSchedule.explicit_timestamps([datetime(2024, 1, 2, 16, 0)])

        with pytest.raises(
            ValueError,
            match=r"2024-01-02T21:00:00\+00:00.*nearest observed instant is "
            r"2024-01-02T20:59:00\+00:00",
        ):
            resolve_rebalance_timestamps(
                [
                    datetime(2024, 1, 2, 20, 59, tzinfo=UTC),
                    datetime(2024, 1, 2, 21, 1, tzinfo=UTC),
                ],
                schedule,
                timezone="America/New_York",
            )

    def test_explicit_timestamp_outside_the_observed_date_window_is_ignored(self) -> None:
        observed = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
        schedule = RebalanceSchedule.explicit_timestamps([datetime(2024, 1, 3, 16, 0)])

        result = resolve_rebalance_timestamps(
            [observed],
            schedule,
            timezone="America/New_York",
        )

        assert result.to_list() == []

    def test_explicit_timestamp_outside_a_partial_day_feed_slice_is_ignored(self) -> None:
        schedule = RebalanceSchedule.explicit_timestamps([datetime(2024, 1, 2, 9, 30)])

        result = resolve_rebalance_timestamps(
            [
                datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
                datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
            ],
            schedule,
            timezone="America/New_York",
        )

        assert result.to_list() == []

    def test_explicit_timestamp_reports_a_later_miss_after_an_earlier_match(self) -> None:
        schedule = RebalanceSchedule.explicit_timestamps(
            [datetime(2024, 1, 2, 16, 0), datetime(2024, 1, 2, 17, 0)]
        )

        with pytest.raises(
            ValueError,
            match=r"2024-01-02T22:00:00\+00:00.*nearest observed instant is "
            r"2024-01-02T21:59:00\+00:00",
        ):
            resolve_rebalance_timestamps(
                [
                    datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
                    datetime(2024, 1, 2, 21, 59, tzinfo=UTC),
                    datetime(2024, 1, 2, 22, 1, tzinfo=UTC),
                ],
                schedule,
                timezone="America/New_York",
            )

    def test_stateless_explicit_matching_caches_normalized_schedule_instants(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        start = datetime(2024, 1, 2, 16, 0)
        schedule = RebalanceSchedule.explicit_timestamps(
            [start + timedelta(hours=index) for index in range(2_500)]
        )
        calls = 0
        event_time_utc = schedule_module._event_time_utc

        def count_conversion(timestamp: datetime, timezone: str | None) -> datetime:
            nonlocal calls
            calls += 1
            return event_time_utc(timestamp, timezone)

        monkeypatch.setattr(schedule_module, "_event_time_utc", count_conversion)

        assert is_rebalance_timestamp(
            datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
            schedule,
            session_index=1,
            timezone="America/New_York",
        )
        first_call_count = calls
        assert not is_rebalance_timestamp(
            datetime(2024, 1, 2, 21, 30, tzinfo=UTC),
            schedule,
            session_index=1,
            timezone="America/New_York",
        )

        assert first_call_count == len(schedule.timestamps) + 1
        assert calls == first_call_count + 1

    def test_fixed_n_sessions_thins_session_closes(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-10")

        result = resolve_rebalance_timestamps(timestamps, RebalanceSchedule.fixed_n_sessions(2))

        expected = [datetime.combine(ts, datetime.min.time()) for ts in timestamps.to_list()[::2]]
        assert result.to_list() == expected

    def test_weekly_uses_scheduled_week_end(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-31")

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceSchedule.weekly(),
            calendar="NYSE",
            data_frequency="daily",
        )

        assert result.to_list() == [
            datetime(2024, 1, 5),
            datetime(2024, 1, 12),
            datetime(2024, 1, 19),
            datetime(2024, 1, 26),
        ]

    def test_month_end_uses_scheduled_month_end(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-03-31")

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceSchedule.month_end(),
            calendar="NYSE",
            data_frequency="daily",
        )

        resolved = result.to_list()
        assert len(resolved) == 3
        assert resolved[0].month == 1 and resolved[0].day == 31
        assert resolved[1].month == 2 and resolved[1].day == 29
        assert resolved[2].month == 3 and resolved[2].day == 28

    def test_incomplete_month_does_not_move_rebalance_to_last_available_bar(self) -> None:
        timestamps = _make_weekday_series("2024-01-01", "2024-01-30")

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceSchedule.month_end(),
            calendar="NYSE",
            data_frequency="daily",
        )

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
            data_frequency="1h",
            timestamp_semantics="bar_close",
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

    def test_intraday_batch_schedule_rejects_total_close_alignment_miss(self) -> None:
        timestamps = [datetime(2024, 1, 8, 16, 1), datetime(2024, 1, 8, 16, 2)]

        with pytest.raises(ValueError, match="resolved no session closes"):
            resolve_rebalance_timestamps(
                timestamps,
                RebalanceCadence.EVERY_SESSION,
                calendar="NYSE",
                timezone="America/New_York",
                data_frequency="1m",
                timestamp_semantics="event_time",
            )

    def test_intraday_batch_schedule_rejects_one_missing_required_close(self) -> None:
        timestamps = [
            datetime(2024, 1, 2, 16, 0),
            datetime(2024, 1, 3, 16, 15),
            datetime(2024, 1, 4, 16, 0),
        ]

        with pytest.raises(
            ValueError,
            match=r"required session 2024-01-03.*expected market_close",
        ):
            resolve_rebalance_timestamps(
                timestamps,
                RebalanceCadence.EVERY_SESSION,
                calendar="NYSE",
                timezone="America/New_York",
                data_frequency="15m",
                timestamp_semantics="bar_close",
            )

    def test_intraday_batch_rejects_truncated_interior_session(self) -> None:
        timestamps = [
            datetime(2024, 1, 2, 16, 0),
            datetime(2024, 1, 3, 15, 45),
            datetime(2024, 1, 4, 16, 0),
        ]

        with pytest.raises(ValueError, match="required session 2024-01-03"):
            resolve_rebalance_timestamps(
                timestamps,
                RebalanceSchedule.every_session(),
                calendar="NYSE",
                timezone="America/New_York",
                data_frequency="15m",
                timestamp_semantics="bar_close",
            )

    @pytest.mark.parametrize(
        "schedule,timestamp",
        [
            (RebalanceSchedule.weekly(), datetime(2024, 1, 5, 16, 15)),
            (RebalanceSchedule.month_end(), datetime(2024, 1, 31, 16, 15)),
        ],
    )
    def test_intraday_period_schedule_rejects_required_close_alignment_miss(
        self,
        schedule: RebalanceSchedule,
        timestamp: datetime,
    ) -> None:
        with pytest.raises(ValueError, match="resolved no session closes"):
            resolve_rebalance_timestamps(
                [timestamp],
                schedule,
                calendar="NYSE",
                timezone="America/New_York",
                data_frequency="15m",
                timestamp_semantics="bar_close",
            )

    def test_present_week_end_with_wrong_close_reports_alignment_error(self) -> None:
        timestamps = [
            datetime(2024, 1, 5, 16, 15),
            datetime(2024, 1, 8, 16, 0),
        ]

        with pytest.raises(
            ValueError,
            match=r"required session 2024-01-05.*expected market_close",
        ):
            resolve_rebalance_timestamps(
                timestamps,
                RebalanceSchedule.weekly(),
                calendar="NYSE",
                timezone="America/New_York",
                data_frequency="15m",
                timestamp_semantics="bar_close",
            )

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

    def test_incomplete_trailing_intraday_session_is_not_an_alignment_error(self) -> None:
        timestamps = [datetime(2024, 1, 2, 16, 0), datetime(2024, 1, 3, 12, 0)]

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceSchedule.every_session(),
            calendar="NYSE",
            timezone="America/New_York",
            data_frequency="15m",
            timestamp_semantics="bar_close",
        )

        assert result.to_list() == [datetime(2024, 1, 2, 16, 0)]

    def test_intraday_exchange_holiday_is_ignored(self) -> None:
        timestamps = [
            datetime(2024, 1, 12, 16, 0),
            datetime(2024, 1, 15, 16, 0),
            datetime(2024, 1, 16, 16, 0),
        ]

        result = resolve_rebalance_timestamps(
            timestamps,
            RebalanceSchedule.every_session(),
            calendar="NYSE",
            timezone="America/New_York",
            data_frequency="1m",
            timestamp_semantics="bar_close",
        )

        assert result.to_list() == [timestamps[0], timestamps[2]]

    @pytest.mark.parametrize("data_frequency", ["daily", "1m"])
    def test_calendar_with_no_matching_sessions_fails(self, data_frequency: str) -> None:
        with pytest.raises(ValueError, match="observed no exchange sessions"):
            resolve_rebalance_timestamps(
                [datetime(2024, 1, 15, 16, 0)],
                RebalanceSchedule.every_session(),
                calendar="NYSE",
                timezone="America/New_York",
                data_frequency=data_frequency,
                timestamp_semantics="bar_close",
            )

    def test_every_bar_and_explicit_schedules_include_holiday_timestamps(self) -> None:
        holiday = datetime(2024, 1, 15, 16, 0)

        assert resolve_rebalance_timestamps(
            [holiday],
            RebalanceSchedule.every_bar(),
            calendar="NYSE",
            data_frequency="1m",
        ).to_list() == [holiday]
        assert resolve_rebalance_timestamps(
            [holiday],
            RebalanceSchedule.explicit_timestamps([holiday]),
            calendar="NYSE",
            data_frequency="1m",
        ).to_list() == [holiday]


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

    def test_period_schedules_require_calendar_metadata(self) -> None:
        for schedule in (RebalanceSchedule.weekly(), RebalanceSchedule.month_end()):
            with pytest.raises(ValueError, match="require calendar metadata"):
                is_rebalance_timestamp(
                    datetime(2024, 3, 29),
                    schedule,
                    session_index=20,
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

    def test_batch_and_online_agree_for_unspecified_non_midnight_daily_labels(self) -> None:
        timestamp = datetime(2024, 1, 31, 16, 0)
        schedule = RebalanceSchedule.every_session()
        executor = TargetWeightExecutor(RebalanceConfig(schedule=schedule, calendar="NYSE"))

        with pytest.warns(UserWarning, match="treated as daily session closes"):
            online = executor.should_rebalance(timestamp)
            batch = resolve_rebalance_timestamps([timestamp], schedule, calendar="NYSE")

        assert online
        assert batch.to_list() == [timestamp]

    def test_missing_metadata_rejects_multiple_events_on_one_date(self) -> None:
        timestamps = [datetime(2024, 1, 31, 10, 0), datetime(2024, 1, 31, 16, 0)]
        schedule = RebalanceSchedule.every_session()
        executor = TargetWeightExecutor(RebalanceConfig(schedule=schedule, calendar="NYSE"))

        with pytest.raises(ValueError, match="multiple schedule events.*intraday data"):
            resolve_rebalance_timestamps(timestamps, schedule, calendar="NYSE")

        with pytest.warns(UserWarning, match="treated as daily session closes"):
            assert executor.should_rebalance(timestamps[0])
        with pytest.raises(ValueError, match="multiple schedule events.*intraday data"):
            executor.should_rebalance(timestamps[1])

    def test_missing_metadata_uses_aware_timestamp_dates_for_ambiguity(self) -> None:
        new_york = ZoneInfo("America/New_York")
        same_date = [
            datetime(2024, 1, 2, 9, 0, tzinfo=new_york),
            datetime(2024, 1, 2, 20, 0, tzinfo=new_york),
        ]
        different_dates = [
            datetime(2024, 1, 2, 23, 0, tzinfo=new_york),
            datetime(2024, 1, 3, 2, 0, tzinfo=new_york),
        ]
        schedule = RebalanceSchedule.every_session()

        with pytest.raises(ValueError, match="multiple schedule events observed on 2024-01-02"):
            resolve_rebalance_timestamps(same_date, schedule, timezone="UTC")

        with pytest.warns(UserWarning, match="treated as daily session closes"):
            resolved = resolve_rebalance_timestamps(different_dates, schedule, timezone="UTC")
        assert resolved.to_list() == different_dates

        executor = TargetWeightExecutor(RebalanceConfig(schedule=schedule, timezone="UTC"))
        with pytest.warns(UserWarning, match="treated as daily session closes"):
            assert executor.should_rebalance(same_date[0])
        with pytest.raises(ValueError, match="multiple schedule events observed on 2024-01-02"):
            executor.should_rebalance(same_date[1])

    def test_explicit_boundary_signal_needs_no_intraday_metadata(self) -> None:
        executor = TargetWeightExecutor(RebalanceConfig(schedule=RebalanceSchedule.every_session()))

        assert not executor.should_rebalance(
            datetime(2024, 1, 5, 15, 59),
            is_session_close=False,
        )
        assert executor.should_rebalance(
            datetime(2024, 1, 5, 16, 0),
            is_session_close=True,
        )

    def test_same_instant_rejects_conflicting_explicit_boundary_signals(self) -> None:
        executor = TargetWeightExecutor(RebalanceConfig(schedule=RebalanceSchedule.every_session()))
        timestamp = datetime(2024, 1, 5, 16, 0)

        assert not executor.should_rebalance(timestamp, is_session_close=False)
        with pytest.raises(ValueError, match="conflicting is_session_close"):
            executor.should_rebalance(timestamp, is_session_close=True)

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

    def test_intraday_calendar_closes_are_cached_per_calendar_year(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0
        get_schedule = calendar_module.get_schedule

        def counted_get_schedule(*args, **kwargs):
            nonlocal calls
            calls += 1
            return get_schedule(*args, **kwargs)

        schedule_module._calendar_closes.cache_clear()
        calendar_module.get_calendar_sessions.cache_clear()
        monkeypatch.setattr(calendar_module, "get_schedule", counted_get_schedule)
        for timestamp in (
            datetime(2024, 1, 5, 20, 58, tzinfo=UTC),
            datetime(2024, 1, 8, 20, 59, tzinfo=UTC),
        ):
            is_rebalance_timestamp(
                timestamp,
                RebalanceSchedule.weekly(),
                session_index=5,
                calendar="NYSE",
                timezone="UTC",
                data_frequency="1m",
            )

        assert calls == 1

    def test_calendar_period_end_is_cached_within_period(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0
        get_schedule = calendar_module.get_schedule

        def counted_get_schedule(*args, **kwargs):
            nonlocal calls
            calls += 1
            return get_schedule(*args, **kwargs)

        schedule_module._calendar_period_end.cache_clear()
        calendar_module.get_calendar_sessions.cache_clear()
        monkeypatch.setattr(calendar_module, "get_schedule", counted_get_schedule)
        for day in range(25, 29):
            is_rebalance_timestamp(
                datetime(2024, 3, day),
                RebalanceSchedule.weekly(),
                session_index=day - 24,
                calendar="NYSE",
                data_frequency="daily",
            )

        assert calls == 1
