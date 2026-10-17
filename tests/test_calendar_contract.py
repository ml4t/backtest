"""Public Engine regressions for calendar declaration and observation semantics."""

import inspect
import json
from dataclasses import replace
from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.specs import UNSET_CALENDAR, FeedSpec, TimestampSemantics

from ml4t.backtest import (
    BacktestConfig,
    DataFeed,
    Engine,
    RebalanceConfig,
    RebalanceSchedule,
    Strategy,
    TargetWeightExecutor,
)
from ml4t.backtest.analytics import EquityCurve
from ml4t.backtest.config import DataFrequency
from ml4t.backtest.execution.schedule import is_rebalance_timestamp, resolve_rebalance_timestamps


class RecordBars(Strategy):
    def __init__(self):
        self.seen = []

    def on_data(self, timestamp, data, context, broker):
        self.seen.append(timestamp)
        if len(self.seen) == 1:
            broker.submit_order("TEST", 1)


def run_engine(config, *, metadata=None, times=None, feed_calendar=UNSET_CALENDAR):
    times = times or [datetime(2025, 7, day, tzinfo=UTC) for day in (11, 12, 14)]
    frame = pl.DataFrame({"timestamp": times, "asset": ["TEST"] * 3, "close": [10.0, 11.0, 12.0]})
    kwargs = {} if feed_calendar is UNSET_CALENDAR else {"calendar": feed_calendar}
    feed = DataFeed(prices_df=frame, feed_spec=metadata, **kwargs)
    strategy = RecordBars()
    result = Engine(feed, strategy, config).run()
    return result, strategy.seen


def test_explicit_none_keeps_weekend_bar_despite_declared_feed_calendar():
    result, seen = run_engine(
        BacktestConfig(calendar=None, enforce_sessions=True), metadata=FeedSpec(calendar="NYSE")
    )
    assert len(seen) == 3
    assert result.config.resolved_calendar is None


def test_omission_uses_nyse_when_feed_has_no_declaration():
    result, seen = run_engine(BacktestConfig(enforce_sessions=True))
    assert len(seen) == 2
    assert result.config.resolved_calendar == "NYSE"


def test_omission_inherits_feed_calendar_positive_control():
    result, seen = run_engine(
        BacktestConfig(enforce_sessions=True), metadata=FeedSpec(calendar="crypto")
    )
    assert len(seen) == 3
    assert result.config.resolved_calendar == "crypto"


def test_omission_respects_v2_declared_disable_instead_of_nyse():
    result, seen = run_engine(
        BacktestConfig(enforce_sessions=True),
        metadata=FeedSpec(time_contract_version=2, calendar=None),
    )
    assert len(seen) == 3
    assert result.config.resolved_calendar is None


def test_direct_feed_disable_survives_engine_metadata_merge():
    result, seen = run_engine(
        BacktestConfig(enforce_sessions=True),
        metadata=FeedSpec(calendar="NYSE"),
        feed_calendar=None,
    )
    assert len(seen) == 3
    assert result.config.resolved_calendar is None


@pytest.mark.parametrize("selection", [UNSET_CALENDAR, None, "NYSE"])
def test_config_calendar_declaration_survives_json_yaml_and_copy(tmp_path, selection):
    config = BacktestConfig(calendar=selection, feed_spec=FeedSpec(calendar="CME_Equity"))
    payload = config.to_dict()
    if selection is UNSET_CALENDAR:
        assert "calendar" not in payload["calendar"]
    else:
        assert payload["calendar"]["calendar"] == selection
    expected = (
        None if selection is None else "CME_Equity" if selection is UNSET_CALENDAR else selection
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    for restored in [
        replace(config),
        BacktestConfig.from_dict(json.loads(json.dumps(payload))),
        BacktestConfig.from_yaml(path),
    ]:
        assert restored.resolved_calendar == expected
        assert restored.time_contract_version == 2


def test_legacy_serialized_null_remains_unspecified_not_v2_disable():
    config = BacktestConfig.from_dict(
        {"calendar": {"calendar": None}, "feed": {"calendar": "CME_Equity"}}
    )
    assert config.resolved_calendar == "CME_Equity"
    assert config.time_contract_version == 1


def test_calendar_free_hourly_engine_does_not_assume_nyse_session_length():
    result, _ = run_engine(BacktestConfig(calendar=None, data_frequency=DataFrequency.HOURLY))
    assert result.equity.periods_per_year == 252.0
    assert result.equity.years == pytest.approx(2 / 252)


def test_row_mode_metrics_do_not_depend_on_timestamp_spacing():
    config = BacktestConfig(calendar=None, data_frequency=DataFrequency.HOURLY)
    regular = [datetime(2025, 7, 11, hour, tzinfo=UTC) for hour in (10, 11, 12)]
    irregular = [datetime(2025, 7, day, tzinfo=UTC) for day in (11, 12, 20)]
    a, _ = run_engine(config, times=regular)
    b, _ = run_engine(config, times=irregular)
    assert a.equity.values == b.equity.values
    assert a.equity.cagr == b.equity.cagr
    assert a.equity.periods_per_year == b.equity.periods_per_year == 252.0


def test_explicit_annualization_scale_propagates_through_serialized_engine_config():
    config = BacktestConfig(calendar=None, periods_per_year=365.0)
    result, _ = run_engine(BacktestConfig.from_dict(config.to_dict()))
    assert result.equity.periods_per_year == 365.0
    assert result.equity.years == pytest.approx(2 / 365)


@pytest.mark.parametrize("factor", [0, -1, float("inf"), float("nan"), True])
def test_invalid_annualization_scale_fails_before_engine_mutation(factor):
    with pytest.raises(ValueError, match="periods_per_year"):
        BacktestConfig(calendar=None, periods_per_year=factor)


def test_daily_event_time_keeps_valid_cme_overnight_observation():
    times = [
        datetime(2025, 7, 6, 23, 30, tzinfo=UTC),
        datetime(2025, 7, 7, 0, 30, tzinfo=UTC),
        datetime(2025, 7, 7, 1, 30, tzinfo=UTC),
    ]
    result, seen = run_engine(
        BacktestConfig(calendar="CME_Equity", enforce_sessions=True),
        metadata=FeedSpec(timestamp_semantics=TimestampSemantics.EVENT_TIME),
        times=times,
    )
    assert len(seen) == 3
    assert len(result.equity_curve) == 3


@pytest.mark.parametrize("contract_version", [1, 2])
def test_result_explicit_none_returns_observations_without_daily_aggregation(contract_version):
    times = [datetime(2025, 7, 11, hour, tzinfo=UTC) for hour in (10, 11, 12)]
    declaration = {"calendar": "NYSE"}
    if contract_version == 2:
        declaration["time_contract_version"] = 2
    config = BacktestConfig.from_dict({"calendar": declaration})
    result, _ = run_engine(config, times=times)
    assert len(result.to_daily_returns()) == 1
    assert result.to_daily_returns(calendar=None).to_list() == result.to_returns_series().to_list()
    with pytest.raises(ValueError, match="requires a calendar"):
        result.to_daily_returns(calendar=None, session_aligned=True)


def test_named_result_calendar_override_controls_actual_overnight_aggregation():
    times = [
        datetime(2025, 7, 6, 23, 30, tzinfo=UTC),
        datetime(2025, 7, 7, 0, 30, tzinfo=UTC),
        datetime(2025, 7, 7, 1, 30, tzinfo=UTC),
    ]
    result, _ = run_engine(
        BacktestConfig(calendar="NYSE"),
        metadata=FeedSpec(timestamp_semantics=TimestampSemantics.EVENT_TIME),
        times=times,
    )
    assert len(result.to_daily_returns(calendar="CME_Equity")) == 1
    assert len(result.to_daily_returns(calendar="NYSE")) == 2
    assert len(result.to_daily_returns(calendar=None)) == 3


def test_schedule_explicit_none_does_not_inherit_calendar_metadata():
    times = [datetime(2025, 7, day, tzinfo=UTC) for day in (7, 8, 9, 10, 11)]
    with pytest.raises(ValueError, match="require calendar"):
        resolve_rebalance_timestamps(
            times,
            RebalanceSchedule.weekly(),
            calendar=None,
            feed_spec=FeedSpec(time_contract_version=2, calendar="NYSE"),
        )


def test_schedule_omission_inherits_distinct_continuous_metadata():
    times = [datetime(2025, 7, day, tzinfo=UTC) for day in (11, 12, 14)]
    resolved = resolve_rebalance_timestamps(
        times,
        RebalanceSchedule.every_session(),
        feed_spec=FeedSpec(time_contract_version=2, calendar="crypto", data_frequency="daily"),
    )
    assert resolved.to_list() == times


def test_schedule_omission_without_metadata_uses_nyse():
    times = [datetime(2025, 7, day, tzinfo=UTC) for day in (11, 12, 14)]
    resolved = resolve_rebalance_timestamps(
        times, RebalanceSchedule.every_session(), data_frequency="daily"
    )
    assert resolved.to_list() == [times[0], times[2]]


def test_executor_default_uses_nyse_but_explicit_none_requires_weekly_calendar():
    executor = TargetWeightExecutor(
        RebalanceConfig(schedule=RebalanceSchedule.weekly(), data_frequency="daily")
    )
    assert executor.should_rebalance(datetime(2025, 7, 11, tzinfo=UTC))
    with pytest.raises(ValueError, match="require calendar"):
        TargetWeightExecutor(RebalanceConfig(schedule=RebalanceSchedule.weekly(), calendar=None))


def test_standalone_observation_equity_is_invariant_to_timestamp_spacing():
    curves = [
        EquityCurve(
            values=[100.0, 110.0, 120.0],
            timestamps=[datetime(2025, 7, d, tzinfo=UTC) for d in days],
            observation_based=True,
        )
        for days in ((11, 12, 13), (11, 12, 20))
    ]
    assert [curve.periods_per_year for curve in curves] == [252.0, 252.0]
    assert curves[0].cagr == curves[1].cagr


def test_daily_event_time_executor_waits_for_actual_cme_close():
    executor = TargetWeightExecutor(
        RebalanceConfig(
            schedule=RebalanceSchedule.every_session(),
            calendar="CME_Equity",
            data_frequency="daily",
            timestamp_semantics=TimestampSemantics.EVENT_TIME,
        )
    )
    assert not executor.should_rebalance(datetime(2025, 7, 6, 23, 30, tzinfo=UTC))
    assert not executor.should_rebalance(datetime(2025, 7, 7, 0, 30, tzinfo=UTC))
    assert executor.should_rebalance(datetime(2025, 7, 7, 21, tzinfo=UTC))


def test_direct_schedule_nyse_default_does_not_rebalance_on_saturday():
    timestamp = datetime(2025, 7, 12, tzinfo=UTC)
    assert not is_rebalance_timestamp(
        timestamp, RebalanceSchedule.every_session(), session_index=1, data_frequency="daily"
    )
    assert is_rebalance_timestamp(
        timestamp,
        RebalanceSchedule.every_session(),
        session_index=1,
        calendar=None,
        data_frequency="daily",
    )


def test_positional_timezone_and_frequency_override_feed_metadata_in_engine():
    arguments = []
    overrides = {"calendar": None, "timezone": "UTC", "data_frequency": DataFrequency.HOURLY}
    for name, parameter in inspect.signature(BacktestConfig.__init__).parameters.items():
        if name == "self" or parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            continue
        arguments.append(overrides.get(name, parameter.default))
        if name == "data_frequency":
            break
    config = BacktestConfig(
        *arguments,
        feed_spec=FeedSpec(timezone="America/New_York", data_frequency="daily"),
    )
    result, _ = run_engine(config)
    assert result.config.resolved_timezone == "UTC"
    assert result.config.resolved_data_frequency == DataFrequency.HOURLY


@pytest.mark.parametrize("metadata_version", [1, 2])
def test_undeclared_config_feed_calendar_does_not_mask_input_time_metadata(metadata_version):
    declared = FeedSpec(time_contract_version=metadata_version)
    config = BacktestConfig(feed_spec=declared, enforce_sessions=True)
    times = [
        datetime(2025, 7, 6, 23, 30, tzinfo=UTC),
        datetime(2025, 7, 7, 0, 30, tzinfo=UTC),
        datetime(2025, 7, 7, 1, 30, tzinfo=UTC),
    ]
    result, seen = run_engine(
        config,
        metadata=FeedSpec(
            time_contract_version=2,
            calendar="CME_Equity",
            timezone="America/Chicago",
            data_frequency="1h",
            timestamp_semantics=TimestampSemantics.EVENT_TIME,
        ),
        times=times,
    )
    assert result.config.resolved_calendar == "CME_Equity"
    assert result.config.resolved_timezone == "America/Chicago"
    assert result.config.resolved_data_frequency == DataFrequency.HOURLY
    assert result.config.resolved_timestamp_semantics == TimestampSemantics.EVENT_TIME
    assert len(seen) == 3
    assert config.feed_spec is declared
    assert declared.calendar is (None if metadata_version == 1 else UNSET_CALENDAR)


@pytest.mark.parametrize("selection,expected,count", [(None, None, 3), ("NYSE", "NYSE", 2)])
def test_declared_config_feed_calendar_still_overrides_input_metadata(selection, expected, count):
    result, seen = run_engine(
        BacktestConfig(
            feed_spec=FeedSpec(time_contract_version=2, calendar=selection), enforce_sessions=True
        ),
        metadata=FeedSpec(time_contract_version=2, calendar="crypto"),
    )
    assert result.config.resolved_calendar == expected
    assert len(seen) == count


def test_declared_input_disable_survives_legacy_undeclared_config_feed_metadata():
    result, seen = run_engine(
        BacktestConfig(feed_spec=FeedSpec(), enforce_sessions=True),
        metadata=FeedSpec(time_contract_version=2, calendar=None),
    )
    assert result.config.resolved_calendar is None
    assert len(seen) == 3
    restored = BacktestConfig.from_dict(json.loads(json.dumps(result.config.to_dict())))
    assert restored.resolved_calendar is None


@pytest.mark.parametrize("frequency", [DataFrequency.DAILY, DataFrequency.HOURLY])
@pytest.mark.parametrize("semantics", [TimestampSemantics.EVENT_TIME, TimestampSemantics.BAR_CLOSE])
def test_explicit_instants_reject_preopen_and_afterclose_at_each_frequency(frequency, semantics):
    times = [datetime(2025, 7, 11, hour, tzinfo=UTC) for hour in (12, 15, 22)]
    _, seen = run_engine(
        BacktestConfig(calendar="NYSE", enforce_sessions=True, data_frequency=frequency),
        metadata=FeedSpec(timestamp_semantics=semantics),
        times=times,
    )
    assert seen == [times[1]]


@pytest.mark.parametrize("frequency", [DataFrequency.DAILY, DataFrequency.HOURLY])
def test_explicit_session_labels_use_dates_instead_of_intraday_hours(frequency):
    result, seen = run_engine(
        BacktestConfig(calendar="NYSE", enforce_sessions=True, data_frequency=frequency),
        metadata=FeedSpec(timestamp_semantics=TimestampSemantics.SESSION_LABEL),
    )
    assert [timestamp.day for timestamp in seen] == [11, 14]
    assert result.metrics["skipped_bars"] == 1


@pytest.mark.parametrize("frequency", [DataFrequency.DAILY, DataFrequency.HOURLY])
def test_offsession_funding_is_rejected_before_strategy_starts_for_instants(frequency):
    class RecordStart(RecordBars):
        started = False

        def on_start(self, broker):
            self.started = True

    times = [datetime(2025, 7, 11, hour, tzinfo=UTC) for hour in (12, 15, 22)]
    prices = pl.DataFrame({"timestamp": times, "asset": ["TEST"] * 3, "close": [10.0, 11.0, 12.0]})
    strategy = RecordStart()
    engine = Engine(
        DataFeed(
            prices_df=prices,
            feed_spec=FeedSpec(timestamp_semantics=TimestampSemantics.EVENT_TIME),
        ),
        strategy,
        BacktestConfig(calendar="NYSE", enforce_sessions=True, data_frequency=frequency),
        funding_df=pl.DataFrame({"timestamp": [times[0]], "asset": ["TEST"], "rate": [0.01]}),
    )
    with pytest.raises(ValueError, match="funding event timestamp is excluded"):
        engine.run()
    assert strategy.started is False
    assert strategy.seen == []
    assert engine.broker.fills == []
