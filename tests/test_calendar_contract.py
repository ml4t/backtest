"""Public Engine regressions for calendar declaration and observation semantics."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.specs import UNSET_CALENDAR, FeedSpec, TimestampSemantics

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import DataFrequency


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
        BacktestConfig(enforce_sessions=True), metadata=FeedSpec(calendar="NYSE")
    )
    assert len(seen) == 2
    assert result.config.resolved_calendar == "NYSE"


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


def test_result_explicit_none_returns_observations_without_daily_aggregation():
    times = [datetime(2025, 7, 11, hour, tzinfo=UTC) for hour in (10, 11, 12)]
    result, _ = run_engine(BacktestConfig(calendar="NYSE"), times=times)
    assert result.to_daily_returns(calendar=None).to_list() == result.to_returns_series().to_list()


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
