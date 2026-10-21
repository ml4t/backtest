"""Declared right-end bars must retain real close instants, including interval boundaries."""

from datetime import UTC, date, datetime

import polars as pl
from ml4t.specs import BarLabel, FeedSpec
from ml4t.specs.market_data import TimestampSemantics

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import DataFrequency


class Cash(Strategy):
    def on_data(self, timestamp, data, context, broker):
        pass


def run_closing_bars(calendar, timestamps):
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset": ["TEST"] * len(timestamps),
            "close": [100.0] * len(timestamps),
        }
    )
    spec = FeedSpec(
        calendar=calendar,
        timezone="UTC",
        data_frequency=DataFrequency.MINUTE_1,
        timestamp_semantics=TimestampSemantics.BAR_CLOSE,
        bar_label=BarLabel.RIGHT,
        time_contract_version=2,
    )
    return Engine(
        DataFeed(prices_df=frame, feed_spec=spec), Cash(), BacktestConfig(enforce_sessions=True)
    ).run()


def test_bar_close_metadata_keeps_exchange_close_and_excludes_open_boundary():
    times = [datetime(2025, 1, 2, h, m, tzinfo=UTC) for h, m in [(14, 30), (14, 31), (21, 0)]]
    result = run_closing_bars("NYSE", times)
    assert result.to_equity_dataframe()["timestamp"].to_list() == times[1:]
    assert result.metrics["skipped_bars"] == 1


def test_lunch_break_end_bars_include_morning_close_and_exclude_afternoon_open():
    times = [datetime(2025, 1, 2, h, m, tzinfo=UTC) for h, m in [(3, 59), (4, 0), (5, 0), (5, 1)]]
    result = run_closing_bars("XHKG", times)
    assert result.to_equity_dataframe()["timestamp"].to_list() == [times[0], times[1], times[3]]


def test_midnight_bar_close_keeps_its_instant_but_belongs_to_the_previous_crypto_session():
    times = [datetime(2025, 1, 2, 0, m, tzinfo=UTC) for m in [0, 1]]
    result = run_closing_bars("crypto", times)
    assert result.to_equity_dataframe()["timestamp"].to_list() == times
    assert result.to_daily_pnl(session_aligned=True)["session_date"].dt.date().to_list() == [
        date(2025, 1, 1),
        date(2025, 1, 2),
    ]
