"""Session controls require the actual close clock and retain real mark provenance."""

from datetime import UTC, datetime
from io import StringIO

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.controls import AccountConstraints, AccountControlViolation


def _feed(*, close_row=True, calendar="NYSE"):
    csv = "timestamp,asset,open,high,low,close,volume\n"
    csv += "2025-07-02T13:31:00Z,A,90,101,89,100,1000\n"
    csv += "2025-07-02T13:32:00Z,A,90,101,89,100,1000\n"
    if close_row:
        csv += "2025-07-02T20:00:00Z,A,,,,,0\n"
    return DataFeed(
        prices_df=pl.read_csv(StringIO(csv), try_parse_dates=True),
        feed_spec={
            "calendar": calendar,
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "data_frequency": "1m",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )


def _engine(strategy, **feed_args):
    return Engine(
        _feed(**feed_args),
        strategy,
        BacktestConfig(
            initial_cash=1000,
            allow_leverage=True,
            skip_cash_validation=True,
            execution_price="close",
            next_bar_execution_price="close",
        ),
        holdings_sampling="bar",
    )


@pytest.mark.parametrize("maximum,breach", [(3, False), (1, True)])
def test_session_control_uses_real_close_clock_without_inventing_a_quote(maximum, breach):
    observed = []

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_account_control(
                AccountConstraints(max_leverage=maximum), boundary="session_end"
            )

        def on_data(self, timestamp, data, context, broker):
            observed.append(timestamp)
            if len(observed) == 1:
                broker.submit_order("A", 20)

    engine = _engine(Submit())
    if breach:
        with pytest.raises(AccountControlViolation, match="maximum"):
            engine.run()
        assert observed[-1] == datetime(2025, 7, 2, 20, tzinfo=UTC)
    else:
        result = engine.run()
        assert result.metrics["final_value"] == 1000
        last = result.to_holdings_dataframe().row(-1, named=True)
        assert last["quantity"] == 20 and last["mark"] == 100
        assert last["mark_timestamp"] == datetime(2025, 7, 2, 13, 32, tzinfo=UTC)
        assert result.to_fills_dataframe().height == 1
        assert engine.broker.cash == -1000
    assert len(observed) == 3


def test_session_control_rejects_missing_close_instead_of_silently_skipping_validation():
    class Observe(Strategy):
        def on_data(self, *args):
            pass

        def on_start(self, broker):
            broker.register_account_control(
                AccountConstraints(max_leverage=3), boundary="session_end"
            )

    with pytest.raises(ValueError, match="close observation"):
        _engine(Observe(), close_row=False).run()


def test_session_control_requires_calendar_and_rejects_late_registration():
    class Register(Strategy):
        def on_data(self, *args):
            pass

        def on_start(self, broker):
            broker.register_account_control(
                AccountConstraints(max_leverage=3), boundary="session_end"
            )

    with pytest.raises(ValueError, match="calendar"):
        _engine(Register(), calendar=None).run()

    class Late(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.register_account_control(
                AccountConstraints(max_leverage=3), boundary="session_end"
            )

    with pytest.raises(ValueError, match="before.*market"):
        _engine(Late()).run()


def test_session_control_rejects_session_labels_instead_of_treating_midnight_as_close():
    frame = pl.DataFrame({"timestamp": [datetime(2025, 7, 2)], "asset": ["A"], "close": [100.0]})
    feed = DataFeed(
        prices_df=frame,
        feed_spec={
            "calendar": "NYSE",
            "timestamp_semantics": "session_label",
            "data_frequency": "1d",
        },
    )

    class Observe(Strategy):
        def on_start(self, broker):
            broker.register_account_control(
                AccountConstraints(max_leverage=1), boundary="session_end"
            )

        def on_data(self, *args):
            pass

    with pytest.raises(ValueError, match="physical instants"):
        Engine(feed, Observe()).run()


def test_later_valid_close_does_not_hide_an_earlier_sessions_missing_close():
    frame = _feed(close_row=False).prices
    # Append a real early close on the next session, leaving the first incomplete.
    extra = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 7, 3, 17, tzinfo=UTC)],
            "asset": ["A"],
            "open": [90],
            "high": [101],
            "low": [89],
            "close": [100],
            "volume": [1000],
        }
    )
    feed = DataFeed(
        prices_df=pl.concat([frame, extra], how="vertical_relaxed"), feed_spec=_feed().feed_spec
    )

    class Observe(Strategy):
        def on_start(self, broker):
            broker.register_account_control(
                AccountConstraints(max_leverage=1), boundary="session_end"
            )

        def on_data(self, *args):
            pass

    with pytest.raises(ValueError, match="2025, 7, 2"):
        Engine(feed, Observe()).run()


@pytest.mark.parametrize(
    "calendar,timezone,points",
    [
        ("NYSE", "America/New_York", [datetime(2025, 7, 2, 9, 31), datetime(2025, 7, 2, 16)]),
        (
            "CME_Equity",
            "UTC",
            [
                datetime(2025, 7, 6, 23, tzinfo=UTC),
                datetime(2025, 7, 7, 0, tzinfo=UTC),
                datetime(2025, 7, 7, 21, tzinfo=UTC),
            ],
        ),
    ],
)
def test_session_validation_uses_declared_timezone_and_overnight_session_not_civil_date(
    calendar, timezone, points
):
    checks = []

    class Capture(AccountConstraints):
        def validate(self, account):
            checks.append(account.timestamp)
            super().validate(account)

    class Observe(Strategy):
        def on_start(self, broker):
            broker.register_account_control(Capture(max_leverage=1), boundary="session_end")

        def on_data(self, *args):
            pass

    feed = DataFeed(
        prices_df=pl.DataFrame(
            {"timestamp": points, "asset": ["A"] * len(points), "close": [100.0] * len(points)}
        ),
        feed_spec={
            "calendar": calendar,
            "timezone": timezone,
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "data_frequency": "1m",
            "time_contract_version": 2,
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )
    Engine(feed, Observe()).run()
    assert checks == [points[-1]]
