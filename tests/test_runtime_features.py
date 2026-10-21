"""Prepared features and context must obey their own publication clocks."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from ml4t.specs import LifecyclePhase

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import ExecutionMode


def _prices(times):
    return pl.DataFrame(
        {"timestamp": times, "asset": ["TEST"] * len(times), "close": [100.0] * len(times)}
    )


@pytest.mark.parametrize("append", [False, True])
def test_signal_publication_masks_actual_strategy_delivery_and_appended_future_features(append):
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [2, 3, 6]]
    signals = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 3,
            "alpha": [10.0, -1.0, 999.0],
            "published": [times[1], times[1], times[2]],
        }
    )
    source = signals if append else signals.head(2)
    runtime = RuntimeServices(signal_known_at_cols={"alpha": "published"}, end=times[1])
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            alpha = data["TEST"]["signals"]["alpha"]
            seen.append(alpha)
            if alpha is not None and alpha > 0:
                broker.submit_order("TEST", 1)

    result = Engine(
        DataFeed(
            prices_df=_prices(times if append else times[:2]),
            signals_df=source,
            feed_spec={"timestamp_semantics": "bar_close"},
        ),
        Read(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=runtime,
    ).run()
    assert seen == [None, -1.0]
    assert result.to_fills_dataframe().is_empty()
    assert source["alpha"].to_list() == ([10.0, -1.0, 999.0] if append else [10.0, -1.0])


@pytest.mark.parametrize("delay,expected", [(0, 10.0), (1, None)])
def test_pre_open_feature_queries_use_available_decision_rows_without_reading_opening_quotes(
    delay, expected
):
    pre = datetime(2025, 1, 6, 13, 45, tzinfo=UTC)
    opened = datetime(2025, 1, 6, 14, 30, tzinfo=UTC)
    signals = pl.DataFrame(
        {
            "timestamp": [pre],
            "asset": ["TEST"],
            "alpha": [10.0],
            "published": [pre + timedelta(minutes=delay)],
        }
    )
    runtime = RuntimeServices(
        start=pre - timedelta(minutes=1), signal_known_at_cols={"alpha": "published"}
    )
    seen = []

    def before(timestamp, data, context, broker):
        seen.append(
            (
                runtime.signals.current(["TEST"], ["alpha"])["alpha"].to_list(),
                runtime.market.current(["TEST"], ["close"])["close"].to_list(),
            )
        )

    runtime.scheduler.register(
        before, phase=LifecyclePhase.PRE_OPEN, at="session_open", minutes_before=45
    )

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == opened:
                latest = runtime.signals.current(["TEST"], ["alpha"], forward_fill=True)
                assert latest["alpha"].to_list() == [10.0]
                assert latest["alpha_observed_at"].to_list() == [pre]

    Engine(
        DataFeed(
            prices_df=_prices([opened]),
            signals_df=signals,
            feed_spec={"timestamp_semantics": "event_time"},
        ),
        Read(),
        BacktestConfig(calendar="NYSE"),
        runtime=runtime,
    ).run()
    assert seen == [([expected], [])]


def test_context_publication_masks_the_strategy_payload_without_changing_market_quotes():
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [2, 3]]
    context = pl.DataFrame(
        {"timestamp": times, "rate": [0.05, 0.1], "published": [times[1], times[1]]}
    )
    runtime = RuntimeServices(context_known_at_cols={"rate": "published"})
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(context["rate"])
            assert runtime.market.can_trade("TEST")

    Engine(
        DataFeed(
            prices_df=_prices(times),
            context_df=context,
            feed_spec={"timestamp_semantics": "event_time"},
        ),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == [None, 0.1]
    assert context["rate"].to_list() == [0.05, 0.1]


def test_a_feature_named_close_is_a_feature_and_cannot_supply_an_executable_quote():
    times = [datetime(2025, 1, 2, 15, m, tzinfo=UTC) for m in [0, 1]]
    runtime = RuntimeServices(signal_known_at_cols={"close": "published"})
    prices = pl.DataFrame({"timestamp": times[1:], "asset": ["TEST"], "px": [100.0]})
    signals = pl.DataFrame(
        {"timestamp": times[:1], "asset": ["TEST"], "close": [999.0], "published": times[:1]}
    )
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                seen.append(runtime.signals.current(["TEST"], ["close"])["close"].to_list())
                assert not runtime.market.can_trade("TEST")
                assert broker.get_mark_price("TEST") is None
                broker.submit_order("TEST", 1)

    result = Engine(
        DataFeed(
            prices_df=prices,
            signals_df=signals,
            feed_spec={"timestamp_semantics": "event_time"},
            price_col="px",
            close_col="px",
        ),
        Read(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=runtime,
    ).run()
    assert seen == [[999.0]]
    assert result.to_fills_dataframe()["timestamp"].to_list() == times[1:]
    assert result.to_fills_dataframe()["price"].to_list() == [100.0]


def test_retained_feature_views_cannot_read_a_later_reuse_of_the_runtime():
    times = [datetime(2025, 1, 2, 21, tzinfo=UTC)]
    runtime = RuntimeServices(signal_known_at_cols={"alpha": "published"})
    saved = []

    class Read(Strategy):
        def on_data(self, *args):
            if saved:
                with pytest.raises(RuntimeError, match="active run"):
                    saved[0].current(["TEST"], ["alpha"])
            saved.append(runtime.signals)

    for value in [1.0, 999.0]:
        signals = pl.DataFrame(
            {"timestamp": times, "asset": ["TEST"], "alpha": [value], "published": times}
        )
        Engine(
            DataFeed(
                prices_df=_prices(times),
                signals_df=signals,
                feed_spec={"timestamp_semantics": "event_time"},
            ),
            Read(),
            BacktestConfig(calendar=None),
            runtime=runtime,
        ).run()
    with pytest.raises(RuntimeError, match="active run"):
        saved[-1].current(["TEST"], ["alpha"])


@pytest.mark.parametrize("late", [False, True])
def test_known_ineligible_features_are_not_replaced_with_stale_values_but_future_rows_do_not_erase_known_history(
    late,
):
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [2, 3, 6]]
    signals = pl.DataFrame(
        {
            "timestamp": times[:2],
            "asset": ["TEST"] * 2,
            "alpha": [10.0, 99.0],
            "published": [times[0], times[2] if late else times[1]],
            "eligible": [True, False],
        }
    )
    runtime = RuntimeServices(
        signal_known_at_cols={"alpha": "published"}, signal_eligible_cols={"alpha": "eligible"}
    )
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[1]:
                assert data["TEST"]["signals"]["alpha"] is None
                seen.append(
                    runtime.signals.current(["TEST"], ["alpha"], forward_fill=True)[
                        "alpha"
                    ].to_list()
                )
                assert runtime.signals.history(["TEST"], ["alpha"], bar_count=2)[
                    "alpha"
                ].to_list() == [10.0, None]

    Engine(
        DataFeed(
            prices_df=_prices(times[:2]),
            signals_df=signals,
            feed_spec={"timestamp_semantics": "event_time"},
        ),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == [[10.0 if late else None]]


@pytest.mark.parametrize("bad", ["asset_type", "publication_column", "duplicate_context"])
def test_invalid_signal_and_context_declarations_fail_before_initialization(bad):
    times = [datetime(2025, 1, 2, 21, tzinfo=UTC)]
    signals = pl.DataFrame(
        {
            "timestamp": times,
            "asset": [17 if bad == "asset_type" else "TEST"],
            "alpha": [1.0],
            "published": times,
        }
    )
    context = (
        pl.DataFrame({"timestamp": times * 2, "rate": [0.1, 0.2], "published": times * 2})
        if bad == "duplicate_context"
        else None
    )
    runtime = RuntimeServices(
        signal_known_at_cols={"alpha": "missing" if bad == "publication_column" else "published"},
        context_known_at_cols={"rate": "published"} if bad == "duplicate_context" else None,
    )
    started = []

    class Read(Strategy):
        def on_start(self, broker):
            started.append(True)

        def on_data(self, *args):
            pass

    with pytest.raises(ValueError, match="dtypes|Known-at|unique"):
        Engine(
            DataFeed(
                prices_df=_prices(times),
                signals_df=signals,
                context_df=context,
                feed_spec={"timestamp_semantics": "event_time"},
            ),
            Read(),
            BacktestConfig(calendar=None),
            runtime=runtime,
        ).run()
    assert started == []


def test_signal_publication_does_not_round_an_adjacent_nanosecond_back_into_the_decision():
    epoch = 1735851600000000000
    times = pl.Series("timestamp", [epoch + 1, epoch + 2]).cast(pl.Datetime("ns", "UTC"))
    published = pl.Series("published", [epoch + 2, epoch + 2]).cast(pl.Datetime("ns", "UTC"))
    prices = pl.DataFrame({"timestamp": times, "asset": ["TEST"] * 2, "close": [100.0] * 2})
    signals = pl.DataFrame(
        {"timestamp": times, "asset": ["TEST"] * 2, "alpha": [10.0, -1.0], "published": published}
    )
    runtime = RuntimeServices(signal_known_at_cols={"alpha": "published"})
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(data["TEST"]["signals"]["alpha"])

    Engine(
        DataFeed(
            prices_df=prices, signals_df=signals, feed_spec={"timestamp_semantics": "event_time"}
        ),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == [None, -1.0]


def test_context_can_declare_its_observation_clock_as_its_publication_clock():
    times = [datetime(2025, 1, 2, 21, tzinfo=UTC)]
    context = pl.DataFrame({"timestamp": times, "rate": [0.2]})
    runtime = RuntimeServices(context_known_at_cols={"rate": "timestamp"})
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(context["rate"])

    Engine(
        DataFeed(
            prices_df=_prices(times),
            context_df=context,
            feed_spec={"timestamp_semantics": "event_time"},
        ),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == [0.2]
