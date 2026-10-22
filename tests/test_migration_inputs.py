"""Caller storage formats must retain simulation keys, precision and financial outcomes."""

from io import StringIO

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode

EPOCHS = [1736197200000000000, 1736197200000000001, 1736197201000000000]
SPEC = FeedSpec(
    time_contract_version=2,
    calendar=None,
    timestamp_semantics="event_time",
    timestamp_col="event_at",
    entity_col="instrument_id",
    price_col="settle",
    close_col="settle",
)


def panels(kind):
    prices = pl.DataFrame(
        {
            "event_at": pl.Series(EPOCHS).cast(pl.Datetime("ns", "UTC")),
            "instrument_id": ["capture:A"] * 3,
            "settle": [100.0, 101.0, 102.0],
        }
    )
    context = prices.select("event_at").with_columns(pl.Series("risk_limit", [1.0, None, 2.0]))
    if kind.startswith("pandas"):
        prices = pd.DataFrame(
            {
                "event_at": pd.to_datetime(EPOCHS, utc=True),
                "instrument_id": pd.array(["capture:A"] * 3, dtype="string"),
                "settle": pd.array([100.0, 101.0, 102.0], dtype="Float64"),
            }
        )
        context = pd.DataFrame(
            {
                "event_at": pd.to_datetime(EPOCHS, utc=True),
                "risk_limit": pd.array([1.0, None, 2.0], dtype="Float64"),
            }
        )
        if kind == "pandas-indexed":
            prices = prices.set_index(["event_at", "instrument_id"])
            context = context.set_index("event_at")
    elif kind == "lazy":
        prices, context = prices.lazy(), context.lazy()
    return prices, context


@pytest.mark.parametrize("kind", ["polars", "pandas", "pandas-indexed", "lazy"])
def test_native_input_formats_preserve_ns_keys_nullable_context_and_cash(kind):
    prices, context = panels(kind)
    seen = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(context["risk_limit"])
            if len(seen) == 1:
                broker.submit_order("capture:A", 10)

    result = Engine(
        DataFeed(prices_df=prices, context_df=context, feed_spec=SPEC),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=10000,
            execution_mode=ExecutionMode.SAME_BAR,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
        ),
        holdings_sampling="bar",
    ).run()
    state = result.to_portfolio_state_dataframe()
    assert state["timestamp"].cast(pl.Int64).to_list() == EPOCHS
    assert state["cash"].to_list() == [8998.0] * 3
    assert state["equity"].to_list() == [9998.0, 10008.0, 10018.0]
    assert result.to_fills_dataframe()["asset"].to_list() == ["capture:A"]
    assert seen == [1.0, None, 2.0]


@pytest.mark.parametrize("duplicate", ["prices", "signals", "context"])
def test_ambiguous_csv_keys_are_rejected_instead_of_silently_overwriting_an_event(duplicate):
    class Noop(Strategy):
        def on_data(self, *args):
            pass

    frame = pl.read_csv(
        StringIO("timestamp,asset,close\n2025-01-06T21:00:00,AA,100\n2025-01-06T21:00:00,AA,200\n"),
        try_parse_dates=True,
    )
    args = {"prices_df": frame.head(1), f"{duplicate}_df": frame}
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        Engine(DataFeed(**args), Noop(), BacktestConfig(calendar=None)).run()


def test_direct_pandas_weekday_deprecation_remains_an_error_outside_calendar_dependency():
    pandas = pytest.importorskip("pandas", minversion="3.1")
    dates = pandas.Series(pandas.to_datetime(["2025-01-06"]))
    with pytest.raises(pandas.errors.Pandas4Warning, match="Series.dt.weekday is deprecated"):
        _ = dates.dt.weekday


def test_unsorted_cross_sections_keep_original_asset_order_at_equal_event_instants():
    times = pd.to_datetime([EPOCHS[2], EPOCHS[2], EPOCHS[0], EPOCHS[0]], utc=True)
    prices = pd.DataFrame(
        {
            "event_at": times,
            "instrument_id": ["capture:B", "capture:A"] * 2,
            "settle": [102.0, 202.0, 100.0, 200.0],
        }
    )
    observed = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            observed.append(list(data))

    Engine(DataFeed(prices_df=prices, feed_spec=SPEC), Read(), BacktestConfig(calendar=None)).run()
    assert observed == [["capture:B", "capture:A"], ["capture:B", "capture:A"]]
