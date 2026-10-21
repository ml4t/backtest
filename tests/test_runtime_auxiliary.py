"""Offline research sources must not reveal future values or create market events."""

from datetime import UTC, datetime

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec, LifecyclePhase

import ml4t.backtest as bt
from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import ExecutionMode

SOURCE = "https://example.org/research/original.csv"


def _clocks():
    return pd.to_datetime(
        [
            "2025-01-06T21:00:00.000000001Z",
            "2025-01-06T21:00:00.000000002Z",
            "2025-01-06T21:00:00.000000003Z",
        ]
    )


def _panel():
    t = _clocks()
    return pl.DataFrame(
        {
            "observed": pl.Series([t[0].value, t[1].value, t[1].value]).cast(
                pl.Datetime("ns", "UTC")
            ),
            "key": ["A", "A", "B"],
            "score": [10.0, 20.0, 7.0],
            "published": pl.Series([t[1].value, t[2].value, None]).cast(pl.Datetime("ns", "UTC")),
            "undeclared_future_label": [999.0] * 3,
        }
    )


def _source(frame=None):
    return bt.AuxiliaryData(
        SOURCE,
        _panel() if frame is None else frame,
        feed_spec=FeedSpec(
            time_contract_version=2,
            timestamp_col="observed",
            entity_col="key",
            timestamp_semantics="event_time",
            calendar=None,
        ),
        known_at_cols={"score": "published"},
        preparation={"date_format": "%Y-%m-%dT%H:%M:%S%z", "symbol_mapping": "captured stable IDs"},
    )


def _feed():
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": pl.Series(_clocks().asi8).cast(pl.Datetime("ns", "UTC")),
                "asset": ["A"] * 3,
                "close": [100.0] * 3,
            }
        ),
        feed_spec={"time_contract_version": 2, "timestamp_semantics": "event_time"},
    )


@pytest.mark.parametrize("container", ["polars", "pandas", "indexed_pandas", "lazy", "chunked"])
def test_delayed_research_values_drive_only_causal_orders_without_extra_market_events(container):
    panel = _panel()
    if container in {"pandas", "indexed_pandas"}:
        panel = pd.DataFrame({name: panel[name].to_list() for name in panel.columns})
        # Build clocks from integer epochs so Python datetime conversion cannot truncate ns.
        panel["observed"] = pd.to_datetime(_panel()["observed"].cast(pl.Int64).to_list(), utc=True)
        panel["published"] = pd.to_datetime(
            _panel()["published"].cast(pl.Int64).to_list(), utc=True
        )
        if container == "indexed_pandas":
            panel = panel.set_index(["observed", "key"])
    elif container == "lazy":
        panel = panel.lazy()
    elif container == "chunked":
        panel = pl.concat([panel.head(1), panel.tail(2)], rechunk=False)
    source = _source(panel)
    runtime = RuntimeServices(auxiliary=[source])
    seen, saved = [], []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            view = runtime.auxiliary(SOURCE)
            saved.append(view)
            current = view.current(["A"], ["score"], forward_fill=True)
            seen.append(current["score"][0])
            assert view.assets == (() if timestamp == _clocks()[0] else ("A",))
            if timestamp == _clocks()[1]:
                history = view.history(["A"], ["score"], bar_count=2)
                assert history["score"].to_list() == [10.0, None]
                assert view.last_observed("A", "score") == _clocks()[0]
                assert current["score_observed_at"].cast(pl.Int64).to_list() == [_clocks()[0].value]
            if current["score"][0] is not None:
                broker.submit_order("A", 1)

    result = Engine(
        _feed(),
        Read(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR, initial_cash=1000),
        runtime=runtime,
    ).run()
    assert seen == [None, 10.0, 20.0]
    assert (
        result.to_equity_dataframe()["timestamp"].cast(pl.Int64).to_list()
        == _clocks().asi8.tolist()
    )
    assert result.to_fills_dataframe()["quantity"].to_list() == [1.0, 1.0]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [1000, 900, 800]
    with pytest.raises(RuntimeError, match="active run"):
        _ = saved[0].metadata
    with pytest.raises(RuntimeError, match="active run"):
        _ = saved[0].assets


def test_original_source_lookup_and_declared_fields_cannot_silently_substitute_data():
    source = _source()
    runtime = RuntimeServices(auxiliary=[source])
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            with pytest.raises(ValueError, match="No prepared auxiliary source"):
                runtime.auxiliary(SOURCE + "?different=1")
            view = runtime.auxiliary(SOURCE)
            for field in ["undeclared_future_label", "published", "price"]:
                with pytest.raises(ValueError, match="declared payload"):
                    view.current(["A"], [field])
            metadata = view.metadata
            assert metadata["source_id"] == SOURCE
            assert len(metadata["content_sha256"]) == 64
            metadata["preparation"]["symbol_mapping"] = "changed"
            assert view.metadata["preparation"]["symbol_mapping"] == "captured stable IDs"
            seen.append(timestamp)

    Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()
    assert len(seen) == 3


def test_appended_future_rows_and_caller_frame_mutation_cannot_change_earlier_decisions():
    source = _source()
    later = (
        _panel()
        .head(1)
        .with_columns(
            pl.lit(datetime(2025, 1, 7, tzinfo=UTC))
            .cast(pl.Datetime("ns", "UTC"))
            .alias("observed"),
            pl.lit(datetime(2025, 1, 7, tzinfo=UTC))
            .cast(pl.Datetime("ns", "UTC"))
            .alias("published"),
            pl.lit(99999.0).alias("score"),
        )
    )
    future = _source(pl.concat([_panel(), later], how="vertical_relaxed"))

    def run(dataset):
        frame = dataset.frame
        frame.replace_column(
            frame.get_column_index("score"), pl.Series("score", [0.0] * len(frame))
        )
        runtime = RuntimeServices(auxiliary=[dataset])
        seen = []

        class Read(Strategy):
            def on_data(self, timestamp, data, context, broker):
                seen.append(
                    runtime.auxiliary(SOURCE).current(["A"], ["score"], forward_fill=True)["score"][
                        0
                    ]
                )

        Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()
        return seen

    outputs = [run(dataset) for dataset in [source, future]]
    assert outputs == [[None, 10.0, 20.0]] * 2


def test_preopen_published_auxiliary_data_does_not_publish_the_opening_quote():
    pre = datetime(2025, 1, 6, 13, 45, tzinfo=UTC)
    opened = datetime(2025, 1, 6, 14, 30, tzinfo=UTC)
    source = _source(
        pl.DataFrame({"observed": [pre], "key": ["A"], "score": [10.0], "published": [pre]})
    )
    runtime = RuntimeServices(start=pre, auxiliary=[source])
    seen = []

    def before(timestamp, data, context, broker):
        assert runtime.market.current(["A"], ["price"]).is_empty()
        seen.append(runtime.auxiliary(SOURCE).current(["A"], ["score"])["score"][0])

    runtime.scheduler.register(
        before, phase=LifecyclePhase.PRE_OPEN, at="session_open", minutes_before=45
    )

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    Engine(
        DataFeed(
            prices_df=pl.DataFrame({"timestamp": [opened], "asset": ["A"], "close": [100.0]}),
            feed_spec={"timestamp_semantics": "event_time"},
        ),
        Read(),
        BacktestConfig(calendar="NYSE"),
        runtime=runtime,
    ).run()
    assert seen == [10.0]


@pytest.mark.parametrize(
    "fault,error",
    [
        ("duplicate", "Duplicate auxiliary"),
        ("null_entity", "cannot be null"),
        ("naive", "aware instants"),
        ("date", "aware instants"),
        ("missing_publication", "missing declared"),
    ],
)
def test_invalid_preparation_fails_before_any_engine_or_account_exists(fault, error):
    panel = _panel()
    if fault == "duplicate":
        panel = pl.concat([panel, panel.head(1)])
    elif fault == "null_entity":
        panel = panel.with_columns(pl.lit(None, dtype=pl.String).alias("key"))
    elif fault == "naive":
        panel = panel.with_columns(pl.col("published").dt.replace_time_zone(None))
    elif fault == "date":
        panel = panel.with_columns(pl.col("observed").dt.date())
    else:
        panel = panel.drop("published")
    with pytest.raises(ValueError, match=error):
        _source(panel)


def test_empty_unsigned_source_metadata_and_views_expire_with_the_owning_run():
    frame = _panel().head(0).with_columns(pl.col("key").cast(pl.UInt64))
    runtime = RuntimeServices(auxiliary=[_source(frame)])
    saved = []

    class Read(Strategy):
        def __init__(self):
            self.previous = saved[0] if saved else None

        def on_data(self, timestamp, data, context, broker):
            if self.previous is not None:
                with pytest.raises(RuntimeError, match="active run"):
                    _ = self.previous.metadata
                with pytest.raises(RuntimeError, match="active run"):
                    _ = self.previous.assets
            view = runtime.auxiliary(SOURCE)
            assert view.assets == ()
            assert view.metadata["source_id"] == SOURCE
            assert view.current([2**63 + 1], ["score"])["asset"].dtype == pl.UInt64
            if not saved:
                saved.append(view)

    Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()
    Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()


def test_ambiguous_source_registration_is_rejected_and_no_implicit_registry_exists():
    with pytest.raises(ValueError, match="Duplicate auxiliary source"):
        RuntimeServices(auxiliary=[_source(), _source()])
    with pytest.raises(TypeError, match="AuxiliaryData"):
        RuntimeServices(auxiliary=[object()])
    with pytest.raises(RuntimeError, match="active run"):
        RuntimeServices().auxiliary(SOURCE)


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_saved_prepared_sources_preserve_nanoseconds_nulls_and_original_identity(tmp_path, format):
    panel = _panel()
    path = tmp_path / ("capture." + format)
    if format == "csv":
        panel.write_csv(path)
        frame = pl.read_csv(
            path,
            schema_overrides={
                "observed": pl.Datetime("ns", "UTC"),
                "published": pl.Datetime("ns", "UTC"),
            },
        )
    else:
        panel.write_parquet(path)
        frame = pl.read_parquet(path)
    source = _source(frame)
    assert (
        source.frame["observed"].cast(pl.Int64).to_list()
        == panel["observed"].cast(pl.Int64).to_list()
    )
    assert source.frame["published"].null_count() == 1
    runtime = RuntimeServices(auxiliary=[source])
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(
                runtime.auxiliary(SOURCE).current(["A"], ["score"], forward_fill=True)["score"][0]
            )

    Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()
    assert seen == [None, 10.0, 20.0]


def test_each_payload_field_keeps_its_own_publication_cutoff():
    panel = _panel().with_columns(pl.lit(5.0).alias("other"))
    source = bt.AuxiliaryData(
        SOURCE,
        panel,
        feed_spec=FeedSpec(
            timestamp_col="observed", entity_col="key", timestamp_semantics="event_time"
        ),
        known_at_cols={"score": "published", "other": "observed"},
    )
    runtime = RuntimeServices(auxiliary=[source])
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(runtime.auxiliary(SOURCE).current(["A"], ["score", "other"]).row(0)[2:])

    Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()
    assert seen == [(None, 5.0), (None, 5.0), (None, None)]


def test_payload_column_cannot_overwrite_generated_forward_fill_provenance():
    panel = _panel().with_columns(pl.lit(999.0).alias("score_observed_at"))
    source = bt.AuxiliaryData(
        SOURCE,
        panel,
        feed_spec=FeedSpec(
            timestamp_col="observed", entity_col="key", timestamp_semantics="event_time"
        ),
        known_at_cols={"score": "published", "score_observed_at": "observed"},
    )
    runtime = RuntimeServices(auxiliary=[source])

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            with pytest.raises(ValueError, match="provenance columns collide"):
                runtime.auxiliary(SOURCE).current(
                    ["A"], ["score", "score_observed_at"], forward_fill={"score": True}
                )
            assert broker.get_orders() == ()

    Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()


def test_native_market_payload_cannot_replace_its_requested_fill_provenance():
    prices = _feed().prices.with_columns(pl.lit(999.0).alias("close_observed_at"))
    runtime = RuntimeServices()

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            with pytest.raises(ValueError, match="provenance columns collide"):
                runtime.market.current(
                    ["A"], ["close", "close_observed_at"], forward_fill={"close": True}
                )

    Engine(
        DataFeed(prices_df=prices, feed_spec={"timestamp_semantics": "event_time"}),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()


def test_auxiliary_eligibility_blocks_orders_and_prevents_stale_values_crossing_exclusion():
    from datetime import UTC, datetime

    clocks = [datetime(2025, 7, 2, 14, minute, tzinfo=UTC) for minute in [0, 1, 2]]
    source = bt.AuxiliaryData(
        "screened-panel",
        pl.DataFrame(
            {
                "timestamp": clocks[:2],
                "asset": ["A", "A"],
                "score": [10.0, 20.0],
                "known": clocks[:2],
                "eligible": [True, False],
            }
        ),
        feed_spec={
            "timestamp_semantics": "event_time",
            "calendar": None,
            "time_contract_version": 2,
            "entity_col": "asset",
        },
        known_at_cols={"score": "known"},
        eligible_cols={"score": "eligible"},
    )
    runtime = RuntimeServices(auxiliary=(source,))
    seen = []

    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            value = runtime.auxiliary("screened-panel").current(
                ["A"], ["score"], forward_fill=True
            )["score"][0]
            seen.append(value)
            if value is not None:
                broker.submit_order("A", 1)

    prices = pl.DataFrame({"timestamp": clocks, "asset": ["A"] * 3, "close": [100.0] * 3})
    result = Engine(
        DataFeed(prices_df=prices, feed_spec={"timestamp_semantics": "event_time"}),
        Trade(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=runtime,
    ).run()
    assert seen == [10.0, None, None]
    assert result.to_fills_dataframe()["quantity"].to_list() == [1.0]
    assert source.metadata["eligible_cols"] == {"score": "eligible"}


def test_scoped_auxiliary_sources_restore_caller_inputs_after_success_and_binding_error():
    runtime = RuntimeServices()
    source = _source()

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            runtime.auxiliary(SOURCE).current(["A"], ["score"])

    def execute():
        return Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()

    with runtime.auxiliary_scope((source, source)):
        execute()
    with pytest.raises(ValueError, match="No prepared auxiliary"):
        execute()
    with (
        pytest.raises(ValueError, match="Duplicate auxiliary"),
        runtime.auxiliary_scope((source, _source())),
    ):
        pass
    with pytest.raises(ValueError, match="No prepared auxiliary"):
        execute()
