"""Causal Data selections must preserve dated identity through migration callbacks."""

from dataclasses import replace
from datetime import timedelta

import polars as pl
import pytest

from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Future
from tests.test_data_dated_rolls import SPEC, TIMES, contract_inputs


def _no_fees(context):
    from ml4t.backtest.compat.zipline.finance import commission

    context.set_commission(us_futures=commission.PerFutureTrade(0))


def _inputs(*, adjustment=None, offset=0, roll="volume"):
    from ml4t.backtest.compat.zipline import ContinuousFuture, PreparedFutures

    futures = pytest.importorskip("ml4t.data.futures")
    bars, definitions = contract_inputs()
    assets = {
        identity: Future(
            f"capture:{identity}",
            f"TEST{identity}",
            identity,
            root_symbol="TEST",
            multiplier=50,
            tick_size=0.25,
        )
        for identity in [17, 23]
    }
    handle = ContinuousFuture(
        f"research:TEST:{offset}:{roll}:{adjustment}",
        root_symbol="TEST",
        offset=offset,
        roll_style=roll,
        adjustment=adjustment,
    )

    def provider(*, as_of):
        return futures.prepare_futures_inputs(
            bars,
            definitions,
            as_of=as_of,
            feed_spec=SPEC,
            roll_strategy=futures.HighestVolumeRoll()
            if roll == "volume"
            else futures.CalendarRoll(),
        )

    prepared = provider(as_of=TIMES[-1])
    feed = DataFeed(
        prices_df=prepared.bars.with_columns(
            pl.concat_str(pl.lit("capture:"), pl.col("asset").cast(pl.String)).alias("asset")
        ),
        feed_spec=prepared.metadata["normalized_feed_spec"],
    )
    return feed, assets, PreparedFutures(handle, provider, assets, "CME", "frozen-r1")


@pytest.mark.parametrize("roll,expected", [("volume", [17, 23, 23, 23]), ("calendar", [17] * 4)])
def test_actual_data_current_chain_and_raw_contract_orders_keep_independent_dated_cash(
    roll, expected
):
    from ml4t.backtest.compat.zipline.api import continuous_future
    from ml4t.backtest.compat.zipline.finance import commission, slippage

    feed, assets, source = _inputs(roll=roll)
    seen = []

    def initialize(context):
        context.cf = continuous_future("TEST", roll=roll, adjustment=None)
        assert context.cf is source.handle
        context.set_commission(us_futures=commission.PerFutureTrade(0))
        context.set_slippage(us_futures=slippage.NoSlippage())

    def handle(context, data):
        dated = data.current(context.cf, "contract")
        seen.append((dated.legacy_sid, data.current(context.cf, "price")))
        assert data.current_chain(context.cf)[0] is dated
        assert data.can_trade(context.cf) and not data.is_stale(context.cf)
        assert data.current([context.cf], ["contract", "price"]).iloc[0, 0] is dated
        with pytest.raises(TypeError, match="dated Future"):
            context.order(context.cf, 1)
        if len(seen) == 1:
            context.order(dated, 1)

    result = run_algorithm(
        feed=feed,
        assets=list(assets.values()),
        continuous_futures=[source],
        initialize=initialize,
        handle_data=handle,
        runtime=RuntimeServices(start=TIMES[1]),
    ).attrs["backtest_result"]
    assert [sid for sid, _ in seen] == expected
    assert [price for _, price in seen] == [
        4010,
        4030 if roll == "volume" else 4020,
        4040 if roll == "volume" else 4030,
        4050 if roll == "volume" else 4040,
    ]
    assert [(fill.asset, fill.price, fill.quantity) for fill in result.fills] == [
        ("capture:17", 4020, 1)
    ]
    # The continuous switch alone never rolls the held physical position.
    assert result.to_portfolio_state_dataframe()["cash"][-1] == -101000
    assert result.to_equity_dataframe()["equity"][-1] == 101000
    assert result.config.metadata["migration_continuous_futures"][0]["source_id"] == "frozen-r1"


def test_continuous_offset_is_next_maturity_not_next_volume_rank_and_missing_tail_is_empty():
    feed, assets, source = _inputs(offset=1)
    seen = []

    def handle(context, data):
        contract = data.current(source.handle, "contract")
        seen.append(None if contract is None else contract.legacy_sid)
        assert [a.legacy_sid for a in data.current_chain(source.handle)] == (
            [23] if len(seen) == 1 else []
        )

    run_algorithm(
        feed=feed,
        assets=list(assets.values()),
        continuous_futures=[source],
        handle_data=handle,
        initialize=_no_fees,
        runtime=RuntimeServices(start=TIMES[1]),
    )
    assert seen == [23, None, None, None]


def test_future_provider_vintage_is_rejected_at_actual_callback_before_any_order():
    feed, assets, source = _inputs()
    original = source.provider
    source = replace(source, provider=lambda *, as_of: original(as_of=as_of + timedelta(days=1)))
    orders = []

    def handle(context, data):
        data.current(source.handle, "contract")
        orders.append(context.order(assets[17], 1))

    with pytest.raises(ValueError, match="actual causal as_of"):
        run_algorithm(
            feed=feed,
            assets=list(assets.values()),
            continuous_futures=[source],
            handle_data=handle,
            initialize=_no_fees,
            runtime=RuntimeServices(start=TIMES[1], end=TIMES[1]),
        )
    assert not orders


def test_continuous_source_cannot_silently_drop_real_unmapped_contracts():
    feed, assets, source = _inputs()
    source = replace(source, contracts={17: assets[17]})
    with pytest.raises(ValueError, match="unmapped stable instrument"):
        run_algorithm(
            feed=feed,
            assets=list(assets.values()),
            continuous_futures=[source],
            handle_data=lambda _c, d: d.current_chain(source.handle),
            initialize=_no_fees,
            runtime=RuntimeServices(start=TIMES[1]),
        )


def test_colliding_continuous_research_keys_fail_before_initialize():
    feed, assets, source = _inputs()
    duplicate = replace(source, handle=replace(source.handle, offset=1))
    initialized = []
    with pytest.raises(ValueError, match="ambiguous"):
        run_algorithm(
            feed=feed,
            assets=list(assets.values()),
            continuous_futures=[source, duplicate],
            initialize=lambda c: initialized.append(c),
        )
    assert not initialized


@pytest.mark.parametrize("adjustment", [None, "add", "mul"])
def test_data_adjusted_continuous_history_never_replaces_raw_execution_quotes(adjustment):
    from ml4t.backtest.compat.zipline.finance import slippage

    futures = pytest.importorskip("ml4t.data.futures")
    adjustments = pytest.importorskip("ml4t.data.futures.adjustment")
    roll = pytest.importorskip("ml4t.data.futures.roll")
    history = pytest.importorskip("ml4t.data.adjustments")
    feed, assets, source = _inputs(adjustment=adjustment)
    bars, _ = contract_inputs()

    def provide(*, as_of, observation_before=None):
        selected = []
        known = bars.filter(pl.col("timestamp") <= as_of)
        if observation_before is not None:
            known = known.filter(pl.col("timestamp") < observation_before)
        for point in known["timestamp"].unique().sort():
            prepared = source.provider(as_of=point)
            if prepared.selections.is_empty():
                continue
            identity = prepared.selections["selected_instrument_id"][0]
            selected.append({"date": point.date(), "symbol": str(identity)})
        raw = known.with_columns(
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("asset").cast(pl.String).alias("symbol"),
            *[pl.col("close").alias(name) for name in ["open", "high", "low"]],
        )
        selection = pl.DataFrame(selected)
        events = roll.build_roll_events(raw, selection)
        method = {
            None: adjustments.NoAdjustment,
            "add": adjustments.BackAdjustment,
            "mul": adjustments.RatioAdjustment,
        }[adjustment]()
        continuous = method.adjust(raw.join(selection, on=["date", "symbol"]), events)
        prices = continuous.select(
            "timestamp",
            pl.lit(as_of).alias("available_at"),
            "source",
            "revision_id",
            pl.lit(source.handle.key).alias("asset"),
            pl.col("adjusted_close").alias("close"),
            "volume",
        )
        # Data owns adjustments and validates physical source availability again.
        return history.prepare_adjusted_history(
            prices,
            None,
            as_of=as_of,
            feed_spec=SPEC.with_overrides(entity_col="asset"),
            basis="raw",
        )

    observed = []

    def initialize(context):
        _no_fees(context)
        context.set_slippage(us_futures=slippage.NoSlippage())

    def handle(context, data):
        value = data.history(source.handle, "price", 2, "1d")
        assert value.name is source.handle
        observed.append(value.tolist())
        assert data.current(source.handle, "price") == 4030 + (len(observed) - 1) * 10
        if len(observed) == 1:
            context.order(data.current(source.handle, "contract"), 1)

    result = run_algorithm(
        feed=feed,
        assets=list(assets.values()),
        continuous_futures=[source],
        initialize=initialize,
        handle_data=handle,
        runtime=RuntimeServices(
            start=TIMES[2],
            history_sources={"continuous-r1": provide},
            history_assets={"continuous-r1": [source.handle.key]},
        ),
        history_sources={"1d": "continuous-r1"},
        history_unit="observation",
    ).attrs["backtest_result"]
    first = {None: 4010, "add": 4020, "mul": 4010 * 4030 / 4020}[adjustment]
    assert observed[0] == pytest.approx([first, 4030])
    assert observed[1:] == [[4030, 4040], [4040, 4050]]
    assert [(fill.asset, fill.price) for fill in result.fills] == [("capture:23", 4040)]


def test_research_history_key_requires_explicit_read_only_declaration():
    with pytest.raises(ValueError, match="registered sources"):
        RuntimeServices(history_assets={"unknown": ["research:TEST"]})


def test_actual_data_normalized_units_cannot_be_replaced_by_wrong_declared_contract_size():
    feed, assets, source = _inputs()
    wrong = {identity: replace(asset, multiplier=25) for identity, asset in assets.items()}
    source = replace(source, contracts=wrong)
    with pytest.raises(ValueError, match="Normalized source units"):
        run_algorithm(
            feed=feed,
            assets=list(wrong.values()),
            continuous_futures=[source],
            initialize=_no_fees,
            handle_data=lambda _c, d: d.current(source.handle, "price"),
            runtime=RuntimeServices(start=TIMES[1]),
        )
