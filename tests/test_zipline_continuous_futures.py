"""Causal Data selections must preserve dated identity through migration callbacks."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from runpy import run_path

import polars as pl
import pytest

from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Future
from tests.test_data_dated_rolls import SPEC, TIMES, contract_inputs


def _no_fees(context):
    from ml4t.backtest.compat.zipline.finance import commission

    context.set_commission(us_futures=commission.PerFutureTrade(0))


def _inputs(*, adjustment=None, offset=0, roll="volume", notice=False):
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
            roll_strategy=(
                futures.HighestVolumeRoll()
                if roll == "volume"
                else futures.FirstNoticeDateRoll(0, calendar="CME_Equity")
                if notice
                else futures.CalendarRoll()
            ),
        )

    prepared = provider(as_of=TIMES[-1])
    feed = DataFeed(
        prices_df=prepared.bars.with_columns(
            pl.concat_str(pl.lit("capture:"), pl.col("asset").cast(pl.String)).alias("asset")
        ),
        feed_spec=prepared.metadata["normalized_feed_spec"],
    )
    return feed, assets, PreparedFutures(handle, provider, assets, "CME", "frozen-r1")


@pytest.mark.parametrize(
    "roll,notice,expected",
    [
        ("volume", False, [17, 23, 23, 23]),
        ("calendar", False, [17] * 4),
        ("calendar", True, [17, 17, 23, 23]),
    ],
)
def test_actual_data_current_chain_and_raw_contract_orders_keep_independent_dated_cash(
    roll, notice, expected
):
    from ml4t.backtest.compat.zipline.api import continuous_future
    from ml4t.backtest.compat.zipline.finance import commission, slippage

    feed, assets, source = _inputs(roll=roll, notice=notice)
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
        4010 + 10 * i + (10 if sid == 23 else 0) for i, sid in enumerate(expected)
    ]
    if notice or roll == "volume":
        reference = json.loads(
            (Path(__file__).parent / "fixtures/dated_futures/zipline_3_2.json").read_text()
        )
        rows = next(c["trace"] for c in reference["cases"] if c["name"] == roll)
        assert [sid for sid, _ in seen] == [r["selected_sid"] for r in rows[1:5]]
        assert [price for _, price in seen] == [r["continuous_price"] for r in rows[1:5]]
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


@pytest.mark.parametrize("roll_marks", ["previous_session", "same_session"])
@pytest.mark.parametrize("adjustment", [None, "add", "mul"])
def test_data_adjusted_continuous_history_never_replaces_raw_execution_quotes(
    adjustment, roll_marks
):
    from ml4t.backtest.compat.zipline.finance import slippage

    feed, assets, source = _inputs(adjustment=adjustment)
    bars, _ = contract_inputs()

    recipe = run_path(str(Path(__file__).parents[1] / "examples/continuous_futures_migration.py"))
    provide = recipe["continuous_history_provider"](bars, source, SPEC, roll_marks=roll_marks)

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
    first = {
        None: 4010,
        "add": 4020,
        "mul": 4020 if roll_marks == "previous_session" else 4010 * 4030 / 4020,
    }[adjustment]
    assert observed[0] == pytest.approx([first, 4030])
    assert observed[1:] == [[4030, 4040], [4040, 4050]]
    if roll_marks == "previous_session":
        reference = json.loads(
            (Path(__file__).parent / "fixtures/dated_futures/zipline_3_2.json").read_text()
        )
        rows = next(c["trace"] for c in reference["cases"] if c["name"] == "volume")
        assert observed == [r["adjusted_history"][str(adjustment)] for r in rows[2:5]]
    assert [(fill.asset, fill.price) for fill in result.fills] == [("capture:23", 4040)]


def test_research_history_key_requires_explicit_read_only_declaration():
    with pytest.raises(ValueError, match="registered sources"):
        RuntimeServices(history_assets={"unknown": ["research:TEST"]})


@pytest.mark.parametrize("read", ["current", "current_chain", "history"])
def test_same_key_cannot_silently_substitute_an_unregistered_continuous_query(read):
    feed, assets, source = _inputs()
    wrong = replace(source.handle, offset=1)
    orders = []

    def handle(context, data):
        if read == "history":
            data.history(wrong, "price", 2, "1d")
        elif read == "current":
            data.current(wrong, "contract")
        else:
            data.current_chain(wrong)
        orders.append(context.order(assets[17], 1))

    with pytest.raises(ValueError, match="query"):
        run_algorithm(
            feed=feed,
            assets=list(assets.values()),
            continuous_futures=[source],
            initialize=_no_fees,
            handle_data=handle,
            runtime=RuntimeServices(start=TIMES[1]),
        )
    assert not orders


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


@pytest.mark.parametrize("adjustment", [None, "add", "mul"])
def test_documented_continuous_workflow_saves_actual_snapshot_and_native_account(
    adjustment, tmp_path
):
    storage = pytest.importorskip("ml4t.data.storage")
    from ml4t.backtest import BacktestResult

    recipe = run_path(str(Path(__file__).parents[1] / "examples/continuous_futures_migration.py"))
    output = tmp_path / "continuous"
    report = recipe["run"](output, adjustment=adjustment)
    inputs = storage.read_snapshot(output / "inputs")
    assert report["input_snapshot_id"] == inputs.snapshot_id
    assert report["trace"][0]["history"] == ([4010, 4030] if adjustment is None else [4020, 4030])
    assert [row["raw_price"] for row in report["trace"]] == [4030, 4040, 4050]
    assert [row["chain"] for row in report["trace"]] == [["capture:23"]] * 3
    result = BacktestResult.from_parquet(output / "result")
    assert [(fill.asset, fill.price, fill.quantity) for fill in result.fills] == [
        ("capture:23", 4040, 1)
    ]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [100000, -102000, -102000]
    assert result.to_equity_dataframe()["equity"].to_list() == [100000, 100000, 100500]
