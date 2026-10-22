"""Real Data identities must constrain simulation reads and orders causally."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import ExecutionMode
from ml4t.backtest.controls import TradingControlViolation
from ml4t.backtest.types import ContractSpec, OrderStatus, OrderType

ROOT = Path(__file__).parent / "fixtures" / "prepared_instruments"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


class CapturedRegistry:
    """Replay literal Data outputs, without reproducing its selection algorithm."""

    def __init__(self, group):
        self.group = MANIFEST["groups"][group]
        self.calls = []

    def lifetimes(self, on=None, *, as_of=None, universe=None):
        assert universe is None
        key = (on.isoformat(), as_of.isoformat())
        self.calls.append(key)
        item = next(s for s in self.group["snapshots"] if (s["on"], s["as_of"]) == key)
        return pl.read_parquet(ROOT / item["file"])

    def resolve(self, symbol, on, exchange=None, *, as_of=None):
        item = next(
            r
            for r in self.group["resolutions"]
            if (r["symbol"], r["exchange"], r["on"], r["as_of"])
            == (symbol, exchange, on.isoformat(), as_of.isoformat())
        )
        if "error" in item:
            raise ValueError(item["error"])
        return item["asset"]


def _source(group="equity", actual=False):
    spec = FeedSpec.from_any(MANIFEST["groups"][group]["feed_spec"])
    if actual:
        assets = pytest.importorskip("ml4t.data.assets")
        frame = pl.read_parquet(ROOT / MANIFEST["groups"][group]["identities"])
        return assets.InstrumentRegistry.from_frame(frame, feed_spec=spec), spec
    return CapturedRegistry(group), spec


def _times(group="equity"):
    return [datetime.fromisoformat(t) for t in MANIFEST["groups"][group]["times"]]


def _feed(frame=None):
    t = _times()
    if frame is None:
        frame = pl.DataFrame(
            {
                "timestamp": [t[0], t[1], t[1], t[2], t[3]],
                "asset": ["OLD", "OLD", "NEW", "NEW", "NEW"],
                "close": [100.0, 110.0, 50.0, 50.0, 60.0],
            }
        )
    return DataFeed(
        prices_df=frame,
        feed_spec={
            "time_contract_version": 2,
            "calendar": None,
            "timestamp_semantics": "event_time",
        },
    )


class Noop(Strategy):
    def on_data(self, timestamp, data, context, broker):
        pass


@pytest.mark.parametrize("actual", [False, True], ids=["captured", "installed-data"])
def test_effective_cash_conversion_removes_expiring_holdings_before_require_flat_validation(actual):
    source, spec = _source(actual=actual)
    runtime = RuntimeServices(
        instruments=source, instrument_feed_spec=spec, inactive_position_policy="require_flat"
    )
    times = _times()
    feed = _feed()
    feed.corporate_actions = pl.DataFrame(
        [
            {
                "source": "declared-conversion",
                "event_id": "old-cash",
                "revision_id": "r1",
                "asset": "OLD",
                "action_kind": "merger",
                "currency": "USD",
                "available_at": times[0],
                "effective_at": times[2],
                "payment_at": times[2],
                "share_basis": "pre_event",
                "quantity_ratio": 0.0,
                "cash_component": 110.0,
            }
        ]
    )
    feed.input_manifest = {"price_basis": "raw"}

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("OLD", 1)
            if timestamp == times[2]:
                assert broker.get_position("OLD") is None
                assert broker.cash == 1010

    result = Engine(
        feed,
        Buy(),
        BacktestConfig(
            initial_cash=1000,
            calendar=None,
            commission_rate=0,
            slippage_rate=0,
            execution_mode=ExecutionMode.SAME_BAR,
        ),
        runtime=runtime,
        contract_specs={"OLD": ContractSpec("OLD")},
    ).run()
    assert result.to_equity_dataframe()["equity"].to_list() == [1000, 1010, 1010, 1010]


@pytest.mark.parametrize("actual", [False, True], ids=["captured", "installed-data"])
@pytest.mark.parametrize("group", ["equity", "future_revision"])
def test_rename_reuse_and_expiry_preserve_ids_cash_and_last_eligible_mark(actual, group):
    source, spec = _source(group, actual)
    runtime = RuntimeServices(instruments=source, instrument_feed_spec=spec)
    times = _times()
    seen, saved = [], []

    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if len(saved) >= 4:
                with pytest.raises(RuntimeError, match="active run"):
                    saved[0].is_active("OLD")
            saved.append(runtime.assets)
            metadata = runtime.assets.metadata
            metadata["feed_spec"]["entity_col"] = "changed"
            assert runtime.assets.metadata["feed_spec"]["entity_col"] == "asset"
            seen.append(runtime.assets.resolve("ALPHA"))
            if timestamp == times[0]:
                broker.submit_order("OLD", 1)
            if timestamp == times[1]:
                assert runtime.assets.resolve("BETA") == "OLD"
                broker.submit_order("OLD", 1, order_type=OrderType.LIMIT, limit_price=90)
            if timestamp == times[2]:
                assert not runtime.assets.is_active("OLD")
                assert not runtime.market.can_trade("OLD")
                with pytest.raises(TradingControlViolation, match="active identity"):
                    broker.submit_order("OLD", 1)
                broker.submit_order("NEW", 1)
                history = runtime.market.history(["OLD"], ["close"], bar_count=3)
                assert history["close"].to_list() == [100.0, 110.0, None]
            if timestamp == times[3]:
                orders = broker.get_orders()
                assert len(orders) == 3
                assert len(broker.get_orders("OLD")) == 2
                expired = orders[1]
                assert expired.status is OrderStatus.CANCELLED
                assert expired.cancellation_reason == "instrument_inactive"
                assert expired.cancelled_at == times[2]
                orders[0].quantity = 999
                assert broker.get_orders()[0].quantity == 1

    engine = Engine(
        _feed(),
        Trade(),
        BacktestConfig(initial_cash=1000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=runtime,
        holdings_sampling="bar",
    )
    result = engine.run()
    assert seen == ["OLD", "NEW", "NEW", "NEW"]
    assert result.to_fills_dataframe()["asset"].to_list() == ["OLD", "NEW"]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [900, 900, 850, 850]
    assert result.to_equity_dataframe()["equity"].to_list() == [1000, 1010, 1010, 1020]
    old = result.to_holdings_dataframe().filter(pl.col("asset") == "OLD")
    assert old["mark"].to_list() == [100, 110, 110, 110]
    assert old["mark_timestamp"].to_list() == [times[0], times[1], times[1], times[1]]
    assert old["mark_source"][-1] == "last_observed"
    with pytest.raises(RuntimeError, match="active run"):
        saved[0].is_active("OLD")
    # A new run binds a new view, without retaining the first run's control/cache.
    second = Engine(
        _feed(),
        Trade(),
        BacktestConfig(initial_cash=1000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=runtime,
        holdings_sampling="bar",
    ).run()
    assert second.to_equity_dataframe().equals(result.to_equity_dataframe())
    with pytest.raises(RuntimeError, match="active run"):
        saved[0].is_active("OLD")


def test_require_flat_rejects_expiry_before_mark_cash_or_pending_order_mutation():
    source, spec = _source()
    runtime = RuntimeServices(
        instruments=source, instrument_feed_spec=spec, inactive_position_policy="require_flat"
    )
    times = _times()

    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("OLD", 1)
            if timestamp == times[1]:
                broker.submit_order("OLD", 1, order_type=OrderType.LIMIT, limit_price=90)

    engine = Engine(
        _feed(),
        Trade(),
        BacktestConfig(initial_cash=1000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=runtime,
    )
    with pytest.raises(ValueError, match="Positions lack active"):
        engine.run()
    assert engine.broker.cash == 900
    assert engine.broker.get_position("OLD").quantity == 1
    assert engine.broker.get_position("OLD").current_price == 110
    assert engine.broker.get_orders(pending_only=True)[0].status is OrderStatus.PENDING
    with pytest.raises(RuntimeError, match="active run"):
        _ = runtime.assets


@pytest.mark.parametrize("asset", ["OLD", "GHOST"])
def test_unknown_or_expired_real_quote_fails_before_account_mark_mutation(asset):
    source, spec = _source()
    runtime = RuntimeServices(instruments=source, instrument_feed_spec=spec)
    frame = _feed().prices
    frame = pl.concat(
        [frame, pl.DataFrame({"timestamp": [_times()[2]], "asset": [asset], "close": [999.0]})]
    ).sort("timestamp", maintain_order=True)

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == _times()[0]:
                broker.submit_order("OLD", 1)

    engine = Engine(
        _feed(frame),
        Buy(),
        BacktestConfig(initial_cash=1000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=runtime,
    )
    with pytest.raises(ValueError, match="inactive or unknown instrument"):
        engine.run()
    assert engine.broker.cash == 900
    assert len(engine.broker.get_orders()) == 1
    assert engine.broker.get_position("OLD").current_price == 110
    assert engine.feed.prices["close"].max() == 999


def test_actual_later_snapshot_returned_early_is_rejected_before_strategy_starts():
    source, spec = _source()
    later = pl.read_parquet(ROOT / "equity-1-1.parquet")
    source.lifetimes = lambda **_kwargs: later
    started = []

    class Observe(Noop):
        def on_start(self, broker):
            started.append(True)

    with pytest.raises(ValueError, match="unavailable future"):
        Engine(
            _feed(),
            Observe(),
            BacktestConfig(calendar=None),
            runtime=RuntimeServices(instruments=source, instrument_feed_spec=spec),
        ).run()
    assert not started


@pytest.mark.parametrize("actual", [False, True], ids=["captured", "installed-data"])
def test_cme_overnight_identity_uses_declared_session_labels_not_civil_date(actual):
    source, spec = _source("cme", actual)
    times = _times("cme")
    runtime = RuntimeServices(instruments=source, instrument_feed_spec=spec)
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(runtime.assets.is_active("CME1"))
            assert runtime.assets.resolve("ES") == "CME1"

    Engine(
        _feed(
            pl.DataFrame({"timestamp": times[:2], "asset": ["CME1"] * 2, "close": [100.0, 101.0]})
        ),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == [True, True]


def test_naive_simulation_clock_uses_declared_timezone_for_registry_cutoff():
    source, spec = _source()
    times = _times()
    naive = [t.replace(tzinfo=None, hour=16) for t in times]
    prices = pl.DataFrame(
        {"timestamp": naive, "asset": ["NEW"] * 4, "close": [None, 50.0, 50.0, 60.0]},
        schema_overrides={"close": pl.Float64},
    )
    seen = []
    runtime = RuntimeServices(instruments=source, instrument_feed_spec=spec)

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(runtime.assets.metadata["as_of"])

    Engine(
        _feed(prices),
        Read(),
        BacktestConfig(calendar=None, timezone="America/New_York"),
        runtime=runtime,
    ).run()
    assert seen == [t.isoformat() for t in times]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"instruments": object()},
        {"instrument_feed_spec": FeedSpec()},
        {"inactive_position_policy": "require_flat"},
        {"inactive_position_policy": "sell_at_zero"},
    ],
)
def test_incomplete_or_invented_instrument_policy_fails_before_run(kwargs):
    with pytest.raises(ValueError):
        RuntimeServices(**kwargs)


def test_preopen_order_uses_known_identity_at_eligible_open_without_exposing_future_quote():
    from ml4t.specs import LifecyclePhase

    source, spec = _source(actual=True)
    closed = _times()[0]
    opened = datetime(2025, 1, 8, 14, 30, tzinfo=UTC)
    runtime = RuntimeServices(start=closed, instruments=source, instrument_feed_spec=spec)
    submitted = []

    def before(timestamp, data, context, broker):
        if timestamp.date() != opened.date():
            return
        assert runtime.market.current(["NEW"], ["price"]).is_empty()
        assert not runtime.market.can_trade("NEW")
        assert runtime.assets.resolve("ALPHA") == "NEW"
        submitted.append(broker.submit_order("NEW", 1))

    runtime.scheduler.register(
        before, phase=LifecyclePhase.PRE_OPEN, at="session_open", minutes_before=45
    )
    result = Engine(
        _feed(
            pl.DataFrame(
                {"timestamp": [closed, opened], "asset": ["OLD", "NEW"], "close": [100.0, 50.0]}
            )
        ),
        Noop(),
        BacktestConfig(calendar="NYSE", execution_mode=ExecutionMode.NEXT_BAR),
        runtime=runtime,
    ).run()
    assert len(submitted) == 1
    assert result.to_fills_dataframe()["timestamp"].to_list() == [opened]
    assert result.to_fills_dataframe()["price"].to_list() == [50.0]


def test_actual_data_universe_query_retains_membership_publication_and_identity_keys():
    assets = pytest.importorskip("ml4t.data.assets")
    _, spec = _source(actual=True)
    times = _times()
    identities = pl.read_parquet(ROOT / "equity-identities.parquet")
    memberships = pl.DataFrame(
        {
            "asset": ["NEW"],
            "universe": ["chosen"],
            "valid_from": [datetime(2025, 1, 8).date()],
            "valid_to": pl.Series([None], dtype=pl.Date),
            "available_at": [datetime(2025, 1, 7, 12, tzinfo=UTC)],
            "source": ["captured"],
            "revision_id": ["r1"],
        }
    )
    source = assets.InstrumentRegistry.from_frame(identities, memberships, feed_spec=spec)
    runtime = RuntimeServices(instruments=source, instrument_feed_spec=spec)
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(runtime.assets.active(universe="chosen")["asset"].to_list())

    Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()
    assert seen == [[], ["NEW"], ["NEW"], ["NEW"]]
    assert times[0] < memberships["available_at"][0] < times[1]


def test_failed_callback_restores_removed_user_control_and_unbind_removes_only_runtime_control():
    from ml4t.backtest.controls import OrderConstraints

    source, spec = _source()
    runtime = RuntimeServices(instruments=source, instrument_feed_spec=spec)
    rule = OrderConstraints(max_order_quantity=1)

    class Fail(Noop):
        def on_start(self, broker):
            broker.register_trading_control(rule)

        def on_data(self, timestamp, data, context, broker):
            assert not broker.unregister_trading_control(OrderConstraints(max_order_quantity=1))
            assert broker.unregister_trading_control(rule)
            raise ValueError("callback failed after removal")

    engine = Engine(_feed(), Fail(), BacktestConfig(calendar=None), runtime=runtime)
    with pytest.raises(ValueError, match="callback failed after removal"):
        engine.run()
    with pytest.raises(TradingControlViolation, match="quantity"):
        engine.broker.submit_order("OLD", 2)
    assert engine.broker.get_orders() == ()
    assert engine.broker.unregister_trading_control(rule)
    assert not engine.broker.unregister_trading_control(rule)
