"""Mixed assets must use their declared fill basis, impact model and shared volume cap."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import (
    AssetClass,
    BacktestConfig,
    ContractSpec,
    DataFeed,
    Engine,
    Strategy,
    run_backtest,
)
from ml4t.backtest.config import ExecutionMode, ExecutionPrice


def _prices():
    return pl.DataFrame(
        [
            {
                "timestamp": datetime(2025, 3, day, 21, tzinfo=UTC),
                "asset": asset,
                "close": 100.0,
                "high": 120.0,
                "low": 100.0,
                "volume": 4.0,
            }
            for day in [3, 4, 5, 6]
            for asset in ["A", "ES"]
        ]
    )


@pytest.mark.parametrize("entry", ["engine", "from_config", "run_backtest"])
@pytest.mark.parametrize("side", [1.0, -1.0], ids=["long", "short"])
def test_mixed_models_preserve_distinct_fill_basis_caps_cash_and_marked_nav(entry, side):
    from ml4t.backtest.execution.limits import AssetExecutionLimits, SharedVolumeParticipationLimit
    from ml4t.backtest.models import AssetSlippage, FixedSlippage, PercentageSlippage

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                broker.submit_order("A", 3 * side)
                broker.submit_order("ES", 3 * side)

    config = BacktestConfig(
        calendar=None,
        execution_mode=ExecutionMode.NEXT_BAR,
        execution_price=ExecutionPrice.CLOSE,
        next_bar_execution_price=ExecutionPrice.CLOSE,
        allow_short_selling=True,
    )
    kwargs = {
        "slippage_model": AssetSlippage(
            {"ES": FixedSlippage(0.5)},
            default=PercentageSlippage(0.01),
            price_sources={"ES": ExecutionPrice.MID},
        ),
        "execution_limits": AssetExecutionLimits(
            {"ES": SharedVolumeParticipationLimit(0.25)},
            default=SharedVolumeParticipationLimit(0.5),
        ),
        "contract_specs": {"ES": ContractSpec("ES", AssetClass.FUTURE, multiplier=50.0)},
    }
    result = (
        run_backtest(_prices(), Submit(), config=config, **kwargs)
        if entry == "run_backtest"
        else (Engine if entry == "engine" else Engine.from_config)(
            DataFeed(prices_df=_prices()), Submit(), config, **kwargs
        ).run()
    )
    fills = result.to_fills_dataframe()
    equity = fills.filter(pl.col("asset") == "A")
    future = fills.filter(pl.col("asset") == "ES")
    assert equity["quantity"].to_list() == [2.0, 1.0]
    assert equity["price"].to_list() == [100 + side] * 2
    assert future["quantity"].to_list() == [1.0, 1.0, 1.0]
    assert future["price"].to_list() == [110 + 0.5 * side] * 3
    assert equity["price_source"].to_list() == ["close"] * 2
    assert future["price_source"].to_list() == ["mid"] * 3
    state = result.to_portfolio_state_dataframe()
    assert state["cash"][-1] == (83122.0 if side > 0 else 116722.0)
    assert state["equity"].to_list() == (
        [100000.0, 99473.0, 98947.0, 98422.0]
        if side > 0
        else [100000.0, 100473.0, 100947.0, 101422.0]
    )


def test_routed_shared_limits_do_not_reuse_the_bar_capacity_for_competing_orders():
    from ml4t.backtest.execution.limits import AssetExecutionLimits, SharedVolumeParticipationLimit
    from ml4t.backtest.models import AssetSlippage, NoSlippage

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                broker.submit_order("ES", 2)
                broker.submit_order("ES", 2)
                broker.submit_order("A", 2)

    result = Engine(
        DataFeed(prices_df=_prices()),
        Submit(),
        BacktestConfig(
            calendar=None,
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_execution_price=ExecutionPrice.CLOSE,
        ),
        slippage_model=AssetSlippage({}, default=NoSlippage()),
        execution_limits=AssetExecutionLimits(
            {"ES": SharedVolumeParticipationLimit(0.25)},
            default=SharedVolumeParticipationLimit(0.5),
        ),
    ).run()
    fills = result.to_fills_dataframe()
    future = fills.filter(pl.col("asset") == "ES")
    assert future["quantity"].to_list() == [1.0, 1.0, 1.0]
    assert future["timestamp"].n_unique() == 3
    assert fills.filter(pl.col("asset") == "A")["quantity"].to_list() == [2.0]


def test_routed_quotes_copy_only_the_selected_model_and_never_consume_its_fill_state():
    from ml4t.backtest.execution.limits import SharedVolumeParticipationLimit
    from ml4t.backtest.models import AssetSlippage, NoSlippage

    class CountingSlippage:
        def __init__(self):
            self.fills = 0

        def calculate(self, asset, quantity, price, volume):
            self.fills += 1
            return float(self.fills)

    class Unused:
        def __deepcopy__(self, memo):
            raise AssertionError("The unselected model must not be copied")

        def calculate(self, asset, quantity, price, volume):
            raise AssertionError("The unselected model must not execute")

    selected = CountingSlippage()

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                broker.submit_order("ES", 3)

    result = Engine(
        DataFeed(prices_df=_prices()),
        Submit(),
        BacktestConfig(
            calendar=None,
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_execution_price=ExecutionPrice.CLOSE,
        ),
        slippage_model=AssetSlippage({"ES": selected, "UNUSED": Unused()}, default=NoSlippage()),
        execution_limits=SharedVolumeParticipationLimit(0.25),
    ).run()
    assert result.to_fills_dataframe()["slippage"].to_list() == [1.0, 2.0, 3.0]
    assert selected.fills == 3


def test_trade_volume_requirement_applies_only_to_the_selected_asset_model():
    from ml4t.backtest.models import AssetSlippage, NoSlippage

    class RequireVolume(NoSlippage):
        requires_traded_volume = True

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                broker.submit_order("ES", 1)
                broker.submit_order("A", 1)

    prices = _prices().with_columns(
        pl.when(pl.col("timestamp").dt.day() < 5)
        .then(None)
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    result = Engine(
        DataFeed(prices_df=prices),
        Submit(),
        BacktestConfig(
            calendar=None,
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_execution_price=ExecutionPrice.CLOSE,
        ),
        slippage_model=AssetSlippage({"ES": RequireVolume()}, default=NoSlippage()),
    ).run()
    fills = result.to_fills_dataframe()
    assert fills["asset"].to_list() == ["A", "ES"]
    assert [point.day for point in fills["timestamp"]] == [4, 5]


@pytest.mark.parametrize("kind", ["slippage", "commission"])
def test_explicit_false_valued_model_is_used_instead_of_silently_replaced_by_zero_costs(kind):
    class FalseValued:
        def __bool__(self):
            return False

        def calculate(self, asset, quantity, price, volume=None):
            return 1.0

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                broker.submit_order("A", 1)

    result = Engine(
        DataFeed(prices_df=_prices()),
        Submit(),
        BacktestConfig(
            calendar=None,
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_execution_price=ExecutionPrice.CLOSE,
        ),
        **{f"{kind}_model": FalseValued()},
    ).run()
    fills = result.to_fills_dataframe()
    assert fills["price"].to_list() == ([101.0] if kind == "slippage" else [100.0])
    assert fills["commission"].to_list() == ([0.0] if kind == "slippage" else [1.0])
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 99899.0


def test_invalid_slippage_override_fails_before_strategy_callbacks():
    seen = []

    class Read(Strategy):
        def on_data(self, *args):
            seen.append("market")

    with pytest.raises(TypeError, match="SlippageModel"):
        Engine(DataFeed(prices_df=_prices()), Read(), slippage_model=object()).run()
    assert seen == []


def test_routed_midpoint_does_not_replace_an_explicit_market_on_close_basis():
    from ml4t.backtest.models import AssetSlippage, FixedSlippage, NoSlippage
    from ml4t.backtest.types import OrderType

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                broker.submit_order("ES", 1, order_type=OrderType.MOC)

    result = Engine(
        DataFeed(prices_df=_prices(), feed_spec={"timestamp_semantics": "bar_close"}),
        Submit(),
        BacktestConfig(calendar="NYSE", execution_mode=ExecutionMode.NEXT_BAR),
        slippage_model=AssetSlippage(
            {"ES": FixedSlippage(0.5)},
            default=NoSlippage(),
            price_sources={"ES": ExecutionPrice.MID},
        ),
    ).run()
    assert result.to_fills_dataframe()["price"].to_list() == [100.5]
    assert result.to_fills_dataframe()["price_source"].to_list() == ["close"]


def test_routed_midpoint_does_not_replace_an_opening_child_intents_declared_open_price():
    from datetime import date

    from ml4t.specs import (
        AssetTarget,
        CanonicalTargetIntent,
        IntentReason,
        LifecyclePhase,
        ResidualPolicy,
        RoundingPolicy,
        TargetMeasure,
    )

    from ml4t.backtest.models import AssetSlippage, NoSlippage

    class Submit(Strategy):
        def on_prepare(self, broker, config=None):
            decision = datetime(2025, 3, 2, 23, tzinfo=UTC)
            broker.register_target_intent(
                CanonicalTargetIntent(
                    intent_id="opening",
                    decision_time=decision,
                    information_cutoff=decision,
                    effective_session=date(2025, 3, 3),
                    effective_phase=LifecyclePhase.PRE_OPEN,
                    targets=(AssetTarget("A", TargetMeasure.WEIGHT, 0.1),),
                    idempotency_key="opening-key",
                    measure=TargetMeasure.WEIGHT,
                    cash_buffer=0.0,
                    rounding=RoundingPolicy.TOWARD_ZERO,
                    residual=ResidualPolicy.KEEP_CASH,
                    reason=IntentReason.REBALANCE,
                )
            )

        def on_data(self, *args):
            pass

    result = Engine(
        DataFeed(
            prices_df=_prices().with_columns(pl.lit(90.0).alias("open")),
            feed_spec={"timestamp_semantics": "bar_close", "data_frequency": "daily"},
        ),
        Submit(),
        BacktestConfig(calendar="NYSE"),
        slippage_model=AssetSlippage(
            {}, default=NoSlippage(), price_sources={"A": ExecutionPrice.MID}
        ),
    ).run()
    fills = result.to_fills_dataframe()
    assert fills["price"].to_list() == [90.0]
    assert fills["price_source"].to_list() == ["open"]
    assert fills["quantity"].to_list() == [111.0]


@pytest.mark.parametrize("truthy", [True, False], ids=["ordinary_spec", "false_valued_spec"])
def test_partial_fills_preserve_declared_units_without_requesting_new_contract_snapshots(truthy):
    from dataclasses import dataclass

    from ml4t.backtest.execution.limits import VolumeParticipationLimit

    snapshots = []

    @dataclass
    class ProviderContract(ContractSpec):
        def __bool__(self):
            return truthy

        def __deepcopy__(self, memo):
            snapshots.append(self.symbol)
            result = type(self)(
                self.symbol,
                self.asset_class,
                self.multiplier,
                self.tick_size,
                self.margin,
                self.margin_pct,
                self.currency,
            )
            memo[id(self)] = result
            return result

    contract = ProviderContract("ES", AssetClass.FUTURE, multiplier=50.0, currency="USD")

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                broker.submit_order("ES", 2)

    engine = Engine(
        DataFeed(prices_df=_prices().filter(pl.col("timestamp").dt.day() <= 5)),
        Submit(),
        BacktestConfig(
            calendar=None,
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_execution_price=ExecutionPrice.CLOSE,
        ),
        contract_specs={"ES": contract},
        execution_limits=VolumeParticipationLimit(0.25),
    )
    owned_snapshots = list(snapshots)
    result = engine.run()
    assert [fill.price for fill in result.fills] == [100.0, 100.0]
    assert [fill.multiplier for fill in result.fills] == [50.0, 50.0]
    assert [fill.currency for fill in result.fills] == ["USD", "USD"]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [100000.0, 95000.0, 90000.0]
    assert result.to_portfolio_state_dataframe()["equity"].to_list() == [100000.0] * 3
    assert snapshots == owned_snapshots
