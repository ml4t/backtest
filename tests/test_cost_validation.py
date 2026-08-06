from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from ml4t.backtest import (
    BacktestConfig,
    DataFeed,
    Engine,
    ExecutionMode,
    OrderSide,
    Strategy,
    run_backtest,
)
from ml4t.backtest.execution.result import ExecutionResult


class _NoOpStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker) -> None:
        return None


class _BuyOnceStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False
        self.broker_snapshot: dict[str, Any] | None = None

    def on_data(self, timestamp, data, context, broker) -> None:
        if self.submitted:
            return
        broker.submit_order("AAPL", 10)
        self.submitted = True
        self.broker_snapshot = _financial_snapshot(broker)


class _RoundTripStrategy(Strategy):
    def __init__(self) -> None:
        self.entered = False
        self.exit_submitted = False
        self.broker_snapshot: dict[str, Any] | None = None

    def on_data(self, timestamp, data, context, broker) -> None:
        position = broker.get_position("AAPL")
        if not self.entered:
            broker.submit_order("AAPL", 10)
            self.entered = True
        elif position is not None and not self.exit_submitted:
            broker.close_position("AAPL")
            self.exit_submitted = True
            self.broker_snapshot = _financial_snapshot(broker)


class _FlipStrategy(Strategy):
    def __init__(self, invalid_commission: _ConstantCommission) -> None:
        self.invalid_commission = invalid_commission
        self.entered = False
        self.flip_submitted = False
        self.broker_snapshot: dict[str, Any] | None = None

    def on_data(self, timestamp, data, context, broker) -> None:
        position = broker.get_position("AAPL")
        if not self.entered:
            broker.submit_order("AAPL", 10)
            self.entered = True
        elif position is not None and not self.flip_submitted:
            broker.submit_order("AAPL", 20, OrderSide.SELL)
            broker.commission_model = self.invalid_commission
            broker.gatekeeper.commission_model = self.invalid_commission
            broker.skip_cash_validation = True
            self.flip_submitted = True
            self.broker_snapshot = _financial_snapshot(broker)


class _ConstantCommission:
    def __init__(self, value: float) -> None:
        self.value = value

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        return self.value


class _SnapshotCommission(_ConstantCommission):
    def __init__(self, value: float) -> None:
        super().__init__(value)
        self.broker = None
        self.broker_snapshot: dict[str, Any] | None = None

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        self.broker_snapshot = _financial_snapshot(self.broker)
        return super().calculate(asset, quantity, price)


class _ConstantSlippage:
    def __init__(self, value: float) -> None:
        self.value = value

    def calculate(
        self,
        asset: str,
        quantity: float,
        price: float,
        volume: float | None,
    ) -> float:
        return self.value


class _ConstantImpact:
    def __init__(self, value: float) -> None:
        self.value = value

    def calculate(
        self,
        quantity: float,
        price: float,
        volume: float | None,
        is_buy: bool,
    ) -> float:
        return self.value


class _InvalidSellPriceImpact:
    def __init__(self) -> None:
        self.broker = None
        self.broker_snapshot: dict[str, Any] | None = None

    def calculate(
        self,
        quantity: float,
        price: float,
        volume: float | None,
        is_buy: bool,
    ) -> float:
        if not is_buy:
            self.broker_snapshot = _financial_snapshot(self.broker)
        return 0.0 if is_buy else -200.0


class _InvalidExecutionLimits:
    def __init__(self, value: float) -> None:
        self.value = value

    def calculate(
        self,
        order_quantity: float,
        bar_volume: float | None,
        price: float,
    ) -> ExecutionResult:
        return ExecutionResult(
            fillable_quantity=self.value,
            remaining_quantity=order_quantity,
            adjusted_price=price,
        )


def _prices() -> pl.DataFrame:
    start = datetime(2024, 1, 2)
    timestamps = [start + timedelta(days=offset) for offset in range(4)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["AAPL"] * len(timestamps),
            "open": [100.0] * len(timestamps),
            "high": [100.0] * len(timestamps),
            "low": [100.0] * len(timestamps),
            "close": [100.0] * len(timestamps),
            "volume": [1_000_000.0] * len(timestamps),
        }
    )


def _financial_snapshot(broker) -> dict[str, Any]:
    return {
        "cash": broker.cash,
        "account_cash": broker.account.cash,
        "positions": copy.deepcopy(broker.positions),
        "account_positions": copy.deepcopy(broker.account.positions),
        "orders": copy.deepcopy(broker.orders),
        "pending_orders": copy.deepcopy(broker.pending_orders),
        "fills": copy.deepcopy(broker.fills),
        "trades": copy.deepcopy(broker.trades),
        "partial_orders": copy.deepcopy(broker._partial_orders),
        "filled_this_bar": copy.deepcopy(broker._filled_this_bar),
    }


@pytest.mark.parametrize(
    "field",
    [
        "commission_rate",
        "commission_per_share",
        "commission_per_trade",
        "commission_minimum",
        "slippage_rate",
        "slippage_fixed",
        "slippage_spread",
        "stop_slippage_rate",
    ],
)
@pytest.mark.parametrize("value", [-0.01, math.nan, math.inf])
def test_engine_rejects_invalid_builtin_cost_config(field: str, value: float) -> None:
    config = BacktestConfig(**{field: value})

    with pytest.raises(ValueError, match=field):
        Engine(DataFeed(prices_df=_prices()), _NoOpStrategy(), config)


@pytest.mark.parametrize("value", [-0.01, math.nan, math.inf])
def test_engine_rejects_invalid_asset_spread(value: float) -> None:
    config = BacktestConfig(slippage_spread_by_asset={"AAPL": value})

    with pytest.raises(ValueError, match=r"slippage_spread_by_asset\['AAPL'\]"):
        Engine(DataFeed(prices_df=_prices()), _NoOpStrategy(), config)


def test_run_backtest_enforces_config_validation() -> None:
    config = BacktestConfig(commission_per_trade=-10.0)

    with pytest.raises(ValueError, match=r"commission_per_trade.*-10\.0"):
        run_backtest(_prices(), _NoOpStrategy(), config=config)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_invalid_custom_commission_is_fail_atomic(value: float) -> None:
    strategy = _BuyOnceStrategy()
    engine = Engine(DataFeed(prices_df=_prices()), strategy)
    model = _ConstantCommission(value)
    engine.broker.commission_model = model
    engine.broker.gatekeeper.commission_model = model

    with pytest.raises(ValueError, match=r"commission.*_ConstantCommission"):
        engine.run()

    assert strategy.broker_snapshot is not None
    assert _financial_snapshot(engine.broker) == strategy.broker_snapshot


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_invalid_custom_slippage_is_fail_atomic(value: float) -> None:
    strategy = _BuyOnceStrategy()
    engine = Engine(DataFeed(prices_df=_prices()), strategy)
    engine.broker.slippage_model = _ConstantSlippage(value)

    with pytest.raises(ValueError, match=r"slippage.*_ConstantSlippage"):
        engine.run()

    assert strategy.broker_snapshot is not None
    assert _financial_snapshot(engine.broker) == strategy.broker_snapshot


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_invalid_custom_impact_is_fail_atomic(value: float) -> None:
    strategy = _BuyOnceStrategy()
    engine = Engine(
        DataFeed(prices_df=_prices()),
        strategy,
        market_impact_model=_ConstantImpact(value),
    )

    with pytest.raises(ValueError, match=r"market impact.*_ConstantImpact"):
        engine.run()

    assert strategy.broker_snapshot is not None
    assert _financial_snapshot(engine.broker) == strategy.broker_snapshot


def test_invalid_execution_price_is_fail_atomic() -> None:
    strategy = _RoundTripStrategy()
    impact = _InvalidSellPriceImpact()
    engine = Engine(
        DataFeed(prices_df=_prices()),
        strategy,
        BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
        market_impact_model=impact,
    )
    impact.broker = engine.broker

    with pytest.raises(ValueError, match=r"execution price.*-100\.0"):
        engine.run()

    assert impact.broker_snapshot is not None
    assert _financial_snapshot(engine.broker) == impact.broker_snapshot


def test_invalid_flip_commission_is_fail_atomic() -> None:
    invalid_commission = _SnapshotCommission(math.nan)
    strategy = _FlipStrategy(invalid_commission)
    engine = Engine(
        DataFeed(prices_df=_prices()),
        strategy,
        BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
    )
    invalid_commission.broker = engine.broker

    with pytest.raises(ValueError, match=r"commission.*_SnapshotCommission"):
        engine.run()

    assert invalid_commission.broker_snapshot is not None
    assert _financial_snapshot(engine.broker) == invalid_commission.broker_snapshot


@pytest.mark.parametrize("value", [-0.5, math.nan, math.inf])
def test_invalid_execution_limit_quantity_is_fail_atomic(value: float) -> None:
    strategy = _BuyOnceStrategy()
    limits = _InvalidExecutionLimits(value)
    engine = Engine(
        DataFeed(prices_df=_prices()),
        strategy,
        BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
        execution_limits=limits,
    )

    with pytest.raises(ValueError, match=r"execution quantity.*_InvalidExecutionLimits"):
        engine.run()

    assert strategy.broker_snapshot is not None
    assert _financial_snapshot(engine.broker) == strategy.broker_snapshot


def test_non_positive_base_execution_price_is_rejected() -> None:
    prices = _prices().with_columns(
        pl.when(pl.col("timestamp") == pl.col("timestamp").min())
        .then(pl.col("open"))
        .otherwise(0.0)
        .alias("open")
    )
    strategy = _BuyOnceStrategy()
    engine = Engine(
        DataFeed(prices_df=prices),
        strategy,
        BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
    )

    with pytest.raises(ValueError, match=r"base execution price.*0\.0"):
        engine.run()
