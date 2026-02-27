from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest.config import (
    BacktestConfig,
    CommissionModel,
    FillOrdering,
    FillTiming,
    RebalanceMode,
    ShareType,
    SlippageModel,
)
from ml4t.backtest.engine import run_backtest
from ml4t.backtest.execution import RebalanceConfig, TargetWeightExecutor
from ml4t.backtest.strategy import Strategy
from ml4t.backtest.types import ExecutionMode


def _bar(ts: datetime, asset: str, close: float, volume: float = 1_000_000.0) -> dict:
    return {
        "timestamp": ts,
        "asset": asset,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
    }


def _qty_for_symbol(result, symbol: str) -> float:
    for trade in result.trades:
        if trade.symbol == symbol and trade.status == "open":
            return abs(trade.quantity)
    return 0.0


class _RebalanceByMode(Strategy):
    def __init__(self, mode: RebalanceMode) -> None:
        self.mode = mode
        self.bar = 0
        self.msft_order_qty = 0.0
        self.executor = TargetWeightExecutor(
            RebalanceConfig(
                rebalance_mode=mode,
                min_trade_value=0.0,
                min_weight_change=0.0,
                allow_fractional=False,
            )
        )

    def on_data(self, timestamp, data, context, broker) -> None:
        self.bar += 1
        if self.bar == 1:
            broker.submit_order("AAPL", 1900.0)
        elif self.bar == 2:
            orders = self.executor.execute({"AAPL": 0.5, "MSFT": 0.5}, data, broker)
            for order in orders:
                if order.asset == "MSFT":
                    self.msft_order_qty = order.quantity


def test_snapshot_value_freezes_targets_vs_incremental_recompute() -> None:
    start = datetime(2024, 1, 1)
    prices = pl.DataFrame(
        [
            _bar(start, "AAPL", 100.0),
            _bar(start, "MSFT", 100.0),
            _bar(start + timedelta(days=1), "AAPL", 100.0),
            _bar(start + timedelta(days=1), "MSFT", 100.0),
        ]
    )
    cfg = BacktestConfig(
        initial_cash=200_000.0,
        fill_timing=FillTiming.SAME_BAR,
        execution_mode=ExecutionMode.SAME_BAR,
        share_type=ShareType.INTEGER,
        fill_ordering=FillOrdering.FIFO,
        commission_model=CommissionModel.PERCENTAGE,
        commission_rate=0.01,
        slippage_model=SlippageModel.NONE,
    )

    snapshot_strategy = _RebalanceByMode(RebalanceMode.SNAPSHOT)
    incremental_strategy = _RebalanceByMode(RebalanceMode.INCREMENTAL)
    run_backtest(prices=prices, strategy=snapshot_strategy, config=cfg)
    run_backtest(prices=prices, strategy=incremental_strategy, config=cfg)

    assert snapshot_strategy.msft_order_qty > incremental_strategy.msft_order_qty


class _RotateSellThenBuy(Strategy):
    def __init__(self) -> None:
        self.bar = 0

    def on_data(self, timestamp, data, context, broker) -> None:
        self.bar += 1
        if self.bar == 1:
            broker.submit_order("AAPL", 100.0)
        elif self.bar == 2:
            broker.rebalance_to_weights({"MSFT": 1.0})


def test_sell_before_buy_rotation_works_under_tight_cash() -> None:
    start = datetime(2024, 1, 1)
    prices = pl.DataFrame(
        [
            _bar(start, "AAPL", 100.0),
            _bar(start, "MSFT", 100.0),
            _bar(start + timedelta(days=1), "AAPL", 100.0),
            _bar(start + timedelta(days=1), "MSFT", 100.0),
        ]
    )
    cfg = BacktestConfig(
        initial_cash=10_000.0,
        fill_timing=FillTiming.SAME_BAR,
        execution_mode=ExecutionMode.SAME_BAR,
        share_type=ShareType.INTEGER,
        fill_ordering=FillOrdering.FIFO,
        commission_model=CommissionModel.NONE,
        slippage_model=SlippageModel.NONE,
    )
    result = run_backtest(prices=prices, strategy=_RotateSellThenBuy(), config=cfg)

    assert _qty_for_symbol(result, "AAPL") == 0.0
    assert _qty_for_symbol(result, "MSFT") > 0.0


class _MissingBarsRebalance(Strategy):
    def __init__(self) -> None:
        self.bar = 0
        self.orders_by_bar: dict[int, list[str]] = {}

    def on_data(self, timestamp, data, context, broker) -> None:
        self.bar += 1
        if self.bar == 1:
            broker.submit_order("MSFT", 10.0)
        else:
            orders = broker.rebalance_to_weights({"AAPL": 0.5, "MSFT": 0.5})
            self.orders_by_bar[self.bar] = [o.asset for o in orders]


def test_missing_bar_skips_orders_for_unpriced_asset() -> None:
    start = datetime(2024, 1, 1)
    prices = pl.DataFrame(
        [
            _bar(start, "AAPL", 100.0),
            _bar(start, "MSFT", 100.0),
            _bar(start + timedelta(days=1), "AAPL", 101.0),  # MSFT missing this bar
            _bar(start + timedelta(days=2), "AAPL", 102.0),
            _bar(start + timedelta(days=2), "MSFT", 102.0),
        ]
    )
    strategy = _MissingBarsRebalance()
    cfg = BacktestConfig(
        initial_cash=20_000.0,
        fill_timing=FillTiming.SAME_BAR,
        execution_mode=ExecutionMode.SAME_BAR,
        share_type=ShareType.INTEGER,
        fill_ordering=FillOrdering.FIFO,
        commission_model=CommissionModel.NONE,
        slippage_model=SlippageModel.NONE,
    )
    run_backtest(prices=prices, strategy=strategy, config=cfg)

    # On bar 2 MSFT has no price, so no MSFT rebalance order should be submitted.
    assert "MSFT" not in strategy.orders_by_bar.get(2, [])


class _LateAssetRebalance(Strategy):
    def on_data(self, timestamp, data, context, broker) -> None:
        broker.rebalance_to_weights({"AAPL": 0.5, "MSFT": 0.5})


def test_late_asset_start_only_trades_after_first_price() -> None:
    start = datetime(2024, 1, 1)
    msft_start = start + timedelta(days=2)
    prices = pl.DataFrame(
        [
            _bar(start, "AAPL", 100.0),
            _bar(start + timedelta(days=1), "AAPL", 101.0),
            _bar(msft_start, "AAPL", 102.0),
            _bar(msft_start, "MSFT", 50.0),  # late-start asset
            _bar(start + timedelta(days=3), "AAPL", 103.0),
            _bar(start + timedelta(days=3), "MSFT", 51.0),
        ]
    )
    cfg = BacktestConfig(
        initial_cash=10_000.0,
        fill_timing=FillTiming.SAME_BAR,
        execution_mode=ExecutionMode.SAME_BAR,
        share_type=ShareType.INTEGER,
        fill_ordering=FillOrdering.FIFO,
        commission_model=CommissionModel.NONE,
        slippage_model=SlippageModel.NONE,
    )
    result = run_backtest(prices=prices, strategy=_LateAssetRebalance(), config=cfg)

    msft_fills = [f for f in result.fills if f.asset == "MSFT"]
    assert msft_fills
    assert min(f.timestamp for f in msft_fills) >= msft_start
