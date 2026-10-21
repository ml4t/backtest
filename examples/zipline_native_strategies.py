"""Native Strategy ports of the published Zipline 3.2.0 OLMAR and dual-MA examples.

Copyright 2014 Quantopian, Inc. Derived calculations are Apache-2.0,
with original notices retained in
LICENSE.zipline and tests/fixtures/zipline_original/. These demonstrations use
explicit native history, integer sizing and execution settings. They are examples,
not additional public strategy templates or a dependency on Zipline.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl

from ml4t.backtest import BacktestConfig, RuntimeServices, Strategy
from ml4t.backtest.config import (
    CommissionType,
    ConditionalOrderMode,
    ExecutionMode,
    ExecutionPrice,
    FillOrdering,
    SlippageType,
)
from ml4t.backtest.execution.limits import SharedVolumeParticipationLimit
from ml4t.backtest.execution.sizing import quantity_to_target


def reference_execution(
    commission_per_share: float = 0.0075,
) -> tuple[BacktestConfig, SharedVolumeParticipationLimit]:
    """Select the examples' close fills, FIFO, fee minimum and volume impact."""
    config = BacktestConfig(
        initial_cash=100000,
        allow_short_selling=True,
        allow_leverage=True,
        skip_cash_validation=True,
        partial_fills_allowed=True,
        execution_mode=ExecutionMode.NEXT_BAR,
        execution_price=ExecutionPrice.CLOSE,
        next_bar_execution_price=ExecutionPrice.CLOSE,
        conditional_order_mode=ConditionalOrderMode.EXECUTION_PRICE,
        fill_ordering=FillOrdering.FIFO,
        commission_type=CommissionType.PER_ORDER_PER_SHARE,
        commission_per_share=commission_per_share,
        commission_minimum=1,
        slippage_type=SlippageType.CUMULATIVE_VOLUME_SHARE,
        slippage_rate=0.1,
    )
    return config, SharedVolumeParticipationLimit(max_participation=0.025)


class DualMovingAverage(Strategy):
    """Use native long-form windows, recording the original three observed series."""

    def __init__(self, runtime: RuntimeServices, asset: str) -> None:
        self.runtime, self.asset = runtime, asset

    def on_start(self, broker) -> None:
        self.observations = 0

    def on_data(self, timestamp, data, context, broker) -> None:
        self.observations += 1
        if self.observations < 300:
            return
        short = self.runtime.market.history(
            [self.asset], ["close"], bar_count=100, unit="expected_bar"
        )["close"].mean()
        long = self.runtime.market.history(
            [self.asset], ["close"], bar_count=300, unit="expected_bar"
        )["close"].mean()
        held = broker.get_position(self.asset)
        quantity = held.quantity if held is not None else 0
        if short != long:
            target = 100 if short > long else 0
            delta = quantity_to_target(target, quantity, rounding="near_integer")
            if delta:
                broker.submit_order(self.asset, delta)
        price = self.runtime.market.current([self.asset], ["close"])["close"][0]
        broker.record(AAPL=price, short_mavg=short, long_mavg=long)


def _simplex_projection(values: np.ndarray) -> np.ndarray:
    """Duchi projection, Python port copyright 2013 Thomas Wiecki, Apache-2.0."""
    values = (values > 0) * values
    descending = np.sort(values)[::-1]
    cumulative = np.cumsum(descending)
    rho = np.where(descending > (cumulative - 1) / np.arange(1, len(values) + 1))[0][-1]
    theta = max(0, (cumulative[rho] - 1) / (rho + 1))
    return np.maximum(values - theta, 0)


class OnlineMovingAverageReversion(Strategy):
    """Native OLMAR example, using the original rounded per-asset order deltas."""

    def __init__(
        self,
        runtime: RuntimeServices,
        assets: Sequence[str],
        *,
        epsilon: float = 1.05,
        window: int = 5,
    ) -> None:
        self.runtime, self.assets = runtime, tuple(assets)
        self.epsilon, self.window = epsilon, window

    def on_start(self, broker) -> None:
        self.observations = 0
        self.weights = np.ones(len(self.assets)) / len(self.assets)
        self.initial = True
        self.starting_cash = broker.get_cash()

    def on_data(self, timestamp, data, context, broker) -> None:
        self.observations += 1
        if self.observations < self.window:
            return
        current = self.runtime.market.current(self.assets, ["close"])
        prices_by_asset = dict(current.select("asset", "close").iter_rows())
        prices = np.array([prices_by_asset[asset] for asset in self.assets])
        if not self.initial:
            history = self.runtime.market.history(
                self.assets, ["close"], bar_count=self.window, unit="expected_bar"
            )
            averages = dict(history.group_by("asset").agg(pl.col("close").mean()).iter_rows())
            relatives = np.array([averages[asset] for asset in self.assets]) / prices
            deviation = relatives - relatives.mean()
            variance = np.linalg.norm(deviation) ** 2
            step = (
                max(0, (self.epsilon - np.dot(self.weights, relatives)) / variance)
                if variance
                else 0
            )
            self.weights = _simplex_projection(self.weights + step * deviation)
        state = broker.get_account_snapshot()
        capital = self.starting_cash if self.initial else state.positions_value + state.cash
        desired = np.round(self.weights * capital / prices)
        for asset, target in zip(self.assets, desired, strict=True):
            position = broker.get_position(asset)
            held = position.quantity if position is not None else 0
            if target != held:
                broker.submit_order(asset, float(target - held))
        self.initial = False
