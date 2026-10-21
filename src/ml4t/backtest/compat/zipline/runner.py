"""Original callback signatures over one native Engine and authoritative ledger."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from ... import BacktestConfig, Broker, DataFeed, Engine, RuntimeServices, Strategy
from ..._temporal import temporal_rows
from ...config import (
    CommissionType,
    ConditionalOrderMode,
    ExecutionMode,
    ExecutionPrice,
    FillOrdering,
    SlippageType,
)
from ...execution.limits import PositiveVolumeLimit, SharedVolumeParticipationLimit
from ._scope import callback_scope
from .assets import Asset
from .data import BarData
from .finance import commission, slippage


@dataclass(frozen=True)
class PositionView:
    asset: Asset
    amount: float
    cost_basis: float
    last_sale_price: float


class _Positions(Mapping[Asset, PositionView]):
    def __init__(self, context: MigrationContext):
        self.context = context

    def __getitem__(self, asset: Asset) -> PositionView:
        self.context._validate_asset(asset)
        position = self.context._require_broker().get_position(asset.key)
        return PositionView(
            asset,
            position.quantity if position is not None else 0,
            position.commission_adjusted_cost_basis if position is not None else 0.0,
            (position.current_price if position.current_price is not None else position.entry_price)
            if position is not None
            else 0.0,
        )

    def __iter__(self) -> Iterator[Asset]:
        positions = self.context._require_broker().get_positions()
        return iter(self.context._by_key[key] for key in positions)

    def __len__(self) -> int:
        return len(self.context._require_broker().get_positions())


class _Portfolio:
    def __init__(self, context: MigrationContext):
        self._context = context
        self.positions = _Positions(context)

    @property
    def starting_cash(self) -> float:
        return self._context._config.initial_cash

    @property
    def cash(self) -> float:
        return self._context._require_broker().get_account_snapshot().cash

    @property
    def positions_value(self) -> float:
        value = self._context._require_broker().get_account_snapshot().positions_value
        assert value is not None
        return value

    @property
    def portfolio_value(self) -> float:
        return self._context._require_broker().get_account_snapshot().equity


class MigrationContext:
    """User attributes plus native-backed methods during migration callbacks."""

    def __init__(self, assets: Sequence[Asset], config: BacktestConfig) -> None:
        if not assets or any(not isinstance(asset, Asset) for asset in assets):
            raise TypeError("assets must contain explicitly declared migration Assets")
        self._by_key = {asset.key: asset for asset in assets}
        if len(self._by_key) != len(assets):
            raise ValueError("Duplicate native asset keys")
        sids = [asset.legacy_sid for asset in assets if asset.legacy_sid is not None]
        if len(set(sids)) != len(sids):
            raise ValueError("Ambiguous legacy SID mapping")
        self._config = config
        self._broker: Broker | None = None
        self._initializing = True
        self._running = False
        self._slippage: Any = slippage.VolumeShareSlippage()
        self._set_commission(commission.PerShare())
        self.portfolio = _Portfolio(self)

    def _require_broker(self) -> Broker:
        if self._broker is None:
            raise RuntimeError("Account and order APIs require a running native Engine")
        return self._broker

    def _validate_asset(self, asset: Asset) -> None:
        if not isinstance(asset, Asset) or asset.key not in self._by_key:
            raise ValueError("Asset must belong to the explicit migration mapping")

    def symbol(self, symbol_str: str) -> Asset:
        matches = [asset for asset in self._by_key.values() if asset.symbol == symbol_str]
        if len(matches) != 1:
            raise ValueError(f"Symbol {symbol_str!r} has no unique explicit asset mapping")
        return matches[0]

    def symbols(self, *symbol_strs: str) -> tuple[Asset, ...]:
        return tuple(self.symbol(name) for name in symbol_strs)

    def sid(self, sid: int) -> Asset:
        matches = [asset for asset in self._by_key.values() if asset.legacy_sid == sid]
        if len(matches) != 1:
            raise ValueError(f"SID {sid!r} has no explicit asset mapping")
        return matches[0]

    def order(self, asset: Asset, amount: float) -> str | None:
        self._validate_asset(asset)
        if not self._running:
            raise RuntimeError("Orders require a running market callback")
        if not isfinite(amount):
            raise ValueError("Order amount must be finite")
        # The historical share-order boundary rounds near integers, then truncates.
        rounded = round(amount)
        quantity = int(rounded if abs(amount - rounded) <= 0.0001 else amount)
        broker = self._require_broker()
        if quantity == 0:
            return None
        submitted = broker.submit_order(asset.key, quantity)
        return submitted.order_id if submitted is not None else None

    def record(self, **values: Any) -> None:
        if not self._running:
            raise RuntimeError("record requires a running market callback")
        self._require_broker().record(**values)

    def get_datetime(self, tz: str | None = None) -> pd.Timestamp:
        timestamp = self._require_broker().get_account_snapshot().timestamp
        if timestamp is None:
            raise RuntimeError("No simulation instant has been observed")
        stamp = pd.Timestamp(timestamp)
        if stamp.tz is None:
            stamp = stamp.tz_localize(self._config.resolved_timezone)
        return stamp.tz_convert(tz or "UTC")

    def _setup_only(self) -> None:
        if not self._initializing:
            raise RuntimeError("Cost settings must be declared during initialize")

    def _set_commission(self, model: Any) -> None:
        config = self._config
        config.commission_rate = config.commission_per_share = config.commission_per_trade = 0.0
        config.commission_minimum = 0.0
        if isinstance(model, commission.PerShare):
            config.commission_type = CommissionType.PER_ORDER_PER_SHARE
            config.commission_per_share, config.commission_minimum = (
                model.cost,
                model.min_trade_cost,
            )
        elif isinstance(model, commission.PerTrade):
            config.commission_type = CommissionType.PER_ORDER
            config.commission_per_trade = model.cost
        elif isinstance(model, commission.PerDollar):
            config.commission_type = CommissionType.PERCENTAGE
            config.commission_rate = model.cost
        else:
            raise TypeError("Use a declared migration commission model")

    def set_commission(self, model: Any) -> None:
        self._setup_only()
        self._set_commission(model)

    def set_slippage(self, model: Any) -> None:
        self._setup_only()
        if not isinstance(
            model, slippage.VolumeShareSlippage | slippage.FixedSlippage | slippage.NoSlippage
        ):
            raise TypeError("Use a declared migration slippage model")
        self._slippage = model

    def set_benchmark(self, asset: Asset | None) -> None:
        self._setup_only()
        if asset is not None:
            raise ValueError("A benchmark asset requires explicitly prepared benchmark returns")
        self._config.metadata["benchmark_status"] = "explicitly_disabled"

    def _execution_limits(self):
        config, model = self._config, self._slippage
        if isinstance(model, slippage.VolumeShareSlippage):
            config.slippage_type = SlippageType.CUMULATIVE_VOLUME_SHARE
            config.slippage_rate = model.price_impact
            if not isfinite(model.volume_limit) or not 0 < model.volume_limit <= 1:
                raise ValueError("volume_limit must be finite and in (0, 1]")
            config.metadata["execution_limits"] = {
                "model": "shared_observed_bar_volume",
                "max_participation": model.volume_limit,
            }
            return SharedVolumeParticipationLimit(max_participation=model.volume_limit)
        if isinstance(model, slippage.FixedSlippage):
            if not isfinite(model.spread) or model.spread < 0:
                raise ValueError("spread must be finite and nonnegative")
            config.slippage_type = SlippageType.FIXED
            config.slippage_fixed = model.spread / 2
        else:
            config.slippage_type = SlippageType.NONE
        config.metadata["execution_limits"] = {"model": "positive_observed_bar_volume"}
        return PositiveVolumeLimit()


class _Callbacks(Strategy):
    def __init__(
        self, context: MigrationContext, data: BarData, handle_data: Callable[..., Any] | None
    ):
        self.context, self.data, self.handle_data = context, data, handle_data

    def on_start(self, broker: Broker) -> None:
        self.context._broker = broker
        self.context._running = True

    def on_end(self, broker: Broker) -> None:
        self.context._running = False

    def on_data(self, timestamp, data, context, broker) -> None:
        if self.handle_data is not None:
            with callback_scope(self.context):
                self.handle_data(self.context, self.data)


def _performance(result) -> pd.DataFrame:
    frame = result.to_portfolio_state_dataframe()
    rows = list(temporal_rows(frame, named=True))
    index = pd.DatetimeIndex([row["timestamp"] for row in rows])
    if index.tz is None:
        index = index.tz_localize(result.config.resolved_timezone)
    performance = pd.DataFrame(
        {
            "portfolio_value": [row["equity"] for row in rows],
            "cash": [row["cash"] for row in rows],
            "returns": result.to_returns_series().to_list(),
        },
        index=index.tz_convert("UTC").as_unit("ns"),
    )
    if result.recorded_variables is not None:
        records = result.to_recorded_records()
        by_observation = {row["bar_index"]: row["values"] for row in records}
        names = set().union(*(row["values"] for row in records)) if records else set()
        for name in sorted(names):
            if name in performance.columns:
                raise ValueError(f"Recorded name {name!r} conflicts with a native account field")
            performance[name] = [
                by_observation.get(row["observation_id"], {}).get(name) for row in rows
            ]
    performance.attrs["backtest_result"] = result
    return performance


def run_algorithm(
    *,
    feed: DataFeed,
    assets: Sequence[Asset],
    initialize: Callable[..., Any] | None = None,
    handle_data: Callable[..., Any] | None = None,
    analyze: Callable[..., Any] | None = None,
    capital_base: float = 100000.0,
    runtime: RuntimeServices | None = None,
) -> pd.DataFrame:
    """Run imported callback functions on explicit bundle-free native inputs.

    Import migration API and finance names from this namespace. No legacy package,
    bundle discovery, network download or module rewriting occurs. Runtime bounds
    are physical instants, with pre-start rows available for warmup history.
    """
    config = BacktestConfig(
        feed_spec=feed.feed_spec,
        initial_cash=capital_base,
        allow_short_selling=True,
        allow_leverage=True,
        skip_cash_validation=True,
        partial_fills_allowed=True,
        execution_mode=ExecutionMode.NEXT_BAR,
        execution_price=ExecutionPrice.CLOSE,
        next_bar_execution_price=ExecutionPrice.CLOSE,
        conditional_order_mode=ConditionalOrderMode.EXECUTION_PRICE,
        fill_ordering=FillOrdering.FIFO,
        metadata={"migration_adapter": "zipline", "benchmark_status": "not_supplied"},
    )
    context = MigrationContext(assets, config)
    if initialize is not None:
        with callback_scope(context):
            initialize(context)
    context._initializing = False
    limits = context._execution_limits()
    runtime = runtime if runtime is not None else RuntimeServices()
    result = Engine(
        feed,
        _Callbacks(context, BarData(runtime), handle_data),
        config,
        runtime=runtime,
        execution_limits=limits,
        holdings_sampling="bar",
    ).run()
    performance = _performance(result)
    if analyze is not None:
        with callback_scope(context):
            analyze(context, performance)
    return performance
