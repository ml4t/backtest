"""Original callback signatures over one native Engine and authoritative ledger."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite
from typing import Any
from uuid import uuid4

import pandas as pd

from ... import BacktestConfig, Broker, DataFeed, Engine, RuntimeServices, Strategy
from ...config import (
    CommissionType,
    ConditionalOrderMode,
    ExecutionMode,
    ExecutionPrice,
    FillOrdering,
    SlippageType,
)
from ...controls import (
    AccountConstraints,
    AccountControlContext,
    OrderConstraints,
    OrderRequestCount,
    TradingControl,
)
from ...core.shared import SubmitOrderOptions
from ...execution.limits import PositiveVolumeLimit, SharedVolumeParticipationLimit
from ...execution.sizing import quantity_for_value, quantity_to_target, round_quantity
from ...types import Order, OrderSide, OrderStatus
from ._scope import callback_scope
from .assets import Asset
from .controls import _register_account_controls, _TradingRule
from .data import BarData
from .events import _DateRule, _register, _Schedule, _TimeRule, date_rules, time_rules
from .finance import commission, slippage
from .finance.execution import _native_style
from .finance.order import ORDER_STATUS, OrderView
from .results import to_performance_frame


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
        self._order_prefix = uuid4().hex
        self._broker: Broker | None = None
        self._initializing = True
        self._running = False
        self._in_before_trading_start = False
        self._schedules: list[_Schedule] = []
        self._trading_controls: list[TradingControl] = []
        self._account_controls: list[tuple[str, float, timedelta | None]] = []
        self._slippage: Any = slippage.VolumeShareSlippage()
        self._set_commission(commission.PerShare())
        self.portfolio = _Portfolio(self)

    @property
    def account(self) -> AccountControlContext:
        return self._require_broker().get_account_snapshot()

    def set_max_leverage(self, max_leverage: float) -> None:
        self._setup_only()
        AccountConstraints(max_leverage=max_leverage)
        self._account_controls.append(("maximum", max_leverage, None))
        self._config.metadata.setdefault("migration_account_controls", []).append(
            {"max_leverage": max_leverage, "boundary": "session_end"}
        )

    def set_min_leverage(self, min_leverage: float, grace_period: timedelta) -> None:
        self._setup_only()
        if not isinstance(grace_period, timedelta) or not isfinite(grace_period.total_seconds()):
            raise ValueError("grace_period must be a finite timedelta")
        self._account_controls.append(("minimum", min_leverage, grace_period))
        self._config.metadata.setdefault("migration_account_controls", []).append(
            {
                "min_leverage": min_leverage,
                "grace_seconds": grace_period.total_seconds(),
                "deadline_origin": "start_session_midnight_UTC",
                "boundary": "session_end",
            }
        )

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

    def order(
        self,
        asset: Asset,
        amount: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        style: Any | None = None,
    ) -> str | None:
        self._require_order_phase()
        self._validate_asset(asset)
        kind, limit_price, stop_price = _native_style(style, limit_price, stop_price)
        quantity = round_quantity(amount, rounding="near_integer")
        broker = self._require_broker()
        submitted = broker.submit_order(
            asset.key,
            quantity,
            order_type=kind,
            limit_price=limit_price,
            stop_price=stop_price,
            _options=SubmitOrderOptions(validate_zero_quantity=True),
        )
        return self._external_order_id(submitted.order_id) if submitted is not None else None

    def _require_order_phase(self) -> None:
        if self._in_before_trading_start:
            raise RuntimeError("Orders are disallowed in before_trading_start")
        if not self._running:
            raise RuntimeError("Orders require a running market callback")

    def _value_quantity(self, asset: Asset, value: float) -> float | None:
        self._require_order_phase()
        self._validate_asset(asset)
        broker = self._require_broker()
        price = broker.get_last_price(asset.key)
        return (
            None
            if price is None
            else quantity_for_value(value, price, broker.get_multiplier(asset.key))
        )

    def order_value(
        self,
        asset: Asset,
        value: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        style: Any | None = None,
    ) -> str | None:
        amount = self._value_quantity(asset, value)
        return None if amount is None else self.order(asset, amount, limit_price, stop_price, style)

    def order_percent(
        self,
        asset: Asset,
        percent: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        style: Any | None = None,
    ) -> str | None:
        return self.order_value(
            asset, self._require_broker().equity() * percent, limit_price, stop_price, style
        )

    def order_target(
        self,
        asset: Asset,
        target: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        style: Any | None = None,
    ) -> str | None:
        self._require_order_phase()
        self._validate_asset(asset)
        position = self._require_broker().get_position(asset.key)
        amount = quantity_to_target(
            target, position.quantity if position is not None else 0, rounding="near_integer"
        )
        return self.order(asset, amount, limit_price, stop_price, style)

    def order_target_value(
        self,
        asset: Asset,
        target: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        style: Any | None = None,
    ) -> str | None:
        amount = self._value_quantity(asset, target)
        return (
            None
            if amount is None
            else self.order_target(asset, amount, limit_price, stop_price, style)
        )

    def order_target_percent(
        self,
        asset: Asset,
        target: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        style: Any | None = None,
    ) -> str | None:
        return self.order_target_value(
            asset, self._require_broker().equity() * target, limit_price, stop_price, style
        )

    def _control(self, constraint: TradingControl, on_error: str, metadata: dict[str, Any]) -> None:
        self._setup_only()
        if on_error not in {"fail", "log"}:
            raise ValueError("on_error must be fail or log")
        self._trading_controls.append(_TradingRule(constraint, on_error))
        self._config.metadata.setdefault("migration_trading_controls", []).append(
            {**metadata, "on_error": on_error}
        )

    def set_max_order_count(self, max_count: int, on_error: str = "fail") -> None:
        self._control(
            OrderRequestCount(max_count, timezone="UTC"),
            on_error,
            {"method": "validated_requests", "max_count": max_count, "timezone": "UTC"},
        )

    def set_max_order_size(
        self,
        asset: Asset | None = None,
        max_shares: float | None = None,
        max_notional: float | None = None,
        on_error: str = "fail",
    ) -> None:
        if max_shares is None and max_notional is None:
            raise ValueError("Supply max_shares or max_notional")
        if asset is not None:
            self._validate_asset(asset)
        key = None if asset is None else asset.key
        self._control(
            OrderConstraints(
                asset=key,
                max_order_quantity=max_shares,
                max_order_notional=max_notional,
                include_pending=False,
                notional_basis="quoted_units",
            ),
            on_error,
            {
                "method": "max_order_size",
                "asset": key,
                "max_shares": max_shares,
                "max_notional": max_notional,
                "notional_basis": "quoted_units",
            },
        )

    def set_max_position_size(
        self,
        asset: Asset | None = None,
        max_shares: float | None = None,
        max_notional: float | None = None,
        on_error: str = "fail",
    ) -> None:
        if max_shares is None and max_notional is None:
            raise ValueError("Supply max_shares or max_notional")
        if asset is not None:
            self._validate_asset(asset)
        key = None if asset is None else asset.key
        self._control(
            OrderConstraints(
                asset=key,
                max_position_quantity=max_shares,
                max_position_notional=max_notional,
                include_pending=False,
                notional_basis="quoted_units",
            ),
            on_error,
            {
                "method": "max_position_size",
                "asset": key,
                "max_shares": max_shares,
                "max_notional": max_notional,
                "include_pending": False,
                "notional_basis": "quoted_units",
            },
        )

    def set_long_only(self, on_error: str = "fail") -> None:
        self._control(
            OrderConstraints(long_only=True, include_pending=False),
            on_error,
            {"method": "long_only", "include_pending": False},
        )

    def schedule_function(
        self,
        func: Callable[..., Any],
        date_rule: _DateRule | _TimeRule | None = None,
        time_rule: _TimeRule | None = None,
        half_days: bool = True,
        calendar: str | None = None,
    ) -> None:
        if not self._initializing:
            raise RuntimeError("Register callbacks during initialize")
        if not callable(func):
            raise TypeError("Scheduled callback must be callable")
        if isinstance(date_rule, _TimeRule) and time_rule is None:
            warnings.warn("Pass a standalone time rule with time_rule=", UserWarning, stacklevel=2)
            time_rule, date_rule = date_rule, None
        date_rule = date_rule if date_rule is not None else date_rules.every_day()
        time_rule = time_rule if time_rule is not None else time_rules.every_minute()
        if not isinstance(date_rule, _DateRule) or not isinstance(time_rule, _TimeRule):
            raise TypeError("Use migration date_rules/time_rules factories")
        if not isinstance(half_days, bool):
            raise TypeError("half_days must be boolean")
        self._schedules.append(_Schedule(func, date_rule, time_rule, half_days, calendar))

    def _external_order_id(self, native: str) -> str:
        return f"{self._order_prefix}:{native}"

    def _native_order_id(self, external: str) -> str | None:
        if not isinstance(external, str):
            raise TypeError("order_id must be the migration string identifier")
        prefix, delimiter, native = external.partition(":")
        return native if delimiter and prefix == self._order_prefix and native else None

    def _order_view(self, order: Order) -> OrderView:
        broker = self._require_broker()
        events = [order.created_at, order.filled_at, order.cancelled_at]
        fills = broker.get_fills(order.order_id)
        if fills:
            events.append(fills[-1].timestamp)
        observed = [pd.Timestamp(t) for t in events if t is not None]

        def utc(value):
            if value is None:
                return None
            stamp = pd.Timestamp(value)
            if stamp.tz is None:
                stamp = stamp.tz_localize(self._config.resolved_timezone)
            return stamp.tz_convert("UTC")

        sign = 1 if order.side is OrderSide.BUY else -1
        statuses = {
            OrderStatus.PENDING: ORDER_STATUS.OPEN,
            OrderStatus.FILLED: ORDER_STATUS.FILLED,
            OrderStatus.CANCELLED: ORDER_STATUS.CANCELLED,
            OrderStatus.REJECTED: ORDER_STATUS.REJECTED,
        }
        assert order.requested_quantity is not None
        return OrderView(
            self._external_order_id(order.order_id),
            order.order_id,
            self._by_key[order.asset],
            sign * order.requested_quantity,
            sign * order.filled_quantity,
            order.commission_paid,
            statuses[order.status],
            utc(order.created_at),
            utc(max(observed)) if observed else None,
            order.limit_price,
            order.stop_price,
            order.stop_triggered,
            order.rejection_reason,
        )

    def get_order(self, order_id: str) -> OrderView | None:
        native = self._native_order_id(order_id)
        if native is None:
            return None
        order = self._require_broker().get_order(native)
        return self._order_view(order) if order is not None else None

    def get_open_orders(
        self, asset: Asset | None = None
    ) -> dict[Asset, list[OrderView]] | list[OrderView]:
        if asset is not None:
            self._validate_asset(asset)
        orders = self._require_broker().get_pending_orders(None if asset is None else asset.key)
        if asset is not None:
            return [self._order_view(order) for order in orders]
        groups: dict[Asset, list[OrderView]] = {}
        for order in orders:
            groups.setdefault(self._by_key[order.asset], []).append(self._order_view(order))
        return groups

    def cancel_order(self, order_param: str | OrderView) -> None:
        if not self._running:
            raise RuntimeError("Cancellation requires a running market callback")
        external = order_param.id if isinstance(order_param, OrderView) else order_param
        native = self._native_order_id(external)
        if native is not None:
            self._require_broker().cancel_order(native)

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
            raise RuntimeError("Settings must be declared during initialize")

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
        self,
        context: MigrationContext,
        data: BarData,
        handle_data: Callable[..., Any] | None,
        runtime: RuntimeServices,
        before_trading_start: Callable[..., Any] | None,
    ):
        self.context, self.data, self.handle_data = context, data, handle_data
        self.runtime, self.before_trading_start = runtime, before_trading_start

    def on_start(self, broker: Broker) -> None:
        self.context._broker = broker
        self.context._running = True
        for control in self.context._trading_controls:
            broker.register_trading_control(control)
        _register_account_controls(self.context, self.runtime)
        _register(self.context, self.runtime, self.data, self.before_trading_start)

    def on_end(self, broker: Broker) -> None:
        self.context._running = False

    def on_data(self, timestamp, data, context, broker) -> None:
        if self.handle_data is not None:
            with callback_scope(self.context):
                self.handle_data(self.context, self.data)


def run_algorithm(
    *,
    feed: DataFeed,
    assets: Sequence[Asset],
    initialize: Callable[..., Any] | None = None,
    handle_data: Callable[..., Any] | None = None,
    analyze: Callable[..., Any] | None = None,
    before_trading_start: Callable[..., Any] | None = None,
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
        metadata={
            "migration_adapter": "zipline",
            "benchmark_status": "not_supplied",
            "quantity_rounding": "near_integer",
            "target_pending_orders": "ignored",
        },
    )
    context = MigrationContext(assets, config)
    config.metadata["migration_result"] = {
        "assets": [
            {"key": asset.key, "symbol": asset.symbol, "legacy_sid": asset.legacy_sid}
            for asset in assets
        ],
        "order_id_prefix": context._order_prefix,
    }
    if initialize is not None:
        with callback_scope(context):
            initialize(context)
    context._initializing = False
    limits = context._execution_limits()
    runtime = runtime if runtime is not None else RuntimeServices()
    result = Engine(
        feed,
        _Callbacks(context, BarData(runtime), handle_data, runtime, before_trading_start),
        config,
        runtime=runtime,
        execution_limits=limits,
        holdings_sampling="bar",
    ).run()
    performance = to_performance_frame(result)
    if analyze is not None:
        with callback_scope(context):
            analyze(context, performance)
    return performance
