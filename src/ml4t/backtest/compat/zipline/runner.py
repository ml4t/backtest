"""Original callback signatures over one native Engine and authoritative ledger."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from math import isfinite
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from ml4t.specs import WindowUnit

from ... import BacktestConfig, Broker, DataFeed, Engine, RuntimeServices, Strategy
from ...benchmark import BenchmarkData
from ...config import (
    CancelPolicy,
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
from ...models import AssetCommission, PercentageCommission, PerOrderCommission
from ...sessions import SessionConfig
from ...types import AssetClass, ContractSpec, Order, OrderSide, OrderStatus
from ._scope import callback_scope
from ._slippage import compile_slippage
from .assets import Asset, Future
from .continuous import ContinuousFuture, PreparedFutures
from .controls import _register_account_controls, _TradingRule
from .csv import PreparedCSV
from .data import BarData
from .events import _DateRule, _register, _Schedule, _TimeRule, date_rules, time_rules
from .finance import commission, slippage
from .finance.asset_restrictions import (
    NoRestrictions,
    Restrictions,
    SecurityListRestrictions,
    StaticRestrictions,
)
from .finance.cancel_policy import EODCancel, NeverCancel, _configure_cancellation
from .finance.execution import _native_style
from .finance.order import ORDER_STATUS, OrderView
from .lookup import PreparedAssetLookup, _lookup_instant, _symbol_parts
from .pipeline import PreparedPipeline, _PipelineState
from .results import to_performance_frame
from .window import _borrow_window, _execution_windows, _window


@dataclass(frozen=True)
class PositionView:
    asset: Asset
    amount: float
    cost_basis: float
    last_sale_price: float


class _Positions(Mapping[Asset, PositionView]):
    def __init__(self, context: MigrationContext):
        self.context = context

    def _flat_distributions(self):
        broker = self.context._require_broker()
        corporate = broker.account.corporate
        if corporate is None:
            return {}
        return {
            key: basis
            for key, (_index, basis) in corporate.flat_deliveries.items()
            if key not in broker.positions
        }

    def __getitem__(self, asset: Asset) -> PositionView:
        self.context._validate_asset(asset)
        position = self.context._require_broker().get_position(asset.key)
        flat = self._flat_distributions().get(asset.key) if position is None else None
        if flat is not None:
            broker = self.context._require_broker()
            mark = broker.get_mark_price(asset.key)
            if mark is None:
                mark = broker.get_last_price(asset.key)
            return PositionView(asset, 0.0, flat, mark if mark is not None else 0.0)

        return PositionView(
            asset,
            position.quantity if position is not None else 0,
            position.commission_adjusted_cost_basis if position is not None else 0.0,
            self.context._require_broker()._portfolio_ledger.position_mark(asset.key, position)[0]
            if position is not None
            else 0.0,
        )

    def __iter__(self) -> Iterator[Asset]:
        positions = self.context._require_broker().get_positions()
        return iter(self.context._by_key[key] for key in [*positions, *self._flat_distributions()])

    def __len__(self) -> int:
        return len(self.context._require_broker().get_positions()) + len(self._flat_distributions())


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
        self._environment: dict[str, Any] = {}
        self._asset_lookup: PreparedAssetLookup | None = None
        self._symbol_lookup_date: pd.Timestamp | None = None
        self._default_lookup_date: pd.Timestamp | None = None
        self._lookup_overrides: list[dict[str, str | None]] = []
        self._order_prefix = uuid4().hex
        self._broker: Broker | None = None
        self._initializing = True
        self._running = False
        self._in_before_trading_start = False
        self._schedules: list[_Schedule] = []
        self._pipelines: dict[str, _PipelineState] = {}
        self._csv_sources: dict[str, PreparedCSV] = {}
        self._continuous_futures: dict[tuple[str, int, str, str | None], ContinuousFuture] = {}
        self._active_csv: dict[str, PreparedCSV] = {}
        self._benchmark_input: BenchmarkData | None = None
        self._benchmark: BenchmarkData | None = None
        self._trading_controls: list[TradingControl] = []
        self._account_controls: list[tuple[str, float, timedelta | None]] = []
        self._restrictions: Restrictions = NoRestrictions()
        self._slippage: Any = slippage.FixedBasisPointsSlippage()
        self._futures_slippage: Any = slippage.VolatilityVolumeShare()
        self._futures_commission = commission.PerContract(0.85, {})
        self._cancel_policy: EODCancel | NeverCancel = NeverCancel()
        self._set_commission(commission.PerShare())
        self.portfolio = _Portfolio(self)

    @property
    def account(self) -> AccountControlContext:
        return self._require_broker().get_account_snapshot()

    def fetch_csv(self, url: str) -> PreparedCSV:
        self._setup_only()
        if url not in self._csv_sources:
            raise ValueError(
                "No frozen CSV for this original source identity; prepare it before the run"
            )
        source = self._csv_sources[url]
        self._active_csv.pop(url, None)
        self._active_csv[url] = source
        return source

    def attach_pipeline(
        self, pipeline: PreparedPipeline, name: str, chunks: Any = None, eager: bool = True
    ) -> PreparedPipeline:
        self._setup_only()
        if not isinstance(pipeline, PreparedPipeline):
            raise TypeError("Prepare feature calculations in Engineer and pass PreparedPipeline")
        if not isinstance(name, str) or not name or name in self._pipelines:
            raise ValueError("Pipeline names must be nonempty and unique")
        if chunks is not None:
            raise ValueError(
                "chunks controls factor calculation; prepare batches upstream and attach the prepared panel without chunks"
            )
        if not isinstance(eager, bool):
            raise TypeError("eager must be Boolean")
        self._pipelines[name] = _PipelineState(pipeline, eager)
        return pipeline

    def pipeline_output(self, name: str) -> pd.DataFrame:
        if self._initializing:
            raise RuntimeError("pipeline_output is unavailable during initialize")
        if name not in self._pipelines:
            raise ValueError(f"No pipeline named {name!r}")
        return self._pipelines[name].output()

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

    def get_environment(self, field: str = "platform") -> Any:
        if field == "*":
            return dict(self._environment)
        try:
            return self._environment[field]
        except KeyError as error:
            raise ValueError(f"Unknown environment field {field!r}") from error

    def _require_broker(self) -> Broker:
        if self._broker is None:
            raise RuntimeError("Account and order APIs require a running native Engine")
        return self._broker

    def _validate_asset(self, asset: Asset) -> None:
        if isinstance(asset, ContinuousFuture):
            raise TypeError("Order the actual dated Future returned by current(handle, 'contract')")
        if not isinstance(asset, Asset) or asset.key not in self._by_key:
            raise ValueError("Asset must belong to the explicit migration mapping")

    def symbol(self, symbol_str: str, country_code: str | None = None) -> Asset:
        if self._asset_lookup is not None:
            on = (
                self._symbol_lookup_date
                if self._symbol_lookup_date is not None
                else self._default_lookup_date
            )
            assert on is not None
            resolved = self._asset_lookup.resolve(symbol_str, on, country_code=country_code)
            return self._by_key[resolved.key]
        if country_code is not None:
            raise ValueError("Country selection requires a PreparedAssetLookup")
        parts = _symbol_parts(symbol_str)
        matches = [
            asset
            for asset in self._by_key.values()
            if not isinstance(asset, Future)
            and asset.symbol is not None
            and _symbol_parts(asset.symbol) == parts
        ]
        if len(matches) != 1:
            raise ValueError(f"Symbol {symbol_str!r} has no unique explicit asset mapping")
        return matches[0]

    def symbols(self, *symbol_strs: str, country_code: str | None = None) -> list[Asset]:
        return [self.symbol(name, country_code) for name in symbol_strs]

    def set_symbol_lookup_date(self, dt: date | datetime | str) -> None:
        if self._asset_lookup is None:
            raise ValueError("Dated symbol lookup requires a PreparedAssetLookup")
        self._symbol_lookup_date = _lookup_instant(dt)
        self._lookup_overrides.append(
            {
                "lookup_date": self._symbol_lookup_date.isoformat(),
                "simulation_time": self.get_datetime().isoformat()
                if self._broker is not None
                else None,
            }
        )

    def sid(self, sid: int) -> Asset:
        matches = [asset for asset in self._by_key.values() if asset.legacy_sid == sid]
        if len(matches) != 1:
            raise ValueError(f"SID {sid!r} has no explicit asset mapping")
        return matches[0]

    def continuous_future(
        self,
        root_symbol_str: str,
        offset: int = 0,
        roll: str = "volume",
        adjustment: str | None = "mul",
    ) -> ContinuousFuture:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Continuous offset must be a nonnegative integer")
        key = root_symbol_str, offset, roll, adjustment
        if key not in self._continuous_futures:
            raise ValueError(
                "Register an explicit PreparedFutures source for this root/offset/roll/adjustment"
            )
        return self._continuous_futures[key]

    def future_symbol(self, symbol: str) -> Future:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("future_symbol requires a nonempty contract symbol")
        matches = [
            asset
            for asset in self._by_key.values()
            if isinstance(asset, Future)
            and asset.symbol is not None
            and asset.symbol.upper() == symbol.upper()
        ]
        if len(matches) != 1:
            raise ValueError(f"Future {symbol!r} has no unique explicit contract mapping")
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

    def _is_restricted(self, asset: Asset, timestamp) -> bool:
        point = pd.Timestamp(timestamp)
        if point.tz is None:
            point = point.tz_localize(self._config.resolved_timezone)
        return bool(self._restrictions.is_restricted(asset, point.tz_convert("UTC")))

    def set_asset_restrictions(self, restrictions: Restrictions, on_error: str = "fail") -> None:
        self._setup_only()
        if not isinstance(restrictions, Restrictions):
            raise TypeError(
                "Supply a migration Restrictions object; native strategies use OrderConstraints"
            )

        def restricted(request):
            asset = self._by_key[request.asset]
            point = pd.Timestamp(request.timestamp)
            if point.tz is None:
                point = point.tz_localize(self._config.resolved_timezone)
            return bool(restrictions.is_restricted(asset, point.tz_convert("UTC")))

        self._control(
            OrderConstraints(restriction=restricted),
            on_error,
            {"rule": "asset_restrictions", "definition": restrictions.metadata},
        )
        self._restrictions = self._restrictions | restrictions

    def set_do_not_order_list(self, restricted_list: Any, on_error: str = "fail") -> None:
        self._setup_only()
        restrictions = (
            SecurityListRestrictions(restricted_list)
            if callable(getattr(restricted_list, "current_securities", None))
            else StaticRestrictions(restricted_list)
        )
        self.set_asset_restrictions(restrictions, on_error)

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
            sign
            * (
                order.action_adjusted_quantity
                if order.action_adjusted_quantity is not None
                else order.requested_quantity
            ),
            sign * order.filled_quantity,
            order.commission_paid,
            statuses[order.status],
            utc(order.created_at),
            utc(max(observed)) if observed else None,
            order.limit_price,
            None if order.stop_triggered else order.stop_price,
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

    def set_commission(
        self, model: Any = None, *, us_equities: Any = None, us_futures: Any = None
    ) -> None:
        self._setup_only()
        if model is not None and us_equities is not None:
            raise ValueError("Declare the equity commission once, positionally or as us_equities")
        model = us_equities if us_equities is not None else model
        if model is not None:
            self._set_commission(model)
        if us_futures is not None:
            if not isinstance(us_futures, commission.PerContract | commission.PerFutureTrade):
                raise TypeError("Use PerContract or PerFutureTrade for us_futures commissions")
            self._futures_commission = us_futures

    def _compile_commission(self) -> AssetCommission | None:
        futures = [asset for asset in self._by_key.values() if isinstance(asset, Future)]
        if not futures:
            return None
        config = self._config
        equity = (
            PercentageCommission(config.commission_rate)
            if config.commission_type == CommissionType.PERCENTAGE
            else PerOrderCommission(
                per_share=config.commission_per_share,
                per_trade=config.commission_per_trade,
                minimum=config.commission_minimum,
            )
        )
        models = {
            asset.key: commission._future_model(self._futures_commission, asset.root_symbol)
            for asset in futures
        }
        config.metadata["migration_future_fees"] = {
            key: {
                "per_contract": model.per_share,
                "exchange_or_trade": model.per_trade,
                "minimum": model.minimum,
            }
            for key, model in models.items()
        }
        return AssetCommission(models, default=equity)

    def set_slippage(
        self, model: Any = None, *, us_equities: Any = None, us_futures: Any = None
    ) -> None:
        self._setup_only()
        if model is not None and us_equities is not None:
            raise ValueError("Declare equity slippage once, positionally or as us_equities")
        model = us_equities if us_equities is not None else model
        if model is not None and not isinstance(
            model,
            slippage.VolumeShareSlippage
            | slippage.FixedBasisPointsSlippage
            | slippage.FixedSlippage
            | slippage.NoSlippage,
        ):
            raise TypeError("Use a declared migration slippage model")
        if model is not None:
            self._slippage = model
        if us_futures is not None:
            if not isinstance(
                us_futures,
                slippage.VolatilityVolumeShare
                | slippage.FixedBasisPointsSlippage
                | slippage.FixedSlippage
                | slippage.NoSlippage,
            ):
                raise TypeError("Use a declared futures slippage model")
            self._futures_slippage = us_futures

    def set_benchmark(self, asset: Asset | None) -> None:
        self._setup_only()
        if asset is None:
            self._benchmark = None
            self._config.metadata["benchmark_status"] = "explicitly_disabled"
            return
        self._validate_asset(asset)
        if self._benchmark_input is None or self._benchmark_input.asset != asset.key:
            raise ValueError(
                "A benchmark asset requires explicitly prepared benchmark returns for its stable key"
            )
        self._benchmark = self._benchmark_input
        self._config.metadata["benchmark_status"] = "prepared_returns"

    def set_cancel_policy(self, cancel_policy: EODCancel | NeverCancel) -> None:
        self._setup_only()
        if not isinstance(cancel_policy, EODCancel | NeverCancel):
            raise TypeError("Use a declared EODCancel or NeverCancel policy")
        if isinstance(cancel_policy, EODCancel) and not isinstance(
            cancel_policy.warn_on_cancel, bool
        ):
            raise TypeError("warn_on_cancel must be boolean")
        self._cancel_policy = cancel_policy
        self._config.cancel_policy = (
            CancelPolicy.SESSION_END if isinstance(cancel_policy, EODCancel) else CancelPolicy.NEVER
        )

    def _execution_limits(self):
        config, model = self._config, self._slippage
        if isinstance(model, slippage.FixedBasisPointsSlippage):
            if not isfinite(model.basis_points) or model.basis_points < 0:
                raise ValueError("basis_points must be finite and nonnegative")
            if not isfinite(model.volume_limit) or not 0 < model.volume_limit <= 1:
                raise ValueError("volume_limit must be finite and in (0, 1]")
            config.slippage_type = SlippageType.PERCENTAGE
            config.slippage_rate = model.basis_points / 10000.0
            config.metadata["execution_limits"] = {
                "model": "shared_observed_bar_volume",
                "max_participation": model.volume_limit,
            }
            return SharedVolumeParticipationLimit(max_participation=model.volume_limit)
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
        _register(
            self.context,
            self.runtime,
            self.data,
            self._before if self.context._pipelines else self.before_trading_start,
        )

    def _before(self, context, data) -> None:
        for state in context._pipelines.values():
            state.capture(context, self.runtime)
        if self.before_trading_start is not None:
            self.before_trading_start(context, data)

    def on_end(self, broker: Broker) -> None:
        self.context._running = False

    def on_data(self, timestamp, data, context, broker) -> None:
        if self.handle_data is not None:
            with callback_scope(self.context):
                self.handle_data(self.context, self.data)


def run_algorithm(
    *,
    feed: DataFeed,
    assets: Sequence[Asset] | None = None,
    asset_lookup: PreparedAssetLookup | None = None,
    initialize: Callable[..., Any] | None = None,
    handle_data: Callable[..., Any] | None = None,
    analyze: Callable[..., Any] | None = None,
    before_trading_start: Callable[..., Any] | None = None,
    capital_base: float = 100000.0,
    runtime: RuntimeServices | None = None,
    csv_sources: Sequence[PreparedCSV] = (),
    continuous_futures: Sequence[PreparedFutures] = (),
    benchmark: BenchmarkData | None = None,
    history_view: Literal["raw", "prepared"] = "raw",
    history_unit: WindowUnit | str | None = None,
    history_sources: Mapping[str, str] | None = None,
    contract_specs: dict[str, ContractSpec] | None = None,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Run imported callback functions on explicit bundle-free native inputs.

    Import migration API and finance names from this namespace. No legacy package,
    bundle discovery, network download or module rewriting occurs. Runtime bounds
    are physical instants, with pre-start rows available for warmup history.
    Optional start/end instead accept timezone-naive midnight session labels and
    select the declared calendar's first tradable minute through its last close.
    """
    if asset_lookup is not None:
        if not isinstance(asset_lookup, PreparedAssetLookup):
            raise TypeError("asset_lookup must be a PreparedAssetLookup")
        if assets is not None:
            raise ValueError("Pass assets or asset_lookup, not both")
        assets = asset_lookup.assets
    if assets is None:
        raise ValueError("Declare assets or a PreparedAssetLookup")
    specs = dict(contract_specs or {})
    for asset in assets:
        if isinstance(asset, ContinuousFuture):
            raise ValueError("Declare continuous research keys in continuous_futures, not assets")
        if isinstance(asset, Future):
            declared = ContractSpec(
                asset.key, AssetClass.FUTURE, multiplier=asset.multiplier, tick_size=asset.tick_size
            )
            supplied = specs.get(asset.key)
            if supplied is not None and (
                supplied.asset_class != declared.asset_class
                or supplied.multiplier != declared.multiplier
                or supplied.tick_size != declared.tick_size
            ):
                raise ValueError(f"ContractSpec conflicts with declared Future {asset.key!r}")
            specs.setdefault(asset.key, declared)
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
        corporate_action_policy="zipline_3_2",
        fill_ordering=FillOrdering.FIFO,
        metadata={
            "migration_adapter": "zipline",
            "benchmark_status": "not_supplied",
            "quantity_rounding": "near_integer",
            "target_pending_orders": "ignored",
        },
    )
    runtime = runtime if runtime is not None else RuntimeServices()
    effective_config = config.merge_feed_spec(feed.feed_spec)
    execution_windows = _execution_windows(feed, effective_config, runtime)
    with _borrow_window(runtime, runtime.start, runtime.end, execution_windows):
        bounded_start, bounded_end, environment, window_mode = _window(
            feed, effective_config, runtime, start, end
        )
    if execution_windows is not None:
        config.metadata["migration_execution_windows"] = [
            [opened.isoformat(), closed.isoformat()] for opened, closed in execution_windows
        ]
    config.metadata["migration_window"] = {
        "mode": window_mode,
        "start": environment["start"].isoformat(),
        "end": environment["end"].isoformat(),
    }
    context = MigrationContext(assets, config)
    for source in continuous_futures:
        if not isinstance(source, PreparedFutures):
            raise TypeError("continuous_futures requires PreparedFutures declarations")
        handle = source.handle
        if (
            handle.query in context._continuous_futures
            or handle.key in context._by_key
            or handle in context._continuous_futures.values()
        ):
            raise ValueError("Continuous source lookup or research key is ambiguous")
        for contract in source.contracts.values():
            known = context._by_key.get(contract.key)
            if not isinstance(known, Future) or (
                known.root_symbol,
                known.multiplier,
                known.tick_size,
            ) != (contract.root_symbol, contract.multiplier, contract.tick_size):
                raise ValueError("Continuous contracts must match registered dated Future units")
        context._continuous_futures[handle.query] = handle
    if continuous_futures:
        config.metadata["migration_continuous_futures"] = [
            {
                "source_id": source.source_id,
                "research_key": source.handle.key,
                "query": list(source.handle.query),
                "exchange": source.exchange,
                "contracts": [
                    {"source_id": identity, "native_key": asset.key, "legacy_sid": asset.legacy_sid}
                    for identity, asset in source.contracts.items()
                ],
            }
            for source in continuous_futures
        ]
    if benchmark is not None and not isinstance(benchmark, BenchmarkData):
        raise TypeError("benchmark requires prepared BenchmarkData")
    context._benchmark_input = context._benchmark = benchmark
    if benchmark is not None:
        config.metadata["benchmark_status"] = "prepared_returns"
    for source in csv_sources:
        if not isinstance(source, PreparedCSV):
            raise TypeError("csv_sources requires PreparedCSV inputs")
        if source.source.source_id in context._csv_sources:
            raise ValueError("Duplicate original CSV source identity")
        context._csv_sources[source.source.source_id] = source
    context._environment = environment
    context._asset_lookup = asset_lookup
    if asset_lookup is not None:
        resolved = config.merge_feed_spec(feed.feed_spec)
        session = (
            SessionConfig(
                resolved.resolved_calendar,
                resolved.resolved_timezone,
                resolved.resolved_session_start_time,
                resolved.resolved_timestamp_semantics,
                resolved.resolved_feed_spec.bar_label,
                resolved.resolved_data_frequency,
            )
            .date_for_timestamp(environment["end"])
            .date()
            if resolved.resolved_calendar is not None
            else environment["end"].date()
        )
        context._default_lookup_date = pd.Timestamp(session)
        config.metadata["migration_symbol_lookup"] = {
            **asset_lookup.metadata,
            "default_policy": "end_session"
            if resolved.resolved_calendar is not None
            else "end_observation_UTC_date",
            "default_date": context._default_lookup_date.isoformat(),
        }

    config.metadata["migration_result"] = {
        "assets": [asdict(asset) for asset in assets],
        "order_id_prefix": context._order_prefix,
    }
    with _borrow_window(runtime, bounded_start, bounded_end, execution_windows):
        if initialize is not None:
            with callback_scope(context):
                initialize(context)
        context._initializing = False
        config.metadata["migration_read_policy"] = {
            "method": "zipline",
            "price_view": "known_price_stale",
            "execution_quotes": "raw_observed",
            "history_view": history_view,
            "history_unit": getattr(history_unit, "value", history_unit),
            "lifetimes": "runtime_instrument_source"
            if runtime.instruments is not None
            else "not_declared",
        }
        limits = context._execution_limits()
        impact, limits = compile_slippage(context, runtime, history_view, history_sources, limits)
        fees = context._compile_commission()
        if context._pipelines:
            config.metadata["migration_pipelines"] = {
                name: {"definition": state.definition.metadata, "eager": state.eager}
                for name, state in context._pipelines.items()
            }
        if context._active_csv:
            config.metadata["migration_csv"] = {
                key: source.metadata for key, source in context._active_csv.items()
            }
        with runtime.auxiliary_scope(
            (
                *[state.definition.source for state in context._pipelines.values()],
                *[source.source for source in context._active_csv.values()],
            )
        ):
            engine = Engine(
                feed,
                _Callbacks(
                    context,
                    BarData(
                        runtime,
                        csv_sources=tuple(context._active_csv.values()),
                        restriction=context._is_restricted,
                        read_policy="zipline",
                        history_view=history_view,
                        history_unit=history_unit,
                        history_sources=history_sources,
                        continuous_futures=continuous_futures,
                    ),
                    handle_data,
                    runtime,
                    before_trading_start,
                ),
                config,
                runtime=runtime,
                execution_limits=limits,
                commission_model=fees,
                slippage_model=impact,
                contract_specs=specs,
                benchmark=context._benchmark,
                holdings_sampling="bar",
            )
            if isinstance(context._cancel_policy, EODCancel):
                _configure_cancellation(engine, context._cancel_policy)
            result = engine.run()
    if asset_lookup is not None and result.config is not None:
        result.config.metadata["migration_symbol_lookup"]["date_overrides"] = list(
            context._lookup_overrides
        )
    performance = to_performance_frame(result)
    if analyze is not None:
        with callback_scope(context):
            analyze(context, performance)
    return performance
