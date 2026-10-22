"""Pre-submission and observed-account validators over the native broker state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import ClassVar, Literal, Protocol
from zoneinfo import ZoneInfo


class TradingControlViolation(ValueError):
    """An order failed validation before submission changed broker state."""


class AccountControlViolation(ValueError):
    """The actual observed account failed a registered boundary validation."""


@dataclass(frozen=True)
class OrderControlContext:
    timestamp: datetime | None
    asset: str
    quantity: float
    position_quantity: float
    pending_buy_quantity: float
    pending_sell_quantity: float
    price: float | None
    multiplier: float
    session_date: date | None
    orders_in_session: int
    orders_in_day: int
    is_update: bool = False

    @property
    def pending_quantity(self) -> float:
        return self.pending_buy_quantity - self.pending_sell_quantity


class TradingControl(Protocol):
    """Validators inspect immutable values and raise without side effects."""

    def validate(self, request: OrderControlContext) -> None: ...


def _nonnegative(name: str, value: float | None) -> None:
    if value is not None and (isinstance(value, bool) or not isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class OrderConstraints:
    """Native order rules, with explicit pending-order and notional conventions.

    Session limits require a calendar. Day limits count the delivered timestamp's
    civil date. Only submitted orders consume counts; validation failures do not.
    Economic notional includes the instrument multiplier. ``quoted_units`` is an
    explicit compatibility choice for legacy price-times-quantity controls.
    """

    asset: str | None = None
    max_order_quantity: float | None = None
    max_order_notional: float | None = None
    max_position_quantity: float | None = None
    max_position_notional: float | None = None
    max_orders_per_session: int | None = None
    max_orders_per_day: int | None = None
    long_only: bool = False
    include_pending: bool = True
    notional_basis: Literal["economic", "quoted_units"] = "economic"
    blocked_assets: frozenset[str] = field(default_factory=frozenset)
    restriction: Callable[[OrderControlContext], bool] | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_order_quantity",
            "max_order_notional",
            "max_position_quantity",
            "max_position_notional",
        ):
            _nonnegative(name, getattr(self, name))
        for name in ("max_orders_per_session", "max_orders_per_day"):
            count = getattr(self, name)
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.notional_basis not in {"economic", "quoted_units"}:
            raise ValueError("notional_basis must be economic or quoted_units")
        if isinstance(self.blocked_assets, str):
            raise TypeError("blocked_assets must be a collection of stable asset IDs")
        if self.restriction is not None and not callable(self.restriction):
            raise TypeError("restriction must be callable")
        object.__setattr__(self, "blocked_assets", frozenset(self.blocked_assets))

    def validate(self, request: OrderControlContext) -> None:
        if self.asset is not None and self.asset != request.asset:
            return
        if not isfinite(request.quantity):
            raise TradingControlViolation("Order quantity must be finite")
        if (
            request.asset in self.blocked_assets
            or self.restriction is not None
            and self.restriction(request)
        ):
            raise TradingControlViolation(
                f"Asset {request.asset!r} is restricted at {request.timestamp}"
            )
        position = request.position_quantity + request.quantity
        if self.include_pending:
            position += (
                request.pending_buy_quantity
                if request.quantity > 0
                else -request.pending_sell_quantity
            )
        minimum_position = request.position_quantity + request.quantity
        if self.include_pending:
            minimum_position -= request.pending_sell_quantity
        if self.long_only and minimum_position < 0:
            raise TradingControlViolation("Order would violate the long-only control")
        for kind, amount, maximum in [
            ("order quantity", abs(request.quantity), self.max_order_quantity),
            ("position quantity", abs(position), self.max_position_quantity),
        ]:
            if maximum is not None and amount > maximum:
                raise TradingControlViolation(f"{kind} {amount} exceeds {maximum}")
        if self.max_order_notional is not None or self.max_position_notional is not None:
            if request.price is None or not isfinite(request.price) or request.price <= 0:
                raise TradingControlViolation(
                    "Notional controls require a current positive observed price"
                )
            unit = request.price * (
                request.multiplier if self.notional_basis == "economic" else 1.0
            )
            for kind, amount, maximum in [
                ("order notional", abs(request.quantity) * unit, self.max_order_notional),
                ("position notional", abs(position) * unit, self.max_position_notional),
            ]:
                if maximum is not None and amount > maximum:
                    raise TradingControlViolation(f"{kind} {amount} exceeds {maximum}")
        if self.max_orders_per_session is not None and not request.is_update:
            if request.session_date is None:
                raise ValueError(
                    "max_orders_per_session requires a declared calendar and current timestamp"
                )
            if request.orders_in_session >= self.max_orders_per_session:
                raise TradingControlViolation("Maximum orders in the exchange session exceeded")
        if self.max_orders_per_day is not None and not request.is_update:
            if request.timestamp is None:
                raise ValueError("max_orders_per_day requires a current timestamp")
            if request.orders_in_day >= self.max_orders_per_day:
                raise TradingControlViolation("Maximum orders in the civil day exceeded")


@dataclass(frozen=True)
class OrderRequestCount:
    """Limit requests that reach this rule, including zero and later-rejected requests.

    The broker owns counts per validator identity and civil date in ``timezone``.
    Validation is pure; the broker advances this audit count only after this rule
    returns. Earlier rule failures do not consume it, while later failures do.
    Existing OrderConstraints day/session counts continue to count submissions.
    """

    max_count: int
    timezone: str = "UTC"
    counts_order_requests: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_count, bool)
            or not isinstance(self.max_count, int)
            or self.max_count < 0
        ):
            raise ValueError("max_count must be a nonnegative integer")
        ZoneInfo(self.timezone)

    def validate(self, request: OrderControlContext) -> None:
        if request.is_update:
            return
        if request.timestamp is None:
            raise ValueError("Request counts require a simulation timestamp")
        if request.orders_in_day >= self.max_count:
            raise TradingControlViolation(
                "Maximum validated order requests in the civil day exceeded"
            )


@dataclass(frozen=True)
class AccountControlContext:
    timestamp: datetime | None
    equity: float
    cash: float
    gross_exposure: float
    net_exposure: float
    positions_value: float | None = field(default=None, kw_only=True)
    initial_margin_required: float | None = field(default=None, kw_only=True)
    maintenance_margin_required: float | None = field(default=None, kw_only=True)
    available_funds: float | None = field(default=None, kw_only=True)
    futures_unsettled_pnl: float | None = field(default=None, kw_only=True)
    settlement_holds: float | None = field(default=None, kw_only=True)
    cash_buffer: float | None = field(default=None, kw_only=True)
    margin_breach: bool | None = field(default=None, kw_only=True)

    @property
    def leverage(self) -> float:
        if self.equity > 0:
            return self.gross_exposure / self.equity
        return float("inf") if self.gross_exposure else 0.0


class AccountControl(Protocol):
    """Validate observed account values without placing orders or mutating the account."""

    def validate(self, account: AccountControlContext) -> None: ...


@dataclass(frozen=True)
class AccountConstraints:
    max_leverage: float | None = None
    min_leverage: float | None = None
    deadline: datetime | None = None

    def __post_init__(self) -> None:
        _nonnegative("max_leverage", self.max_leverage)
        _nonnegative("min_leverage", self.min_leverage)
        if self.min_leverage is not None and not isinstance(self.deadline, datetime):
            raise ValueError("Minimum leverage requires a datetime deadline")

    def validate(self, account: AccountControlContext) -> None:
        if any(
            not isfinite(value)
            for value in (
                account.equity,
                account.cash,
                account.gross_exposure,
                account.net_exposure,
            )
        ):
            raise AccountControlViolation("Account control received non-finite account values")
        if self.max_leverage is not None and account.leverage > self.max_leverage:
            raise AccountControlViolation("Account exceeds the maximum gross leverage")
        if self.min_leverage is not None:
            if account.timestamp is None:
                raise ValueError("Minimum leverage requires a current timestamp")
            assert self.deadline is not None
            if (account.timestamp.tzinfo is None) != (self.deadline.tzinfo is None):
                raise ValueError(
                    "Minimum leverage deadline and account timestamps require compatible timezone awareness"
                )
            if account.timestamp > self.deadline and account.leverage < self.min_leverage:
                raise AccountControlViolation(
                    "Account is below the minimum gross leverage after the deadline"
                )
