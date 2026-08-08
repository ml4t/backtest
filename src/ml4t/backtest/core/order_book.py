"""Order-book operations extracted from Broker."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from ..models import calculate_commission
from ..types import ExecutionMode, Order, OrderSide, OrderStatus, OrderType, Position
from .shared import SubmitOrderOptions, is_exit_order, quantity_zero_tolerance
from .state import MarketState, OrderState, RiskState

if TYPE_CHECKING:
    from ..accounting import AccountState
    from ..broker import Broker
    from .fill_engine import FillEngine


class OrderBook:
    """Handles order submission/mutation/retrieval."""

    _UPDATABLE_ORDER_FIELDS: frozenset[str] = frozenset(
        {"quantity", "limit_price", "stop_price", "trail_amount"}
    )
    _MIN_ORDER_SIZE: float = 1e-8

    def __init__(
        self,
        broker: Broker,
        *,
        account: AccountState,
        market: MarketState,
        orders: OrderState,
        risk: RiskState,
        fill_engine: FillEngine,
    ) -> None:
        self.broker = broker
        self.account = account
        self.market = market
        self.orders = orders
        self.risk = risk
        self.fill_engine = fill_engine
        self._submission_shadow_bar: object | None = None
        self._submission_shadow_cash: float = 0.0
        self._submission_shadow_positions: dict[str, tuple[float, float]] = {}

    def submit_order(
        self,
        asset: str,
        quantity: float,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_amount: float | None = None,
        options: SubmitOrderOptions | None = None,
    ) -> Order | None:
        broker = self.broker

        if side is None:
            if quantity == 0:
                return None
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
        quantity = abs(quantity)
        if quantity <= self._MIN_ORDER_SIZE:
            return None

        if asset in self.risk.stop_exits_this_bar:
            existing_pos = self.account.positions.get(asset)
            if existing_pos is None:
                return None

        self.orders.counter += 1
        order = Order(
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            rebalance_id=options.rebalance_id if options is not None else None,
            order_id=f"ORD-{self.orders.counter}",
            created_at=self.market.time,
            _created_bar_index=self.market.bar_index,
            _risk_exit_reason=options.risk_exit_reason if options is not None else None,
            _exit_reason=options.exit_reason if options is not None else None,
            _risk_fill_price=options.risk_fill_price if options is not None else None,
        )

        order._signal_price = self.market.prices.get(asset)

        self.orders.orders.append(order)

        # Immediate fill: same-bar market orders fill during submit_order()
        # instead of being queued for later _process_orders(). Each order
        # validates against real cash (no shadow), matching LEAN's atomic
        # SetHoldings() model.
        if self._should_fill_immediately(order, options):
            return self._fill_immediately(order)

        if self._should_apply_submission_precheck(order) and not self._passes_submission_precheck(
            order
        ):
            if not order.rejection_reason:
                order.rejection_reason = "Insufficient cash (submission precheck)"
            order.reject(order.rejection_reason, order._rejection_code or "insufficient_cash")
            return order

        if self._should_apply_buying_power_reservation(
            order
        ) and not self._passes_buying_power_check(order):
            if not order.rejection_reason:
                order.rejection_reason = "Insufficient buying power"
            order.reject(
                order.rejection_reason,
                order._rejection_code or "insufficient_buying_power",
            )
            return order

        self.orders.pending.append(order)

        if broker.execution_mode is ExecutionMode.NEXT_BAR and (
            options is None or not options.eligible_in_next_bar_mode
        ):
            self.orders.current_bar.append(order)
            self.orders.current_bar_ids.add(order.order_id)

        return order

    def _should_fill_immediately(self, order: Order, options: SubmitOrderOptions | None) -> bool:
        """Check if an order qualifies for immediate fill.

        Immediate fill applies to same-bar market orders when the broker has
        immediate_fill enabled. Limit/stop orders still queue for later fill
        checks. Risk exits (from evaluate_position_rules) are not submitted
        through this path — they go through _process_orders() directly.
        """
        broker = self.broker
        return (
            broker.immediate_fill
            and broker.execution_mode is ExecutionMode.SAME_BAR
            and order.order_type is OrderType.MARKET
        )

    def _fill_immediately(self, order: Order) -> Order:
        """Fill a same-bar market order immediately during submit_order().

        Validates entries against real cash via the gatekeeper (single
        validation, no shadow), then executes the fill. Exits always pass.

        Returns the order with status FILLED or REJECTED.
        """
        broker = self.broker
        fill = self.fill_engine

        # Apply share rounding
        fill.apply_share_rounding(order)
        if order.quantity <= 0:
            order.reject(
                "Quantity rounds to zero (share_type=INTEGER)",
                "quantity_rounds_to_zero",
            )
            return order

        # Get fill price (close price for same-bar)
        price = fill.get_fill_price_for_order(order, use_open=False)
        if price is None:
            order.reject("No price available", "price_unavailable")
            return order

        fill_price = fill.check_fill(order, price)
        if fill_price is None:
            order.reject("Fill check failed", "fill_check_failed")
            return order

        # Determine if this is an exit (reduces existing position)
        is_exit = self._is_exit_order(order)

        if is_exit:
            # Exits always fill (they free capital)
            fully_filled = fill.execute_fill(order, fill_price)
            if fully_filled:
                self.orders.partial_quantities.pop(order.order_id, None)
            else:
                fill.update_partial_order(order)
        else:
            # Entries: validate against real cash via gatekeeper
            if not broker.skip_cash_validation:
                valid, rejection_reason, rejection_code = (
                    broker.gatekeeper.validate_order_with_code(order, fill_price)
                )
                if not valid:
                    allow_rebalance_partial = (
                        order.rebalance_id is not None and broker.share_type.value == "integer"
                    )
                    if broker.partial_fills_allowed and "insufficient" in rejection_reason.lower():
                        if not fill.try_partial_fill(order, fill_price):
                            order.reject(
                                rejection_reason,
                                rejection_code or "order_validation_failed",
                            )
                            return order
                        self.orders.partial_quantities.pop(order.order_id, None)
                        return order
                    if allow_rebalance_partial and "insufficient" in rejection_reason.lower():
                        if not fill.try_partial_fill(order, fill_price):
                            order.reject(
                                rejection_reason,
                                rejection_code or "order_validation_failed",
                            )
                            return order
                        self.orders.partial_quantities.pop(order.order_id, None)
                        return order
                    order.reject(
                        rejection_reason,
                        rejection_code or "order_validation_failed",
                    )
                    return order

            fully_filled = fill.execute_fill(order, fill_price)
            if fully_filled:
                self.orders.partial_quantities.pop(order.order_id, None)
            else:
                fill.update_partial_order(order)

        return order

    def _is_exit_order(self, order: Order) -> bool:
        """Check if an order reduces an existing position without reversing."""
        return is_exit_order(order, self.account.positions)

    def update_order(self, order_id: str, **kwargs) -> bool:
        invalid_fields = set(kwargs.keys()) - self._UPDATABLE_ORDER_FIELDS
        if invalid_fields:
            raise ValueError(
                f"Cannot update order fields: {invalid_fields}. "
                f"Updatable fields: {sorted(self._UPDATABLE_ORDER_FIELDS)}"
            )

        for order in self.orders.pending:
            if order.order_id == order_id:
                for key, value in kwargs.items():
                    setattr(order, key, value)
                return True
        return False

    def cancel_order(self, order_id: str) -> bool:
        for order in self.orders.pending:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self.orders.pending.remove(order)
                return True
        return False

    def get_order(self, order_id: str) -> Order | None:
        for order in self.orders.orders:
            if order.order_id == order_id:
                return order
        return None

    def get_pending_orders(self, asset: str | None = None) -> list[Order]:
        if asset is None:
            return list(self.orders.pending)
        return [o for o in self.orders.pending if o.asset == asset]

    def _should_apply_submission_precheck(self, order: Order) -> bool:
        broker = self.broker
        return (
            broker.execution_mode is ExecutionMode.NEXT_BAR
            and broker.next_bar_submission_precheck
            and order.order_type in {OrderType.MARKET, OrderType.MOC}
        )

    def _reset_submission_shadow_if_needed(self) -> None:
        broker = self.broker
        ts = self.market.time
        bar_key = ts.date() if ts is not None else None
        if bar_key == self._submission_shadow_bar:
            return

        self._submission_shadow_bar = bar_key
        self._submission_shadow_cash = broker.cash
        self._submission_shadow_positions = {
            asset: (
                pos.quantity,
                broker.get_mark_price(asset, quantity=pos.quantity)
                or pos.current_price
                or pos.entry_price,
            )
            for asset, pos in self.account.positions.items()
            if abs(pos.quantity) > quantity_zero_tolerance(pos.quantity)
        }

    def _build_shadow_policy_positions(self) -> dict[str, Position]:
        broker = self.broker
        ts = self.market.time or datetime(1970, 1, 1)
        positions: dict[str, Position] = {}
        for asset, (qty, basis_price) in self._submission_shadow_positions.items():
            if abs(qty) <= quantity_zero_tolerance(qty):
                continue
            mark_price = broker.get_mark_price(asset, quantity=qty) or basis_price
            positions[asset] = Position(
                asset=asset,
                quantity=qty,
                entry_price=basis_price,
                current_price=mark_price,
                entry_time=ts,
                multiplier=broker.get_multiplier(asset),
            )
        return positions

    @staticmethod
    def _simulate_position_update(
        old_qty: float, old_price: float, size: float, price: float
    ) -> tuple[float, float, float, float]:
        """Mirror Backtrader Position.update for pseudo-exec prechecks."""
        new_qty = old_qty + size
        if abs(new_qty) <= quantity_zero_tolerance(old_qty, size):
            return 0.0, 0.0, 0.0, size

        if abs(old_qty) <= quantity_zero_tolerance(old_qty):
            return new_qty, price, size, 0.0

        if old_qty > 0:
            if size > 0:
                new_price = (old_price * old_qty + size * price) / new_qty
                return new_qty, new_price, size, 0.0
            if new_qty > 0:
                return new_qty, old_price, 0.0, size
            return new_qty, price, new_qty, -old_qty

        # old short position
        if size < 0:
            new_price = (old_price * old_qty + size * price) / new_qty
            return new_qty, new_price, size, 0.0
        if new_qty < 0:
            return new_qty, old_price, 0.0, size
        return new_qty, price, new_qty, -old_qty

    def _passes_submission_precheck(self, order: Order) -> bool:
        broker = self.broker

        if broker.share_type.value == "integer":
            order.quantity = float(int(order.quantity))
            if order.quantity <= 0:
                order.rejection_reason = "Quantity rounds to zero (share_type=INTEGER)"
                order._rejection_code = "quantity_rounds_to_zero"
                return False

        signal_price = getattr(order, "_signal_price", None)
        if signal_price is None:
            signal_price = self.market.prices.get(order.asset)
        if signal_price is None:
            # No local price to precheck with; keep order eligible.
            return True

        self._reset_submission_shadow_if_needed()

        if broker.allow_leverage and not broker.next_bar_simple_cash_check:
            return self._passes_margin_submission_precheck(order, signal_price)

        size = order.quantity if order.side is OrderSide.BUY else -order.quantity
        old_qty, old_price = self._submission_shadow_positions.get(order.asset, (0.0, signal_price))
        new_qty, new_price, opened, closed = self._simulate_position_update(
            old_qty, old_price, size, signal_price
        )

        # Match Backtrader default stock-like, shortcash=True submission check semantics.
        shadow_cash = self._submission_shadow_cash

        if closed != 0.0:
            close_cash = (-closed) * signal_price
            shadow_cash += close_cash
            closed_commission = calculate_commission(
                broker.commission_model, order.asset, abs(closed), signal_price
            )
            shadow_cash -= closed_commission

        if opened != 0.0:
            open_cash = opened * signal_price
            shadow_cash -= open_cash
            opened_commission = calculate_commission(
                broker.commission_model, order.asset, abs(opened), signal_price
            )
            shadow_cash -= opened_commission

        # Keep shadow effects even for rejected orders to mirror Backtrader's
        # sequential submitted-queue pseudo-execution behavior.
        self._submission_shadow_cash = shadow_cash
        if abs(new_qty) <= quantity_zero_tolerance(old_qty, size):
            self._submission_shadow_positions.pop(order.asset, None)
        else:
            self._submission_shadow_positions[order.asset] = (new_qty, new_price)

        if shadow_cash < 0.0:
            trace_counts = getattr(broker, "_trace_submission_precheck_counts", None)
            if isinstance(trace_counts, dict) and self.market.time is not None:
                ts = self.market.time.date()
                counts = trace_counts.setdefault(ts, {"accepted": 0, "rejected": 0})
                counts["rejected"] += 1
            return False

        trace_counts = getattr(broker, "_trace_submission_precheck_counts", None)
        if isinstance(trace_counts, dict) and self.market.time is not None:
            ts = self.market.time.date()
            counts = trace_counts.setdefault(ts, {"accepted": 0, "rejected": 0})
            counts["accepted"] += 1
        return True

    def _should_apply_buying_power_reservation(self, order: Order) -> bool:
        broker = self.broker
        return broker.buying_power_reservation and order.order_type in {
            OrderType.MARKET,
            OrderType.MOC,
        }

    def _passes_buying_power_check(self, order: Order) -> bool:
        """LEAN-style buying power reservation at submission time.

        Unlike Backtrader precheck, rejected orders do NOT consume shadow
        buying power — only accepted orders update the shadow pool.

        When allow_leverage is True, delegates to the margin-aware precheck
        that uses policy validation (matching LEAN's margin-mode buying power
        model). When False, uses simple cash accounting.
        """
        broker = self.broker

        if broker.share_type.value == "integer":
            order.quantity = float(int(order.quantity))
            if order.quantity <= 0:
                order.rejection_reason = "Quantity rounds to zero (share_type=INTEGER)"
                order._rejection_code = "quantity_rounds_to_zero"
                return False

        signal_price = getattr(order, "_signal_price", None)
        if signal_price is None:
            signal_price = self.market.prices.get(order.asset)
        if signal_price is None:
            return True

        self._reset_submission_shadow_if_needed()

        size = order.quantity if order.side is OrderSide.BUY else -order.quantity
        old_qty, old_price = self._submission_shadow_positions.get(order.asset, (0.0, signal_price))
        new_qty, new_price, opened, closed = self._simulate_position_update(
            old_qty, old_price, size, signal_price
        )

        shadow_cash = self._submission_shadow_cash

        if closed != 0.0:
            closed_value = (-closed) * signal_price
            shadow_cash += closed_value
            closed_commission = calculate_commission(
                broker.commission_model, order.asset, abs(closed), signal_price
            )
            shadow_cash -= closed_commission

        if opened != 0.0:
            # LEAN semantics: both longs and shorts consume buying power.
            # Longs cost notional; shorts also require notional (not credit).
            # This prevents credit-model inflation where short proceeds
            # artificially inflate shadow cash.
            shadow_cash -= abs(opened) * signal_price
            opened_commission = calculate_commission(
                broker.commission_model, order.asset, abs(opened), signal_price
            )
            shadow_cash -= opened_commission

        if shadow_cash < 0.0:
            # Rejected — do NOT update shadow state (LEAN semantics)
            return False

        # Accepted — commit shadow changes
        self._submission_shadow_cash = shadow_cash
        if abs(new_qty) <= quantity_zero_tolerance(old_qty, size):
            self._submission_shadow_positions.pop(order.asset, None)
        else:
            self._submission_shadow_positions[order.asset] = (new_qty, new_price)
        return True

    def _passes_margin_submission_precheck(self, order: Order, signal_price: float) -> bool:
        """Margin-aware submission precheck for NEXT_BAR market orders.

        Simulates sequential submission-time acceptance using policy validation at
        signal-bar prices and commits only accepted orders into shadow state.
        """
        broker = self.broker

        size = order.quantity if order.side is OrderSide.BUY else -order.quantity
        old_qty, old_price = self._submission_shadow_positions.get(order.asset, (0.0, signal_price))
        new_qty, new_price, _opened, _closed = self._simulate_position_update(
            old_qty, old_price, size, signal_price
        )

        commission = calculate_commission(
            broker.commission_model, order.asset, order.quantity, signal_price
        )
        shadow_positions = self._build_shadow_policy_positions()
        available_cash = self._submission_shadow_cash
        if (
            self.account.policy.short_cash_policy == "lock_notional"
            and not self.account.policy.allow_leverage
        ):
            available_cash = self.account.policy.get_spendable_cash(
                self._submission_shadow_cash, shadow_positions
            )
        if broker.cash_buffer_pct > 0 and available_cash > 0:
            available_cash *= 1.0 - broker.cash_buffer_pct

        is_reversal = (
            abs(old_qty) > quantity_zero_tolerance(old_qty)
            and abs(new_qty) > quantity_zero_tolerance(old_qty, size)
            and ((old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0))
        )

        if abs(old_qty) <= quantity_zero_tolerance(old_qty):
            valid, reason = self.account.policy.validate_new_position(
                asset=order.asset,
                quantity=size,
                price=signal_price,
                current_positions=shadow_positions,
                cash=available_cash - commission,
                multiplier=broker.get_multiplier(order.asset),
            )
        elif is_reversal:
            valid, reason = self.account.policy.handle_reversal(
                asset=order.asset,
                current_quantity=old_qty,
                order_quantity_delta=size,
                price=signal_price,
                current_positions=shadow_positions,
                cash=available_cash,
                commission=commission,
                multiplier=broker.get_multiplier(order.asset),
            )
        else:
            valid, reason = self.account.policy.validate_position_change(
                asset=order.asset,
                current_quantity=old_qty,
                quantity_delta=size,
                price=signal_price,
                current_positions=shadow_positions,
                cash=available_cash - commission,
                multiplier=broker.get_multiplier(order.asset),
            )

        trace_counts = getattr(broker, "_trace_submission_precheck_counts", None)
        if isinstance(trace_counts, dict) and self.market.time is not None:
            ts = self.market.time.date()
            counts = trace_counts.setdefault(ts, {"accepted": 0, "rejected": 0})
            counts["accepted" if valid else "rejected"] += 1

        if not valid:
            rejection_code = broker.gatekeeper.classify_rejection(old_qty + size)
            fallback_reason = {
                "account_restriction": "Account restriction (submission precheck)",
                "insufficient_cash": "Insufficient cash (submission precheck)",
                "insufficient_buying_power": "Insufficient buying power (submission precheck)",
            }[rejection_code]
            order.rejection_reason = reason or fallback_reason
            order._rejection_code = rejection_code
            return False

        multiplier = broker.get_multiplier(order.asset)
        self._submission_shadow_cash += -size * signal_price * multiplier - commission
        if abs(new_qty) <= quantity_zero_tolerance(old_qty, size):
            self._submission_shadow_positions.pop(order.asset, None)
        else:
            self._submission_shadow_positions[order.asset] = (new_qty, new_price)
        return True
