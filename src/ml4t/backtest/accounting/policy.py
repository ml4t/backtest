"""Account policy implementations for different account types.

This module defines the AccountPolicy interface and UnifiedAccountPolicy
for cash, crypto, and margin accounts, enabling flexible constraint enforcement
based on account parameters.

UnifiedAccountPolicy uses parameters to control behavior:
- allow_short_selling: Whether short positions are allowed
- allow_leverage: Whether margin/leverage is enabled

Account types map to these parameters:
- Cash: allow_short_selling=False, allow_leverage=False
- Crypto: allow_short_selling=True, allow_leverage=False
- Margin: allow_short_selling=True, allow_leverage=True
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import BacktestConfig
    from ..types import Position


class AccountPolicy(ABC):
    """Abstract base class for account-specific trading constraints.

    Different account types (cash, margin, portfolio margin) have different rules
    for what trades are allowed. This interface defines the contract that all
    account policies must implement.

    The policy pattern allows the engine to support multiple account types without
    complex conditional logic or parallel systems.
    """

    @abstractmethod
    def calculate_buying_power(self, cash: float, positions: dict[str, Position]) -> float:
        """Calculate available buying power for new long positions.

        Args:
            cash: Current cash balance (can be negative for margin accounts)
            positions: Dictionary of current positions {asset: Position}

        Returns:
            Available buying power in dollars. Must be >= 0.

        Note:
            This is used to determine if a new BUY order can be placed.
            For cash accounts: buying_power = max(0, cash)
            For margin accounts: buying_power = (NLV - MM) / initial_margin_rate
        """
        pass

    @abstractmethod
    def allows_short_selling(self) -> bool:
        """Whether this account type allows short selling.

        Returns:
            True if short selling is allowed, False otherwise.

        Note:
            Cash accounts: False (cannot short)
            Margin accounts: True (can short with margin requirements)
        """
        pass

    @abstractmethod
    def validate_new_position(
        self,
        asset: str,
        quantity: float,
        price: float,
        current_positions: dict[str, Position],
        cash: float,
    ) -> tuple[bool, str]:
        """Validate whether a new position can be opened.

        This is the core validation method called by the Gatekeeper before
        executing any order.

        Args:
            asset: Asset identifier (e.g., "AAPL")
            quantity: Desired position size (positive=long, negative=short)
            price: Expected fill price
            current_positions: Current positions {asset: Position}
            cash: Current cash balance

        Returns:
            (is_valid, reason) tuple:
                - is_valid: True if order can proceed, False if rejected
                - reason: Human-readable explanation (empty if valid)

        Examples:
            Cash account rejecting short:
                (False, "Short selling not allowed in cash account")

            Cash account rejecting insufficient funds:
                (False, "Insufficient cash: need $10,000, have $5,000")

            Margin account allowing trade:
                (True, "")

        Note:
            This method must be fast (called on every order). Keep validation
            logic simple and avoid unnecessary calculations.
        """
        pass

    @abstractmethod
    def handle_reversal(
        self,
        asset: str,
        current_quantity: float,
        order_quantity_delta: float,
        price: float,
        current_positions: dict[str, Position],
        cash: float,
        commission: float,
    ) -> tuple[bool, str]:
        """Handle position reversal validation (long→short or short→long).

        This method is called by the Gatekeeper when a reversal is detected.
        Each account policy implements this according to its rules:
        - Cash accounts reject all reversals (no short selling)
        - Margin accounts validate buying power for the new opposite position

        Args:
            asset: Asset identifier
            current_quantity: Current position quantity (non-zero)
            order_quantity_delta: Order quantity delta causing reversal
            price: Expected fill price
            current_positions: Current positions dict
            cash: Current cash balance
            commission: Pre-calculated commission for the order

        Returns:
            (is_valid, reason) tuple
        """
        pass

    @abstractmethod
    def validate_position_change(
        self,
        asset: str,
        current_quantity: float,
        quantity_delta: float,
        price: float,
        current_positions: dict[str, Position],
        cash: float,
    ) -> tuple[bool, str]:
        """Validate a change to an existing position.

        This handles adding to or reducing existing positions, including
        position reversals (long -> short or short -> long).

        Args:
            asset: Asset identifier
            current_quantity: Current position size (0 if no position)
            quantity_delta: Change in position (positive=buy, negative=sell)
            price: Expected fill price
            current_positions: All current positions
            cash: Current cash balance

        Returns:
            (is_valid, reason) tuple

        Examples:
            Adding to long position: current=100, delta=+50
            Closing long position: current=100, delta=-100
            Reversing position: current=100, delta=-200 (cash account rejects)

        Note:
            Position reversals (sign change) are particularly important for
            cash accounts, which must reject them.
        """
        pass


class UnifiedAccountPolicy(AccountPolicy):
    """Unified account policy with parameter-driven behavior.

    This class consolidates all account policy logic into a single implementation.
    Behavior is controlled by two primary flags:
    - allow_short_selling: Enables short positions
    - allow_leverage: Enables margin calculations

    Parameter Combinations:
        allow_short_selling=False, allow_leverage=False -> Cash account behavior
        allow_short_selling=True, allow_leverage=False -> Crypto account behavior
        allow_short_selling=True, allow_leverage=True -> Margin account behavior

    Examples:
        # Cash account (no shorts, no leverage)
        policy = UnifiedAccountPolicy()

        # Crypto account (shorts OK, no leverage)
        policy = UnifiedAccountPolicy(allow_short_selling=True)

        # Margin account (shorts and leverage)
        policy = UnifiedAccountPolicy(
            allow_short_selling=True,
            allow_leverage=True,
            initial_margin=0.5,
        )

        # Create from BacktestConfig
        config = BacktestConfig.from_preset("backtrader")
        policy = UnifiedAccountPolicy.from_config(config)
    """

    def __init__(
        self,
        allow_short_selling: bool = False,
        allow_leverage: bool = False,
        initial_margin: float = 0.5,
        long_maintenance_margin: float = 0.25,
        short_maintenance_margin: float = 0.30,
        fixed_margin_schedule: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize unified account policy.

        Args:
            allow_short_selling: If True, short positions are allowed.
            allow_leverage: If True, margin calculations are enabled.
            initial_margin: Initial margin requirement (0.0-1.0).
                - 0.5 = 50% = Reg T standard (2x leverage)
                - 1.0 = 100% = no leverage
            long_maintenance_margin: Maintenance margin for long positions.
                - 0.25 = 25% = Reg T standard for longs
            short_maintenance_margin: Maintenance margin for short positions.
                - 0.30 = 30% = Reg T standard for shorts (higher!)
            fixed_margin_schedule: Per-asset fixed dollar margin for futures.
                - Dict mapping asset symbol to (initial, maintenance) tuple
                - Example: {"ES": (12000, 6000)}

        Raises:
            ValueError: If margin parameters are invalid when leverage is enabled.
        """
        self.allow_short_selling = allow_short_selling
        self.allow_leverage = allow_leverage
        self.initial_margin = initial_margin
        self.long_maintenance_margin = long_maintenance_margin
        self.short_maintenance_margin = short_maintenance_margin
        self.fixed_margin_schedule = fixed_margin_schedule or {}

        # Validate margin parameters if leverage is enabled
        if allow_leverage:
            if not 0.0 < initial_margin <= 1.0:
                raise ValueError(f"Initial margin must be in (0.0, 1.0], got {initial_margin}")
            if not 0.0 < long_maintenance_margin <= 1.0:
                raise ValueError(
                    f"Long maintenance margin must be in (0.0, 1.0], got {long_maintenance_margin}"
                )
            if not 0.0 < short_maintenance_margin <= 1.0:
                raise ValueError(
                    f"Short maintenance margin must be in (0.0, 1.0], got {short_maintenance_margin}"
                )
            if long_maintenance_margin >= initial_margin:
                raise ValueError(
                    f"Long maintenance margin ({long_maintenance_margin}) must be < "
                    f"initial margin ({initial_margin})"
                )
            if short_maintenance_margin >= initial_margin:
                raise ValueError(
                    f"Short maintenance margin ({short_maintenance_margin}) must be < "
                    f"initial margin ({initial_margin})"
                )

    @classmethod
    def from_config(cls, config: BacktestConfig) -> UnifiedAccountPolicy:
        """Create policy from BacktestConfig.

        Args:
            config: BacktestConfig instance

        Returns:
            UnifiedAccountPolicy configured from the config
        """
        allow_shorts, allow_leverage = config.get_effective_account_settings()
        return cls(
            allow_short_selling=allow_shorts,
            allow_leverage=allow_leverage,
            initial_margin=config.initial_margin,
            long_maintenance_margin=config.long_maintenance_margin,
            short_maintenance_margin=config.short_maintenance_margin,
            fixed_margin_schedule=config.fixed_margin_schedule,
        )

    def allows_short_selling(self) -> bool:
        """Whether this policy allows short selling."""
        return self.allow_short_selling

    def calculate_buying_power(self, cash: float, positions: dict[str, Position]) -> float:
        """Calculate available buying power.

        For non-leverage accounts (cash/crypto): buying_power = max(0, cash)
        For margin accounts: buying_power = (NLV - required_IM) / initial_margin
        """
        if not self.allow_leverage:
            # Cash or Crypto: simple cash balance
            return max(0.0, cash)

        # Margin account: calculate based on equity and margin requirements
        total_market_value = sum(pos.market_value for pos in positions.values())
        nlv = cash + total_market_value

        # Calculate required initial margin for all existing positions
        required_initial_margin = 0.0
        for pos in positions.values():
            price = pos.current_price if pos.current_price is not None else pos.entry_price
            required_initial_margin += self.get_margin_requirement(
                pos.asset, pos.quantity, price, for_initial=True
            )

        # Buying power = excess equity / initial margin rate
        excess_equity = nlv - required_initial_margin
        return max(0.0, excess_equity / self.initial_margin)

    def get_margin_requirement(
        self,
        asset: str,
        quantity: float,
        price: float,
        for_initial: bool = True,
    ) -> float:
        """Calculate margin requirement for a position.

        Uses fixed dollar margin for futures, percentage for equities.

        Args:
            asset: Asset symbol
            quantity: Position quantity (signed)
            price: Current price
            for_initial: True for initial margin, False for maintenance

        Returns:
            Margin required in dollars
        """
        # Check for fixed margin (futures)
        if asset in self.fixed_margin_schedule:
            initial, maintenance = self.fixed_margin_schedule[asset]
            margin_per_contract = initial if for_initial else maintenance
            return abs(quantity) * margin_per_contract

        # Percentage-based margin (equities)
        market_value = abs(quantity * price)
        if for_initial:
            return market_value * self.initial_margin
        else:
            # Maintenance margin depends on position direction
            if quantity > 0:
                return market_value * self.long_maintenance_margin
            else:
                return market_value * self.short_maintenance_margin

    def is_margin_call(self, cash: float, positions: dict[str, Position]) -> bool:
        """Check if account is in margin call territory.

        Only applicable for margin accounts.
        """
        if not self.allow_leverage or not positions:
            return False

        total_market_value = sum(pos.market_value for pos in positions.values())
        nlv = cash + total_market_value

        required_maintenance = 0.0
        for pos in positions.values():
            price = pos.current_price if pos.current_price is not None else pos.entry_price
            required_maintenance += self.get_margin_requirement(
                pos.asset, pos.quantity, price, for_initial=False
            )

        return nlv < required_maintenance

    def handle_reversal(
        self,
        asset: str,
        current_quantity: float,
        order_quantity_delta: float,
        price: float,
        current_positions: dict[str, Position],
        cash: float,
        commission: float,
    ) -> tuple[bool, str]:
        """Handle position reversal validation (long→short or short→long)."""
        if not self.allow_short_selling:
            # Cash account: never allow reversals
            return False, "Position reversal not allowed in cash account"

        # Simulate close: proceeds from closing the position
        close_proceeds = abs(current_quantity * price)

        # Calculate cash after close (depends on leverage)
        if self.allow_leverage:
            # Margin: long receives value, short pays back
            if current_quantity > 0:
                cash_after_close = cash + close_proceeds
            else:
                cash_after_close = cash - close_proceeds
        else:
            # Crypto: simple proceeds
            cash_after_close = cash + close_proceeds

        cash_after_close -= commission

        # Calculate the new opposite position
        new_qty = current_quantity + order_quantity_delta
        new_position_cost = abs(new_qty * price)

        if self.allow_leverage:
            # Margin: check buying power for new position
            positions_after_close = {k: v for k, v in current_positions.items() if k != asset}
            return self.validate_new_position(
                asset=asset,
                quantity=new_qty,
                price=price,
                current_positions=positions_after_close,
                cash=cash_after_close,
            )
        else:
            # Crypto: check cash covers new position
            # Floating-point tolerance for cash comparison ($0.01 = 1 cent)
            CASH_TOLERANCE = 0.01
            if new_position_cost > cash_after_close + CASH_TOLERANCE:
                return (
                    False,
                    f"Insufficient cash for reversal: need ${new_position_cost:.2f}, "
                    f"have ${cash_after_close:.2f} after closing",
                )
            return True, ""

    def validate_new_position(
        self,
        asset: str,
        quantity: float,
        price: float,
        current_positions: dict[str, Position],
        cash: float,
    ) -> tuple[bool, str]:
        """Validate whether a new position can be opened."""
        # Check short selling permission
        if quantity < 0 and not self.allow_short_selling:
            return False, "Short selling not allowed in cash account"

        order_cost = abs(quantity * price)

        # Floating-point tolerance for cash comparison ($0.01 = 1 cent)
        # This prevents rejections due to floating-point arithmetic rounding
        # e.g., when equity/price * price != equity exactly
        CASH_TOLERANCE = 0.01

        if self.allow_leverage:
            # Margin: check buying power
            buying_power = self.calculate_buying_power(cash, current_positions)
            if order_cost > buying_power + CASH_TOLERANCE:
                return (
                    False,
                    f"Insufficient buying power: need ${order_cost:.2f}, "
                    f"have ${buying_power:.2f} (IM={self.initial_margin:.1%})",
                )
        else:
            # Cash/Crypto: check cash directly
            if order_cost > cash + CASH_TOLERANCE:
                direction = "long" if quantity > 0 else "short"
                if self.allow_short_selling:
                    return (
                        False,
                        f"Insufficient cash for {direction}: need ${order_cost:.2f}, have ${cash:.2f}",
                    )
                else:
                    return (
                        False,
                        f"Insufficient cash: need ${order_cost:.2f}, have ${cash:.2f}",
                    )

        return True, ""

    def validate_position_change(
        self,
        asset: str,
        current_quantity: float,
        quantity_delta: float,
        price: float,
        current_positions: dict[str, Position],
        cash: float,
    ) -> tuple[bool, str]:
        """Validate a change to an existing position."""
        new_quantity = current_quantity + quantity_delta

        # Check for reversal (sign change)
        is_reversal = current_quantity != 0 and (
            (current_quantity > 0 and new_quantity < 0)
            or (current_quantity < 0 and new_quantity > 0)
        )

        if is_reversal and not self.allow_short_selling:
            return (
                False,
                f"Position reversal not allowed in cash account "
                f"(current: {current_quantity:.2f}, delta: {quantity_delta:.2f})",
            )

        # Check for shorts
        if new_quantity < 0 and not self.allow_short_selling:
            return False, "Short positions not allowed in cash account"

        # Determine if this is reducing or increasing risk
        is_closing = (current_quantity > 0 and quantity_delta < 0) or (
            current_quantity < 0 and quantity_delta > 0
        )

        # For partial close without reversal, always allowed
        if is_closing and not is_reversal:
            # Check not over-closing
            if abs(quantity_delta) > abs(current_quantity):
                # This shouldn't happen without reversal, but check anyway
                pass
            else:
                return True, ""

        # Calculate order cost for validation
        if current_quantity == 0:
            # Opening new position
            order_cost = abs(quantity_delta * price)
        elif is_reversal:
            # Reversing: need resources for the new opposite portion
            order_cost = abs(new_quantity * price)
        else:
            # Adding to position
            order_cost = abs(quantity_delta * price)

        # Floating-point tolerance for cash comparison ($0.01 = 1 cent)
        CASH_TOLERANCE = 0.01

        if self.allow_leverage:
            # Margin: check buying power
            buying_power = self.calculate_buying_power(cash, current_positions)
            if order_cost > buying_power + CASH_TOLERANCE:
                return (
                    False,
                    f"Insufficient buying power: need ${order_cost:.2f}, "
                    f"have ${buying_power:.2f} (IM={self.initial_margin:.1%})",
                )
        else:
            # Cash/Crypto: check cash
            if order_cost > cash + CASH_TOLERANCE:
                return (
                    False,
                    f"Insufficient cash: need ${order_cost:.2f}, have ${cash:.2f}",
                )

        return True, ""
