"""Account policy implementations for different account types.

This module defines the AccountPolicy interface and implementations for cash
and margin accounts, enabling flexible constraint enforcement based on account type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


class CashAccountPolicy(AccountPolicy):
    """Account policy for cash accounts (no leverage, no shorts).

    Cash accounts are the simplest account type:
    - Cannot go negative (no borrowing)
    - Cannot short sell (no borrowing shares)
    - Buying power = available cash only
    - Position reversals not allowed (must close, then re-open)

    This is appropriate for:
    - Retail investors with no margin approval
    - Tax-advantaged accounts (IRA, 401k)
    - Conservative risk management
    """

    def calculate_buying_power(self, cash: float, positions: dict[str, Position]) -> float:
        """Cash account buying power is simply positive cash balance.

        Args:
            cash: Current cash balance
            positions: Ignored for cash accounts

        Returns:
            max(0, cash) - Cannot use margin
        """
        return max(0.0, cash)

    def allows_short_selling(self) -> bool:
        """Cash accounts cannot short sell.

        Returns:
            False - Short selling not allowed
        """
        return False

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
        """Cash accounts do not allow position reversals.

        Returns:
            (False, rejection_reason) - Always rejects reversals
        """
        return False, "Position reversal not allowed in cash account"

    def validate_new_position(
        self,
        asset: str,
        quantity: float,
        price: float,
        current_positions: dict[str, Position],
        cash: float,
    ) -> tuple[bool, str]:
        """Validate new position for cash account.

        Checks:
        1. No short positions (quantity must be > 0)
        2. Sufficient cash to cover purchase

        Args:
            asset: Asset identifier
            quantity: Desired position size
            price: Expected fill price
            current_positions: Current positions (unused)
            cash: Current cash balance

        Returns:
            (is_valid, reason) tuple
        """
        # Check 1: No short selling
        if quantity < 0:
            return False, "Short selling not allowed in cash account"

        # Check 2: Sufficient cash
        order_cost = quantity * price
        if order_cost > cash:
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
        """Validate position change for cash account.

        Checks:
        1. No position reversals (sign change)
        2. For increases: sufficient cash
        3. For decreases: not exceeding current position

        Args:
            asset: Asset identifier
            current_quantity: Current position size (0 if none)
            quantity_delta: Change in position
            price: Expected fill price
            current_positions: All current positions
            cash: Current cash balance

        Returns:
            (is_valid, reason) tuple
        """
        new_quantity = current_quantity + quantity_delta

        # Check 1: No position reversals (long -> short or short -> long)
        if current_quantity != 0 and (
            (current_quantity > 0 and new_quantity < 0)
            or (current_quantity < 0 and new_quantity > 0)
        ):
            return (
                False,
                f"Position reversal not allowed in cash account "
                f"(current: {current_quantity:.2f}, delta: {quantity_delta:.2f})",
            )

        # Check 2: No short positions
        if new_quantity < 0:
            return False, "Short positions not allowed in cash account"

        # Check 3: For increases (buying), check cash
        if quantity_delta > 0:
            order_cost = quantity_delta * price
            if order_cost > cash:
                return (
                    False,
                    f"Insufficient cash: need ${order_cost:.2f}, have ${cash:.2f}",
                )

        # Check 4: For decreases (selling), check position size
        if quantity_delta < 0 and abs(quantity_delta) > abs(current_quantity):
            return (
                False,
                f"Cannot sell {abs(quantity_delta):.2f}, only have {abs(current_quantity):.2f}",
            )

        return True, ""


class MarginAccountPolicy(AccountPolicy):
    """Account policy for margin accounts (leverage enabled, shorts allowed).

    Supports both percentage-based margin (equities) and fixed-dollar margin (futures).

    **Equities (Reg T style):**
    - Initial margin: 50% for both longs and shorts
    - Maintenance margin: 25% for longs, 30% for shorts (asymmetric!)
    - Buying power = excess equity / initial margin rate

    **Futures (SPAN style):**
    - Fixed dollar margin per contract (not percentage of notional)
    - Specify via fixed_margin_schedule: {"ES": (12000, 6000)}
    - First value = initial margin, second = maintenance margin

    Key Formulas:
        NLV = cash + sum(position.market_value)
        Required IM = sum(get_margin_requirement(pos, for_initial=True))
        Excess Equity = NLV - Required IM
        BP = Excess Equity / initial_margin_rate

    Examples:
        >>> # Standard Reg T margin (realistic asymmetric maintenance)
        >>> policy = MarginAccountPolicy()  # Uses defaults: 50% IM, 25%/30% MM
        >>>
        >>> # With futures margin schedule
        >>> policy = MarginAccountPolicy(
        ...     fixed_margin_schedule={
        ...         "ES": (12_000, 6_000),   # $12k initial, $6k maintenance per contract
        ...         "NQ": (15_000, 7_500),   # $15k initial, $7.5k maintenance
        ...     }
        ... )
        >>>
        >>> # Custom margin requirements
        >>> policy = MarginAccountPolicy(
        ...     initial_margin=0.25,           # 4x leverage
        ...     long_maintenance_margin=0.15,
        ...     short_maintenance_margin=0.20,
        ... )
    """

    def __init__(
        self,
        initial_margin: float = 0.5,
        long_maintenance_margin: float = 0.25,
        short_maintenance_margin: float = 0.30,
        fixed_margin_schedule: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize margin account policy.

        Args:
            initial_margin: Initial margin requirement (0.0-1.0)
                - 0.5 = 50% = Reg T standard (2x leverage)
                - 1.0 = 100% = no leverage
                - Lower values = more leverage (higher risk)

            long_maintenance_margin: Maintenance margin for long positions
                - 0.25 = 25% = Reg T standard for longs
                - Below this triggers margin call

            short_maintenance_margin: Maintenance margin for short positions
                - 0.30 = 30% = Reg T standard for shorts (higher than longs!)
                - Below this triggers margin call

            fixed_margin_schedule: Per-asset fixed dollar margin for futures
                - Dict mapping asset symbol to (initial, maintenance) tuple
                - Example: {"ES": (12000, 6000)} = $12k initial, $6k maintenance
                - Assets in this dict use fixed margin, others use percentage

        Raises:
            ValueError: If margin parameters are invalid
        """
        # Validate parameters
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

        self.initial_margin = initial_margin
        self.long_maintenance_margin = long_maintenance_margin
        self.short_maintenance_margin = short_maintenance_margin
        self.fixed_margin_schedule = fixed_margin_schedule or {}

    def get_margin_requirement(
        self,
        asset: str,
        quantity: float,
        price: float,
        for_initial: bool = True,
    ) -> float:
        """Calculate margin requirement for a position.

        Automatically selects the appropriate margin model:
        - Fixed dollar margin for assets in fixed_margin_schedule (futures)
        - Percentage margin for all other assets (equities)

        Args:
            asset: Asset symbol
            quantity: Position quantity (signed: positive=long, negative=short)
            price: Current price per unit
            for_initial: True for initial margin, False for maintenance

        Returns:
            Margin required in dollars

        Examples:
            >>> policy = MarginAccountPolicy(fixed_margin_schedule={"ES": (12000, 6000)})
            >>> # Equity: 100 shares @ $150 = $15,000 market value
            >>> policy.get_margin_requirement("AAPL", 100, 150.0, for_initial=True)
            7500.0  # 50% of $15,000
            >>> # Futures: 2 ES contracts
            >>> policy.get_margin_requirement("ES", 2, 5000.0, for_initial=True)
            24000.0  # 2 × $12,000 per contract
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

        A margin call occurs when equity falls below the maintenance margin
        requirement for current positions.

        Args:
            cash: Current cash balance
            positions: Current positions

        Returns:
            True if account equity is below maintenance requirement
        """
        if not positions:
            return False

        # Calculate NLV
        total_market_value = sum(pos.market_value for pos in positions.values())
        nlv = cash + total_market_value

        # Calculate total maintenance margin required
        required_maintenance = 0.0
        for pos in positions.values():
            price = pos.current_price if pos.current_price is not None else pos.entry_price
            required_maintenance += self.get_margin_requirement(
                pos.asset, pos.quantity, price, for_initial=False
            )

        return nlv < required_maintenance

    def calculate_buying_power(self, cash: float, positions: dict[str, Position]) -> float:
        """Calculate buying power for margin account.

        Handles both percentage-based margin (equities) and fixed-dollar margin
        (futures) via get_margin_requirement().

        Formula:
            NLV = cash + sum(position.market_value)
            Required IM = sum(get_margin_requirement(pos, for_initial=True))
            Excess Equity = NLV - Required IM
            BP = Excess Equity / initial_margin_rate

        Args:
            cash: Current cash balance (can be negative)
            positions: Dictionary of current positions {asset: Position}

        Returns:
            Available buying power in dollars. Returns 0 if account is
            underwater (below initial margin requirement).

        Examples:
            Cash only account (no positions):
                cash=$100k, positions={}
                NLV = $100k, Required IM = $0
                BP = $100k / 0.5 = $200k (2x leverage)

            Long equity position (Reg T 50%):
                cash=$0, long 1000 shares @ $100 = $100k market value
                NLV = $0 + $100k = $100k
                Required IM = $100k × 0.5 = $50k
                Excess Equity = $100k - $50k = $50k
                BP = $50k / 0.5 = $100k (can buy $100k more)

            Futures position (fixed margin):
                cash=$50k, long 2 ES contracts @ 5000 (notional $500k)
                NLV = $50k + $0 = $50k (futures have no market_value impact)
                Required IM = 2 × $12,000 = $24k
                Excess Equity = $50k - $24k = $26k
                BP = $26k / 0.5 = $52k

        Note:
            For mixed portfolios (equities + futures), buying power is calculated
            using the percentage-based initial margin rate for the excess equity
            conversion. This assumes new positions are equities; for futures,
            check excess_equity directly against the fixed margin requirement.
        """
        # Calculate Net Liquidation Value (NLV)
        total_market_value = sum(pos.market_value for pos in positions.values())
        nlv = cash + total_market_value

        # Calculate required initial margin for all existing positions
        # Uses get_margin_requirement() to handle both equities and futures
        required_initial_margin = 0.0
        for pos in positions.values():
            price = pos.current_price if pos.current_price is not None else pos.entry_price
            required_initial_margin += self.get_margin_requirement(
                pos.asset, pos.quantity, price, for_initial=True
            )

        # Calculate excess equity (equity above required margin)
        excess_equity = nlv - required_initial_margin

        # Calculate buying power (excess equity leveraged by initial margin)
        # Note: This assumes buying equities. For futures, the user should
        # check: excess_equity >= fixed_margin_schedule[asset][0] * contracts
        buying_power = max(0.0, excess_equity / self.initial_margin)

        return buying_power

    def allows_short_selling(self) -> bool:
        """Margin accounts allow short selling.

        Returns:
            True - Short selling is allowed with appropriate margin
        """
        return True

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
        """Handle position reversal for margin account.

        Reversal is conceptually split into:
        1. Close existing position (always approved - reduces risk)
        2. Open new opposite position (must validate buying power)

        We simulate the close first, then validate the new position.
        """
        # Simulate close: Create positions dict without the closing asset
        positions_after_close = {k: v for k, v in current_positions.items() if k != asset}

        # Calculate cash after close:
        # Long position: receive market value; Short: pay back borrowed value
        close_proceeds = abs(current_quantity * price)
        cash_after_close = cash + close_proceeds if current_quantity > 0 else cash - close_proceeds

        # Account for commission
        cash_after_close -= commission

        # Calculate the new opposite position quantity
        new_qty = current_quantity + order_quantity_delta

        # Validate the new position as if opening fresh (with post-close state)
        return self.validate_new_position(
            asset=asset,
            quantity=new_qty,
            price=price,
            current_positions=positions_after_close,
            cash=cash_after_close,
        )

    def validate_new_position(
        self,
        asset: str,
        quantity: float,
        price: float,
        current_positions: dict[str, Position],
        cash: float,
    ) -> tuple[bool, str]:
        """Validate new position for margin account.

        Checks:
        1. Sufficient buying power for the order
        2. Order doesn't create excessive leverage

        Args:
            asset: Asset identifier
            quantity: Desired position size (positive=long, negative=short)
            price: Expected fill price
            current_positions: Current positions
            cash: Current cash balance

        Returns:
            (is_valid, reason) tuple

        Note:
            Unlike cash accounts, margin accounts allow:
            - Short positions (negative quantity)
            - Negative cash (borrowing)
            - Multiple positions simultaneously
        """
        # Calculate order cost (positive for both long and short)
        order_cost = abs(quantity * price)

        # Calculate current buying power
        buying_power = self.calculate_buying_power(cash, current_positions)

        # Check: Sufficient buying power
        if order_cost > buying_power:
            return (
                False,
                f"Insufficient buying power: need ${order_cost:.2f}, "
                f"have ${buying_power:.2f} (IM={self.initial_margin:.1%})",
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
        """Validate position change for margin account.

        Margin accounts are more permissive than cash accounts:
        - Allow position reversals (long -> short, short -> long)
        - Allow adding to short positions
        - Only constraint is buying power

        Args:
            asset: Asset identifier
            current_quantity: Current position size (0 if none)
            quantity_delta: Change in position
            price: Expected fill price
            current_positions: All current positions
            cash: Current cash balance

        Returns:
            (is_valid, reason) tuple

        Examples:
            Adding to long: current=100, delta=+50 -> OK if BP sufficient
            Closing long: current=100, delta=-100 -> Always OK (reduces risk)
            Reversing long->short: current=100, delta=-200 -> OK if BP sufficient
            Adding to short: current=-100, delta=-50 -> OK if BP sufficient
        """
        new_quantity = current_quantity + quantity_delta

        # Determine if this is increasing or reducing risk
        is_closing = (current_quantity > 0 and quantity_delta < 0) or (
            current_quantity < 0 and quantity_delta > 0
        )

        # For closing trades, check we're not over-closing
        if is_closing and abs(new_quantity) < abs(current_quantity):
            # Partial close - always allowed (reduces risk)
            return True, ""
            # Position reversal or over-close - validate new portion

        # For opening or reversing, check buying power
        # Calculate the portion that increases risk
        if current_quantity == 0:
            # Opening new position
            risk_increase = abs(quantity_delta * price)
        elif (current_quantity > 0 and new_quantity > current_quantity) or (
            current_quantity < 0 and new_quantity < current_quantity
        ):
            # Adding to existing position
            risk_increase = abs(quantity_delta * price)
        else:
            # Reversing position - need margin for the new opposite position
            # Example: long 100 -> short 100 requires margin for short 100
            risk_increase = abs(new_quantity * price)

        # Calculate buying power
        buying_power = self.calculate_buying_power(cash, current_positions)

        # Validate sufficient buying power
        if risk_increase > buying_power:
            return (
                False,
                f"Insufficient buying power: need ${risk_increase:.2f}, "
                f"have ${buying_power:.2f} (IM={self.initial_margin:.1%})",
            )

        return True, ""
