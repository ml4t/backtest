"""Account policy implementations for different account types.

This module defines the AccountPolicy interface and implementations for cash
and margin accounts, enabling flexible constraint enforcement based on account type.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple


class AccountPolicy(ABC):
    """Abstract base class for account-specific trading constraints.

    Different account types (cash, margin, portfolio margin) have different rules
    for what trades are allowed. This interface defines the contract that all
    account policies must implement.

    The policy pattern allows the engine to support multiple account types without
    complex conditional logic or parallel systems.
    """

    @abstractmethod
    def calculate_buying_power(
        self, cash: float, positions: Dict[str, "Position"]
    ) -> float:
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
        current_positions: Dict[str, "Position"],
        cash: float,
    ) -> Tuple[bool, str]:
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
    def validate_position_change(
        self,
        asset: str,
        current_quantity: float,
        quantity_delta: float,
        price: float,
        current_positions: Dict[str, "Position"],
        cash: float,
    ) -> Tuple[bool, str]:
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

    def calculate_buying_power(
        self, cash: float, positions: Dict[str, "Position"]
    ) -> float:
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

    def validate_new_position(
        self,
        asset: str,
        quantity: float,
        price: float,
        current_positions: Dict[str, "Position"],
        cash: float,
    ) -> Tuple[bool, str]:
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
            return False, f"Short selling not allowed in cash account"

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
        current_positions: Dict[str, "Position"],
        cash: float,
    ) -> Tuple[bool, str]:
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
        if quantity_delta < 0:
            if abs(quantity_delta) > abs(current_quantity):
                return (
                    False,
                    f"Cannot sell {abs(quantity_delta):.2f}, only have {abs(current_quantity):.2f}",
                )

        return True, ""


class MarginAccountPolicy(AccountPolicy):
    """Account policy for margin accounts (leverage enabled, shorts allowed).

    Margin accounts enable more sophisticated trading strategies:
    - Can use leverage (borrow cash to increase buying power)
    - Can short sell (borrow shares to sell)
    - Buying power calculated from Net Liquidation Value and margin requirements
    - Subject to initial margin (IM) and maintenance margin (MM) requirements

    Key Formulas:
        NLV = cash + sum(position.market_value)
        MM = sum(abs(position.market_value) × maintenance_margin_rate)
        BP = (NLV - MM) / initial_margin_rate

    This is appropriate for:
    - Experienced traders with margin approval
    - Hedge funds and institutional accounts
    - Strategies requiring leverage or short selling
    - Market-neutral and pairs trading strategies

    Args:
        initial_margin: Initial margin requirement (default 0.5 = 50% = Reg T)
        maintenance_margin: Maintenance margin requirement (default 0.25 = 25%)

    Examples:
        >>> # Standard Reg T margin (50% initial, 25% maintenance)
        >>> policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        >>>
        >>> # Conservative margin (100% initial = no leverage)
        >>> policy = MarginAccountPolicy(initial_margin=1.0, maintenance_margin=0.5)
        >>>
        >>> # Aggressive margin (lower requirements)
        >>> policy = MarginAccountPolicy(initial_margin=0.25, maintenance_margin=0.15)
    """

    def __init__(
        self, initial_margin: float = 0.5, maintenance_margin: float = 0.25
    ) -> None:
        """Initialize margin account policy.

        Args:
            initial_margin: Initial margin requirement (0.0-1.0)
                - 0.5 = 50% = Reg T standard (2x leverage)
                - 1.0 = 100% = no leverage
                - Lower values = more leverage (higher risk)

            maintenance_margin: Maintenance margin requirement (0.0-1.0)
                - 0.25 = 25% = Reg T standard
                - Must be < initial_margin
                - Below this triggers margin call

        Raises:
            ValueError: If margin parameters are invalid
        """
        if not 0.0 < initial_margin <= 1.0:
            raise ValueError(
                f"Initial margin must be in (0.0, 1.0], got {initial_margin}"
            )
        if not 0.0 < maintenance_margin <= 1.0:
            raise ValueError(
                f"Maintenance margin must be in (0.0, 1.0], got {maintenance_margin}"
            )
        if maintenance_margin >= initial_margin:
            raise ValueError(
                f"Maintenance margin ({maintenance_margin}) must be < "
                f"initial margin ({initial_margin})"
            )

        self.initial_margin = initial_margin
        self.maintenance_margin = maintenance_margin

    def calculate_buying_power(
        self, cash: float, positions: Dict[str, "Position"]
    ) -> float:
        """Calculate buying power for margin account.

        Formula:
            NLV = cash + sum(position.market_value for all positions)
            MM = sum(abs(position.market_value) × maintenance_margin for all positions)
            BP = (NLV - MM) / initial_margin

        Args:
            cash: Current cash balance (can be negative)
            positions: Dictionary of current positions {asset: Position}

        Returns:
            Available buying power in dollars. Can be negative if account is
            underwater (below maintenance margin).

        Examples:
            Cash only account (no positions):
                cash=$100k, positions={}
                NLV = $100k, MM = $0
                BP = ($100k - $0) / 0.5 = $200k (2x leverage)

            Long position:
                cash=$50k, long 1000 shares @ $100 = $100k market value
                NLV = $50k + $100k = $150k
                MM = $100k × 0.25 = $25k
                BP = ($150k - $25k) / 0.5 = $250k

            Short position:
                cash=$150k, short 1000 shares @ $100 = -$100k market value
                NLV = $150k + (-$100k) = $50k
                MM = |-$100k| × 0.25 = $25k
                BP = ($50k - $25k) / 0.5 = $50k

            Underwater account (margin call):
                cash=-$10k, long 1000 shares @ $50 = $50k market value
                NLV = -$10k + $50k = $40k
                MM = $50k × 0.25 = $12.5k
                BP = ($40k - $12.5k) / 0.5 = $55k
                (Still has buying power, but NLV < initial investment)

        Note:
            Buying power can be negative if the account is severely underwater,
            indicating that positions must be liquidated to meet margin requirements.
        """
        # Calculate Net Liquidation Value (NLV)
        total_market_value = sum(pos.market_value for pos in positions.values())
        nlv = cash + total_market_value

        # Calculate Maintenance Margin requirement (MM)
        # Use absolute value because short positions have negative market value
        maintenance_margin_requirement = sum(
            abs(pos.market_value) * self.maintenance_margin
            for pos in positions.values()
        )

        # Calculate Buying Power (BP)
        # Available equity above maintenance margin, leveraged by initial margin
        buying_power = (nlv - maintenance_margin_requirement) / self.initial_margin

        return buying_power

    def allows_short_selling(self) -> bool:
        """Margin accounts allow short selling.

        Returns:
            True - Short selling is allowed with appropriate margin
        """
        return True

    def validate_new_position(
        self,
        asset: str,
        quantity: float,
        price: float,
        current_positions: Dict[str, "Position"],
        cash: float,
    ) -> Tuple[bool, str]:
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
        current_positions: Dict[str, "Position"],
        cash: float,
    ) -> Tuple[bool, str]:
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
        if is_closing:
            if abs(new_quantity) < abs(current_quantity):
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
