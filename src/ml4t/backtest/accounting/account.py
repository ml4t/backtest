"""Account state management.

This module provides the AccountState class that tracks cash, positions, and
delegates validation to the appropriate AccountPolicy.
"""

from ..types import Position
from .policy import AccountPolicy


class AccountState:
    """Account state ledger with policy-based constraints.

    AccountState is the central ledger that tracks:
    - Cash balance
    - Open positions
    - Account policy (cash vs margin)

    It delegates all validation and constraint checking to the AccountPolicy,
    making it easy to support different account types.

    Example:
        >>> from ml4t.backtest.accounting import AccountState, CashAccountPolicy
        >>> policy = CashAccountPolicy()
        >>> account = AccountState(initial_cash=100000.0, policy=policy)
        >>> account.buying_power
        100000.0
    """

    def __init__(self, initial_cash: float, policy: AccountPolicy):
        """Initialize account state.

        Args:
            initial_cash: Starting cash balance
            policy: AccountPolicy instance (CashAccountPolicy or MarginAccountPolicy)
        """
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.policy = policy

    @property
    def total_equity(self) -> float:
        """Calculate total account equity (Net Liquidating Value).

        For both cash and margin accounts:
            NLV = Cash + Σ(position.market_value)

        Returns:
            Total account equity
        """
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def buying_power(self) -> float:
        """Calculate available buying power for new long positions.

        Delegates to policy:
        - Cash account: buying_power = max(0, cash)
        - Margin account: buying_power = (NLV - MM) / initial_margin_rate

        Returns:
            Available buying power in dollars
        """
        return self.policy.calculate_buying_power(self.cash, self.positions)

    def allows_short_selling(self) -> bool:
        """Check if short selling is allowed.

        Delegates to policy:
        - Cash account: False
        - Margin account: True

        Returns:
            True if short selling allowed, False otherwise
        """
        return self.policy.allows_short_selling()

    def mark_to_market(self, current_prices: dict[str, float]) -> None:
        """Update positions with current market prices.

        This is called at the end of each bar to update unrealized P&L.

        Args:
            current_prices: Dictionary mapping asset -> current_price
        """
        for asset, position in self.positions.items():
            if asset in current_prices:
                position.current_price = current_prices[asset]

    def get_position(self, asset: str) -> Position | None:
        """Get position for a specific asset.

        Args:
            asset: Asset identifier

        Returns:
            Position object if exists, None otherwise
        """
        return self.positions.get(asset)

    def get_position_quantity(self, asset: str) -> float:
        """Get quantity for a specific asset (0 if no position).

        Args:
            asset: Asset identifier

        Returns:
            Position quantity (positive=long, negative=short, 0=flat)
        """
        pos = self.positions.get(asset)
        return pos.quantity if pos else 0.0

    def apply_fill(self, asset: str, quantity_delta: float, fill_price: float, timestamp) -> float:
        """Apply a fill to the account, updating position and cash.

        This method handles both long and short positions correctly:
        - Long positions (quantity > 0): Cash decreases when buying, increases when selling
        - Short positions (quantity < 0): Cash increases when shorting, decreases when covering

        Args:
            asset: Asset identifier
            quantity_delta: Signed quantity change (positive=buy, negative=sell/short)
            fill_price: Fill price per unit
            timestamp: Fill timestamp

        Returns:
            Cash change (positive=cash in, negative=cash out)

        Examples:
            Open long (buy 100 @ $150):
                quantity_delta=+100, fill_price=$150
                cash_change = -$15,000 (paid to buy)

            Close long (sell 100 @ $160):
                quantity_delta=-100, fill_price=$160
                cash_change = +$16,000 (received from sale)

            Open short (sell 100 @ $150):
                quantity_delta=-100, fill_price=$150
                cash_change = +$15,000 (proceeds from short sale)

            Close short (buy 100 @ $145):
                quantity_delta=+100, fill_price=$145
                cash_change = -$14,500 (paid to cover short)

        Note:
            Commission should be handled separately by the caller.
            This method only handles the asset position and base cash flow.
        """

        # Calculate cash flow: negative for buys, positive for sells/shorts
        # Formula: cash_change = -quantity_delta × fill_price
        # Works for both longs and shorts:
        #   - Buy (quantity_delta > 0): cash decreases (negative change)
        #   - Sell/Short (quantity_delta < 0): cash increases (positive change)
        cash_change = -quantity_delta * fill_price

        # Update cash balance
        self.cash += cash_change

        # Update position
        pos = self.positions.get(asset)
        if pos is None:
            # New position (long or short)
            if quantity_delta != 0:
                self.positions[asset] = Position(
                    asset=asset,
                    quantity=quantity_delta,
                    entry_price=fill_price,
                    current_price=fill_price,
                    entry_time=timestamp,
                    bars_held=0,
                )
        else:
            # Existing position - update quantity and cost basis
            old_qty = pos.quantity
            new_qty = old_qty + quantity_delta

            if new_qty == 0:
                # Position fully closed
                del self.positions[asset]
            elif (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
                # Position reversal (long → short or short → long)
                # Close old position, open new position in opposite direction
                del self.positions[asset]
                self.positions[asset] = Position(
                    asset=asset,
                    quantity=new_qty,
                    entry_price=fill_price,
                    current_price=fill_price,
                    entry_time=timestamp,
                    bars_held=0,
                )
            elif abs(new_qty) > abs(old_qty):
                # Adding to existing position (same direction)
                # Update weighted average entry price
                old_cost = abs(old_qty) * pos.entry_price
                new_cost = abs(quantity_delta) * fill_price
                total_cost = old_cost + new_cost
                pos.entry_price = total_cost / abs(new_qty)
                pos.quantity = new_qty
            else:
                # Partial close (reducing position size)
                # Entry price remains unchanged for partial closes
                pos.quantity = new_qty

        return cash_change

    def __repr__(self) -> str:
        """String representation for debugging."""
        policy_name = self.policy.__class__.__name__
        num_positions = len(self.positions)
        return (
            f"AccountState("
            f"cash=${self.cash:,.2f}, "
            f"equity=${self.total_equity:,.2f}, "
            f"positions={num_positions}, "
            f"policy={policy_name})"
        )
