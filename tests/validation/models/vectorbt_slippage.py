"""VectorBT Pro compatible slippage model for validation testing.

This module exists ONLY for validation purposes - to ensure ml4t.backtest produces
identical slippage calculations to VectorBT Pro during comparative testing.

DO NOT import this into production code (src/ml4t.backtest). This is a test fixture.
"""

from typing import TYPE_CHECKING

# Import from production for base class
from ml4t.backtest.execution.slippage import SlippageModel

if TYPE_CHECKING:
    from ml4t.backtest.core.types import Price, Quantity
    from ml4t.backtest.execution.order import Order


class VectorBTSlippage(SlippageModel):
    """VectorBT Pro compatible slippage model.

    Implements VectorBT's multiplicative slippage formula exactly:
    - Buy orders: adj_price = base_price * (1 + slippage)
    - Sell orders: adj_price = base_price * (1 - slippage)

    This differs from additive models where slippage is added/subtracted.
    VectorBT's approach ensures slippage scales proportionally with price.

    Args:
        slippage: Slippage as a decimal (e.g., 0.0002 = 0.02% = 2 basis points)

    Example:
        >>> # 0.02% slippage on BTC at $50,000
        >>> slippage_model = VectorBTSlippage(slippage=0.0002)
        >>>
        >>> # Buy order: pay 0.02% MORE
        >>> # adj_price = 50000 * (1 + 0.0002) = 50000 * 1.0002 = $50,010
        >>>
        >>> # Sell order: receive 0.02% LESS
        >>> # adj_price = 50000 * (1 - 0.0002) = 50000 * 0.9998 = $49,990

    References:
        - TASK-010: VectorBT slippage documentation
        - vectorbtpro/portfolio/nb/core.py: long_buy_nb() and long_sell_nb()
    """

    def __init__(self, slippage: float = 0.0):
        """Initialize with slippage rate.

        Args:
            slippage: Slippage as a decimal (e.g., 0.0002 for 0.02%)

        Raises:
            ValueError: If slippage is negative
        """
        if slippage < 0:
            raise ValueError("Slippage must be non-negative")
        self.slippage = slippage

    def calculate_fill_price(self, order: "Order", market_price: "Price") -> "Price":
        """Calculate fill price using VectorBT's multiplicative formula.

        Buy orders pay MORE (worse price for buyer):
            adj_price = market_price * (1 + slippage)

        Sell orders receive LESS (worse price for seller):
            adj_price = market_price * (1 - slippage)

        Args:
            order: The order being filled
            market_price: Current market price (base price)

        Returns:
            Adjusted fill price with slippage applied
        """
        if order.is_buy:
            # Buy at higher price (pay more) - multiplicative
            return market_price * (1 + self.slippage)
        # Sell at lower price (receive less) - multiplicative
        return market_price * (1 - self.slippage)

    def calculate_slippage_cost(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        market_price: "Price",
        fill_price: "Price",
    ) -> float:
        """Calculate slippage cost in currency terms.

        Cost is the absolute difference between market and fill price,
        multiplied by the quantity filled.

        Args:
            order: The order being filled
            fill_quantity: Quantity being filled
            market_price: Market price before slippage
            fill_price: Actual fill price after slippage

        Returns:
            Slippage cost in currency terms (always positive)
        """
        return abs(fill_price - market_price) * fill_quantity
