"""Corporate actions handling for QEngine.

Corporate actions are events that affect the equity structure of a company,
requiring adjustments to positions, prices, and orders. This module provides
a comprehensive framework for handling:

1. Dividends (cash dividends, special dividends)
2. Stock splits and stock dividends
3. Mergers and acquisitions (cash, stock, mixed)
4. Spin-offs
5. Symbol changes/reorganizations
6. Rights offerings

All actions maintain point-in-time correctness and properly adjust positions,
orders, and price histories.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from qengine.core.types import AssetId, Price, Quantity
    from qengine.execution.order import Order

logger = logging.getLogger(__name__)


@dataclass
class CorporateAction:
    """Base class for corporate actions."""

    action_id: str
    asset_id: "AssetId"
    ex_date: date  # Ex-dividend date (when action takes effect)
    record_date: date | None = None  # Record date for eligibility
    payment_date: date | None = None  # When payment/distribution occurs
    announcement_date: date | None = None  # When action was announced
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dates."""
        if self.record_date and self.ex_date and self.record_date > self.ex_date:
            raise ValueError("Record date must be before ex-date")


class CashDividend(CorporateAction):
    """Cash dividend corporate action."""

    def __init__(
        self,
        action_id: str,
        asset_id: "AssetId",
        ex_date: date,
        dividend_per_share: float,
        currency: str = "USD",
        record_date: date | None = None,
        payment_date: date | None = None,
        announcement_date: date | None = None,
        metadata: dict[str, str] | None = None,
    ):
        super().__init__(
            action_id=action_id,
            asset_id=asset_id,
            ex_date=ex_date,
            record_date=record_date,
            payment_date=payment_date,
            announcement_date=announcement_date,
            metadata=metadata or {},
        )
        self.dividend_per_share = dividend_per_share
        self.currency = currency

    @property
    def action_type(self) -> str:
        return "DIVIDEND"


class StockSplit(CorporateAction):
    """Stock split corporate action."""

    def __init__(
        self,
        action_id: str,
        asset_id: "AssetId",
        ex_date: date,
        split_ratio: float,
        record_date: date | None = None,
        payment_date: date | None = None,
        announcement_date: date | None = None,
        metadata: dict[str, str] | None = None,
    ):
        super().__init__(
            action_id=action_id,
            asset_id=asset_id,
            ex_date=ex_date,
            record_date=record_date,
            payment_date=payment_date,
            announcement_date=announcement_date,
            metadata=metadata or {},
        )
        if split_ratio <= 0:
            raise ValueError("Split ratio must be positive")
        self.split_ratio = split_ratio

    @property
    def action_type(self) -> str:
        return "SPLIT"


class StockDividend(CorporateAction):
    """Stock dividend corporate action."""

    def __init__(
        self,
        action_id: str,
        asset_id: "AssetId",
        ex_date: date,
        dividend_ratio: float,
        record_date: date | None = None,
        payment_date: date | None = None,
        announcement_date: date | None = None,
        metadata: dict[str, str] | None = None,
    ):
        super().__init__(
            action_id=action_id,
            asset_id=asset_id,
            ex_date=ex_date,
            record_date=record_date,
            payment_date=payment_date,
            announcement_date=announcement_date,
            metadata=metadata or {},
        )
        self.dividend_ratio = dividend_ratio

    @property
    def action_type(self) -> str:
        return "STOCK_DIVIDEND"


class Merger(CorporateAction):
    """Merger/acquisition corporate action."""

    def __init__(
        self,
        action_id: str,
        asset_id: "AssetId",
        ex_date: date,
        target_asset_id: "AssetId",
        cash_consideration: float = 0.0,
        stock_consideration: float = 0.0,
        record_date: date | None = None,
        payment_date: date | None = None,
        announcement_date: date | None = None,
        metadata: dict[str, str] | None = None,
    ):
        super().__init__(
            action_id=action_id,
            asset_id=asset_id,
            ex_date=ex_date,
            record_date=record_date,
            payment_date=payment_date,
            announcement_date=announcement_date,
            metadata=metadata or {},
        )
        if cash_consideration == 0.0 and stock_consideration == 0.0:
            raise ValueError("Must have either cash or stock consideration")
        self.target_asset_id = target_asset_id
        self.cash_consideration = cash_consideration
        self.stock_consideration = stock_consideration

    @property
    def action_type(self) -> str:
        return "MERGER"


class SpinOff(CorporateAction):
    """Spin-off corporate action."""

    def __init__(
        self,
        action_id: str,
        asset_id: "AssetId",
        ex_date: date,
        new_asset_id: "AssetId",
        distribution_ratio: float,
        record_date: date | None = None,
        payment_date: date | None = None,
        announcement_date: date | None = None,
        metadata: dict[str, str] | None = None,
    ):
        super().__init__(
            action_id=action_id,
            asset_id=asset_id,
            ex_date=ex_date,
            record_date=record_date,
            payment_date=payment_date,
            announcement_date=announcement_date,
            metadata=metadata or {},
        )
        self.new_asset_id = new_asset_id
        self.distribution_ratio = distribution_ratio

    @property
    def action_type(self) -> str:
        return "SPINOFF"


class SymbolChange(CorporateAction):
    """Symbol change/reorganization."""

    def __init__(
        self,
        action_id: str,
        asset_id: "AssetId",
        ex_date: date,
        new_asset_id: "AssetId",
        conversion_ratio: float = 1.0,
        record_date: date | None = None,
        payment_date: date | None = None,
        announcement_date: date | None = None,
        metadata: dict[str, str] | None = None,
    ):
        super().__init__(
            action_id=action_id,
            asset_id=asset_id,
            ex_date=ex_date,
            record_date=record_date,
            payment_date=payment_date,
            announcement_date=announcement_date,
            metadata=metadata or {},
        )
        self.new_asset_id = new_asset_id
        self.conversion_ratio = conversion_ratio

    @property
    def action_type(self) -> str:
        return "SYMBOL_CHANGE"


class RightsOffering(CorporateAction):
    """Rights offering corporate action."""

    def __init__(
        self,
        action_id: str,
        asset_id: "AssetId",
        ex_date: date,
        subscription_price: float,
        rights_ratio: float,
        shares_per_right: float,
        expiration_date: date,
        record_date: date | None = None,
        payment_date: date | None = None,
        announcement_date: date | None = None,
        metadata: dict[str, str] | None = None,
    ):
        super().__init__(
            action_id=action_id,
            asset_id=asset_id,
            ex_date=ex_date,
            record_date=record_date,
            payment_date=payment_date,
            announcement_date=announcement_date,
            metadata=metadata or {},
        )
        self.subscription_price = subscription_price
        self.rights_ratio = rights_ratio
        self.shares_per_right = shares_per_right
        self.expiration_date = expiration_date

    @property
    def action_type(self) -> str:
        return "RIGHTS_OFFERING"


class CorporateActionProcessor:
    """Processes corporate actions and adjusts positions/orders."""

    def __init__(self):
        """Initialize corporate action processor."""
        self.pending_actions: list[CorporateAction] = []
        self.processed_actions: list[CorporateAction] = []

    def add_action(self, action: CorporateAction) -> None:
        """Add a corporate action for processing.

        Args:
            action: Corporate action to add
        """
        self.pending_actions.append(action)
        # Sort by ex-date to ensure proper processing order
        self.pending_actions.sort(key=lambda a: a.ex_date)
        logger.info(
            f"Added corporate action: {action.action_id} ({action.action_type}) for {action.asset_id}",
        )

    def get_pending_actions(self, as_of_date: date) -> list[CorporateAction]:
        """Get actions that should be processed on the given date.

        Args:
            as_of_date: Date to check for pending actions

        Returns:
            List of actions to process
        """
        return [action for action in self.pending_actions if action.ex_date <= as_of_date]

    def process_actions(
        self,
        as_of_date: date,
        positions: dict["AssetId", "Quantity"],
        orders: list["Order"],
        cash: float,
    ) -> tuple[dict["AssetId", "Quantity"], list["Order"], float, list[str]]:
        """Process all pending corporate actions as of the given date.

        Args:
            as_of_date: Date to process actions through
            positions: Current position quantities by asset
            orders: List of open orders
            cash: Current cash balance

        Returns:
            Tuple of (updated_positions, updated_orders, updated_cash, notifications)
        """
        notifications = []
        # Handle different types of positions objects
        if hasattr(positions, 'clone'):  # Polars DataFrame
            updated_positions = positions.clone()
        elif hasattr(positions, 'copy'):  # Dict or pandas DataFrame
            updated_positions = positions.copy()
        else:
            updated_positions = positions  # Fallback
        updated_orders = orders.copy() if hasattr(orders, 'copy') else list(orders)
        updated_cash = cash

        pending = self.get_pending_actions(as_of_date)

        for action in pending:
            logger.info(f"Processing {action.action_type} for {action.asset_id} on {as_of_date}")

            if isinstance(action, CashDividend):
                updated_cash, notification = self._process_cash_dividend(
                    action,
                    updated_positions,
                    updated_cash,
                )
                notifications.append(notification)

            elif isinstance(action, StockSplit):
                updated_positions, updated_orders, notification = self._process_stock_split(
                    action,
                    updated_positions,
                    updated_orders,
                )
                notifications.append(notification)

            elif isinstance(action, StockDividend):
                updated_positions, notification = self._process_stock_dividend(
                    action,
                    updated_positions,
                )
                notifications.append(notification)

            elif isinstance(action, Merger):
                updated_positions, updated_cash, notification = self._process_merger(
                    action,
                    updated_positions,
                    updated_cash,
                )
                notifications.append(notification)

            elif isinstance(action, SpinOff):
                updated_positions, notification = self._process_spinoff(
                    action,
                    updated_positions,
                )
                notifications.append(notification)

            elif isinstance(action, SymbolChange):
                updated_positions, updated_orders, notification = self._process_symbol_change(
                    action,
                    updated_positions,
                    updated_orders,
                )
                notifications.append(notification)

            elif isinstance(action, RightsOffering):
                # Rights offerings are complex and typically require user decision
                # For now, just notify
                notifications.append(
                    f"Rights offering for {action.asset_id}: "
                    f"{action.rights_ratio} rights per share, "
                    f"subscription price ${action.subscription_price:.2f}",
                )

            # Move to processed
            self.processed_actions.append(action)
            self.pending_actions.remove(action)

        return updated_positions, updated_orders, updated_cash, notifications

    def _process_cash_dividend(
        self,
        dividend: CashDividend,
        positions: dict["AssetId", "Quantity"],
        cash: float,
    ) -> tuple[float, str]:
        """Process cash dividend.

        Args:
            dividend: Dividend action
            positions: Current positions
            cash: Current cash balance

        Returns:
            Tuple of (updated_cash, notification)
        """
        position = positions.get(dividend.asset_id, 0.0)
        if position > 0:
            dividend_payment = position * dividend.dividend_per_share
            cash += dividend_payment
            notification = (
                f"Dividend received: {position:.0f} shares of {dividend.asset_id} "
                f"× ${dividend.dividend_per_share:.4f} = ${dividend_payment:.2f}"
            )
            logger.info(notification)
            return cash, notification

        return cash, f"No position in {dividend.asset_id} for dividend"

    def _process_stock_split(
        self,
        split: StockSplit,
        positions: dict["AssetId", "Quantity"],
        orders: list["Order"],
    ) -> tuple[dict["AssetId", "Quantity"], list["Order"], str]:
        """Process stock split.

        Args:
            split: Stock split action
            positions: Current positions
            orders: Open orders

        Returns:
            Tuple of (updated_positions, updated_orders, notification)
        """
        # Adjust position
        if split.asset_id in positions:
            old_position = positions[split.asset_id]
            positions[split.asset_id] = old_position * split.split_ratio
            notification = (
                f"Stock split: {split.asset_id} {split.split_ratio}:1 split - "
                f"Position adjusted from {old_position:.0f} to {positions[split.asset_id]:.0f} shares"
            )
        else:
            notification = f"No position in {split.asset_id} for stock split"

        # Adjust open orders
        updated_orders = []
        for order in orders:
            if order.asset_id == split.asset_id:
                # Adjust both total quantity and filled quantity for partial fills
                order.quantity *= split.split_ratio
                order.filled_quantity *= split.split_ratio

                # Adjust prices (inverse of split ratio)
                if order.limit_price is not None:
                    order.limit_price /= split.split_ratio
                if order.stop_price is not None:
                    order.stop_price /= split.split_ratio

                # Also adjust average fill price for partial fills
                if order.average_fill_price is not None and order.average_fill_price > 0:
                    order.average_fill_price /= split.split_ratio

                order.metadata["corporate_action"] = (
                    f"Split {split.split_ratio}:1 on {split.ex_date}"
                )
            updated_orders.append(order)

        logger.info(notification)
        return positions, updated_orders, notification

    def _process_stock_dividend(
        self,
        stock_div: StockDividend,
        positions: dict["AssetId", "Quantity"],
    ) -> tuple[dict["AssetId", "Quantity"], str]:
        """Process stock dividend.

        Args:
            stock_div: Stock dividend action
            positions: Current positions

        Returns:
            Tuple of (updated_positions, notification)
        """
        if stock_div.asset_id in positions:
            old_position = positions[stock_div.asset_id]
            additional_shares = old_position * stock_div.dividend_ratio
            positions[stock_div.asset_id] += additional_shares

            notification = (
                f"Stock dividend: {stock_div.asset_id} "
                f"{stock_div.dividend_ratio * 100:.1f}% stock dividend - "
                f"Received {additional_shares:.0f} additional shares"
            )
        else:
            notification = f"No position in {stock_div.asset_id} for stock dividend"

        logger.info(notification)
        return positions, notification

    def _process_merger(
        self,
        merger: Merger,
        positions: dict["AssetId", "Quantity"],
        cash: float,
    ) -> tuple[dict["AssetId", "Quantity"], float, str]:
        """Process merger/acquisition.

        Args:
            merger: Merger action
            positions: Current positions
            cash: Current cash balance

        Returns:
            Tuple of (updated_positions, updated_cash, notification)
        """
        if merger.asset_id not in positions or positions[merger.asset_id] <= 0:
            return positions, cash, f"No position in {merger.asset_id} for merger"

        old_shares = positions[merger.asset_id]

        # Remove old position
        del positions[merger.asset_id]

        # Add cash consideration
        cash_received = old_shares * merger.cash_consideration
        cash += cash_received

        # Add stock consideration
        if merger.stock_consideration > 0:
            new_shares = old_shares * merger.stock_consideration
            if merger.target_asset_id in positions:
                positions[merger.target_asset_id] += new_shares
            else:
                positions[merger.target_asset_id] = new_shares

        notification = (
            f"Merger: {merger.asset_id} → {merger.target_asset_id} - "
            f"{old_shares:.0f} shares converted to "
        )

        if cash_received > 0 and merger.stock_consideration > 0:
            notification += f"${cash_received:.2f} cash + {old_shares * merger.stock_consideration:.0f} {merger.target_asset_id} shares"
        elif cash_received > 0:
            notification += f"${cash_received:.2f} cash"
        else:
            notification += (
                f"{old_shares * merger.stock_consideration:.0f} {merger.target_asset_id} shares"
            )

        logger.info(notification)
        return positions, cash, notification

    def _process_spinoff(
        self,
        spinoff: SpinOff,
        positions: dict["AssetId", "Quantity"],
    ) -> tuple[dict["AssetId", "Quantity"], str]:
        """Process spin-off.

        Args:
            spinoff: Spin-off action
            positions: Current positions

        Returns:
            Tuple of (updated_positions, notification)
        """
        if spinoff.asset_id not in positions or positions[spinoff.asset_id] <= 0:
            return positions, f"No position in {spinoff.asset_id} for spin-off"

        parent_shares = positions[spinoff.asset_id]
        spinoff_shares = parent_shares * spinoff.distribution_ratio

        # Add spin-off shares
        if spinoff.new_asset_id in positions:
            positions[spinoff.new_asset_id] += spinoff_shares
        else:
            positions[spinoff.new_asset_id] = spinoff_shares

        notification = (
            f"Spin-off: {spinoff.asset_id} distributed {spinoff_shares:.0f} shares of "
            f"{spinoff.new_asset_id} ({spinoff.distribution_ratio} per share)"
        )

        logger.info(notification)
        return positions, notification

    def _process_symbol_change(
        self,
        symbol_change: SymbolChange,
        positions: dict["AssetId", "Quantity"],
        orders: list["Order"],
    ) -> tuple[dict["AssetId", "Quantity"], list["Order"], str]:
        """Process symbol change.

        Args:
            symbol_change: Symbol change action
            positions: Current positions
            orders: Open orders

        Returns:
            Tuple of (updated_positions, updated_orders, notification)
        """
        # Update position
        if symbol_change.asset_id in positions:
            old_shares = positions[symbol_change.asset_id]
            new_shares = old_shares * symbol_change.conversion_ratio

            del positions[symbol_change.asset_id]
            positions[symbol_change.new_asset_id] = new_shares

            notification = (
                f"Symbol change: {symbol_change.asset_id} → {symbol_change.new_asset_id} "
                f"({old_shares:.0f} → {new_shares:.0f} shares)"
            )
        else:
            notification = f"Symbol change: {symbol_change.asset_id} → {symbol_change.new_asset_id} (no position)"

        # Update orders
        for order in orders:
            if order.asset_id == symbol_change.asset_id:
                order.asset_id = symbol_change.new_asset_id
                # Adjust both total quantity and filled quantity for partial fills
                order.quantity *= symbol_change.conversion_ratio
                order.filled_quantity *= symbol_change.conversion_ratio

                if symbol_change.conversion_ratio != 1.0:
                    # Adjust prices (inverse of conversion ratio)
                    if order.limit_price is not None:
                        order.limit_price /= symbol_change.conversion_ratio
                    if order.stop_price is not None:
                        order.stop_price /= symbol_change.conversion_ratio
                    # Also adjust average fill price for partial fills
                    if order.average_fill_price is not None and order.average_fill_price > 0:
                        order.average_fill_price /= symbol_change.conversion_ratio

                order.metadata["corporate_action"] = f"Symbol change on {symbol_change.ex_date}"

        logger.info(notification)
        return positions, orders, notification

    def adjust_price_for_actions(
        self,
        asset_id: "AssetId",
        price: "Price",
        as_of_date: date,
    ) -> "Price":
        """Adjust historical price for corporate actions.

        This is used to maintain price continuity in backtesting by adjusting
        historical prices for splits, dividends, etc.

        Args:
            asset_id: Asset to adjust price for
            price: Original price
            as_of_date: Date the price is from

        Returns:
            Adjusted price
        """
        adjusted_price = price

        # Apply adjustments for all actions after this date
        for action in self.processed_actions:
            if action.asset_id != asset_id or action.ex_date <= as_of_date:
                continue

            if isinstance(action, StockSplit):
                # Adjust price downward for future splits
                adjusted_price /= action.split_ratio

            elif isinstance(action, CashDividend):
                # Adjust price downward for future dividends
                adjusted_price -= action.dividend_per_share

            elif isinstance(action, StockDividend):
                # Adjust price for stock dividend
                adjusted_price /= 1 + action.dividend_ratio

        return max(adjusted_price, 0.01)  # Minimum price floor

    def get_processed_actions(
        self,
        asset_id: Optional["AssetId"] = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[CorporateAction]:
        """Get processed corporate actions with optional filtering.

        Args:
            asset_id: Filter by asset ID
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)

        Returns:
            List of matching corporate actions
        """
        filtered_actions = self.processed_actions

        if asset_id:
            filtered_actions = [a for a in filtered_actions if a.asset_id == asset_id]

        if start_date:
            filtered_actions = [a for a in filtered_actions if a.ex_date >= start_date]

        if end_date:
            filtered_actions = [a for a in filtered_actions if a.ex_date <= end_date]

        return filtered_actions

    def reset(self) -> None:
        """Reset processor state."""
        self.pending_actions.clear()
        self.processed_actions.clear()
        logger.info("Corporate action processor reset")


class CorporateActionDataProvider:
    """Provides corporate action data from various sources."""

    def __init__(self):
        """Initialize data provider."""
        self.actions: dict[str, CorporateAction] = {}

    def load_from_csv(self, file_path: str) -> None:
        """Load corporate actions from CSV file.

        Expected CSV format:
        action_id,asset_id,action_type,ex_date,dividend_per_share,split_ratio,...

        Args:
            file_path: Path to CSV file
        """
        import pandas as pd

        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            action = self._create_action_from_row(row)
            if action:
                self.actions[action.action_id] = action
                logger.info(f"Loaded corporate action: {action.action_id}")

    def _create_action_from_row(self, row) -> CorporateAction | None:
        """Create corporate action from CSV row."""
        try:
            action_type = row["action_type"].upper()
            import pandas as pd

            ex_date = pd.to_datetime(row["ex_date"]).date()

            base_args = {
                "action_id": row["action_id"],
                "asset_id": row["asset_id"],
                "ex_date": ex_date,
                "record_date": pd.to_datetime(row.get("record_date")).date()
                if pd.notna(row.get("record_date"))
                else None,
                "payment_date": pd.to_datetime(row.get("payment_date")).date()
                if pd.notna(row.get("payment_date"))
                else None,
            }

            if action_type == "DIVIDEND":
                return CashDividend(
                    dividend_per_share=float(row["dividend_per_share"]),
                    **base_args,
                )
            if action_type == "SPLIT":
                return StockSplit(
                    split_ratio=float(row["split_ratio"]),
                    **base_args,
                )
            if action_type == "MERGER":
                return Merger(
                    target_asset_id=row["target_asset_id"],
                    cash_consideration=float(row.get("cash_consideration", 0)),
                    stock_consideration=float(row.get("stock_consideration", 0)),
                    **base_args,
                )
            if action_type == "SPINOFF":
                return SpinOff(
                    new_asset_id=row["new_asset_id"],
                    distribution_ratio=float(row["distribution_ratio"]),
                    **base_args,
                )
            if action_type == "SYMBOL_CHANGE":
                return SymbolChange(
                    new_asset_id=row["new_asset_id"],
                    conversion_ratio=float(row.get("conversion_ratio", 1.0)),
                    **base_args,
                )
            logger.warning(f"Unknown action type: {action_type}")
            return None

        except Exception as e:
            logger.error(f"Error creating action from row: {e}")
            return None

    def get_actions_for_asset(
        self,
        asset_id: "AssetId",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[CorporateAction]:
        """Get actions for a specific asset.

        Args:
            asset_id: Asset to get actions for
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of corporate actions
        """
        actions = [action for action in self.actions.values() if action.asset_id == asset_id]

        if start_date:
            actions = [a for a in actions if a.ex_date >= start_date]

        if end_date:
            actions = [a for a in actions if a.ex_date <= end_date]

        return sorted(actions, key=lambda a: a.ex_date)
