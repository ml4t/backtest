"""Asset class definitions and specifications for ml4t.backtest."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ml4t.backtest.core.types import AssetId, Price


class AssetClass(Enum):
    """Supported asset classes."""

    EQUITY = "equity"
    FUTURE = "future"
    OPTION = "option"
    FX = "fx"
    CRYPTO = "crypto"
    BOND = "bond"
    COMMODITY = "commodity"


class ContractType(Enum):
    """Contract types for derivatives."""

    SPOT = "spot"
    FUTURE = "future"
    PERPETUAL = "perpetual"
    CALL = "call"
    PUT = "put"


@dataclass
class AssetSpec:
    """
    Complete specification for an asset.

    This class handles the different requirements for various asset classes:
    - Equities: Simple spot trading with T+2 settlement
    - Futures: Margin requirements, expiry, rolling
    - Options: Greeks, expiry, exercise
    - FX: Currency pairs, pip values
    - Crypto: 24/7 trading, fractional shares
    """

    asset_id: AssetId
    asset_class: AssetClass
    contract_type: ContractType = ContractType.SPOT

    # Common fields
    currency: str = "USD"
    tick_size: float = 0.01
    lot_size: float = 1.0
    min_quantity: float = 1.0

    # Equity-specific
    exchange: str | None = None

    # Futures-specific
    contract_size: float = 1.0  # Multiplier for futures/options
    initial_margin: float = 0.0  # Initial margin requirement
    maintenance_margin: float = 0.0  # Maintenance margin requirement
    expiry: datetime | None = None
    underlying: AssetId | None = None  # For derivatives
    roll_date: datetime | None = None  # When to roll to next contract

    # Options-specific
    strike: Price | None = None
    option_type: str | None = None  # "call" or "put"
    exercise_style: str | None = None  # "american", "european"

    # FX-specific
    base_currency: str | None = None
    quote_currency: str | None = None
    pip_value: float = 0.0001  # Standard pip value

    # Crypto-specific
    is_24_7: bool = False  # Trades 24/7
    network_fees: bool = False  # Has blockchain network fees

    # Trading specifications
    maker_fee: float = 0.001  # 0.1% default
    taker_fee: float = 0.001  # 0.1% default
    short_enabled: bool = True
    leverage_available: float = 1.0  # Max leverage

    # Precision overrides (optional - if None, uses asset class defaults)
    position_decimals: int | None = None  # Override position rounding (None = use asset class default)
    price_decimals: int | None = None  # Override price rounding (None = use asset class default)
    cash_decimals: int | None = None  # Override cash/commission rounding (None = use asset class default)

    @property
    def is_derivative(self) -> bool:
        """Check if asset is a derivative."""
        return self.asset_class in [AssetClass.FUTURE, AssetClass.OPTION]

    @property
    def requires_margin(self) -> bool:
        """Check if asset requires margin."""
        return self.asset_class in [AssetClass.FUTURE, AssetClass.FX] or self.leverage_available > 1

    @property
    def has_expiry(self) -> bool:
        """Check if asset has expiry."""
        return self.expiry is not None

    def get_precision_manager(self) -> "PrecisionManager":
        """Create PrecisionManager for this asset.

        Returns:
            PrecisionManager configured with this asset's precision rules
            (uses asset class defaults with optional per-asset overrides)
        """
        from ml4t.backtest.core.precision import PrecisionManager

        return PrecisionManager.from_asset_spec(self)

    def get_margin_requirement(self, quantity: float, price: Price) -> float:
        """
        Calculate margin requirement for position.

        Args:
            quantity: Position size
            price: Current price

        Returns:
            Required margin
        """
        if self.asset_class == AssetClass.FUTURE:
            # Futures use fixed margin per contract
            return abs(quantity) * self.initial_margin
        if self.asset_class == AssetClass.FX:
            # FX uses percentage of notional
            notional = abs(quantity) * price
            return notional / self.leverage_available if self.leverage_available > 0 else notional
        if self.asset_class == AssetClass.CRYPTO and self.leverage_available > 1:
            # Leveraged crypto trading
            notional = abs(quantity) * price
            return notional / self.leverage_available
        if self.asset_class == AssetClass.OPTION:
            # Options: buyers pay premium, sellers need margin
            if quantity > 0:  # Buying options
                return abs(quantity) * price * self.contract_size
            # Selling options - simplified margin
            return abs(quantity) * self.strike * self.contract_size * 0.2  # 20% of notional
        # Spot trading - full cash required
        return abs(quantity) * price

    def get_notional_value(self, quantity: float, price: Price) -> float:
        """
        Calculate notional value of position.

        Args:
            quantity: Position size
            price: Current price

        Returns:
            Notional value
        """
        if self.asset_class in [AssetClass.FUTURE, AssetClass.OPTION]:
            return abs(quantity) * price * self.contract_size
        if self.asset_class == AssetClass.FX:
            # FX notional in base currency
            return abs(quantity) * price
        return abs(quantity) * price

    def calculate_pnl(
        self,
        entry_price: Price,
        exit_price: Price,
        quantity: float,
        include_costs: bool = True,
    ) -> float:
        """
        Calculate P&L for a trade.

        Args:
            entry_price: Entry price (for options: premium per contract, not underlying price)
            exit_price: Exit price (for options: premium per contract, not underlying price)
            quantity: Position size (positive for long, negative for short)
            include_costs: Whether to include trading costs

        Returns:
            Profit/loss

        Note:
            For options, entry_price and exit_price must be the option premiums,
            NOT the underlying asset prices. Use calculate_pnl_premium_based()
            for explicit premium-based calculation.
        """
        if self.asset_class == AssetClass.FUTURE:
            # Futures P&L includes contract multiplier
            pnl = quantity * (exit_price - entry_price) * self.contract_size
        elif self.asset_class == AssetClass.OPTION:
            # Options P&L based on premium change (not intrinsic value)
            # Note: entry_price and exit_price should be option premiums, not underlying prices
            # For positions closed before expiry, P&L = (exit_premium - entry_premium) * quantity * contract_size
            # This calculation assumes entry_price and exit_price are the option premiums
            pnl = quantity * (exit_price - entry_price) * self.contract_size

            # WARNING: If you need P&L at expiry based on intrinsic value, use a separate method
            # The above calculation is correct for trading options before expiry
        elif self.asset_class == AssetClass.FX:
            # FX P&L in quote currency
            pnl = quantity * (exit_price - entry_price)
            # Note: pip_value is typically the value of one pip in the quote currency
            # No division needed - the P&L is already in the correct currency units
        else:
            # Standard P&L calculation
            pnl = quantity * (exit_price - entry_price)

        # Subtract trading costs if requested
        if include_costs:
            entry_cost = abs(quantity * entry_price) * self.taker_fee
            exit_cost = abs(quantity * exit_price) * self.taker_fee
            pnl -= entry_cost + exit_cost

        return pnl

    def calculate_pnl_premium_based(
        self,
        entry_premium: Price,
        exit_premium: Price,
        quantity: float,
        include_costs: bool = True,
    ) -> float:
        """
        Calculate P&L for options using premium change methodology.

        This is the CORRECT way to calculate options P&L for positions
        closed before expiry. It uses the change in option premium,
        not intrinsic value.

        Args:
            entry_premium: Option premium at entry
            exit_premium: Option premium at exit
            quantity: Position size (positive for long, negative for short)
            include_costs: Whether to include trading costs

        Returns:
            Profit/loss based on premium change

        Raises:
            ValueError: If called on non-option assets

        Example:
            # Long 1 call option: bought at $2.00, sold at $1.50
            # P&L = (1.50 - 2.00) * 1 * 100 = -$50
            pnl = call_spec.calculate_pnl_premium_based(2.00, 1.50, 1.0)
            assert pnl == -50.0
        """
        if self.asset_class != AssetClass.OPTION:
            raise ValueError("Premium-based P&L calculation is only for options")

        # CORRECT: P&L = (exit_premium - entry_premium) * quantity * contract_size
        pnl = (exit_premium - entry_premium) * quantity * self.contract_size

        # Subtract trading costs if requested
        if include_costs:
            entry_cost = abs(quantity * entry_premium) * getattr(self, "taker_fee", 0.0)
            exit_cost = abs(quantity * exit_premium) * getattr(self, "taker_fee", 0.0)
            pnl -= entry_cost + exit_cost

        return pnl

    def calculate_option_pnl_at_expiry(
        self,
        entry_premium: Price,
        underlying_price_at_expiry: Price,
        quantity: float,
        option_type: str = "call",  # "call" or "put"
        include_costs: bool = True,
    ) -> float:
        """
        Calculate P&L for options held to expiry based on intrinsic value.

        Args:
            entry_premium: Premium paid/received when opening position
            underlying_price_at_expiry: Price of underlying asset at expiry
            quantity: Position size (positive for long, negative for short)
            option_type: "call" or "put"
            include_costs: Whether to include trading costs

        Returns:
            Profit/loss at expiry

        Raises:
            ValueError: If called on non-option assets or invalid option type
        """
        if self.asset_class != AssetClass.OPTION:
            raise ValueError("Expiry P&L calculation is only for options")

        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")

        if self.strike is None:
            raise ValueError("Option strike price is required for expiry P&L")

        # Calculate intrinsic value at expiry
        if option_type == "call":
            intrinsic_value = max(0, underlying_price_at_expiry - self.strike)
        else:  # put
            intrinsic_value = max(0, self.strike - underlying_price_at_expiry)

        # For long positions: P&L = (intrinsic_value - entry_premium) * quantity * contract_size
        # For short positions: P&L = (entry_premium - intrinsic_value) * quantity * contract_size
        if quantity > 0:  # Long option
            pnl = (intrinsic_value - entry_premium) * quantity * self.contract_size
        else:  # Short option
            pnl = (entry_premium - intrinsic_value) * abs(quantity) * self.contract_size

        # Subtract trading costs if requested (only entry cost for expiry)
        if include_costs:
            entry_cost = abs(quantity * entry_premium) * getattr(self, "taker_fee", 0.0)
            pnl -= entry_cost

        return pnl

    def calculate_pnl_enhanced(
        self,
        entry_price: Price,
        exit_price: Price,
        quantity: float,
        entry_premium: Price = None,
        exit_premium: Price = None,
        include_costs: bool = True,
    ) -> float:
        """
        Enhanced P&L calculation with options premium support.

        For options, this method will use premium-based calculation when
        premium data is provided, otherwise it assumes entry_price and exit_price
        are premiums (NOT underlying prices).

        Args:
            entry_price: Entry price (premium for options if entry_premium not provided)
            exit_price: Exit price (premium for options if exit_premium not provided)
            quantity: Position size
            entry_premium: Option premium at entry (for options only, overrides entry_price)
            exit_premium: Option premium at exit (for options only, overrides exit_price)
            include_costs: Whether to include trading costs

        Returns:
            Profit/loss

        Note:
            For options without explicit premium parameters, entry_price and exit_price
            are treated as premiums, NOT as underlying asset prices.
        """
        if (
            self.asset_class == AssetClass.OPTION
            and entry_premium is not None
            and exit_premium is not None
        ):
            # Use premium-based calculation for options when premium data available
            return self.calculate_pnl_premium_based(
                entry_premium,
                exit_premium,
                quantity,
                include_costs,
            )
        # Use original calculation method
        return self.calculate_pnl(entry_price, exit_price, quantity, include_costs)


class AssetRegistry:
    """Registry for managing asset specifications."""

    def __init__(self):
        """Initialize asset registry."""
        self._assets: dict[AssetId, AssetSpec] = {}

    def register(self, asset_spec: AssetSpec) -> None:
        """Register an asset specification."""
        self._assets[asset_spec.asset_id] = asset_spec

    def get(self, asset_id: AssetId) -> AssetSpec | None:
        """Get asset specification by ID."""
        return self._assets.get(asset_id)

    def get_or_create_equity(self, asset_id: AssetId) -> AssetSpec:
        """Get or create a default equity specification."""
        if asset_id not in self._assets:
            self._assets[asset_id] = AssetSpec(
                asset_id=asset_id,
                asset_class=AssetClass.EQUITY,
                contract_type=ContractType.SPOT,
            )
        return self._assets[asset_id]

    def create_future(
        self,
        asset_id: AssetId,
        underlying: AssetId,
        expiry: datetime,
        contract_size: float = 1.0,
        initial_margin: float = 0.0,
        maintenance_margin: float = 0.0,
    ) -> AssetSpec:
        """Create a futures contract specification."""
        spec = AssetSpec(
            asset_id=asset_id,
            asset_class=AssetClass.FUTURE,
            contract_type=ContractType.FUTURE,
            underlying=underlying,
            expiry=expiry,
            contract_size=contract_size,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
        )
        self._assets[asset_id] = spec
        return spec

    def create_option(
        self,
        asset_id: AssetId,
        underlying: AssetId,
        strike: Price,
        expiry: datetime,
        option_type: str,
        contract_size: float = 100.0,
        exercise_style: str = "american",
    ) -> AssetSpec:
        """Create an option contract specification."""
        spec = AssetSpec(
            asset_id=asset_id,
            asset_class=AssetClass.OPTION,
            contract_type=ContractType.CALL if option_type == "call" else ContractType.PUT,
            underlying=underlying,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            contract_size=contract_size,
            exercise_style=exercise_style,
        )
        self._assets[asset_id] = spec
        return spec

    def create_fx_pair(
        self,
        asset_id: AssetId,
        base_currency: str,
        quote_currency: str,
        pip_value: float = 0.0001,
        leverage_available: float = 100.0,
    ) -> AssetSpec:
        """Create an FX pair specification."""
        spec = AssetSpec(
            asset_id=asset_id,
            asset_class=AssetClass.FX,
            contract_type=ContractType.SPOT,
            base_currency=base_currency,
            quote_currency=quote_currency,
            currency=quote_currency,
            pip_value=pip_value,
            leverage_available=leverage_available,
            tick_size=pip_value,
            lot_size=1000.0,  # Mini lot
        )
        self._assets[asset_id] = spec
        return spec

    def create_crypto(
        self,
        asset_id: AssetId,
        base_currency: str,
        quote_currency: str = "USD",
        min_quantity: float = 0.00001,
        maker_fee: float = 0.001,
        taker_fee: float = 0.001,
        leverage_available: float = 1.0,
    ) -> AssetSpec:
        """Create a cryptocurrency specification."""
        spec = AssetSpec(
            asset_id=asset_id,
            asset_class=AssetClass.CRYPTO,
            contract_type=ContractType.SPOT,
            base_currency=base_currency,
            quote_currency=quote_currency,
            currency=quote_currency,
            min_quantity=min_quantity,
            tick_size=0.01,
            lot_size=1.0,
            is_24_7=True,
            network_fees=True,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            leverage_available=leverage_available,
        )
        self._assets[asset_id] = spec
        return spec
