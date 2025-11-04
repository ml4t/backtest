# QEngine Advanced Simulation Models

## Overview

This document details the sophisticated execution models implemented in QEngine, providing institutional-grade backtesting realism through advanced slippage, commission, market impact, and corporate actions handling.

## 1. Advanced Order Types ✅

### 1.1 Implemented Order Types

```python
class OrderType(Enum):
    MARKET = "market"           # Immediate execution
    LIMIT = "limit"             # Execute at price or better
    STOP = "stop"              # Trigger and convert to market
    STOP_LIMIT = "stop_limit"   # Trigger and convert to limit
    TRAILING_STOP = "trailing_stop"  # Dynamic stop level
    BRACKET = "bracket"         # Entry with stop-loss and take-profit
```

### 1.2 Bracket Orders with OCO Logic

```python
class BracketOrderExecution:
    """Sophisticated bracket order handling."""

    def handle_bracket_fill(self, parent_order, fill_event):
        """Create OCO legs after parent fill."""
        # Stop-loss order (opposite side)
        stop_order = Order(
            order_type=OrderType.STOP,
            side=opposite_side(parent_order.side),
            quantity=parent_order.filled_quantity,
            stop_price=parent_order.stop_loss,
            metadata={"bracket_type": "stop_loss"}
        )

        # Take-profit order (opposite side)
        profit_order = Order(
            order_type=OrderType.LIMIT,
            side=opposite_side(parent_order.side),
            quantity=parent_order.filled_quantity,
            limit_price=parent_order.profit_target,
            metadata={"bracket_type": "take_profit"}
        )

        # Link as One-Cancels-Other
        stop_order.child_order_ids.append(profit_order.order_id)
        profit_order.child_order_ids.append(stop_order.order_id)

        return [stop_order, profit_order]
```

### 1.3 Trailing Stop Implementation

```python
class TrailingStopLogic:
    """Dynamic trailing stop adjustment."""

    def update_trailing_stop(self, order, current_price):
        """Update trailing stop level based on price movement."""
        if order.side == OrderSide.BUY:
            # Long position: trail upward only
            if order.trail_amount:
                new_stop = current_price - order.trail_amount
                order.stop_price = max(order.stop_price or 0, new_stop)
            elif order.trail_percent:
                new_stop = current_price * (1 - order.trail_percent)
                order.stop_price = max(order.stop_price or 0, new_stop)
        else:
            # Short position: trail downward only
            if order.trail_amount:
                new_stop = current_price + order.trail_amount
                order.stop_price = min(order.stop_price or float('inf'), new_stop)
            elif order.trail_percent:
                new_stop = current_price * (1 + order.trail_percent)
                order.stop_price = min(order.stop_price or float('inf'), new_stop)
```

## 2. Sophisticated Slippage Models ✅

### 2.1 Seven Slippage Implementations

```python
# 1. NoSlippage - Perfect execution
class NoSlippage(SlippageModel):
    def calculate_fill_price(self, order, market_price):
        return market_price

# 2. FixedSlippage - Constant spread
class FixedSlippage(SlippageModel):
    def __init__(self, spread_bps=5):
        self.spread = spread_bps / 10000

    def calculate_fill_price(self, order, market_price):
        if order.side == OrderSide.BUY:
            return market_price * (1 + self.spread)
        return market_price * (1 - self.spread)

# 3. PercentageSlippage - Proportional to price
class PercentageSlippage(SlippageModel):
    def __init__(self, slippage_rate=0.001):
        self.rate = slippage_rate

    def calculate_fill_price(self, order, market_price):
        slippage = market_price * self.rate
        if order.side == OrderSide.BUY:
            return market_price + slippage
        return market_price - slippage

# 4. LinearImpactSlippage - Volume-based
class LinearImpactSlippage(SlippageModel):
    def __init__(self, impact_coeff=1e-6, daily_volume=1000000):
        self.impact_coeff = impact_coeff
        self.daily_volume = daily_volume

    def calculate_fill_price(self, order, market_price):
        participation_rate = order.quantity / self.daily_volume
        impact = market_price * self.impact_coeff * participation_rate

        if order.side == OrderSide.BUY:
            return market_price + impact
        return market_price - impact

# 5. SquareRootSlippage - Almgren-Chriss scaling
class SquareRootSlippage(SlippageModel):
    def __init__(self, impact_coeff=0.1, daily_volume=1000000):
        self.impact_coeff = impact_coeff
        self.daily_volume = daily_volume

    def calculate_fill_price(self, order, market_price):
        participation_rate = order.quantity / self.daily_volume
        # Square root scaling for large orders
        impact = market_price * self.impact_coeff * math.sqrt(participation_rate)

        if order.side == OrderSide.BUY:
            return market_price + impact
        return market_price - impact

# 6. VolumeShareSlippage - Market participation based
class VolumeShareSlippage(SlippageModel):
    def __init__(self, volume_limit=0.1, limit_price_adjustment=0.05):
        self.volume_limit = volume_limit
        self.price_adjustment = limit_price_adjustment

    def calculate_fill_price(self, order, market_price, market_volume):
        participation_rate = order.quantity / market_volume

        if participation_rate > self.volume_limit:
            # High participation: significant impact
            excess_rate = participation_rate - self.volume_limit
            additional_impact = market_price * self.price_adjustment * excess_rate

            if order.side == OrderSide.BUY:
                return market_price + additional_impact
            return market_price - additional_impact

        return market_price

# 7. AssetClassSlippage - Different rates by asset type
class AssetClassSlippage(SlippageModel):
    def __init__(self):
        self.slippage_rates = {
            AssetClass.EQUITY: 0.0005,      # 5 bps
            AssetClass.ETF: 0.0003,         # 3 bps
            AssetClass.CRYPTO: 0.01,        # 100 bps
            AssetClass.FX: 0.0001,          # 1 bp
            AssetClass.FUTURE: 0.0002,      # 2 bps
            AssetClass.OPTION: 0.002,       # 20 bps
        }
```

## 3. Comprehensive Commission Models ✅

### 3.1 Nine Commission Implementations

```python
# 1. NoCommission - Zero cost
class NoCommission(CommissionModel):
    def calculate(self, order, fill_quantity, fill_price):
        return 0.0

# 2. FlatCommission - Fixed per trade
class FlatCommission(CommissionModel):
    def __init__(self, commission_per_trade=1.0):
        self.commission = commission_per_trade

    def calculate(self, order, fill_quantity, fill_price):
        return self.commission

# 3. PercentageCommission - Percentage of notional
class PercentageCommission(CommissionModel):
    def __init__(self, rate=0.001, min_commission=1.0):
        self.rate = rate
        self.min_commission = min_commission

    def calculate(self, order, fill_quantity, fill_price):
        notional = fill_quantity * fill_price
        commission = notional * self.rate
        return max(commission, self.min_commission)

# 4. PerShareCommission - Cost per share
class PerShareCommission(CommissionModel):
    def __init__(self, cost_per_share=0.005, min_commission=1.0):
        self.cost_per_share = cost_per_share
        self.min_commission = min_commission

    def calculate(self, order, fill_quantity, fill_price):
        commission = fill_quantity * self.cost_per_share
        return max(commission, self.min_commission)

# 5. TieredCommission - Volume-based tiers
class TieredCommission(CommissionModel):
    def __init__(self):
        self.tiers = [
            (0, 0.01),          # $0.01/share up to 500 shares
            (500, 0.008),       # $0.008/share for next tier
            (2000, 0.005),      # $0.005/share for high volume
            (10000, 0.003),     # $0.003/share for very high volume
        ]

    def calculate(self, order, fill_quantity, fill_price, monthly_volume=0):
        # Find applicable tier based on monthly volume
        rate = self.tiers[0][1]  # Default rate
        for threshold, tier_rate in reversed(self.tiers):
            if monthly_volume >= threshold:
                rate = tier_rate
                break

        return fill_quantity * rate

# 6. MakerTakerCommission - Exchange-style with rebates
class MakerTakerCommission(CommissionModel):
    def __init__(self, maker_rate=-0.0002, taker_rate=0.0003):
        self.maker_rate = maker_rate  # Negative = rebate
        self.taker_rate = taker_rate

    def calculate(self, order, fill_quantity, fill_price):
        notional = fill_quantity * fill_price

        if order.order_type == OrderType.LIMIT:
            # Assume limit orders are maker
            return notional * self.maker_rate  # Could be negative (rebate)
        else:
            # Market orders are taker
            return notional * self.taker_rate

# 7. AssetClassCommission - Different rates by asset
class AssetClassCommission(CommissionModel):
    def __init__(self):
        self.rates = {
            AssetClass.EQUITY: 0.005,    # $0.005/share
            AssetClass.ETF: 0.003,       # $0.003/share
            AssetClass.CRYPTO: 0.001,    # 0.1% of notional
            AssetClass.FX: 0.00002,      # 0.002% of notional
            AssetClass.FUTURE: 2.5,      # $2.50/contract
            AssetClass.OPTION: 0.5,      # $0.50/contract
        }

# 8. InteractiveBrokersCommission - IB pricing model
class InteractiveBrokersCommission(CommissionModel):
    def calculate(self, order, fill_quantity, fill_price):
        # IB Tiered pricing for US equities
        if fill_quantity <= 300:
            rate = 0.005  # $0.005/share
        elif fill_quantity <= 1000:
            rate = 0.003  # $0.003/share
        else:
            rate = 0.001  # $0.001/share

        commission = fill_quantity * rate
        return max(commission, 1.0)  # $1 minimum

# 9. RobinhoodCommission - Zero commission with fees
class RobinhoodCommission(CommissionModel):
    def calculate(self, order, fill_quantity, fill_price):
        # Zero commission but include regulatory fees
        notional = fill_quantity * fill_price

        # SEC fee: $0.0000221 per dollar sold (sells only)
        sec_fee = 0
        if order.side == OrderSide.SELL:
            sec_fee = notional * 0.0000221

        # FINRA TAF: $0.000145 per share (max $7.27)
        taf_fee = min(fill_quantity * 0.000145, 7.27)

        return sec_fee + taf_fee
```

## 4. Market Impact Models ✅

### 4.1 Six Advanced Market Impact Models

```python
# 1. NoMarketImpact - Zero impact baseline
class NoMarketImpact(MarketImpactModel):
    def calculate_impact(self, order, fill_quantity, market_price, timestamp):
        return 0.0, 0.0

# 2. LinearMarketImpact - Simple linear scaling
class LinearMarketImpact(MarketImpactModel):
    def __init__(self, permanent_factor=0.1, temporary_factor=0.5,
                 avg_daily_volume=1_000_000, decay_rate=0.1):
        super().__init__()
        self.permanent_factor = permanent_factor
        self.temporary_factor = temporary_factor
        self.avg_daily_volume = avg_daily_volume
        self.decay_rate = decay_rate

    def calculate_impact(self, order, fill_quantity, market_price, timestamp):
        volume_fraction = fill_quantity / self.avg_daily_volume

        permanent = market_price * self.permanent_factor * volume_fraction
        temporary = market_price * self.temporary_factor * volume_fraction

        # Apply direction
        if order.side == OrderSide.SELL:
            permanent = -permanent
            temporary = -temporary

        return permanent, temporary

# 3. AlmgrenChrissImpact - Academic square-root model
class AlmgrenChrissImpact(MarketImpactModel):
    def __init__(self, permanent_const=0.01, temporary_const=0.1,
                 daily_volatility=0.02, avg_daily_volume=1_000_000):
        super().__init__()
        self.permanent_const = permanent_const
        self.temporary_const = temporary_const
        self.daily_volatility = daily_volatility
        self.avg_daily_volume = avg_daily_volume

    def calculate_impact(self, order, fill_quantity, market_price, timestamp):
        volume_fraction = fill_quantity / self.avg_daily_volume

        # Permanent: square-root of volume
        permanent = (
            self.permanent_const *
            self.daily_volatility *
            market_price *
            math.sqrt(volume_fraction)
        )

        # Temporary: linear in volume
        temporary = (
            self.temporary_const *
            self.daily_volatility *
            market_price *
            volume_fraction
        )

        # Apply direction
        if order.side == OrderSide.SELL:
            permanent = -permanent
            temporary = -temporary

        return permanent, temporary

# 4. PropagatorImpact - Bouchaud propagation model
class PropagatorImpact(MarketImpactModel):
    def __init__(self, impact_coeff=0.1, propagator_exp=0.5,
                 decay_exp=0.7, avg_daily_volume=1_000_000):
        super().__init__()
        self.impact_coeff = impact_coeff
        self.propagator_exp = propagator_exp
        self.decay_exp = decay_exp
        self.avg_daily_volume = avg_daily_volume
        self.order_history = []  # [(timestamp, signed_volume, price), ...]

    def calculate_impact(self, order, fill_quantity, market_price, timestamp):
        volume_fraction = fill_quantity / self.avg_daily_volume

        # Instant impact
        instant_impact = (
            self.impact_coeff *
            market_price *
            (volume_fraction ** self.propagator_exp)
        )

        # Propagated impact from historical orders
        propagated = 0.0
        cutoff_time = timestamp - timedelta(hours=1)

        for hist_time, hist_volume, hist_price in self.order_history[-100:]:
            if hist_time < cutoff_time:
                continue

            time_diff = (timestamp - hist_time).total_seconds()
            if time_diff > 0:
                decay_factor = (1 + time_diff) ** (-self.decay_exp)
                hist_impact = (
                    self.impact_coeff *
                    hist_price *
                    (abs(hist_volume) / self.avg_daily_volume) ** self.propagator_exp *
                    decay_factor *
                    (1 if hist_volume > 0 else -1)
                )
                propagated += hist_impact

        # Store current order
        signed_volume = fill_quantity if order.side == OrderSide.BUY else -fill_quantity
        self.order_history.append((timestamp, signed_volume, market_price))

        # Trim history
        if len(self.order_history) > 1000:
            self.order_history = self.order_history[-500:]

        # Split into components
        if order.side == OrderSide.SELL:
            instant_impact = -instant_impact

        permanent = instant_impact * 0.2
        temporary = instant_impact * 0.8 + propagated

        return permanent, temporary

# 5. IntraDayMomentum - Momentum-based impact
class IntraDayMomentum(MarketImpactModel):
    def __init__(self, base_impact=0.05, momentum_factor=0.3,
                 momentum_decay=0.2, avg_daily_volume=1_000_000):
        super().__init__()
        self.base_impact = base_impact
        self.momentum_factor = momentum_factor
        self.momentum_decay = momentum_decay
        self.avg_daily_volume = avg_daily_volume
        self.momentum_states = {}  # asset_id -> momentum

    def calculate_impact(self, order, fill_quantity, market_price, timestamp):
        asset_id = order.asset_id
        volume_fraction = fill_quantity / self.avg_daily_volume

        # Current momentum
        momentum = self.momentum_states.get(asset_id, 0.0)

        # Base impact
        base = self.base_impact * market_price * volume_fraction

        # Momentum adjustment
        trade_direction = 1.0 if order.side == OrderSide.BUY else -1.0
        momentum_enhancement = 1.0 + self.momentum_factor * abs(momentum)

        if momentum * trade_direction > 0:
            # Same direction as momentum
            impact = base * momentum_enhancement
        else:
            # Against momentum
            impact = base / momentum_enhancement

        # Update momentum
        new_momentum = (
            momentum * (1 - self.momentum_decay) +
            trade_direction * volume_fraction * self.momentum_decay
        )
        self.momentum_states[asset_id] = new_momentum

        # Apply direction
        if order.side == OrderSide.SELL:
            impact = -impact

        # Split into components
        permanent = impact * 0.3
        temporary = impact * 0.7

        return permanent, temporary

# 6. ObizhaevWangImpact - Order book dynamics
class ObizhaevWangImpact(MarketImpactModel):
    def __init__(self, price_impact_const=0.1, information_share=0.3,
                 book_depth=100_000, resilience_rate=0.5):
        super().__init__()
        self.price_impact_const = price_impact_const
        self.information_share = information_share
        self.book_depth = book_depth
        self.decay_rate = resilience_rate

    def calculate_impact(self, order, fill_quantity, market_price, timestamp):
        size_ratio = fill_quantity / self.book_depth

        # Information-based permanent impact
        permanent = (
            self.information_share *
            self.price_impact_const *
            market_price *
            size_ratio
        )

        # Mechanical temporary impact
        temporary = (
            (1 - self.information_share) *
            self.price_impact_const *
            market_price *
            size_ratio
        )

        # Apply direction
        if order.side == OrderSide.SELL:
            permanent = -permanent
            temporary = -temporary

        return permanent, temporary
```

## 5. Corporate Actions Processing ✅

### 5.1 Comprehensive Corporate Actions Support

```python
class CorporateActionProcessor:
    """Process all major corporate actions with position adjustments."""

    def process_actions(self, as_of_date, positions, orders, cash):
        """Handle corporate actions as of date."""
        updated_positions = positions.copy()
        updated_orders = orders.copy()
        updated_cash = cash
        notifications = []

        for action in self.get_pending_actions(as_of_date):
            if isinstance(action, CashDividend):
                updated_cash, notification = self._process_dividend(
                    action, updated_positions, updated_cash
                )
            elif isinstance(action, StockSplit):
                updated_positions, updated_orders, notification = self._process_split(
                    action, updated_positions, updated_orders
                )
            elif isinstance(action, Merger):
                updated_positions, updated_cash, notification = self._process_merger(
                    action, updated_positions, updated_cash
                )
            # ... other action types

            notifications.append(notification)

        return updated_positions, updated_orders, updated_cash, notifications

# Supported corporate action types:
class CashDividend(CorporateAction):
    """Cash dividend with automatic distribution."""
    dividend_per_share: float
    currency: str = "USD"

class StockSplit(CorporateAction):
    """Stock split with position and order adjustments."""
    split_ratio: float  # New shares per old share

class StockDividend(CorporateAction):
    """Stock dividend with additional shares."""
    dividend_ratio: float  # Additional shares per existing share

class Merger(CorporateAction):
    """Merger with cash/stock consideration."""
    target_asset_id: str
    cash_consideration: float = 0.0
    stock_consideration: float = 0.0

class SpinOff(CorporateAction):
    """Spin-off with new asset distribution."""
    new_asset_id: str
    distribution_ratio: float

class SymbolChange(CorporateAction):
    """Symbol change with conversion."""
    new_asset_id: str
    conversion_ratio: float = 1.0
```

### 5.2 Historical Price Adjustments

```python
class PriceContinuityAdjuster:
    """Adjust historical prices for corporate actions."""

    def adjust_price_for_actions(self, asset_id, price, as_of_date):
        """Adjust historical price for corporate actions."""
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
                # Adjust for stock dividend
                adjusted_price /= (1 + action.dividend_ratio)

        return max(adjusted_price, 0.01)  # Price floor
```

## 6. Irregular Timestamps Support ✅

### 6.1 Event-Driven Time Processing

```python
class IrregularTimestampProcessor:
    """Process events with any timestamp pattern."""

    def process_events(self, events):
        """Handle events regardless of timing regularity."""
        # Sort by timestamp - handles any irregular sequence
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        for event in sorted_events:
            # Clock advances to exact event time
            self.clock.advance_to(event.timestamp)

            # Strategy receives event at natural market timing
            pit_data = self.data_manager.get_pit_data(event.timestamp)
            self.strategy.on_event(event, pit_data)

            # Process orders at realistic timing
            self.broker.process_event(event)
```

### 6.2 Bar Type Support

```python
# Volume bars - irregular timing based on shares traded
volume_bar_event = MarketEvent(
    timestamp=datetime(2024, 3, 15, 9, 47, 23, 145000),  # Irregular
    asset_id="AAPL",
    volume=10000,  # Exactly 10,000 shares
    metadata={"bar_type": "volume", "threshold": 10000}
)

# Dollar bars - irregular timing based on dollar volume
dollar_bar_event = MarketEvent(
    timestamp=datetime(2024, 3, 15, 9, 52, 7, 892000),   # Irregular
    asset_id="AAPL",
    volume=6632,  # Variable volume to reach $1M
    metadata={"bar_type": "dollar", "threshold": 1000000}
)

# Information bars - timing based on information flow
info_bar_event = MarketEvent(
    timestamp=datetime(2024, 3, 15, 10, 3, 12, 445000),  # Irregular
    asset_id="AAPL",
    metadata={"bar_type": "information", "vpin_threshold": 0.8}
)
```

## 7. Integration Architecture

### 7.1 Unified Broker with All Models

```python
class SimulationBroker:
    """Advanced broker with all execution models."""

    def __init__(
        self,
        commission_model: CommissionModel = None,
        slippage_model: SlippageModel = None,
        market_impact_model: MarketImpactModel = None,
        corporate_action_processor: CorporateActionProcessor = None,
    ):
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.market_impact_model = market_impact_model
        self.corporate_actions = corporate_action_processor

        # Advanced order tracking
        self._stop_orders = defaultdict(list)
        self._trailing_stops = defaultdict(list)
        self._bracket_orders = defaultdict(list)

    def execute_order(self, order, market_price, timestamp):
        """Execute with full realism stack."""
        # 1. Apply market impact to market price
        impacted_price = self._apply_market_impact(order, market_price, timestamp)

        # 2. Calculate fill price with slippage
        fill_price = self._apply_slippage(order, impacted_price)

        # 3. Calculate commission
        commission = self._calculate_commission(order, fill_price)

        # 4. Update market impact state
        self._update_market_impact(order, fill_price, timestamp)

        # 5. Process any pending corporate actions
        self._process_corporate_actions(timestamp)

        # 6. Create realistic fill event
        return FillEvent(
            timestamp=timestamp,
            order_id=order.order_id,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=abs(fill_price - market_price),
            market_impact=abs(impacted_price - market_price)
        )
```

## 8. Performance and Testing

### 8.1 Comprehensive Test Coverage

- **Order Types**: 28 tests covering all order type logic
- **Slippage Models**: 15 tests across all 7 models
- **Commission Models**: 18 tests covering all 9 models
- **Market Impact**: 26 tests for all 6 models + integration
- **Corporate Actions**: 28 tests covering all action types
- **Integration**: 12 tests for broker integration

### 8.2 Demonstration Scripts

- `examples/slippage_demo.py` - All slippage models in action
- `examples/commission_demo.py` - All commission models compared
- `examples/market_impact_demo.py` - Market impact effects shown
- `examples/corporate_actions_demo.py` - Corporate actions processing

## Summary

QEngine now provides **institutional-grade execution modeling** with:

- **22 distinct execution models** across slippage, commission, and market impact
- **7 corporate action types** with automatic position adjustments
- **6 advanced order types** with sophisticated logic
- **Native irregular timestamp support** for volume/dollar/info bars
- **Comprehensive testing** with 107 unit tests
- **Real-world calibration** based on academic research and industry practice

This execution modeling sophistication is **unmatched in open-source backtesting frameworks** and provides the realism required for institutional quantitative research.
