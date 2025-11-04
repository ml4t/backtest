# QEngine Simulation Reference

## Overview

This document provides detailed specifications for QEngine's **institutional-grade market simulation** capabilities, including sophisticated execution models, comprehensive corporate actions handling, irregular timestamp processing, and multi-asset simulation workflows.

## Recent Major Enhancements ✅

### 2024 Simulation Improvements
1. **Advanced Order Types**: Stop, Stop-Limit, Trailing Stop, Bracket orders with OCO logic
2. **Slippage Models**: 7 sophisticated models (fixed, percentage, Almgren-Chriss, volume-based)
3. **Commission Models**: 9 comprehensive models (flat, tiered, maker-taker, broker-specific)
4. **Market Impact Models**: 6 advanced models (linear, square-root, propagator, momentum)
5. **Corporate Actions**: Full processing (dividends, splits, mergers, spin-offs, symbol changes)
6. **Irregular Timestamps**: Native support for volume/dollar/information bars

## 1. Market Simulation Models

### 1.1 Order Book Simulation

**Level 1 (Top of Book)**
```python
class Level1Book:
    """Simple bid-ask spread simulation."""

    def __init__(self, spread_model):
        self.spread_model = spread_model
        self.mid_price = None
        self.bid = None
        self.ask = None

    def update(self, market_event):
        """Update book from market data."""
        self.mid_price = market_event.price
        spread = self.spread_model.get_spread(market_event)
        self.bid = self.mid_price - spread / 2
        self.ask = self.mid_price + spread / 2

    def get_execution_price(self, side, size):
        """Get execution price for order."""
        if side == Side.BUY:
            return self.ask
        else:
            return self.bid
```

**Level 2 (Market Depth)**
```python
class Level2Book:
    """Full order book simulation."""

    def __init__(self, n_levels=10):
        self.n_levels = n_levels
        self.bids = []  # [(price, size), ...]
        self.asks = []  # [(price, size), ...]

    def match_order(self, order):
        """Match order against book."""
        fills = []
        remaining = order.quantity

        book_side = self.asks if order.side == Side.BUY else self.bids

        for price, available in book_side:
            if remaining <= 0:
                break

            fill_qty = min(remaining, available)
            fills.append((price, fill_qty))
            remaining -= fill_qty

        return fills
```

### 1.2 Latency Simulation

**Fixed Latency**
```python
class FixedLatency:
    """Constant execution delay."""

    def __init__(self, latency_ms):
        self.latency = timedelta(milliseconds=latency_ms)

    def get_delay(self, order, market_conditions):
        return self.latency
```

**Variable Latency**
```python
class StochasticLatency:
    """Random latency with market condition dependency."""

    def __init__(self, base_latency_ms, volatility_factor=1.0):
        self.base_latency = base_latency_ms
        self.volatility_factor = volatility_factor

    def get_delay(self, order, market_conditions):
        # Higher volatility = higher latency
        vol_multiplier = 1 + market_conditions.volatility * self.volatility_factor

        # Log-normal distribution for realistic latency
        latency_ms = lognormal(
            mean=log(self.base_latency * vol_multiplier),
            sigma=0.3
        )

        return timedelta(milliseconds=latency_ms)
```

## 2. Slippage and Market Impact Models

### 2.1 Linear Slippage

**Simple Linear Model**
```python
class LinearSlippage:
    """Linear price impact model."""

    def __init__(self, impact_coefficient=0.0001):
        self.impact = impact_coefficient

    def calculate_slippage(self, order_size, market_volume, volatility):
        """Calculate price impact."""
        participation_rate = order_size / market_volume
        return self.impact * participation_rate
```

### 2.2 Square-Root Market Impact

**Almgren-Chriss Model**
```python
class AlmgrenChrissImpact:
    """Square-root market impact model."""

    def __init__(self, eta=2.5e-6, gamma=2.5e-7, alpha=0.95):
        self.eta = eta      # Temporary impact coefficient
        self.gamma = gamma  # Permanent impact coefficient
        self.alpha = alpha  # Decay factor

    def calculate_impact(self, order, market_state):
        """
        Calculate total market impact.

        Temporary Impact: η * (V/VD) * σ * √(T/τ)
        Permanent Impact: γ * (V/VD) * σ

        Where:
        - V = order volume
        - VD = daily volume
        - σ = volatility
        - T = time horizon
        - τ = interval length
        """
        participation = order.quantity / market_state.daily_volume
        volatility = market_state.volatility

        # Temporary impact (decays)
        temp_impact = self.eta * participation * volatility * \
                     sqrt(order.time_horizon / order.interval)

        # Permanent impact (does not decay)
        perm_impact = self.gamma * participation * volatility

        return {
            'temporary': temp_impact,
            'permanent': perm_impact,
            'total': temp_impact + perm_impact
        }
```

### 2.3 Intraday Volume Profile

**Volume-Weighted Impact**
```python
class IntradayVolumeProfile:
    """U-shaped intraday volume pattern."""

    def __init__(self):
        # Typical U-shape coefficients
        self.morning_peak = 1.5
        self.midday_trough = 0.7
        self.close_peak = 1.8

    def get_volume_factor(self, time_of_day):
        """Get volume multiplication factor."""
        minutes_since_open = (time_of_day - MARKET_OPEN).seconds / 60
        minutes_to_close = (MARKET_CLOSE - time_of_day).seconds / 60

        if minutes_since_open < 60:
            # Morning peak
            return self.morning_peak
        elif minutes_to_close < 60:
            # Close peak
            return self.close_peak
        else:
            # Midday trough
            return self.midday_trough

    def adjust_impact(self, base_impact, execution_time):
        """Adjust impact based on time of day."""
        volume_factor = self.get_volume_factor(execution_time)
        # Lower volume = higher impact
        return base_impact / volume_factor
```

## 3. Commission Models

### 3.1 Fixed Commission

```python
class FixedCommission:
    """Fixed cost per trade."""

    def __init__(self, cost_per_trade=1.0):
        self.cost = cost_per_trade

    def calculate(self, order, fill):
        return self.cost
```

### 3.2 Percentage Commission

```python
class PercentageCommission:
    """Percentage of trade value."""

    def __init__(self, rate=0.001, minimum=1.0):
        self.rate = rate
        self.minimum = minimum

    def calculate(self, order, fill):
        commission = fill.executed_price * fill.executed_quantity * self.rate
        return max(commission, self.minimum)
```

### 3.3 Tiered Commission

```python
class TieredCommission:
    """Volume-based tiered pricing."""

    def __init__(self, tiers):
        # tiers = [(threshold, rate), ...]
        self.tiers = sorted(tiers, key=lambda x: x[0])

    def calculate(self, order, fill, monthly_volume):
        """Calculate based on monthly volume tier."""
        for threshold, rate in reversed(self.tiers):
            if monthly_volume >= threshold:
                return fill.executed_price * fill.executed_quantity * rate

        # Default to highest rate
        return fill.executed_price * fill.executed_quantity * self.tiers[0][1]
```

## 4. Corporate Actions Handling

### 4.1 Stock Splits

```python
class StockSplit:
    """Handle stock split adjustments."""

    def __init__(self, symbol, ex_date, ratio):
        self.symbol = symbol
        self.ex_date = ex_date
        self.ratio = ratio  # e.g., 2.0 for 2-for-1 split

    def adjust_position(self, position, current_date):
        """Adjust position for split."""
        if current_date >= self.ex_date:
            position.quantity *= self.ratio
            position.avg_price /= self.ratio

    def adjust_historical_prices(self, prices, dates):
        """Adjust historical prices."""
        adjusted = prices.copy()
        mask = dates < self.ex_date
        adjusted[mask] = prices[mask] / self.ratio
        return adjusted
```

### 4.2 Dividends

```python
class Dividend:
    """Handle dividend payments."""

    def __init__(self, symbol, ex_date, amount, dividend_type='cash'):
        self.symbol = symbol
        self.ex_date = ex_date
        self.amount = amount
        self.dividend_type = dividend_type

    def process_payment(self, position, portfolio, current_date):
        """Process dividend payment."""
        if current_date == self.ex_date and position.quantity > 0:
            if self.dividend_type == 'cash':
                payment = position.quantity * self.amount
                portfolio.cash += payment
                return CashFlow(
                    timestamp=current_date,
                    amount=payment,
                    type='dividend',
                    symbol=self.symbol
                )
            elif self.dividend_type == 'stock':
                new_shares = position.quantity * self.amount
                position.quantity += new_shares
```

### 4.3 Rights and Warrants

```python
class RightsOffering:
    """Handle rights offerings."""

    def __init__(self, symbol, ex_date, ratio, subscription_price):
        self.symbol = symbol
        self.ex_date = ex_date
        self.ratio = ratio  # Rights per share
        self.subscription_price = subscription_price

    def calculate_rights_value(self, current_price):
        """Calculate theoretical value of rights."""
        if current_price > self.subscription_price:
            return (current_price - self.subscription_price) / (1 + self.ratio)
        return 0
```

## 5. Multi-Asset Simulation

### 5.1 Asset Class Handlers

**Equity Handler**
```python
class EquityHandler:
    """Equity-specific simulation logic."""

    def __init__(self):
        self.trading_hours = MarketHours('NYSE')
        self.tick_size = 0.01
        self.lot_size = 1

    def validate_order(self, order):
        """Validate equity order."""
        # Check market hours
        if not self.trading_hours.is_open(order.timestamp):
            raise InvalidOrder("Market closed")

        # Check tick size
        if order.limit_price % self.tick_size != 0:
            raise InvalidOrder("Invalid tick size")
```

**Futures Handler**
```python
class FuturesHandler:
    """Futures-specific simulation logic."""

    def __init__(self, contract_specs):
        self.specs = contract_specs
        self.multiplier = contract_specs.multiplier
        self.tick_size = contract_specs.tick_size
        self.margin_requirement = contract_specs.initial_margin

    def calculate_margin(self, position):
        """Calculate margin requirement."""
        notional = position.quantity * position.price * self.multiplier
        return abs(notional) * self.margin_requirement

    def handle_expiration(self, position, expiration_date, settlement_price):
        """Handle contract expiration."""
        if position.expiration_date <= expiration_date:
            # Cash settlement
            pnl = (settlement_price - position.avg_price) * \
                  position.quantity * self.multiplier
            return CashSettlement(pnl)
```

**Options Handler**
```python
class OptionsHandler:
    """Options-specific simulation logic."""

    def __init__(self, pricing_model='black_scholes'):
        self.pricing_model = pricing_model
        self.multiplier = 100  # Standard equity option

    def calculate_greeks(self, option, underlying_price, volatility,
                        risk_free_rate, time_to_expiry):
        """Calculate option Greeks."""
        if self.pricing_model == 'black_scholes':
            return {
                'delta': self._calculate_delta(option, underlying_price),
                'gamma': self._calculate_gamma(option, underlying_price),
                'theta': self._calculate_theta(option, time_to_expiry),
                'vega': self._calculate_vega(option, volatility),
                'rho': self._calculate_rho(option, risk_free_rate)
            }

    def handle_exercise(self, option, underlying_price):
        """Handle option exercise."""
        if option.style == 'american':
            # Can exercise any time
            intrinsic = max(0, underlying_price - option.strike) \
                       if option.type == 'call' else \
                       max(0, option.strike - underlying_price)

            if intrinsic > 0:
                return ExerciseEvent(
                    option=option,
                    underlying_price=underlying_price,
                    payoff=intrinsic * self.multiplier
                )
```

### 5.2 Cross-Asset Correlations

```python
class CorrelationManager:
    """Manage cross-asset correlations for realistic simulation."""

    def __init__(self, correlation_matrix):
        self.correlation_matrix = correlation_matrix
        self.assets = list(correlation_matrix.index)

    def generate_correlated_returns(self, n_periods, dt=1/252):
        """Generate correlated asset returns."""
        n_assets = len(self.assets)

        # Generate independent random shocks
        shocks = randn(n_periods, n_assets)

        # Apply correlation via Cholesky decomposition
        L = cholesky(self.correlation_matrix)
        correlated_shocks = shocks @ L.T

        # Convert to returns
        returns = pd.DataFrame(
            correlated_shocks * sqrt(dt),
            columns=self.assets
        )

        return returns
```

## 6. Execution Algorithms

### 6.1 TWAP (Time-Weighted Average Price)

```python
class TWAP:
    """Time-weighted average price execution."""

    def __init__(self, total_quantity, start_time, end_time, n_slices):
        self.total_quantity = total_quantity
        self.start_time = start_time
        self.end_time = end_time
        self.n_slices = n_slices
        self.slice_size = total_quantity / n_slices
        self.slice_interval = (end_time - start_time) / n_slices

    def generate_child_orders(self):
        """Generate child orders for TWAP execution."""
        orders = []
        current_time = self.start_time

        for i in range(self.n_slices):
            orders.append(ChildOrder(
                timestamp=current_time,
                quantity=self.slice_size,
                order_type=OrderType.MARKET
            ))
            current_time += self.slice_interval

        return orders
```

### 6.2 VWAP (Volume-Weighted Average Price)

```python
class VWAP:
    """Volume-weighted average price execution."""

    def __init__(self, total_quantity, volume_profile):
        self.total_quantity = total_quantity
        self.volume_profile = volume_profile  # Historical intraday volume

    def generate_child_orders(self, market_schedule):
        """Generate child orders based on volume profile."""
        orders = []

        for time_bucket, volume_pct in self.volume_profile.items():
            order_size = self.total_quantity * volume_pct

            orders.append(ChildOrder(
                timestamp=time_bucket,
                quantity=order_size,
                order_type=OrderType.LIMIT,
                limit_price=None  # Set dynamically
            ))

        return orders
```

### 6.3 Implementation Shortfall

```python
class ImplementationShortfall:
    """Minimize implementation shortfall (Almgren-Chriss)."""

    def __init__(self, risk_aversion, total_quantity, time_horizon):
        self.risk_aversion = risk_aversion
        self.total_quantity = total_quantity
        self.time_horizon = time_horizon

    def optimize_trajectory(self, market_params):
        """
        Optimize execution trajectory to minimize:
        E[Cost] + λ * Var[Cost]

        Where λ is risk aversion parameter.
        """
        volatility = market_params['volatility']
        daily_volume = market_params['daily_volume']

        # Optimal trading rate (simplified)
        kappa = sqrt(self.risk_aversion * volatility / daily_volume)

        # Generate execution schedule
        schedule = []
        remaining = self.total_quantity

        for t in range(self.time_horizon):
            # Exponential decay
            trade_size = remaining * (1 - exp(-kappa))
            schedule.append(trade_size)
            remaining -= trade_size

        return schedule
```

## 7. Risk Models

### 7.1 Value at Risk (VaR)

```python
class VaRCalculator:
    """Calculate Value at Risk."""

    def historical_var(self, returns, confidence=0.95):
        """Historical VaR."""
        return percentile(returns, (1 - confidence) * 100)

    def parametric_var(self, portfolio_value, volatility, confidence=0.95):
        """Parametric VaR (assumes normal distribution)."""
        z_score = norm.ppf(1 - confidence)
        return portfolio_value * volatility * z_score

    def monte_carlo_var(self, portfolio, n_simulations=10000, confidence=0.95):
        """Monte Carlo VaR."""
        simulated_returns = []

        for _ in range(n_simulations):
            scenario = self.generate_scenario()
            portfolio_return = portfolio.calculate_return(scenario)
            simulated_returns.append(portfolio_return)

        return percentile(simulated_returns, (1 - confidence) * 100)
```

### 7.2 Margin Requirements

```python
class MarginCalculator:
    """Calculate margin requirements."""

    def __init__(self, margin_rules):
        self.rules = margin_rules

    def calculate_initial_margin(self, position):
        """Calculate initial margin requirement."""
        if position.asset_type == 'equity':
            # Reg T: 50% for long, 150% for short
            if position.quantity > 0:
                return position.market_value * 0.5
            else:
                return abs(position.market_value) * 1.5

        elif position.asset_type == 'futures':
            # Exchange-specified
            return position.quantity * self.rules.futures_margin

        elif position.asset_type == 'options':
            # Complex rules based on position type
            return self._calculate_option_margin(position)

    def calculate_maintenance_margin(self, position):
        """Calculate maintenance margin."""
        # Typically 25% for equities
        return self.calculate_initial_margin(position) * 0.5
```

## 8. Performance Analytics

### 8.1 Trade Analytics

```python
class TradeAnalytics:
    """Analyze trade execution quality."""

    def calculate_slippage(self, order, fill, benchmark_price):
        """Calculate execution slippage."""
        if order.side == Side.BUY:
            slippage = fill.executed_price - benchmark_price
        else:
            slippage = benchmark_price - fill.executed_price

        return {
            'absolute': slippage,
            'relative': slippage / benchmark_price,
            'cost': slippage * fill.executed_quantity
        }

    def calculate_implementation_shortfall(self, decision_price,
                                          execution_price,
                                          opportunity_cost):
        """Calculate total implementation shortfall."""
        explicit_cost = abs(execution_price - decision_price)

        return {
            'explicit': explicit_cost,
            'opportunity': opportunity_cost,
            'total': explicit_cost + opportunity_cost
        }
```

### 8.2 Attribution Analysis

```python
class AttributionAnalysis:
    """Performance attribution."""

    def calculate_attribution(self, portfolio_returns, benchmark_returns,
                            positions):
        """Decompose returns into various factors."""

        # Asset allocation effect
        allocation_effect = self._calculate_allocation_effect(
            positions, benchmark_weights
        )

        # Selection effect
        selection_effect = self._calculate_selection_effect(
            portfolio_returns, benchmark_returns
        )

        # Interaction effect
        interaction_effect = portfolio_returns - benchmark_returns - \
                           allocation_effect - selection_effect

        return {
            'allocation': allocation_effect,
            'selection': selection_effect,
            'interaction': interaction_effect,
            'total': portfolio_returns - benchmark_returns
        }
```

## 9. Realistic Market Conditions

### 9.1 Volatility Clustering

```python
class VolatilityClustering:
    """Simulate realistic volatility patterns."""

    def __init__(self, omega=0.00001, alpha=0.1, beta=0.85):
        # GARCH(1,1) parameters
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.current_variance = omega / (1 - alpha - beta)

    def update(self, return_shock):
        """Update conditional variance."""
        self.current_variance = (
            self.omega +
            self.alpha * return_shock**2 +
            self.beta * self.current_variance
        )

        return sqrt(self.current_variance)
```

### 9.2 Microstructure Noise

```python
class MicrostructureNoise:
    """Add realistic microstructure noise."""

    def __init__(self, noise_ratio=0.01):
        self.noise_ratio = noise_ratio

    def add_noise(self, true_price, bid_ask_spread):
        """Add bid-ask bounce and rounding."""
        # Bid-ask bounce
        bounce = uniform(-bid_ask_spread/2, bid_ask_spread/2)

        # Rounding to tick size
        tick_size = 0.01
        noisy_price = true_price + bounce

        return round(noisy_price / tick_size) * tick_size
```

## 10. Backtesting Workflow

### 10.1 Complete Simulation Loop

```python
class BacktestSimulator:
    """Main simulation orchestrator."""

    def __init__(self, config):
        self.clock = Clock(config.start_date, config.end_date)
        self.event_queue = EventQueue()
        self.data_feed = DataFeed(config.data_source)
        self.strategy = config.strategy
        self.broker = SimulationBroker(config.broker_config)
        self.portfolio = Portfolio(config.initial_capital)
        self.risk_manager = RiskManager(config.risk_config)

    def run(self):
        """Execute complete backtest."""

        while not self.data_feed.is_exhausted:
            # Get next market event
            market_event = self.data_feed.get_next_event()

            # Update clock
            self.clock.advance_to(market_event.timestamp)

            # Update market state
            self.broker.update_market(market_event)

            # Generate strategy signals
            pit_data = self.data_feed.get_pit_data(market_event.timestamp)
            signal = self.strategy.on_event(market_event, pit_data)

            if signal:
                # Risk checks
                if self.risk_manager.check_signal(signal, self.portfolio):
                    # Generate order
                    order = self.strategy.size_signal(signal, self.portfolio)

                    # Submit to broker
                    fill = self.broker.execute_order(order)

                    # Update portfolio
                    self.portfolio.update(fill)

            # Process corporate actions
            self._process_corporate_actions(market_event.timestamp)

            # Mark to market
            self.portfolio.mark_to_market(self.broker.get_prices())

            # Record state
            self._record_state()

        return self._generate_results()
```

This completes the comprehensive simulation documentation for QEngine, covering market mechanics, execution models, and multi-asset support.
