# ml4t-backtest Feature Inventory

**Date**: 2025-12-30
**Library Version**: 0.2.0
**Location**: `/home/stefan/ml4t/software/backtest/`

---

## Executive Summary

ml4t-backtest is a **minimal event-driven backtesting engine** with 80+ public symbols providing:
- Event-driven execution with point-in-time correctness
- Exit-first order processing (realistic broker behavior)
- Comprehensive risk management (position + portfolio rules)
- Multiple account types (cash/margin)
- Market impact and execution limit models
- VectorBT-validated results (exact match)
- 474 tests, 73% coverage, 100k+ events/sec

---

## 1. Core Engine Components

### 1.1 Engine (`engine.py`)
| Feature | Description |
|---------|-------------|
| `Engine` | Main event loop orchestration |
| `BacktestEngine` | Alias for backward compatibility |
| `run_backtest()` | Convenience function for simple backtests |
| Event-driven iteration | Sequential bar processing, no look-ahead |
| Exit-first processing | Exits processed before entries (frees capital) |
| Multi-asset support | Iterate multiple assets in single backtest |

### 1.2 Broker (`broker.py`, ~37KB)
| Feature | Description |
|---------|-------------|
| Order submission | `submit_order()`, `submit_orders()` |
| Position management | `get_position()`, `get_positions()`, `has_position()` |
| Order cancellation | `cancel_order()`, `cancel_all_orders()` |
| Risk rule integration | `set_position_rules()`, `set_portfolio_limits()` |
| Account state access | `account`, `cash`, `equity`, `buying_power` |
| Trade history | `trades`, `fills` lists |

### 1.3 DataFeed (`datafeed.py`)
| Feature | Description |
|---------|-------------|
| `DataFeed` | Iterator over price bars + signals |
| Polars-first | Efficient DataFrame handling |
| Signal columns | Strategy signals passed with price data |
| Multi-asset | Multiple assets in single feed |

### 1.4 Strategy (`strategy.py`)
| Feature | Description |
|---------|-------------|
| `Strategy` (ABC) | Base class for strategies |
| `on_data()` | Called each bar with price/signal data |
| `on_start()` | Called at backtest start |
| `on_end()` | Called at backtest end |

---

## 2. Order Types & Execution

### 2.1 Order Types (`types.py`)
| OrderType | Description |
|-----------|-------------|
| `MARKET` | Execute at current price |
| `LIMIT` | Execute at limit price or better |
| `STOP` | Market order when stop price triggered |
| `STOP_LIMIT` | Limit order when stop price triggered |
| `TRAILING_STOP` | Dynamic stop that follows price |

### 2.2 Execution Modes
| ExecutionMode | Description |
|---------------|-------------|
| `SAME_BAR` | Orders fill at current bar's close (default) |
| `NEXT_BAR` | Orders fill at next bar's open (Backtrader style) |

### 2.3 Stop Fill Modes
| StopFillMode | Description |
|--------------|-------------|
| `STOP_PRICE` | Fill at exact stop price (default, VBT Pro OHLC) |
| `CLOSE_PRICE` | Fill at bar's close (VBT Pro close-only) |
| `BAR_EXTREME` | Fill at bar's low/high (conservative) |
| `NEXT_BAR_OPEN` | Fill at next bar's open (Zipline) |

### 2.4 Other Order Options
| Feature | Description |
|---------|-------------|
| `StopLevelBasis.FILL_PRICE` | Calculate stops from fill price (default) |
| `StopLevelBasis.SIGNAL_PRICE` | Calculate stops from signal close (Backtrader) |

---

## 3. Commission & Slippage Models (`models.py`)

### 3.1 Commission Models
| Model | Description | Parameters |
|-------|-------------|------------|
| `NoCommission` | Zero commission | - |
| `PercentageCommission` | % of trade value | `rate=0.001` (0.1%) |
| `PerShareCommission` | Fixed per share | `per_share=0.005`, `minimum=1.0` |
| `TieredCommission` | Volume-based tiers | `tiers=[(threshold, rate), ...]` |
| `CombinedCommission` | Percentage + fixed | `percentage`, `fixed` |

### 3.2 Slippage Models
| Model | Description | Parameters |
|-------|-------------|------------|
| `NoSlippage` | Zero slippage | - |
| `FixedSlippage` | Fixed per-unit | `amount=0.01` |
| `PercentageSlippage` | % of price | `rate=0.001` (0.1%) |
| `VolumeShareSlippage` | Volume-based impact | `impact_factor=0.1` |

---

## 4. Account System (`accounting/`)

### 4.1 Account State (`account.py`)
| Feature | Description |
|---------|-------------|
| `AccountState` | Unified account tracking |
| Cash balance | Track available cash |
| Position values | Track position market values |
| Margin tracking | Initial + maintenance margin |

### 4.2 Account Policies (`policy.py`)
| Policy | Description |
|--------|-------------|
| `CashAccountPolicy` | No leverage, no shorting |
| `MarginAccountPolicy` | Leverage + short selling |

**MarginAccountPolicy Parameters**:
- `initial_margin=0.5` (50% = Reg T)
- `long_maintenance_margin=0.25` (25% for longs)
- `short_maintenance_margin=0.30` (30% for shorts)
- `fixed_margin_schedule` (futures: fixed $ per contract)

### 4.3 Gatekeeper (`gatekeeper.py`)
| Feature | Description |
|---------|-------------|
| Order validation | Check capital, margin before submission |
| Rejection handling | Reject orders that exceed limits |
| Pre-trade checks | Verify account can support trade |

---

## 5. Risk Management System (`risk/`)

### 5.1 Position-Level Rules (`risk/position/`)

**Static Rules** (`static.py`):
| Rule | Description | Parameters |
|------|-------------|------------|
| `StopLoss` | Exit at max loss % | `pct=0.05` (5%) |
| `TakeProfit` | Exit at target gain % | `pct=0.10` (10%) |
| `TrailingStop` | Dynamic stop follows price | `pct=0.05`, `activation_pct` |

**Dynamic Rules** (`dynamic.py`):
| Rule | Description | Parameters |
|------|-------------|------------|
| `VolatilityStop` | ATR-based stop | `atr_multiplier=2.0` |
| `BreakEvenStop` | Move stop to entry after profit | `activation_pct=0.02` |
| `TimeStop` | Exit after N bars | `max_bars=20` |

**Signal Rules** (`signal.py`):
| Rule | Description |
|------|-------------|
| `SignalBasedExit` | Exit on signal column change |

**Composite Rules** (`composite.py`):
| Rule | Description |
|------|-------------|
| `CompositeRule` | Multiple rules (any triggers) |
| `ConditionalRule` | Rule only active if condition met |
| `SequentialRule` | Rules in priority order |
| `RuleChain` | Bracket orders (OCO: stop + target) |

### 5.2 Portfolio-Level Limits (`risk/portfolio/`)

**Limits** (`limits.py`):
| Limit | Description | Parameters |
|-------|-------------|------------|
| `MaxPositions` | Max number of positions | `max_positions=10` |
| `MaxExposure` | Max total exposure | `max_exposure=2.0` (200%) |
| `MaxDrawdown` | Circuit breaker on drawdown | `max_drawdown=0.20` (20%) |
| `MaxSectorExposure` | Per-sector limits | `sector_limits` dict |
| `MaxConcentration` | Max single position size | `max_weight=0.25` |

**Manager** (`manager.py`):
| Feature | Description |
|---------|-------------|
| `PortfolioRiskManager` | Combines multiple limits |
| Pre-trade validation | Check limits before orders |
| Real-time monitoring | Check limits each bar |

---

## 6. Execution Model (`execution/`)

### 6.1 Market Impact (`impact.py`)
| Model | Description | Parameters |
|-------|-------------|------------|
| `NoImpact` | Zero impact | - |
| `LinearImpact` | Linear in volume fraction | `coefficient=0.1` |
| `SquareRootImpact` | Sqrt in volume fraction | `coefficient=0.1` |

### 6.2 Execution Limits (`limits.py`)
| Limit | Description | Parameters |
|-------|-------------|------------|
| `NoLimits` | No volume constraints | - |
| `VolumeParticipationLimit` | Max % of volume | `max_participation=0.10` |

### 6.3 Rebalancing (`rebalancer.py`)
| Feature | Description |
|---------|-------------|
| `RebalanceConfig` | Rebalance parameters |
| `TargetWeightExecutor` | Execute to target weights |
| Drift tolerance | Only rebalance if drift exceeds threshold |
| Transaction cost awareness | Consider costs in rebalancing |

---

## 7. Analytics (`analytics/`)

### 7.1 Equity Curve (`equity.py`)
| Feature | Description |
|---------|-------------|
| `EquityCurve` | Analyze equity time series |
| Drawdown analysis | Max drawdown, drawdown duration |
| Returns analysis | Daily, monthly, annual returns |

### 7.2 Performance Metrics (`metrics.py`)
| Metric | Description |
|--------|-------------|
| `sharpe_ratio()` | Risk-adjusted return |
| `sortino_ratio()` | Downside risk-adjusted return |
| `calmar_ratio()` | Return / max drawdown |
| `max_drawdown()` | Maximum peak-to-trough decline |
| `cagr()` | Compound annual growth rate |
| `volatility()` | Annualized standard deviation |

### 7.3 Trade Analysis (`trades.py`)
| Feature | Description |
|---------|-------------|
| `TradeAnalyzer` | Analyze completed trades |
| Win rate | % of profitable trades |
| Profit factor | Gross profit / gross loss |
| Average trade | Mean P&L per trade |
| MFE/MAE analysis | Max favorable/adverse excursion |

### 7.4 Diagnostic Integration (`analysis.py`)
| Feature | Description |
|---------|-------------|
| `BacktestAnalyzer` | Comprehensive backtest analysis |
| `TradeStatistics` | Detailed trade metrics |
| `to_trade_records()` | Export trades for analysis |
| `to_equity_dataframe()` | Export equity curve |

---

## 8. Configuration (`config.py`)

### 8.1 BacktestConfig
| Parameter | Description |
|-----------|-------------|
| `initial_cash` | Starting capital |
| `commission_model` | Commission calculation |
| `slippage_model` | Slippage calculation |
| `execution_mode` | SAME_BAR / NEXT_BAR |
| `account_type` | cash / margin |

### 8.2 Enums
| Enum | Values |
|------|--------|
| `DataFrequency` | DAILY, MINUTE, TICK |
| `FillTiming` | CLOSE, OPEN, VWAP |
| `ExecutionPrice` | CLOSE, OPEN, MID |
| `ShareType` | FRACTIONAL, INTEGER |
| `SizingMethod` | FIXED, PERCENT, VOLATILITY |

### 8.3 Presets
Location: `presets/*.yaml`
- Pre-built configurations for common scenarios

---

## 9. Calendar Integration (`calendar.py`)

| Function | Description |
|----------|-------------|
| `get_calendar()` | Get exchange calendar |
| `get_trading_days()` | Trading days in range |
| `is_trading_day()` | Check if date is trading day |
| `is_market_open()` | Check if market is open |
| `get_holidays()` | Get exchange holidays |
| `get_early_closes()` | Get early close days |
| `filter_to_trading_days()` | Filter DataFrame to trading days |
| `generate_trading_minutes()` | Generate intraday timestamps |

**Supported Exchanges**: NYSE, NASDAQ, CME, LSE, etc. via `pandas_market_calendars`

---

## 10. Data Types (`types.py`)

### 10.1 Core Dataclasses
| Type | Description |
|------|-------------|
| `Order` | Order representation |
| `Position` | Open position tracking |
| `Fill` | Order fill record |
| `Trade` | Completed round-trip |
| `ContractSpec` | Asset contract specification |

### 10.2 Position Features
- Weighted average cost basis
- Mark-to-market pricing
- MFE/MAE tracking (max favorable/adverse excursion)
- High/low water marks
- Contract multipliers (futures)

### 10.3 Trade Features
- Entry/exit times and prices
- P&L (absolute and percentage)
- Bars held
- Commission and slippage
- Entry/exit signals preserved
- MFE/MAE from position

---

## 11. Public API Summary

**Total Public Symbols**: 80+

**Core (6)**: Engine, Broker, Strategy, DataFeed, run_backtest, BacktestEngine
**Types (12)**: Order, Position, Fill, Trade, ContractSpec, OrderType, OrderSide, OrderStatus, ExecutionMode, StopFillMode, StopLevelBasis, AssetClass
**Models (10)**: CommissionModel, SlippageModel, NoCommission, PercentageCommission, PerShareCommission, TieredCommission, CombinedCommission, NoSlippage, FixedSlippage, PercentageSlippage, VolumeShareSlippage
**Config (8)**: BacktestConfig, DataFrequency, FillTiming, ExecutionPrice, ShareType, SizingMethod, SignalProcessing, PRESETS_DIR
**Analytics (12)**: EquityCurve, TradeAnalyzer, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown, cagr, volatility, BacktestAnalyzer, TradeStatistics, to_trade_records, to_equity_dataframe
**Calendar (14)**: get_calendar, get_trading_days, is_trading_day, is_market_open, next_trading_day, previous_trading_day, get_holidays, get_early_closes, get_schedule, filter_to_trading_days, filter_to_trading_sessions, generate_trading_minutes, list_calendars, CALENDAR_ALIASES
**Execution (10)**: ExecutionLimits, NoLimits, VolumeParticipationLimit, MarketImpactModel, NoImpact, LinearImpact, SquareRootImpact, ExecutionResult, RebalanceConfig, TargetWeightExecutor

---

## 12. Key Differentiators

1. **Event-Driven**: Sequential bar processing prevents look-ahead bias
2. **Exit-First Processing**: Exits before entries (realistic broker behavior)
3. **Point-in-Time Correctness**: Decisions use only available data
4. **VectorBT Validated**: Exact match with VectorBT Pro results
5. **Comprehensive Risk**: Position + portfolio level rules
6. **Realistic Accounts**: Cash vs margin with proper margin calls
7. **Performance**: 100k+ events/second
8. **Clean Code**: 474 tests, 73% coverage, minimal design

---

*Generated by Book Integration Audit - Phase 1*
