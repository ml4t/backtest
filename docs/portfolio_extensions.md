# Portfolio Extension Guide

How to extend the Portfolio module with custom components.

## Table of Contents

1. [Overview](#overview)
2. [Custom PerformanceAnalyzer](#custom-performanceanalyzer)
3. [Custom TradeJournal](#custom-tradejournal)
4. [Complete Examples](#complete-examples)
5. [Best Practices](#best-practices)

---

## Overview

The Portfolio module's facade pattern makes it easy to extend with custom behavior:

```python
portfolio = Portfolio(
    initial_cash=100000,
    analyzer_class=MyCustomAnalyzer,    # Your analyzer
    journal_class=MyCustomJournal        # Your journal
)
```

### Extension Points

1. **PerformanceAnalyzer**: Add custom metrics, risk calculations
2. **TradeJournal**: Add trade persistence, custom lot matching

### When to Extend

**Use custom PerformanceAnalyzer when:**
- ✅ Need domain-specific metrics (crypto, options, futures)
- ✅ Want custom risk calculations (VaR, CVaR, beta)
- ✅ Integrating with external analytics systems
- ✅ Need real-time alerts on metrics

**Use custom TradeJournal when:**
- ✅ Need to persist trades to database
- ✅ Want custom lot matching (LIFO, specific lots)
- ✅ Integrating with trade reporting systems
- ✅ Need advanced trade analytics

---

## Custom PerformanceAnalyzer

### Basic Template

```python
from qengine.portfolio.analytics import PerformanceAnalyzer
from qengine.portfolio.core import PositionTracker
from qengine.core.event import FillEvent

class MyAnalyzer(PerformanceAnalyzer):
    """Custom analyzer with additional metrics."""

    def __init__(self, tracker: PositionTracker):
        super().__init__(tracker)

        # Add your custom state
        self.custom_metric = 0.0

    def on_fill_event(self, event: FillEvent, tracker: PositionTracker) -> None:
        """Called after every fill."""
        # Call parent to maintain standard metrics
        super().on_fill_event(event, tracker)

        # Calculate your custom metrics
        self.custom_metric = self._calculate_custom()

    def _calculate_custom(self) -> float:
        """Your custom calculation logic."""
        # Example: Custom risk metric
        return self.tracker.equity * 0.1

    def reset(self) -> None:
        """Reset analyzer state."""
        super().reset()
        self.custom_metric = 0.0
```

### Example 1: Volatility Analyzer

Track realized volatility and correlations:

```python
import numpy as np
from collections import deque

class VolatilityAnalyzer(PerformanceAnalyzer):
    """Analyzer with volatility metrics."""

    def __init__(self, tracker: PositionTracker, window: int = 20):
        super().__init__(tracker)

        self.window = window
        self.returns_window = deque(maxlen=window)
        self.realized_vol = 0.0

    def on_fill_event(self, event: FillEvent, tracker: PositionTracker) -> None:
        super().on_fill_event(event, tracker)

        # Calculate period return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]
            curr_equity = self.equity_curve[-1]
            ret = (curr_equity - prev_equity) / prev_equity
            self.returns_window.append(ret)

            # Update realized volatility
            if len(self.returns_window) >= self.window:
                self.realized_vol = np.std(list(self.returns_window))

    def get_volatility_metrics(self) -> dict:
        """Get volatility-specific metrics."""
        annualized_vol = self.realized_vol * np.sqrt(252)  # Daily to annual

        return {
            "realized_vol": self.realized_vol,
            "annualized_vol": annualized_vol,
            "vol_window": self.window,
            "sharpe_ratio": self._calc_vol_adjusted_sharpe()
        }

    def _calc_vol_adjusted_sharpe(self) -> float:
        """Sharpe ratio using realized volatility."""
        if self.realized_vol == 0 or not self.daily_returns:
            return 0.0

        mean_return = np.mean(self.daily_returns)
        return (mean_return / self.realized_vol) * np.sqrt(252)

# Usage
portfolio = Portfolio(
    initial_cash=100000,
    analyzer_class=lambda tracker: VolatilityAnalyzer(tracker, window=30)
)

# Later...
vol_metrics = portfolio.analyzer.get_volatility_metrics()
print(f"Annualized volatility: {vol_metrics['annualized_vol']:.2%}")
```

### Example 2: Crypto-Specific Analyzer

Track funding rates and liquidation risk:

```python
class CryptoAnalyzer(PerformanceAnalyzer):
    """Analyzer for crypto futures trading."""

    def __init__(self, tracker: PositionTracker):
        super().__init__(tracker)

        # Crypto-specific metrics
        self.total_funding_paid = 0.0
        self.max_liquidation_distance = float('inf')
        self.liquidation_events = 0

    def on_fill_event(self, event: FillEvent, tracker: PositionTracker) -> None:
        super().on_fill_event(event, tracker)

        # Update liquidation risk
        self._update_liquidation_risk(tracker)

    def record_funding_payment(self, amount: float) -> None:
        """Record funding rate payment."""
        self.total_funding_paid += amount

    def _update_liquidation_risk(self, tracker: PositionTracker) -> None:
        """Calculate distance to liquidation."""
        # Example: Assuming 10x leverage
        leverage = 10.0
        liquidation_price_distance = tracker.cash / (leverage * tracker.equity)

        if liquidation_price_distance < self.max_liquidation_distance:
            self.max_liquidation_distance = liquidation_price_distance

        # Check if close to liquidation (< 5% margin)
        if liquidation_price_distance < 0.05:
            self.liquidation_events += 1

    def get_crypto_metrics(self) -> dict:
        """Get crypto-specific metrics."""
        base_metrics = self.get_performance_metrics(self.tracker)

        return {
            **base_metrics,
            "total_funding_paid": self.total_funding_paid,
            "max_liquidation_distance": self.max_liquidation_distance,
            "liquidation_events": self.liquidation_events,
            "net_pnl_after_funding": (
                base_metrics["total_pnl"] - self.total_funding_paid
            )
        }

# Usage
portfolio = Portfolio(
    initial_cash=100000,
    analyzer_class=CryptoAnalyzer
)

# During backtest
# portfolio.analyzer.record_funding_payment(-15.50)

# Results
metrics = portfolio.analyzer.get_crypto_metrics()
print(f"Net P&L after funding: ${metrics['net_pnl_after_funding']:,.2f}")
```

### Example 3: Real-Time Alert Analyzer

Trigger alerts when metrics hit thresholds:

```python
import logging

logger = logging.getLogger(__name__)

class AlertAnalyzer(PerformanceAnalyzer):
    """Analyzer with real-time alerts."""

    def __init__(
        self,
        tracker: PositionTracker,
        max_drawdown_threshold: float = 0.10,  # 10%
        min_sharpe_threshold: float = 0.5
    ):
        super().__init__(tracker)

        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_sharpe_threshold = min_sharpe_threshold
        self.alerts = []

    def on_fill_event(self, event: FillEvent, tracker: PositionTracker) -> None:
        super().on_fill_event(event, tracker)

        # Check drawdown alert
        if self.max_drawdown > self.max_drawdown_threshold:
            self._trigger_alert(
                "MAX_DRAWDOWN",
                f"Drawdown {self.max_drawdown:.1%} exceeds {self.max_drawdown_threshold:.1%}"
            )

        # Check Sharpe alert
        sharpe = self.calculate_sharpe_ratio(self.tracker)
        if sharpe is not None and sharpe < self.min_sharpe_threshold:
            self._trigger_alert(
                "LOW_SHARPE",
                f"Sharpe ratio {sharpe:.2f} below threshold {self.min_sharpe_threshold:.2f}"
            )

    def _trigger_alert(self, alert_type: str, message: str) -> None:
        """Trigger an alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": self.timestamps[-1] if self.timestamps else None,
            "equity": self.tracker.equity
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT [{alert_type}]: {message}")

    def get_alerts(self) -> list[dict]:
        """Get all triggered alerts."""
        return self.alerts

# Usage
portfolio = Portfolio(
    initial_cash=100000,
    analyzer_class=lambda tracker: AlertAnalyzer(
        tracker,
        max_drawdown_threshold=0.15,
        min_sharpe_threshold=1.0
    )
)

# After backtest
alerts = portfolio.analyzer.get_alerts()
print(f"Total alerts: {len(alerts)}")
for alert in alerts:
    print(f"  {alert['type']}: {alert['message']}")
```

---

## Custom TradeJournal

### Basic Template

```python
from qengine.portfolio.analytics import TradeJournal
from qengine.core.event import FillEvent

class MyJournal(TradeJournal):
    """Custom trade journal."""

    def __init__(self):
        super().__init__()

        # Add your custom state
        self.custom_data = []

    def on_fill_event(self, event: FillEvent) -> None:
        """Called for every fill."""
        # Call parent to maintain standard tracking
        super().on_fill_event(event)

        # Add your custom logic
        self.custom_data.append(self._extract_custom(event))

    def _extract_custom(self, event: FillEvent) -> dict:
        """Extract custom data from event."""
        return {
            "timestamp": event.timestamp,
            "custom_field": event.asset_id  # Your logic
        }

    def reset(self) -> None:
        """Reset journal state."""
        super().reset()
        self.custom_data.clear()
```

### Example 1: Database Journal

Persist trades to database:

```python
import polars as pl

class DatabaseJournal(TradeJournal):
    """Journal that persists to database."""

    def __init__(self, db_connection):
        super().__init__()
        self.db = db_connection

    def on_fill_event(self, event: FillEvent) -> None:
        # Standard tracking
        super().on_fill_event(event)

        # Persist to database
        self._persist_fill(event)

    def _persist_fill(self, event: FillEvent) -> None:
        """Write fill to database."""
        query = """
            INSERT INTO fills (timestamp, order_id, asset_id, side,
                             quantity, price, commission, slippage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute(query, (
            event.timestamp,
            event.order_id,
            event.asset_id,
            event.side.value,
            float(event.fill_quantity),
            float(event.fill_price),
            event.commission,
            event.slippage
        ))
        self.db.commit()

    def get_trades_from_db(self) -> pl.DataFrame:
        """Load all trades from database."""
        query = "SELECT * FROM fills ORDER BY timestamp"
        return pl.read_database(query, self.db)

# Usage
import sqlite3

db = sqlite3.connect("trades.db")
portfolio = Portfolio(
    initial_cash=100000,
    journal_class=lambda: DatabaseJournal(db)
)

# After backtest
trades_df = portfolio.journal.get_trades_from_db()
```

### Example 2: LIFO Journal

Use LIFO (last-in-first-out) lot matching:

```python
class LIFOJournal(TradeJournal):
    """Journal using LIFO instead of FIFO."""

    def calculate_win_rate(self) -> float:
        """Calculate win rate using LIFO lot matching."""
        if not self.fills:
            return 0.0

        winning_trades = 0
        total_trades = 0

        # Track positions with LIFO (stack)
        position_lots: dict[str, list[dict]] = {}

        for fill in self.fills:
            asset_id = fill.asset_id

            if fill.side.value == "buy":
                # Add to stack
                if asset_id not in position_lots:
                    position_lots[asset_id] = []
                position_lots[asset_id].append({
                    "quantity": fill.fill_quantity,
                    "price": float(fill.fill_price)
                })

            elif fill.side.value == "sell":
                # Pop from stack (LIFO)
                if position_lots.get(asset_id):
                    lot = position_lots[asset_id].pop()  # Last in, first out
                    pnl = (float(fill.fill_price) - lot["price"]) * min(
                        fill.fill_quantity, lot["quantity"]
                    )
                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1

        return winning_trades / total_trades if total_trades > 0 else 0.0

# Usage
portfolio = Portfolio(
    initial_cash=100000,
    journal_class=LIFOJournal
)
```

### Example 3: Tax Lot Journal

Track tax lots for reporting:

```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TaxLot:
    """Tax lot for IRS reporting."""
    asset_id: str
    quantity: float
    cost_basis: float
    purchase_date: datetime
    holding_period: str  # "short" or "long"

class TaxLotJournal(TradeJournal):
    """Journal for tax lot tracking."""

    def __init__(self):
        super().__init__()
        self.open_tax_lots: dict[str, list[TaxLot]] = {}
        self.closed_tax_lots: list[TaxLot] = []

    def on_fill_event(self, event: FillEvent) -> None:
        super().on_fill_event(event)

        if event.side.value == "buy":
            self._open_lot(event)
        else:
            self._close_lot(event)

    def _open_lot(self, event: FillEvent) -> None:
        """Open new tax lot."""
        lot = TaxLot(
            asset_id=event.asset_id,
            quantity=float(event.fill_quantity),
            cost_basis=float(event.fill_price) * float(event.fill_quantity),
            purchase_date=event.timestamp,
            holding_period="short"
        )

        if event.asset_id not in self.open_tax_lots:
            self.open_tax_lots[event.asset_id] = []
        self.open_tax_lots[event.asset_id].append(lot)

    def _close_lot(self, event: FillEvent) -> None:
        """Close tax lot (FIFO)."""
        if event.asset_id not in self.open_tax_lots:
            return

        lots = self.open_tax_lots[event.asset_id]
        if not lots:
            return

        lot = lots.pop(0)  # FIFO for tax purposes

        # Update holding period
        holding_days = (event.timestamp - lot.purchase_date).days
        lot.holding_period = "long" if holding_days > 365 else "short"

        self.closed_tax_lots.append(lot)

    def get_tax_report(self) -> dict:
        """Generate tax report."""
        short_term_lots = [lot for lot in self.closed_tax_lots if lot.holding_period == "short"]
        long_term_lots = [lot for lot in self.closed_tax_lots if lot.holding_period == "long"]

        return {
            "short_term_trades": len(short_term_lots),
            "long_term_trades": len(long_term_lots),
            "short_term_cost_basis": sum(lot.cost_basis for lot in short_term_lots),
            "long_term_cost_basis": sum(lot.cost_basis for lot in long_term_lots),
        }

# Usage
portfolio = Portfolio(
    initial_cash=100000,
    journal_class=TaxLotJournal
)

# After backtest
tax_report = portfolio.journal.get_tax_report()
print(f"Short-term trades: {tax_report['short_term_trades']}")
print(f"Long-term trades: {tax_report['long_term_trades']}")
```

---

## Complete Examples

### Example: Multi-Strategy Analyzer

Track performance of multiple strategies:

```python
class MultiStrategyAnalyzer(PerformanceAnalyzer):
    """Analyzer for portfolios running multiple strategies."""

    def __init__(self, tracker: PositionTracker, strategies: list[str]):
        super().__init__(tracker)

        self.strategies = strategies
        self.strategy_pnl = {s: 0.0 for s in strategies}
        self.strategy_trades = {s: 0 for s in strategies}

    def on_fill_event(self, event: FillEvent, tracker: PositionTracker) -> None:
        super().on_fill_event(event, tracker)

        # Extract strategy from order metadata (if available)
        # strategy_name = event.metadata.get("strategy")
        # if strategy_name in self.strategy_pnl:
        #     ... track per-strategy metrics

    def get_strategy_breakdown(self) -> dict:
        """Get performance breakdown by strategy."""
        return {
            "strategies": self.strategies,
            "pnl_by_strategy": self.strategy_pnl,
            "trades_by_strategy": self.strategy_trades
        }


class MultiStrategyJournal(TradeJournal):
    """Journal for multi-strategy portfolios."""

    def __init__(self, strategies: list[str]):
        super().__init__()
        self.strategy_fills = {s: [] for s in strategies}

    def on_fill_event(self, event: FillEvent) -> None:
        super().on_fill_event(event)

        # Route fills to strategy buckets
        # strategy_name = event.metadata.get("strategy", "unknown")
        # if strategy_name in self.strategy_fills:
        #     self.strategy_fills[strategy_name].append(event)


# Usage
strategies = ["momentum", "mean_reversion", "arbitrage"]

portfolio = Portfolio(
    initial_cash=100000,
    analyzer_class=lambda t: MultiStrategyAnalyzer(t, strategies),
    journal_class=lambda: MultiStrategyJournal(strategies)
)

# After backtest
breakdown = portfolio.analyzer.get_strategy_breakdown()
for strategy, pnl in breakdown["pnl_by_strategy"].items():
    print(f"{strategy}: ${pnl:,.2f}")
```

---

## Best Practices

### 1. Always Call Parent Methods

```python
def on_fill_event(self, event: FillEvent, tracker: PositionTracker) -> None:
    # ✅ GOOD: Call parent first
    super().on_fill_event(event, tracker)

    # Then add your logic
    self.custom_metric = self._calculate()

    # ❌ BAD: Don't skip parent
    # self.custom_metric = self._calculate()  # Missing super()!
```

### 2. Handle Reset Properly

```python
def reset(self) -> None:
    # ✅ GOOD: Reset parent state
    super().reset()

    # Then reset your state
    self.custom_data.clear()
    self.custom_metric = 0.0
```

### 3. Keep Performance in Mind

```python
def on_fill_event(self, event: FillEvent, tracker: PositionTracker) -> None:
    super().on_fill_event(event, tracker)

    # ✅ GOOD: Fast operations
    self.counter += 1
    self.values.append(event.fill_price)

    # ❌ BAD: Expensive operations on every fill
    # self.recalculate_all_metrics()  # Too slow!
    # self.save_to_database()         # Too slow!
```

### 4. Use Type Hints

```python
from typing import Optional

class MyAnalyzer(PerformanceAnalyzer):
    def __init__(self, tracker: PositionTracker, window: int = 20):
        ...

    def get_custom_metrics(self) -> dict[str, float]:
        ...

    def calculate_sharpe_ratio(self) -> Optional[float]:
        ...
```

### 5. Document Your Extensions

```python
class CryptoAnalyzer(PerformanceAnalyzer):
    """Analyzer for cryptocurrency futures trading.

    Tracks crypto-specific metrics including funding rates,
    liquidation risk, and perpetual swap analytics.

    Args:
        tracker: PositionTracker instance
        max_leverage: Maximum allowed leverage (default: 10x)

    Example:
        >>> portfolio = Portfolio(
        ...     initial_cash=100000,
        ...     analyzer_class=CryptoAnalyzer
        ... )
        >>> metrics = portfolio.analyzer.get_crypto_metrics()
    """
```

---

## Testing Your Extensions

### Unit Test Template

```python
import pytest
from datetime import datetime
from decimal import Decimal

from qengine.core.event import FillEvent, OrderSide
from qengine.portfolio.core import PositionTracker
from your_module import MyCustomAnalyzer

def test_custom_analyzer():
    # Setup
    tracker = PositionTracker(initial_cash=100000)
    analyzer = MyCustomAnalyzer(tracker)

    # Create fill event
    fill = FillEvent(
        timestamp=datetime(2025, 1, 1),
        order_id="order_001",
        trade_id="trade_001",
        asset_id="BTC",
        side=OrderSide.BUY,
        fill_quantity=Decimal("1.0"),
        fill_price=Decimal("50000.0"),
        commission=10.0,
        slippage=5.0
    )

    # Process fill
    tracker.on_fill_event(fill)
    analyzer.on_fill_event(fill, tracker)

    # Assert your custom behavior
    assert analyzer.custom_metric > 0
    assert len(analyzer.custom_data) == 1

def test_custom_analyzer_reset():
    tracker = PositionTracker(initial_cash=100000)
    analyzer = MyCustomAnalyzer(tracker)

    # Add some state
    analyzer.custom_metric = 100.0

    # Reset
    analyzer.reset()

    # Assert clean state
    assert analyzer.custom_metric == 0.0
    assert len(analyzer.custom_data) == 0
```

---

## See Also

- [API Reference](portfolio_api.md) - Complete API documentation
- [Architecture Guide](portfolio_architecture.md) - Design patterns
- [Migration Guide](portfolio_migration.md) - Upgrading from legacy
