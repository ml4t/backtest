# Proposal: Portfolio Optimizer Integration (v2 - Refined)

**Date**: 2025-11-23
**Author**: Claude Code
**Status**: Draft v2 - Incorporating External Review Feedback
**Library**: ml4t.backtest
**Reviewer**: Gemini (external)

---

## Changes from v1

This version incorporates feedback from external review:

| Issue | v1 Status | v2 Resolution |
|-------|-----------|---------------|
| Pending order awareness | Not handled | Added `_get_effective_weights()` |
| Double-order risk | Vulnerable | Cancel-before-rebalance option |
| Fractional shares | Not configurable | Added `allow_fractional` config |
| Cash targeting | Implicit | Explicit `sum(weights) < 1.0` support |
| Class vs function | Class | Validated: class is correct |
| Partial fills | Not handled | Validated: natural correction on next bar |
| Constraints | Not handled | Validated: belongs in optimizer, not executor |

---

## Executive Summary

**Recommendation**: Implement Option B (TargetWeightExecutor) - **VALIDATED** by external review.

Key refinements needed:
1. **Pending order awareness**: Account for orders in-flight when calculating current weights
2. **Fractional shares**: Add `allow_fractional` config (default: False)
3. **Cancel-replace logic**: Option to cancel pending orders before rebalancing

---

## Critical Fix: The Pending Order Trap

### The Problem

In v1, `_get_current_weights()` only considers held positions:

```python
# v1 (VULNERABLE)
for asset, pos in broker.positions.items():
    value = pos.quantity * price
    weights[asset] = value / equity
```

**Risk scenario** (especially with `ExecutionMode.NEXT_BAR` or LIMIT orders):

1. T=1: Executor sees 0% AAPL. Target 10%. Submits BUY order.
2. T=2: Order still pending. Executor sees 0% AAPL. Target 10%. Submits ANOTHER BUY.
3. Result: 20% AAPL allocation (double the target).

### The Fix: Effective Weights

Account for pending orders when calculating "effective" position:

```python
def _get_effective_weights(
    self,
    broker: Broker,
    data: dict[str, dict]
) -> dict[str, float]:
    """Get effective portfolio weights including pending orders."""
    equity = broker.get_account_value()
    if equity <= 0:
        return {}

    # 1. Start with actual positions
    effective_value: dict[str, float] = {}
    for asset, pos in broker.positions.items():
        price = data.get(asset, {}).get('close', pos.entry_price)
        effective_value[asset] = pos.quantity * price

    # 2. Add net value of pending orders
    for order in broker.pending_orders:
        price = order.limit_price or data.get(order.asset, {}).get('close')
        if price:
            # BUY adds value, SELL subtracts
            sign = 1 if order.side == OrderSide.BUY else -1
            delta = order.quantity * price * sign
            effective_value[order.asset] = effective_value.get(order.asset, 0) + delta

    # 3. Convert to weights
    return {k: v / equity for k, v in effective_value.items()}
```

### Alternative: Cancel-Before-Rebalance

For simpler logic, cancel all pending orders before rebalancing:

```python
def execute(self, target_weights: dict[str, float], data: dict, broker: Broker) -> list[Order]:
    # Cancel existing orders to prevent race conditions
    if self.config.cancel_before_rebalance:
        for order in list(broker.pending_orders):
            broker.cancel_order(order.order_id)

    # Now safe to use simple _get_current_weights()
    current_weights = self._get_current_weights(broker, data)
    # ... rest of logic
```

---

## Updated RebalanceConfig

```python
@dataclass
class RebalanceConfig:
    """Configuration for rebalancing behavior."""
    # Trade thresholds
    min_trade_value: float = 100.0      # Skip trades smaller than this ($)
    min_weight_change: float = 0.01     # Skip if weight change < 1%

    # Share handling
    allow_fractional: bool = False      # Allow fractional shares (default: whole only)
    round_lots: bool = False            # Round to lot_size increments
    lot_size: int = 100                 # Lot size for rounding

    # Position constraints
    allow_short: bool = False           # Allow short positions
    max_single_weight: float = 1.0      # Max weight per asset

    # Order handling
    cancel_before_rebalance: bool = True  # Cancel pending orders first (safest)
    account_for_pending: bool = True      # Consider pending orders in weights calc
```

---

## Updated TargetWeightExecutor (v2)

```python
# src/ml4t/backtest/execution/rebalancer.py

from dataclasses import dataclass
from typing import Protocol

from ..broker import Broker
from ..types import Order, OrderSide


class WeightProvider(Protocol):
    """Protocol for anything that produces target weights."""
    def get_weights(self, data: dict, broker: Broker) -> dict[str, float]:
        """Return target weights (asset -> weight, should sum to <= 1.0)."""
        ...


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing behavior."""
    # Trade thresholds
    min_trade_value: float = 100.0
    min_weight_change: float = 0.01

    # Share handling
    allow_fractional: bool = False
    round_lots: bool = False
    lot_size: int = 100

    # Position constraints
    allow_short: bool = False
    max_single_weight: float = 1.0

    # Order handling
    cancel_before_rebalance: bool = True
    account_for_pending: bool = True


class TargetWeightExecutor:
    """Convert target portfolio weights to orders.

    Handles the common pattern of rebalancing to target weights:
    - Computes required trades from current vs target positions
    - Accounts for pending orders to prevent double-allocation
    - Applies minimum trade thresholds
    - Handles lot rounding and fractional shares
    - Respects position limits

    Example:
        executor = TargetWeightExecutor(config=RebalanceConfig(
            min_trade_value=500,
            allow_fractional=True,
        ))

        # In strategy:
        target_weights = {'AAPL': 0.3, 'GOOG': 0.3, 'MSFT': 0.35}  # 5% cash
        orders = executor.execute(target_weights, data, broker)
    """

    def __init__(self, config: RebalanceConfig | None = None):
        self.config = config or RebalanceConfig()

    def execute(
        self,
        target_weights: dict[str, float],
        data: dict[str, dict],
        broker: Broker,
    ) -> list[Order]:
        """Execute rebalancing to target weights.

        Args:
            target_weights: Dict of asset -> target weight (0.0 to 1.0)
                            Sum can be < 1.0 to hold cash.
            data: Current bar data (for prices)
            broker: Broker instance for order submission

        Returns:
            List of submitted orders
        """
        # 1. Cancel pending orders if configured (prevents double-allocation)
        if self.config.cancel_before_rebalance:
            for order in list(broker.pending_orders):
                broker.cancel_order(order.order_id)

        equity = broker.get_account_value()
        orders = []

        # 2. Get current weights (effective or actual based on config)
        if self.config.account_for_pending and not self.config.cancel_before_rebalance:
            current_weights = self._get_effective_weights(broker, data)
        else:
            current_weights = self._get_current_weights(broker, data)

        # 3. Validate total weight <= 1.0 (allow cash targeting)
        total_target = sum(target_weights.values())
        if total_target > 1.0 + 1e-6:
            # Scale down to prevent over-allocation
            scale = 1.0 / total_target
            target_weights = {k: v * scale for k, v in target_weights.items()}

        # 4. Process each target asset
        for asset, target_wt in target_weights.items():
            # Apply constraints
            target_wt = min(target_wt, self.config.max_single_weight)
            if target_wt < 0 and not self.config.allow_short:
                target_wt = 0

            current_wt = current_weights.get(asset, 0.0)
            weight_delta = target_wt - current_wt

            # Skip small weight changes
            if abs(weight_delta) < self.config.min_weight_change:
                continue

            # Get price
            price = data.get(asset, {}).get('close')
            if not price or price <= 0:
                continue

            # Compute trade value
            delta_value = equity * weight_delta

            # Skip small trades
            if abs(delta_value) < self.config.min_trade_value:
                continue

            # Compute shares
            shares = delta_value / price

            # Apply share rounding
            if self.config.round_lots:
                shares = round(shares / self.config.lot_size) * self.config.lot_size
            elif not self.config.allow_fractional:
                shares = int(shares)

            if shares == 0:
                continue

            # Submit order
            side = OrderSide.BUY if shares > 0 else OrderSide.SELL
            order = broker.submit_order(asset, abs(shares), side)
            orders.append(order)

        # 5. Close positions not in target
        for asset in current_weights:
            if asset not in target_weights:
                pos = broker.get_position(asset)
                if pos and pos.quantity != 0:
                    order = broker.close_position(asset)
                    if order:
                        orders.append(order)

        return orders

    def _get_current_weights(
        self,
        broker: Broker,
        data: dict[str, dict]
    ) -> dict[str, float]:
        """Get current portfolio weights from held positions only."""
        equity = broker.get_account_value()
        if equity <= 0:
            return {}

        weights = {}
        for asset, pos in broker.positions.items():
            price = data.get(asset, {}).get('close', pos.entry_price)
            value = pos.quantity * price
            weights[asset] = value / equity

        return weights

    def _get_effective_weights(
        self,
        broker: Broker,
        data: dict[str, dict]
    ) -> dict[str, float]:
        """Get effective weights including pending orders."""
        equity = broker.get_account_value()
        if equity <= 0:
            return {}

        # Start with actual positions
        effective_value: dict[str, float] = {}
        for asset, pos in broker.positions.items():
            price = data.get(asset, {}).get('close', pos.entry_price)
            effective_value[asset] = pos.quantity * price

        # Add net value of pending orders
        for order in broker.pending_orders:
            price = order.limit_price or data.get(order.asset, {}).get('close')
            if price:
                sign = 1 if order.side == OrderSide.BUY else -1
                delta = order.quantity * price * sign
                effective_value[order.asset] = effective_value.get(order.asset, 0) + delta

        return {k: v / equity for k, v in effective_value.items()}

    def preview(
        self,
        target_weights: dict[str, float],
        data: dict[str, dict],
        broker: Broker,
    ) -> list[dict]:
        """Preview trades without executing.

        Returns:
            List of trade previews with asset, current_weight, target_weight,
            shares, value, and whether it would be skipped.
        """
        equity = broker.get_account_value()

        if self.config.account_for_pending and not self.config.cancel_before_rebalance:
            current_weights = self._get_effective_weights(broker, data)
        else:
            current_weights = self._get_current_weights(broker, data)

        previews = []

        for asset, target_wt in target_weights.items():
            current_wt = current_weights.get(asset, 0.0)
            price = data.get(asset, {}).get('close', 0)
            weight_delta = target_wt - current_wt

            if price > 0:
                delta_value = equity * weight_delta
                shares = delta_value / price

                # Determine if would be skipped
                skip_reason = None
                if abs(weight_delta) < self.config.min_weight_change:
                    skip_reason = "weight_change_too_small"
                elif abs(delta_value) < self.config.min_trade_value:
                    skip_reason = "trade_value_too_small"

                previews.append({
                    'asset': asset,
                    'current_weight': current_wt,
                    'target_weight': target_wt,
                    'weight_delta': weight_delta,
                    'shares': shares,
                    'value': delta_value,
                    'skip_reason': skip_reason,
                })

        return previews
```

---

## Review Questions - Resolved

### Q1: Class vs Function?
**Answer**: Keep as Class.
- State configuration (`RebalanceConfig`) would make function signature unwieldy
- Future drift tracking would require statefulness
- Class provides clean encapsulation

### Q2: Partial Fills?
**Answer**: No special handling needed.
- Keep executor stateless regarding fill history
- Natural correction on next rebalance:
  - T=1: Target 100 shares, fill 60
  - T=2: Current=60, Target=100, Delta=40 -> order for 40

### Q3: Transaction Cost Awareness?
**Answer**: Keep minimal.
- `min_trade_value` handles 90% of cases
- Complex cost modeling belongs in optimizer, not executor
- Optional: `ignore_if_cost_exceeds_bps` for advanced users

### Q4: Constraints (Sector Limits)?
**Answer**: NO - belongs in Optimizer layer.
- Executor should blindly obey target weights
- Silent constraint overrides would be confusing
- Optimizer knows constraints; executor executes

### Q5: Live Trading?
**Answer**: Translates well.
- Pending order awareness is CRITICAL for live trading
- Higher latency = orders pending longer
- Cancel-before-rebalance pattern is safest

---

## Implementation Plan

### Phase 1: Core Implementation
1. Create `src/ml4t/backtest/execution/__init__.py`
2. Create `src/ml4t/backtest/execution/rebalancer.py`
3. Add to package exports in `src/ml4t/backtest/__init__.py`

### Phase 2: Broker Support
1. Add `cancel_order(order_id)` method if not present
2. Verify `pending_orders` list access works correctly

### Phase 3: Tests
1. Basic rebalancing tests
2. Pending order awareness tests
3. Fractional vs whole share tests
4. Cash targeting tests (weights < 1.0)
5. Cancel-before-rebalance tests

### Phase 4: Documentation
1. Add docstrings with examples
2. Create usage examples in docs/
3. Add to package README

---

## Estimated Effort

| Component | Lines | Complexity |
|-----------|-------|------------|
| RebalanceConfig | ~20 | Low |
| TargetWeightExecutor | ~150 | Medium |
| Tests | ~200 | Medium |
| Documentation | ~100 | Low |
| **Total** | ~470 | Medium |

---

## Changelog

- 2025-11-23 v1: Initial draft
- 2025-11-23 v2: Incorporated external review feedback
  - Added pending order awareness
  - Added `allow_fractional` config
  - Added `cancel_before_rebalance` option
  - Resolved all review questions
  - Added implementation plan
