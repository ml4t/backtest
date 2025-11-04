# Lookahead Bias Fix Documentation

## Problem Statement

The backtesting engine exhibited a critical lookahead bias where strategies could react to a market event and have orders filled at the same event's price with zero latency. This introduced unrealistic trading conditions and could lead to overly optimistic backtest results.

## Root Cause

The SimulationBroker was processing orders immediately upon submission when in immediate execution mode (the previous default). This meant:

1. Strategy receives MarketEvent with price P at time T
2. Strategy submits order based on price P
3. Order fills immediately at price P at time T
4. Zero latency between signal and execution

This violates real-world trading mechanics where there is always a delay between observing a price, making a decision, and having the order executed.

## Solution Implemented

### 1. Execution Delay Mechanism

The SimulationBroker already had an `execution_delay` parameter that could prevent lookahead bias. When enabled:

- Orders submitted on event N are queued in `_pending_orders`
- Orders in pending queue move to `_open_orders` on event N+1
- Orders can only fill starting from event N+1

### 2. Default Behavior Change

Changed the default value of `execution_delay` from `False` to `True` in SimulationBroker:

```python
def __init__(self, ..., execution_delay: bool = True):  # Changed from False
```

### 3. Test Updates

- Created comprehensive test suite in `test_lookahead_prevention.py`
- Updated existing tests to explicitly set `execution_delay=False` where immediate execution is expected for testing purposes
- Tests verify that orders cannot fill on the same event they are submitted

## Impact on Existing Code

### Backward Compatibility

For strategies that require immediate execution (rare, mainly for testing):

```python
broker = SimulationBroker(
    initial_cash=100000,
    execution_delay=False  # Explicitly disable delay
)
```

### Typical Usage (Now Safer by Default)

```python
broker = SimulationBroker(initial_cash=100000)
# execution_delay=True by default, preventing lookahead bias
```

## Testing

The fix includes comprehensive tests that verify:

1. **Default Behavior**: Execution delay is enabled by default
2. **Market Orders**: Execute on the next event after submission
3. **Limit Orders**: Respect delay even when immediately fillable
4. **Stop Orders**: Trigger immediately but convert to market orders that respect delay
5. **Multiple Orders**: Maintain proper sequencing across events
6. **Legacy Mode**: Can still enable immediate execution when explicitly requested

## Benefits

1. **Realistic Backtesting**: Results now reflect achievable real-world performance
2. **Prevents Overfitting**: Strategies cannot exploit unrealistic zero-latency execution
3. **Industry Standard**: Aligns with best practices in backtesting frameworks
4. **Configurable**: Can still use immediate execution for specific testing needs

## Migration Guide

For existing strategies:

1. **No Action Required**: The new default prevents lookahead bias automatically
2. **Review Results**: Backtests may show different (more realistic) results
3. **Special Cases**: If immediate execution is genuinely needed (rare), explicitly set `execution_delay=False`

## Related Issues

This fix addresses one of the P0 (critical) issues identified in the code review:
- **Issue**: Lookahead Bias (Zero Latency)
- **Status**: ✅ Resolved
- **Impact**: High - Affects all backtesting accuracy

## Next Steps

While this fix addresses the immediate lookahead bias issue, consider future enhancements:

1. **Variable Latency**: Model realistic order submission to execution delays
2. **Market Hours**: Ensure orders respect market open/close times
3. **Order Priority**: Implement price-time priority for limit orders
4. **Partial Fills**: More sophisticated fill modeling based on market depth