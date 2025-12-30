# Clock Multi-Feed Synchronization Fix

## Problem Statement

The Clock's logic for merging multiple data feeds was flawed, potentially causing:
1. **Comparison errors**: When timestamps were identical, Python's heapq would try to compare Event objects, which could fail if Events aren't comparable
2. **Non-deterministic ordering**: Even if Events were comparable, the order of events with identical timestamps would be unpredictable
3. **Instability across runs**: The same data could produce different event sequences in different runs

## Root Cause Analysis

The heap-based approach was fundamentally correct - each feed maintains one event in the priority queue, and the heap ensures chronological ordering. However, the implementation stored tuples as `(timestamp, event, source)`.

When multiple events had identical timestamps (common with multi-frequency data like combining tick and daily data), Python's heapq would compare the second element (the Event object) to break ties. This caused two problems:
1. Event objects might not implement comparison operators, causing TypeError
2. Even if they did, the ordering would be based on object properties rather than arrival order

## Solution Implemented

### Stable Ordering with Sequence Counter

Added a monotonically increasing sequence counter to ensure FIFO ordering when timestamps are identical:

```python
# Before: (timestamp, event, source)
# After:  (timestamp, sequence, event, source)

self._event_queue: list[tuple[datetime, int, Event, object]] = []
self._sequence_counter = 0  # Ensures FIFO when timestamps are identical
```

### Updated Methods

1. **Initialization**: Added `_sequence_counter` to track insertion order
2. **_prime_feed**: Inserts `(timestamp, sequence, event, source)` and increments counter
3. **_prime_signal_source**: Same pattern for signal sources
4. **get_next_event**: Unpacks 4-tuple instead of 3-tuple
5. **_replenish_queue**: Maintains sequence counter for replenished events

## Benefits

1. **Deterministic Ordering**: Events with identical timestamps are processed in FIFO order
2. **No Comparison Errors**: Sequence counter is always comparable (integer)
3. **Stable Across Runs**: Same input data produces same event sequence
4. **Multi-Frequency Support**: Correctly handles mixed tick and daily data
5. **Backward Compatible**: No API changes, just internal implementation

## Testing

Comprehensive test suite in `test_clock_multi_feed.py` verifies:

1. **Single Feed**: Basic functionality preserved
2. **Interleaved Feeds**: Correct chronological merging
3. **Identical Timestamps**: Deterministic FIFO ordering
4. **Different Frequencies**: Tick and daily data mixed correctly
5. **End Time Respect**: Events beyond end_time are filtered
6. **Empty Feeds**: Graceful handling of empty data
7. **Complex Scenarios**: Three+ feeds with various timings

## Example: Handling Identical Timestamps

```python
# Two feeds with events at 9:30:00
feed1: AAPL at 9:30:00 (added first)
feed2: GOOGL at 9:30:00 (added second)

# Old behavior: Unpredictable order or TypeError
# New behavior: AAPL processed first (FIFO)
```

## Migration Impact

No changes required for existing code. The fix is purely internal and maintains all existing behavior while adding stability and correctness.

## Performance Considerations

- Minimal overhead: One additional integer per event in heap
- Heap operations remain O(log n)
- No impact on memory usage (one int per event is negligible)

## Related Issues

This fix addresses the P0 (critical) issue from code review:
- **Issue**: Correct Multi-Feed Synchronization (Broken)
- **Status**: ✅ Resolved
- **Impact**: High - Affects all multi-feed backtests

## Future Enhancements

While this fix resolves the immediate synchronization issues, consider future improvements:

1. **Priority Levels**: Allow feeds to have priority beyond timestamp
2. **Event Batching**: Process multiple events atomically when needed
3. **Lazy Loading**: Stream events instead of loading all into memory
4. **Parallel Processing**: Process independent events concurrently
