"""Generate fixed, reproducible signals for validation tests."""
import pandas as pd
import numpy as np


def generate_fixed_entries(
    n_bars: int,
    entry_every: int = 50,
    start_offset: int = 10,
) -> pd.Series:
    """
    Generate fixed entry signals at regular intervals.

    Args:
        n_bars: Total number of bars
        entry_every: Generate entry signal every N bars
        start_offset: Number of bars to skip before first entry

    Returns:
        Boolean Series with True at entry signals
    """
    signals = pd.Series([False] * n_bars)

    # Generate entries at fixed intervals
    for i in range(start_offset, n_bars, entry_every):
        signals.iloc[i] = True

    return signals


def generate_entry_exit_pairs(
    n_bars: int,
    entry_every: int = 50,
    hold_bars: int = 10,
    start_offset: int = 10,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate paired entry and exit signals.

    Each entry is followed by an exit after hold_bars.

    Args:
        n_bars: Total number of bars
        entry_every: Generate entry signal every N bars
        hold_bars: Number of bars to hold before exit
        start_offset: Number of bars to skip before first entry

    Returns:
        Tuple of (entry_signals, exit_signals)
    """
    entries = pd.Series([False] * n_bars)
    exits = pd.Series([False] * n_bars)

    # Generate entry/exit pairs
    for i in range(start_offset, n_bars, entry_every):
        entries.iloc[i] = True

        # Exit after hold_bars
        exit_idx = i + hold_bars
        if exit_idx < n_bars:
            exits.iloc[exit_idx] = True

    return entries, exits


def generate_random_signals(
    n_bars: int,
    entry_probability: float = 0.05,
    exit_probability: float = 0.05,
    seed: int = 42,
    require_flat: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate random entry and exit signals.

    Args:
        n_bars: Total number of bars
        entry_probability: Probability of entry signal at each bar
        exit_probability: Probability of exit signal at each bar
        seed: Random seed for reproducibility
        require_flat: Only allow entry when flat, exit when in position

    Returns:
        Tuple of (entry_signals, exit_signals)
    """
    np.random.seed(seed)

    entries = pd.Series([False] * n_bars)
    exits = pd.Series([False] * n_bars)

    if require_flat:
        # Track position state
        in_position = False

        for i in range(n_bars):
            if not in_position:
                # Can only enter when flat
                if np.random.rand() < entry_probability:
                    entries.iloc[i] = True
                    in_position = True
            else:
                # Can only exit when in position
                if np.random.rand() < exit_probability:
                    exits.iloc[i] = True
                    in_position = False
    else:
        # Independent random signals
        entries = pd.Series(np.random.rand(n_bars) < entry_probability)
        exits = pd.Series(np.random.rand(n_bars) < exit_probability)

    return entries, exits


def generate_exit_on_next_entry(entries: pd.Series, close_final: bool = True) -> pd.Series:
    """
    Generate exit signals one bar before each subsequent entry.

    This creates explicit exit signals that simulate "hold until next entry" behavior
    in a framework-agnostic way. Each entry triggers an exit one bar before the next
    entry signal.

    This approach avoids VectorBT's conflict resolution issues where simultaneous
    exit+entry signals on the same bar cannot both be executed.

    Args:
        entries: Boolean series with True at entry signals
        close_final: If True, add exit signal at final bar to close last position

    Returns:
        Boolean series with True at exit signals (one bar before each subsequent entry,
        plus optionally at the final bar)

    Example:
        >>> entries = pd.Series([False, True, False, False, True, False])
        >>> exits = generate_exit_on_next_entry(entries, close_final=True)
        >>> # exits = [False, False, False, True, False, True]
        >>> # Exit at index 3 (one bar before second entry at index 4)
        >>> # Exit at index 5 (close final position)
    """
    exits = pd.Series([False] * len(entries), index=entries.index)
    entry_indices = entries[entries].index.tolist()

    # For each entry after the first, place exit one bar before
    for i in range(1, len(entry_indices)):
        entry_time = entry_indices[i]
        idx_pos = entries.index.get_loc(entry_time)
        if idx_pos > 0:
            exits.iloc[idx_pos - 1] = True

    # Optionally close the final position at the last bar
    if close_final and len(entry_indices) > 0:
        exits.iloc[-1] = True

    return exits


def validate_signals(
    entries: pd.Series,
    exits: pd.Series,
    allow_simultaneous: bool = False,
) -> bool:
    """
    Validate signal series for common issues.

    Args:
        entries: Entry signal series
        exits: Exit signal series
        allow_simultaneous: Allow entry and exit on same bar

    Returns:
        True if valid, raises AssertionError otherwise
    """
    # Check for NaN
    assert not entries.isnull().any(), "Entry signals contain NaN"
    assert not exits.isnull().any(), "Exit signals contain NaN"

    # Check boolean type
    assert entries.dtype == bool, "Entry signals must be boolean"
    assert exits.dtype == bool, "Exit signals must be boolean"

    # Check same length
    assert len(entries) == len(exits), "Entry and exit signals must have same length"

    # Check for simultaneous signals
    if not allow_simultaneous:
        simultaneous = (entries & exits).sum()
        assert simultaneous == 0, f"Found {simultaneous} bars with both entry and exit signals"

    return True


if __name__ == "__main__":
    # Test signal generation
    print("Testing signal generators...")

    print("\n1. Fixed entries (every 50 bars):")
    entries = generate_fixed_entries(n_bars=200, entry_every=50)
    print(f"   Generated {entries.sum()} entry signals")
    print(f"   Entry indices: {entries[entries].index.tolist()}")

    print("\n2. Entry/exit pairs (hold 10 bars):")
    entries, exits = generate_entry_exit_pairs(n_bars=200, entry_every=50, hold_bars=10)
    print(f"   Generated {entries.sum()} entries, {exits.sum()} exits")

    print("\n3. Random signals:")
    entries, exits = generate_random_signals(n_bars=200, entry_probability=0.05, exit_probability=0.05)
    print(f"   Generated {entries.sum()} entries, {exits.sum()} exits")

    print("\n4. Validating signals...")
    try:
        validate_signals(entries, exits)
        print("   ✅ Signals valid")
    except AssertionError as e:
        print(f"   ❌ Validation failed: {e}")
