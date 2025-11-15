"""Trade matcher for cross-platform comparison.

Matches trades across platforms by entry timestamp (primary key)
and validates exit timestamps, prices, and components.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import sys
from pathlib import Path

# Handle imports when run as script vs module
try:
    from ..core.trade import StandardTrade
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.trade import StandardTrade


@dataclass
class TradeMatch:
    """Result of matching one trade across platforms."""

    # Matched trades (one per platform, or None if missing)
    backtest_trade: StandardTrade | None
    vectorbt_trade: StandardTrade | None
    backtrader_trade: StandardTrade | None
    zipline_trade: StandardTrade | None

    # Match quality metrics
    entry_timestamp_deltas: dict[str, float]  # Platform -> seconds difference from reference
    exit_timestamp_deltas: dict[str, float]   # Platform -> seconds difference from reference
    entry_price_diffs: dict[str, float]       # Platform -> % difference from reference
    exit_price_diffs: dict[str, float]        # Platform -> % difference from reference

    # Component usage
    entry_components: dict[str, str]  # Platform -> 'open'/'close'/etc
    exit_components: dict[str, str]   # Platform -> 'open'/'close'/etc

    # Analysis
    differences: List[str]  # List of detected differences
    severity: str  # 'none', 'minor', 'major', 'critical'

    @property
    def reference_trade(self) -> StandardTrade | None:
        """Get the reference trade (first non-None trade)."""
        for trade in [self.backtest_trade, self.vectorbt_trade, self.backtrader_trade, self.zipline_trade]:
            if trade is not None:
                return trade
        return None

    @property
    def all_trades(self) -> dict[str, StandardTrade]:
        """Get all non-None trades as dict."""
        trades = {}
        if self.backtest_trade:
            trades['ml4t.backtest'] = self.backtest_trade
        if self.vectorbt_trade:
            trades['vectorbt'] = self.vectorbt_trade
        if self.backtrader_trade:
            trades['backtrader'] = self.backtrader_trade
        if self.zipline_trade:
            trades['zipline'] = self.zipline_trade
        return trades


def match_trades(
    trades_by_platform: dict[str, List[StandardTrade]],
    timestamp_tolerance_seconds: int = 60,
) -> List[TradeMatch]:
    """
    Match trades across platforms.

    Strategy:
    1. Group all trades by entry timestamp (with tolerance)
    2. For each group, create a TradeMatch
    3. Calculate deltas and differences

    Args:
        trades_by_platform: Dict mapping platform name to list of trades
        timestamp_tolerance_seconds: Max seconds difference for matching

    Returns:
        List of TradeMatch objects

    Example:
        >>> trades = {
        ...     'ml4t.backtest': extract_backtest_trades(...),
        ...     'vectorbt': extract_vectorbt_trades(...),
        ... }
        >>> matches = match_trades(trades)
        >>> for match in matches:
        ...     print(f"Entry: {match.reference_trade.entry_timestamp}")
    """
    # Flatten all trades with platform labels
    all_trades: List[tuple[str, StandardTrade]] = []
    for platform, trades in trades_by_platform.items():
        for trade in trades:
            all_trades.append((platform, trade))

    # Sort by entry timestamp
    all_trades.sort(key=lambda x: x[1].entry_timestamp)

    # Group by entry timestamp (with tolerance)
    groups: List[List[tuple[str, StandardTrade]]] = []
    tolerance = timedelta(seconds=timestamp_tolerance_seconds)

    for platform, trade in all_trades:
        # Find matching group
        matched = False
        for group in groups:
            # Check if trade's entry is within tolerance of any trade in group
            if any(
                abs((trade.entry_timestamp - t.entry_timestamp).total_seconds()) <= timestamp_tolerance_seconds
                for _, t in group
            ):
                group.append((platform, trade))
                matched = True
                break

        if not matched:
            # Create new group
            groups.append([(platform, trade)])

    # Create TradeMatch for each group
    matches = []
    for group in groups:
        match = _create_match_from_group(group)
        matches.append(match)

    return matches


def _create_match_from_group(group: List[tuple[str, StandardTrade]]) -> TradeMatch:
    """
    Create a TradeMatch from a group of trades.

    Uses the first trade as reference for calculating deltas.
    """
    # Organize by platform
    trades_by_platform = {platform: trade for platform, trade in group}

    ml4t.backtest = trades_by_platform.get('ml4t.backtest')
    vectorbt = trades_by_platform.get('vectorbt')
    backtrader = trades_by_platform.get('backtrader')
    zipline = trades_by_platform.get('zipline')

    # Use first trade as reference
    reference = group[0][1]

    # Calculate deltas
    entry_timestamp_deltas = {}
    exit_timestamp_deltas = {}
    entry_price_diffs = {}
    exit_price_diffs = {}
    entry_components = {}
    exit_components = {}

    for platform, trade in group:
        # Entry timestamp delta
        entry_delta = (trade.entry_timestamp - reference.entry_timestamp).total_seconds()
        entry_timestamp_deltas[platform] = entry_delta

        # Exit timestamp delta
        if trade.exit_timestamp and reference.exit_timestamp:
            exit_delta = (trade.exit_timestamp - reference.exit_timestamp).total_seconds()
            exit_timestamp_deltas[platform] = exit_delta

        # Entry price diff
        if reference.entry_price:
            entry_diff_pct = ((trade.entry_price - reference.entry_price) / reference.entry_price) * 100
            entry_price_diffs[platform] = entry_diff_pct

        # Exit price diff
        if trade.exit_price and reference.exit_price:
            exit_diff_pct = ((trade.exit_price - reference.exit_price) / reference.exit_price) * 100
            exit_price_diffs[platform] = exit_diff_pct

        # Components
        entry_components[platform] = trade.entry_price_component
        if trade.exit_price_component:
            exit_components[platform] = trade.exit_price_component

    # Detect differences
    differences = []

    # Timestamp differences
    entry_deltas_abs = [abs(d) for d in entry_timestamp_deltas.values()]
    if entry_deltas_abs and max(entry_deltas_abs) > 60:
        differences.append(f"Entry timing varies by up to {max(entry_deltas_abs):.0f} seconds")

    exit_deltas_abs = [abs(d) for d in exit_timestamp_deltas.values()]
    if exit_deltas_abs and max(exit_deltas_abs) > 60:
        differences.append(f"Exit timing varies by up to {max(exit_deltas_abs):.0f} seconds")

    # Price differences
    entry_diffs_abs = [abs(d) for d in entry_price_diffs.values()]
    if entry_diffs_abs and max(entry_diffs_abs) > 0.1:
        differences.append(f"Entry prices vary by up to {max(entry_diffs_abs):.2f}%")

    exit_diffs_abs = [abs(d) for d in exit_price_diffs.values()]
    if exit_diffs_abs and max(exit_diffs_abs) > 0.1:
        differences.append(f"Exit prices vary by up to {max(exit_diffs_abs):.2f}%")

    # Component differences
    unique_entry_components = set(entry_components.values())
    if len(unique_entry_components) > 1:
        differences.append(f"Entry uses different OHLC components: {unique_entry_components}")

    unique_exit_components = set(exit_components.values())
    if len(unique_exit_components) > 1:
        differences.append(f"Exit uses different OHLC components: {unique_exit_components}")

    # Determine severity
    if not differences:
        severity = 'none'
    elif max(entry_diffs_abs or [0]) < 1.0 and max(exit_diffs_abs or [0]) < 1.0:
        severity = 'minor'
    elif max(entry_diffs_abs or [0]) < 5.0 and max(exit_diffs_abs or [0]) < 5.0:
        severity = 'major'
    else:
        severity = 'critical'

    return TradeMatch(
        backtest_trade=ml4t.backtest,
        vectorbt_trade=vectorbt,
        backtrader_trade=backtrader,
        zipline_trade=zipline,
        entry_timestamp_deltas=entry_timestamp_deltas,
        exit_timestamp_deltas=exit_timestamp_deltas,
        entry_price_diffs=entry_price_diffs,
        exit_price_diffs=exit_price_diffs,
        entry_components=entry_components,
        exit_components=exit_components,
        differences=differences,
        severity=severity,
    )
