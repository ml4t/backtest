"""Report generation for trade comparison.

Generates human-readable reports showing differences between
platforms at the trade-by-trade level.
"""

from typing import List
import sys
from pathlib import Path

# Handle imports when run as script vs module
try:
    from .matcher import TradeMatch
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from matcher import TradeMatch


def generate_trade_report(match: TradeMatch) -> str:
    """
    Generate detailed report for ONE trade.

    Shows:
    - Entry timing across platforms
    - Entry prices + components
    - Exit timing across platforms
    - Exit prices + components
    - P&L comparison
    - Verdict (match/mismatch/explained)

    Args:
        match: TradeMatch object

    Returns:
        Formatted report string

    Example:
        >>> for match in matches:
        ...     print(generate_trade_report(match))
    """
    ref = match.reference_trade
    if not ref:
        return "No trades to compare"

    lines = []
    lines.append("=" * 80)
    lines.append(f"Trade: Entry {ref.entry_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Entry Timing
    lines.append("Entry Timing:")
    for platform, trade in match.all_trades.items():
        delta = match.entry_timestamp_deltas.get(platform, 0)
        delta_str = f"{delta:+.0f}s" if abs(delta) > 1 else ""
        status = "✅" if abs(delta) <= 1 else "❌"
        lines.append(
            f"  {platform:12s}: {trade.entry_timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{delta_str:>8s} {status}"
        )
    lines.append("")

    # Entry Prices
    lines.append("Entry Prices:")
    for platform, trade in match.all_trades.items():
        component = match.entry_components.get(platform, 'unknown')
        price_diff = match.entry_price_diffs.get(platform, 0)
        diff_str = f"{price_diff:+.2f}%" if abs(price_diff) > 0.01 else ""
        status = "✅" if abs(price_diff) < 0.1 else "⚠️" if abs(price_diff) < 1.0 else "❌"
        lines.append(
            f"  {platform:12s}: ${trade.entry_price:>8.2f} ({component:>6s}) "
            f"{diff_str:>8s} {status}"
        )
    lines.append("")

    # Exit Timing (if trades are closed)
    if ref.exit_timestamp:
        lines.append("Exit Timing:")
        for platform, trade in match.all_trades.items():
            if trade.exit_timestamp:
                delta = match.exit_timestamp_deltas.get(platform, 0)
                delta_str = f"{delta:+.0f}s" if abs(delta) > 1 else ""
                status = "✅" if abs(delta) <= 1 else "❌"
                lines.append(
                    f"  {platform:12s}: {trade.exit_timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"{delta_str:>8s} {status}"
                )
            else:
                lines.append(f"  {platform:12s}: OPEN")
        lines.append("")

    # Exit Prices (if trades are closed)
    if ref.exit_price:
        lines.append("Exit Prices:")
        for platform, trade in match.all_trades.items():
            if trade.exit_price:
                component = match.exit_components.get(platform, 'unknown')
                price_diff = match.exit_price_diffs.get(platform, 0)
                diff_str = f"{price_diff:+.2f}%" if abs(price_diff) > 0.01 else ""
                status = "✅" if abs(price_diff) < 0.1 else "⚠️" if abs(price_diff) < 1.0 else "❌"
                lines.append(
                    f"  {platform:12s}: ${trade.exit_price:>8.2f} ({component:>6s}) "
                    f"{diff_str:>8s} {status}"
                )
            else:
                lines.append(f"  {platform:12s}: N/A (OPEN)")
        lines.append("")

    # Commission
    lines.append("Commission:")
    for platform, trade in match.all_trades.items():
        total_commission = trade.entry_commission + trade.exit_commission
        lines.append(f"  {platform:12s}: ${total_commission:>8.2f}")
    lines.append("")

    # P&L (Net)
    if ref.net_pnl is not None:
        lines.append("P&L (Net):")
        ref_pnl = ref.net_pnl
        for platform, trade in match.all_trades.items():
            if trade.net_pnl is not None:
                pnl_diff_pct = ((trade.net_pnl - ref_pnl) / abs(ref_pnl) * 100) if ref_pnl != 0 else 0
                diff_str = f"{pnl_diff_pct:+.1f}%" if abs(pnl_diff_pct) > 0.1 else ""
                status = "✅" if abs(pnl_diff_pct) < 1.0 else "⚠️" if abs(pnl_diff_pct) < 10.0 else "❌"
                lines.append(
                    f"  {platform:12s}: ${trade.net_pnl:>8.2f} {diff_str:>8s} {status}"
                )
            else:
                lines.append(f"  {platform:12s}: N/A (OPEN)")
        lines.append("")

    # Verdict
    lines.append("VERDICT:")
    if match.severity == 'none':
        lines.append("  ✅ All platforms match exactly")
    elif match.severity == 'minor':
        lines.append("  ⚠️  Minor differences detected:")
        for diff in match.differences:
            lines.append(f"    - {diff}")
    elif match.severity == 'major':
        lines.append("  ⚠️  Major differences detected:")
        for diff in match.differences:
            lines.append(f"    - {diff}")
    else:  # critical
        lines.append("  ❌ Critical differences detected:")
        for diff in match.differences:
            lines.append(f"    - {diff}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def generate_summary_report(matches: List[TradeMatch]) -> str:
    """
    Generate summary across all trades.

    Shows:
    - How many trades match completely
    - Common differences (entry timing, exit timing, price component)
    - Platform configurations needed for parity

    Args:
        matches: List of TradeMatch objects

    Returns:
        Formatted summary string

    Example:
        >>> summary = generate_summary_report(matches)
        >>> print(summary)
    """
    if not matches:
        return "No trades to summarize"

    lines = []
    lines.append("=" * 80)
    lines.append("VALIDATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Count by severity
    severity_counts = {'none': 0, 'minor': 0, 'major': 0, 'critical': 0}
    for match in matches:
        severity_counts[match.severity] += 1

    lines.append(f"Total trades analyzed: {len(matches)}")
    lines.append(f"  ✅ Perfect matches:     {severity_counts['none']}")
    lines.append(f"  ⚠️  Minor differences:   {severity_counts['minor']}")
    lines.append(f"  ⚠️  Major differences:   {severity_counts['major']}")
    lines.append(f"  ❌ Critical differences: {severity_counts['critical']}")
    lines.append("")

    # Platform coverage
    lines.append("Platform Coverage:")
    platforms = ['ml4t.backtest', 'vectorbt', 'backtrader', 'zipline']
    platform_counts = {p: 0 for p in platforms}

    for match in matches:
        if match.backtest_trade:
            platform_counts['ml4t.backtest'] += 1
        if match.vectorbt_trade:
            platform_counts['vectorbt'] += 1
        if match.backtrader_trade:
            platform_counts['backtrader'] += 1
        if match.zipline_trade:
            platform_counts['zipline'] += 1

    for platform in platforms:
        count = platform_counts[platform]
        pct = (count / len(matches) * 100) if matches else 0
        lines.append(f"  {platform:12s}: {count:3d} trades ({pct:5.1f}%)")
    lines.append("")

    # Common differences
    all_differences = []
    for match in matches:
        all_differences.extend(match.differences)

    if all_differences:
        lines.append("Most Common Differences:")
        # Count frequency
        diff_counts: dict[str, int] = {}
        for diff in all_differences:
            diff_counts[diff] = diff_counts.get(diff, 0) + 1

        # Sort by frequency
        sorted_diffs = sorted(diff_counts.items(), key=lambda x: x[1], reverse=True)

        for diff, count in sorted_diffs[:5]:  # Top 5
            pct = (count / len(matches) * 100)
            lines.append(f"  [{count:2d} trades, {pct:5.1f}%] {diff}")
        lines.append("")

    # Platform-specific insights
    lines.append("Platform-Specific Insights:")

    # Check for systematic timing differences
    total_entry_deltas = {p: 0.0 for p in platforms}
    total_exit_deltas = {p: 0.0 for p in platforms}
    count_entry = {p: 0 for p in platforms}
    count_exit = {p: 0 for p in platforms}

    for match in matches:
        for platform, delta in match.entry_timestamp_deltas.items():
            total_entry_deltas[platform] += delta
            count_entry[platform] += 1
        for platform, delta in match.exit_timestamp_deltas.items():
            total_exit_deltas[platform] += delta
            count_exit[platform] += 1

    for platform in platforms:
        if count_entry[platform] > 0:
            avg_entry_delta = total_entry_deltas[platform] / count_entry[platform]
            avg_exit_delta = total_exit_deltas[platform] / count_exit[platform] if count_exit[platform] > 0 else 0

            if abs(avg_entry_delta) > 3600:  # >1 hour systematic difference
                lines.append(
                    f"  {platform:12s}: Systematic {avg_entry_delta/3600:+.1f} hour entry timing offset "
                    f"(same-bar vs next-bar execution)"
                )
            elif abs(avg_entry_delta) > 60:
                lines.append(
                    f"  {platform:12s}: Average {avg_entry_delta:+.0f}s entry timing difference"
                )

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)
