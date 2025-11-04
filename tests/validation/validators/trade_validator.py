"""Trade-level comparison validator."""
from dataclasses import dataclass
from datetime import datetime

import polars as pl

try:
    from ..adapters.base import BacktestResult, Trade
except ImportError:
    # Handle running as script
    from adapters.base import BacktestResult, Trade


@dataclass
class TradeDifference:
    """Represents a difference between trades across platforms."""
    trade_index: int
    platform1: str
    platform2: str
    field: str
    value1: any
    value2: any
    difference: float | str

    def __str__(self) -> str:
        return (f"Trade #{self.trade_index}: {self.field} differs - "
                f"{self.platform1}={self.value1}, {self.platform2}={self.value2} "
                f"(diff: {self.difference})")


@dataclass
class ValidationReport:
    """Comprehensive validation report across platforms."""
    platforms: list[str]
    trade_count_match: bool
    trade_differences: list[TradeDifference]
    pnl_comparison: dict[str, float]
    metrics_comparison: dict[str, dict[str, float]]
    summary: str


class TradeValidator:
    """Validates and compares trades across backtesting platforms."""

    def __init__(self, tolerance_pct: float = 0.1):
        """Initialize validator.

        Args:
            tolerance_pct: Tolerance for price/PnL differences (0.1 = 0.1%)
        """
        self.tolerance_pct = tolerance_pct

    def compare_results(
        self,
        results: dict[str, BacktestResult]
    ) -> ValidationReport:
        """Compare backtest results across all platforms.

        Args:
            results: Dict mapping platform name to BacktestResult

        Returns:
            ValidationReport with detailed comparison
        """
        platforms = list(results.keys())

        # Compare trade counts
        trade_counts = {name: len(res.get_closed_trades()) for name, res in results.items()}
        trade_count_match = len(set(trade_counts.values())) == 1

        # Find trade differences
        trade_diffs = []
        if len(platforms) >= 2:
            # Compare first two platforms in detail
            platform1, platform2 = platforms[0], platforms[1]
            diffs = self._compare_trades(
                results[platform1].get_closed_trades(),
                results[platform2].get_closed_trades(),
                platform1,
                platform2
            )
            trade_diffs.extend(diffs)

        # Compare P&L
        pnl_comparison = {
            name: res.total_pnl()
            for name, res in results.items()
        }

        # Compare metrics
        metrics_comparison = {
            name: res.metrics
            for name, res in results.items()
        }

        # Generate summary
        summary = self._generate_summary(
            trade_counts,
            trade_count_match,
            trade_diffs,
            pnl_comparison
        )

        return ValidationReport(
            platforms=platforms,
            trade_count_match=trade_count_match,
            trade_differences=trade_diffs,
            pnl_comparison=pnl_comparison,
            metrics_comparison=metrics_comparison,
            summary=summary
        )

    def _compare_trades(
        self,
        trades1: list[Trade],
        trades2: list[Trade],
        platform1: str,
        platform2: str
    ) -> list[TradeDifference]:
        """Compare two lists of trades."""
        differences = []

        # Compare by index (assuming same ordering)
        for i in range(min(len(trades1), len(trades2))):
            t1, t2 = trades1[i], trades2[i]

            # Compare entry time
            if t1.entry_time != t2.entry_time:
                differences.append(TradeDifference(
                    trade_index=i,
                    platform1=platform1,
                    platform2=platform2,
                    field='entry_time',
                    value1=t1.entry_time,
                    value2=t2.entry_time,
                    difference=str(abs((t1.entry_time - t2.entry_time).total_seconds())) + 's'
                ))

            # Compare entry price
            if not self._prices_match(t1.entry_price, t2.entry_price):
                diff_pct = abs(t1.entry_price - t2.entry_price) / t1.entry_price * 100
                differences.append(TradeDifference(
                    trade_index=i,
                    platform1=platform1,
                    platform2=platform2,
                    field='entry_price',
                    value1=t1.entry_price,
                    value2=t2.entry_price,
                    difference=f"{diff_pct:.2f}%"
                ))

            # Compare exit price (if both closed)
            if t1.exit_price and t2.exit_price:
                if not self._prices_match(t1.exit_price, t2.exit_price):
                    diff_pct = abs(t1.exit_price - t2.exit_price) / t1.exit_price * 100
                    differences.append(TradeDifference(
                        trade_index=i,
                        platform1=platform1,
                        platform2=platform2,
                        field='exit_price',
                        value1=t1.exit_price,
                        value2=t2.exit_price,
                        difference=f"{diff_pct:.2f}%"
                    ))

            # Compare P&L
            if t1.pnl and t2.pnl:
                if not self._prices_match(t1.pnl, t2.pnl):
                    diff = abs(t1.pnl - t2.pnl)
                    differences.append(TradeDifference(
                        trade_index=i,
                        platform1=platform1,
                        platform2=platform2,
                        field='pnl',
                        value1=t1.pnl,
                        value2=t2.pnl,
                        difference=f"${diff:.2f}"
                    ))

        return differences

    def _prices_match(self, price1: float, price2: float) -> bool:
        """Check if two prices match within tolerance."""
        if price1 == price2:
            return True

        diff_pct = abs(price1 - price2) / max(abs(price1), abs(price2)) * 100
        return diff_pct <= self.tolerance_pct

    def _generate_summary(
        self,
        trade_counts: dict,
        trade_count_match: bool,
        trade_diffs: list,
        pnl_comparison: dict
    ) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("="* 60)
        lines.append("CROSS-PLATFORM VALIDATION SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        # Trade counts
        lines.append("Trade Counts:")
        for platform, count in trade_counts.items():
            lines.append(f"  {platform:20s}: {count:4d} trades")
        lines.append(f"  Match: {'✓ YES' if trade_count_match else '✗ NO'}")
        lines.append("")

        # P&L comparison
        lines.append("Total P&L:")
        for platform, pnl in pnl_comparison.items():
            lines.append(f"  {platform:20s}: ${pnl:,.2f}")
        lines.append("")

        # Differences
        if trade_diffs:
            lines.append(f"Trade Differences Found: {len(trade_diffs)}")
            lines.append("(Showing first 10)")
            for diff in trade_diffs[:10]:
                lines.append(f"  - {diff}")
        else:
            lines.append("✓ No significant trade differences found")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def generate_html_report(self, report: ValidationReport, output_path: str) -> None:
        """Generate detailed HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .match {{ color: green; }}
        .diff {{ color: red; }}
        .summary {{ background-color: #f0f0f0; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Cross-Platform Backtest Validation</h1>

    <div class="summary">
        <h2>Summary</h2>
        <pre>{report.summary}</pre>
    </div>

    <h2>Trade Count Comparison</h2>
    <table>
        <tr><th>Platform</th><th>Trade Count</th></tr>
        {"".join(f"<tr><td>{p}</td><td>{len(report.trade_differences)}</td></tr>" for p in report.platforms)}
    </table>

    <h2>P&L Comparison</h2>
    <table>
        <tr><th>Platform</th><th>Total P&L</th></tr>
        {"".join(f"<tr><td>{p}</td><td>${pnl:,.2f}</td></tr>" for p, pnl in report.pnl_comparison.items())}
    </table>

    <h2>Trade-Level Differences</h2>
    <table>
        <tr><th>#</th><th>Field</th><th>Platform 1</th><th>Value 1</th><th>Platform 2</th><th>Value 2</th><th>Diff</th></tr>
        {"".join(f'''<tr>
            <td>{d.trade_index}</td>
            <td>{d.field}</td>
            <td>{d.platform1}</td>
            <td>{d.value1}</td>
            <td>{d.platform2}</td>
            <td>{d.value2}</td>
            <td>{d.difference}</td>
        </tr>''' for d in report.trade_differences[:100])}
    </table>

</body>
</html>
        """

        with open(output_path, 'w') as f:
            f.write(html)
