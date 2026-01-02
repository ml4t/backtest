"""Efficient export utilities for backtest results.

This module provides batch export capabilities optimized for:
- Parameter sweep results (hundreds of backtests)
- Persistent storage with Parquet
- Summary DataFrames for analysis

Example - Parameter sweep export:
    >>> from ml4t.backtest import BacktestExporter, Engine
    >>>
    >>> results = []
    >>> params = []
    >>> for stop in [0.02, 0.03, 0.05]:
    ...     for target in [0.05, 0.10]:
    ...         result = run_backtest(stop_pct=stop, target_pct=target)
    ...         results.append(result)
    ...         params.append({"stop_pct": stop, "target_pct": target})
    >>>
    >>> summary = BacktestExporter.batch_export(
    ...     results=results,
    ...     base_path="./sweep_results",
    ...     param_names=["stop_pct", "target_pct"],
    ...     param_values=params,
    ... )
    >>>
    >>> # Find best parameters
    >>> best = summary.sort("sharpe", descending=True).head(5)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    from .result import BacktestResult


class BacktestExporter:
    """Utilities for efficient backtest result export."""

    @staticmethod
    def to_parquet(
        result: BacktestResult,
        path: str | Path,
        compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
        include: list[str] | None = None,
    ) -> dict[str, Path]:
        """Export single backtest result to Parquet files.

        Delegates to BacktestResult.to_parquet().

        Args:
            result: BacktestResult to export
            path: Directory path to write files
            compression: Parquet compression codec
            include: Components to include

        Returns:
            Dict mapping component names to file paths
        """
        return result.to_parquet(path, include=include, compression=compression)

    @staticmethod
    def batch_export(
        results: list[BacktestResult],
        base_path: str | Path,
        param_values: list[dict[str, Any]],
        compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
        export_individual: bool = True,
    ) -> pl.DataFrame:
        """Export multiple backtest results efficiently.

        Creates:
            {base_path}/
                sweep_summary.parquet  # All params + key metrics
                run_0001/              # Individual result (if export_individual=True)
                run_0002/
                ...

        Args:
            results: List of BacktestResult objects
            base_path: Base directory for export
            param_values: List of dicts with parameter values for each result
            compression: Parquet compression codec
            export_individual: If True, export each result to subdirectory

        Returns:
            Summary DataFrame with all params and key metrics
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        if len(results) != len(param_values):
            raise ValueError(
                f"results ({len(results)}) and param_values ({len(param_values)}) "
                "must have same length"
            )

        # Build summary records
        summary_records = []

        for i, (result, params) in enumerate(zip(results, param_values)):
            # Start with parameters
            record: dict[str, Any] = dict(params)

            # Add run index
            record["run_id"] = i + 1

            # Add key metrics from result
            metrics = result.metrics
            record["num_trades"] = metrics.get("num_trades", 0)
            record["total_return"] = metrics.get("total_return", 0.0)
            record["total_return_pct"] = metrics.get("total_return_pct", 0.0)
            record["max_drawdown"] = metrics.get("max_drawdown", 0.0)
            record["max_drawdown_pct"] = metrics.get("max_drawdown_pct", 0.0)
            record["sharpe"] = metrics.get("sharpe", 0.0)
            record["sortino"] = metrics.get("sortino", 0.0)
            record["calmar"] = metrics.get("calmar", 0.0)
            record["cagr"] = metrics.get("cagr", 0.0)
            record["win_rate"] = metrics.get("win_rate", 0.0)
            record["profit_factor"] = metrics.get("profit_factor", 0.0)
            record["expectancy"] = metrics.get("expectancy", 0.0)
            record["avg_trade"] = metrics.get("avg_trade", 0.0)
            record["final_value"] = metrics.get("final_value", 0.0)
            record["total_commission"] = metrics.get("total_commission", 0.0)
            record["total_slippage"] = metrics.get("total_slippage", 0.0)

            summary_records.append(record)

            # Export individual result if requested
            if export_individual:
                run_path = base_path / f"run_{i + 1:04d}"
                result.to_parquet(run_path, compression=compression)

        # Build summary DataFrame
        summary_df = pl.DataFrame(summary_records)

        # Write summary
        summary_path = base_path / "sweep_summary.parquet"
        summary_df.write_parquet(summary_path, compression=compression)

        # Also write as CSV for quick viewing
        csv_path = base_path / "sweep_summary.csv"
        summary_df.write_csv(csv_path)

        return summary_df

    @staticmethod
    def from_parquet(path: str | Path) -> BacktestResult:
        """Load backtest result from Parquet directory.

        Delegates to BacktestResult.from_parquet().

        Args:
            path: Directory containing Parquet files

        Returns:
            BacktestResult instance
        """
        from .result import BacktestResult

        return BacktestResult.from_parquet(path)

    @staticmethod
    def load_sweep_summary(base_path: str | Path) -> pl.DataFrame:
        """Load sweep summary DataFrame from batch export.

        Args:
            base_path: Base directory from batch_export()

        Returns:
            Summary DataFrame with all params and metrics
        """
        summary_path = Path(base_path) / "sweep_summary.parquet"
        return pl.read_parquet(summary_path)

    @staticmethod
    def generate_json_report(
        results: list[BacktestResult],
        output_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Generate JSON report for CI/CD integration.

        Args:
            results: List of BacktestResult objects
            output_path: Path to write JSON file
            metadata: Optional metadata to include (e.g., version, git hash)
        """
        import sys

        result_records: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            metrics = result.metrics
            result_records.append(
                {
                    "index": i,
                    "num_trades": metrics.get("num_trades", 0),
                    "total_return_pct": metrics.get("total_return_pct", 0.0),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
                    "sharpe": metrics.get("sharpe", 0.0),
                    "sortino": metrics.get("sortino", 0.0),
                    "calmar": metrics.get("calmar", 0.0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "profit_factor": metrics.get("profit_factor", 0.0),
                    "final_value": metrics.get("final_value", 0.0),
                }
            )

        report = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "python_version": sys.version.split()[0],
                "num_results": len(results),
                **(metadata or {}),
            },
            "results": result_records,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    @staticmethod
    def generate_markdown_report(
        results: list[BacktestResult],
        output_path: str | Path,
        title: str = "Backtest Results",
        param_names: list[str] | None = None,
        param_values: list[dict[str, Any]] | None = None,
    ) -> None:
        """Generate Markdown report for human review.

        Args:
            results: List of BacktestResult objects
            output_path: Path to write Markdown file
            title: Report title
            param_names: Parameter names (for sweep results)
            param_values: Parameter values (for sweep results)
        """
        lines = [
            f"# {title}",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"- Total runs: {len(results)}",
        ]

        if results:
            # Calculate aggregate stats
            sharpes = [r.metrics.get("sharpe", 0.0) for r in results]
            returns = [r.metrics.get("total_return_pct", 0.0) for r in results]

            lines.extend(
                [
                    f"- Best Sharpe: {max(sharpes):.3f}",
                    f"- Best Return: {max(returns):.2f}%",
                    "",
                ]
            )

        lines.extend(
            [
                "## Results",
                "",
            ]
        )

        # Build results table
        if param_names and param_values:
            # Parameter sweep - include params
            header = "| Run | " + " | ".join(param_names) + " | Return % | Sharpe | DD % | Trades |"
            separator = "|" + "|".join(["---"] * (len(param_names) + 5)) + "|"
            lines.extend([header, separator])

            for i, (result, params) in enumerate(zip(results, param_values)):
                metrics = result.metrics
                param_vals = " | ".join(str(params.get(p, "")) for p in param_names)
                lines.append(
                    f"| {i + 1} | {param_vals} | "
                    f"{metrics.get('total_return_pct', 0):.2f} | "
                    f"{metrics.get('sharpe', 0):.3f} | "
                    f"{metrics.get('max_drawdown_pct', 0):.2f} | "
                    f"{metrics.get('num_trades', 0)} |"
                )
        else:
            # Simple list
            header = "| Run | Return % | Sharpe | Max DD % | Trades | Win Rate |"
            separator = "|---|---|---|---|---|---|"
            lines.extend([header, separator])

            for i, result in enumerate(results):
                metrics = result.metrics
                lines.append(
                    f"| {i + 1} | "
                    f"{metrics.get('total_return_pct', 0):.2f} | "
                    f"{metrics.get('sharpe', 0):.3f} | "
                    f"{metrics.get('max_drawdown_pct', 0):.2f} | "
                    f"{metrics.get('num_trades', 0)} | "
                    f"{metrics.get('win_rate', 0) * 100:.1f}% |"
                )

        lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))
