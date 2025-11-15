"""Parquet report generation for ml4t.backtest backtests."""

import json
from pathlib import Path
from typing import Any

import polars as pl

from ml4t.backtest.portfolio.accounting import PortfolioAccounting
from ml4t.backtest.reporting.base import ReportGenerator


class ParquetReportGenerator(ReportGenerator):
    """
    Generates Parquet-based reports for backtest results.

    Creates structured data files optimized for:
    - Data science workflows
    - Further analysis with Polars/Pandas
    - Integration with data pipelines
    - Long-term storage and archival
    """

    def generate(
        self,
        accounting: PortfolioAccounting,
        strategy_params: dict[str, Any] | None = None,
        backtest_params: dict[str, Any] | None = None,
    ) -> Path:
        """
        Generate Parquet report from portfolio accounting data.

        Args:
            accounting: Portfolio accounting with results
            strategy_params: Strategy configuration parameters
            backtest_params: Backtest configuration parameters

        Returns:
            Path to generated report directory
        """
        # Create report directory
        report_dir = self.output_dir / f"{self.report_name}_parquet"
        report_dir.mkdir(exist_ok=True)

        # Prepare report data
        report_data = self._prepare_report_data(accounting, strategy_params, backtest_params)

        # Save metadata as JSON
        self._save_metadata(report_data, report_dir)

        # Save performance metrics
        self._save_performance_metrics(report_data, report_dir)

        # Save time series data
        self._save_equity_curve(report_data, report_dir)

        # Save trades data
        self._save_trades(report_data, report_dir)

        # Save positions data
        self._save_positions(report_data, report_dir)

        # Create summary file
        self._create_summary_file(report_data, report_dir)

        return report_dir

    def _save_metadata(self, report_data: dict[str, Any], report_dir: Path) -> None:
        """Save metadata and configuration as JSON."""
        metadata = {
            "report_info": report_data["metadata"],
            "backtest_config": {
                "strategy_params": report_data["metadata"].get("strategy_params", {}),
                "backtest_params": report_data["metadata"].get("backtest_params", {}),
            },
            "portfolio_config": report_data["portfolio"],
            "file_manifest": {
                "metadata": "metadata.json",
                "performance": "performance_metrics.parquet",
                "equity_curve": "equity_curve.parquet",
                "trades": "trades.parquet",
                "positions": "positions.parquet",
                "summary": "summary.parquet",
            },
        }

        metadata_path = report_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _save_performance_metrics(self, report_data: dict[str, Any], report_dir: Path) -> None:
        """Save performance metrics as Parquet."""
        # Combine all performance data
        metrics_data = []

        # Performance metrics
        for key, value in report_data["performance"].items():
            metrics_data.append(
                {
                    "category": "performance",
                    "metric": key,
                    "value": float(value) if isinstance(value, (int, float)) else str(value),
                    "format": "percentage"
                    if "return" in key or "rate" in key
                    else "currency"
                    if "pnl" in key
                    else "number",
                },
            )

        # Cost metrics
        for key, value in report_data["costs"].items():
            metrics_data.append(
                {
                    "category": "costs",
                    "metric": key,
                    "value": float(value) if isinstance(value, (int, float)) else str(value),
                    "format": "currency" if "commission" in key or "slippage" in key else "number",
                },
            )

        # Risk metrics
        for key, value in report_data["risk"].items():
            metrics_data.append(
                {
                    "category": "risk",
                    "metric": key,
                    "value": float(value) if isinstance(value, (int, float)) else str(value),
                    "format": "percentage" if "concentration" in key else "number",
                },
            )

        # Portfolio metrics
        for key, value in report_data["portfolio"].items():
            metrics_data.append(
                {
                    "category": "portfolio",
                    "metric": key,
                    "value": float(value) if isinstance(value, (int, float)) else str(value),
                    "format": "currency" if "cash" in key or "equity" in key else "number",
                },
            )

        metrics_df = pl.DataFrame(metrics_data)
        metrics_path = report_dir / "performance_metrics.parquet"
        metrics_df.write_parquet(metrics_path)

    def _save_equity_curve(self, report_data: dict[str, Any], report_dir: Path) -> None:
        """Save equity curve as Parquet."""
        equity_df = report_data.get("equity_curve")

        if equity_df is not None and len(equity_df) > 0:
            # Add derived metrics
            enhanced_df = equity_df.with_columns(
                [
                    # Cumulative returns
                    pl.col("returns").cum_sum().alias("cumulative_returns"),
                    # Running maximum for drawdown calculation
                    pl.col("equity").cum_max().alias("running_max"),
                    # Drawdown
                    ((pl.col("equity") / pl.col("equity").cum_max()) - 1).alias("drawdown"),
                    # Volatility (rolling 30-day)
                    pl.col("returns").rolling_std(window_size=30).alias("rolling_volatility_30d"),
                    # Rolling Sharpe (annualized)
                    (
                        pl.col("returns").rolling_mean(window_size=30)
                        / pl.col("returns").rolling_std(window_size=30)
                        * (252**0.5)
                    ).alias("rolling_sharpe_30d"),
                ],
            )

            equity_path = report_dir / "equity_curve.parquet"
            enhanced_df.write_parquet(equity_path)
        else:
            # Create empty DataFrame with schema
            empty_df = pl.DataFrame(
                {
                    "timestamp": [],
                    "equity": [],
                    "returns": [],
                    "cumulative_returns": [],
                    "running_max": [],
                    "drawdown": [],
                    "rolling_volatility_30d": [],
                    "rolling_sharpe_30d": [],
                },
                schema={
                    "timestamp": pl.Datetime,
                    "equity": pl.Float64,
                    "returns": pl.Float64,
                    "cumulative_returns": pl.Float64,
                    "running_max": pl.Float64,
                    "drawdown": pl.Float64,
                    "rolling_volatility_30d": pl.Float64,
                    "rolling_sharpe_30d": pl.Float64,
                },
            )
            empty_df.write_parquet(report_dir / "equity_curve.parquet")

    def _save_trades(self, report_data: dict[str, Any], report_dir: Path) -> None:
        """Save trades data as Parquet."""
        trades_df = report_data.get("trades")

        if trades_df is not None and len(trades_df) > 0:
            # Add derived columns for analysis
            enhanced_df = trades_df.with_columns(
                [
                    # Notional value
                    (pl.col("quantity") * pl.col("price")).alias("notional_value"),
                    # Commission rate
                    (pl.col("commission") / (pl.col("quantity") * pl.col("price"))).alias(
                        "commission_rate",
                    ),
                    # Slippage rate
                    (pl.col("slippage") / (pl.col("quantity") * pl.col("price"))).alias(
                        "slippage_rate",
                    ),
                    # Trade direction
                    pl.when(pl.col("side") == "buy").then(1).otherwise(-1).alias("direction"),
                    # Time-based features
                    pl.col("timestamp").dt.hour().alias("hour_of_day"),
                    pl.col("timestamp").dt.day().alias("day_of_month"),
                    pl.col("timestamp").dt.weekday().alias("day_of_week"),
                    # Size categories
                    pl.when(pl.col("quantity") * pl.col("price") < 1000)
                    .then(pl.lit("small"))
                    .when(pl.col("quantity") * pl.col("price") < 10000)
                    .then(pl.lit("medium"))
                    .otherwise(pl.lit("large"))
                    .alias("trade_size_category"),
                ],
            )

            trades_path = report_dir / "trades.parquet"
            enhanced_df.write_parquet(trades_path)
        else:
            # Create empty DataFrame with schema
            empty_df = pl.DataFrame(
                {
                    "timestamp": [],
                    "order_id": [],
                    "trade_id": [],
                    "asset_id": [],
                    "side": [],
                    "quantity": [],
                    "price": [],
                    "commission": [],
                    "slippage": [],
                    "total_cost": [],
                    "notional_value": [],
                    "commission_rate": [],
                    "slippage_rate": [],
                    "direction": [],
                    "hour_of_day": [],
                    "day_of_month": [],
                    "day_of_week": [],
                    "trade_size_category": [],
                },
                schema={
                    "timestamp": pl.Datetime,
                    "order_id": pl.Utf8,
                    "trade_id": pl.Utf8,
                    "asset_id": pl.Utf8,
                    "side": pl.Utf8,
                    "quantity": pl.Float64,
                    "price": pl.Float64,
                    "commission": pl.Float64,
                    "slippage": pl.Float64,
                    "total_cost": pl.Float64,
                    "notional_value": pl.Float64,
                    "commission_rate": pl.Float64,
                    "slippage_rate": pl.Float64,
                    "direction": pl.Int8,
                    "hour_of_day": pl.UInt32,
                    "day_of_month": pl.UInt32,
                    "day_of_week": pl.UInt32,
                    "trade_size_category": pl.Utf8,
                },
            )
            empty_df.write_parquet(report_dir / "trades.parquet")

    def _save_positions(self, report_data: dict[str, Any], report_dir: Path) -> None:
        """Save positions data as Parquet."""
        positions_df = report_data.get("positions")

        if positions_df is not None and len(positions_df) > 0:
            # Add derived columns
            enhanced_df = positions_df.with_columns(
                [
                    # Position direction
                    pl.when(pl.col("quantity") > 0)
                    .then(pl.lit("long"))
                    .when(pl.col("quantity") < 0)
                    .then(pl.lit("short"))
                    .otherwise(pl.lit("flat"))
                    .alias("position_type"),
                    # Average cost per share
                    (pl.col("cost_basis") / pl.col("quantity")).alias("avg_cost_per_share"),
                    # Unrealized return percentage
                    (pl.col("unrealized_pnl") / pl.col("cost_basis")).alias(
                        "unrealized_return_pct",
                    ),
                    # Total return percentage
                    (pl.col("total_pnl") / pl.col("cost_basis")).alias("total_return_pct"),
                    # Position weight (would need portfolio value for this)
                    pl.col("market_value").alias("position_weight_placeholder"),
                ],
            )

            positions_path = report_dir / "positions.parquet"
            enhanced_df.write_parquet(positions_path)
        else:
            # Create empty DataFrame with schema
            empty_df = pl.DataFrame(
                {
                    "asset_id": [],
                    "quantity": [],
                    "cost_basis": [],
                    "last_price": [],
                    "market_value": [],
                    "unrealized_pnl": [],
                    "realized_pnl": [],
                    "total_pnl": [],
                    "position_type": [],
                    "avg_cost_per_share": [],
                    "unrealized_return_pct": [],
                    "total_return_pct": [],
                    "position_weight_placeholder": [],
                },
                schema={
                    "asset_id": pl.Utf8,
                    "quantity": pl.Float64,
                    "cost_basis": pl.Float64,
                    "last_price": pl.Float64,
                    "market_value": pl.Float64,
                    "unrealized_pnl": pl.Float64,
                    "realized_pnl": pl.Float64,
                    "total_pnl": pl.Float64,
                    "position_type": pl.Utf8,
                    "avg_cost_per_share": pl.Float64,
                    "unrealized_return_pct": pl.Float64,
                    "total_return_pct": pl.Float64,
                    "position_weight_placeholder": pl.Float64,
                },
            )
            empty_df.write_parquet(report_dir / "positions.parquet")

    def _create_summary_file(self, report_data: dict[str, Any], report_dir: Path) -> None:
        """Create a summary file with key statistics."""
        summary_data = [
            {
                "report_name": report_data["metadata"]["report_name"],
                "generated_at": report_data["metadata"]["generated_at"],
                "total_return": report_data["performance"]["total_return"],
                "total_pnl": report_data["performance"]["total_pnl"],
                "sharpe_ratio": report_data["performance"].get("sharpe_ratio", None),
                "max_drawdown": report_data["performance"]["max_drawdown"],
                "win_rate": report_data["performance"]["win_rate"],
                "num_trades": report_data["performance"]["num_trades"],
                "total_commission": report_data["costs"]["total_commission"],
                "total_slippage": report_data["costs"]["total_slippage"],
                "initial_capital": report_data["portfolio"]["initial_cash"],
                "final_equity": report_data["portfolio"]["final_equity"],
                "max_leverage": report_data["risk"]["max_leverage"],
                "max_concentration": report_data["risk"]["max_concentration"],
            },
        ]

        summary_df = pl.DataFrame(summary_data)
        summary_path = report_dir / "summary.parquet"
        summary_df.write_parquet(summary_path)

    def load_report(self, report_dir: Path) -> dict[str, Any]:
        """
        Load a previously generated Parquet report.

        Args:
            report_dir: Directory containing the Parquet report

        Returns:
            Dictionary with loaded report data
        """
        if not report_dir.exists():
            raise FileNotFoundError(f"Report directory not found: {report_dir}")

        # Load metadata
        metadata_path = report_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load data files
        report_data = {
            "metadata": metadata,
            "performance_metrics": pl.read_parquet(report_dir / "performance_metrics.parquet"),
            "equity_curve": pl.read_parquet(report_dir / "equity_curve.parquet"),
            "trades": pl.read_parquet(report_dir / "trades.parquet"),
            "positions": pl.read_parquet(report_dir / "positions.parquet"),
            "summary": pl.read_parquet(report_dir / "summary.parquet"),
        }

        return report_data
