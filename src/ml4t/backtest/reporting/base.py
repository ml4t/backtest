"""Base reporting functionality for ml4t.backtest."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from ml4t.backtest.portfolio.accounting import PortfolioAccounting


class ReportGenerator(ABC):
    """
    Abstract base class for report generation.

    Different report formats (HTML, Parquet, JSON) should implement this interface.
    """

    def __init__(self, output_dir: Path | None = None, report_name: str | None = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
            report_name: Base name for report files
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "reports"
        # Report name will use timestamp from first event if not provided
        self.report_name = report_name or "backtest_report"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate(
        self,
        accounting: PortfolioAccounting,
        strategy_params: dict[str, Any] | None = None,
        backtest_params: dict[str, Any] | None = None,
    ) -> Path:
        """
        Generate report from portfolio accounting data.

        Args:
            accounting: Portfolio accounting with results
            strategy_params: Strategy configuration parameters
            backtest_params: Backtest configuration parameters

        Returns:
            Path to generated report
        """

    def _prepare_report_data(
        self,
        accounting: PortfolioAccounting,
        strategy_params: dict[str, Any] | None = None,
        backtest_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Prepare standardized report data from accounting.

        Args:
            accounting: Portfolio accounting instance
            strategy_params: Strategy parameters
            backtest_params: Backtest parameters

        Returns:
            Dictionary with report data
        """
        # Get performance metrics
        metrics = accounting.get_performance_metrics()

        # Get summary data
        summary = accounting.get_summary()

        # Prepare report data structure
        report_data = {
            "metadata": {
                "report_name": self.report_name,
                "generated_at": datetime.now().isoformat(),  # Wall clock time for report generation
                "strategy_params": strategy_params or {},
                "backtest_params": backtest_params or {},
            },
            "performance": {
                "total_return": metrics.get("total_return", 0.0),
                "total_pnl": metrics.get("total_pnl", 0.0),
                "realized_pnl": metrics.get("realized_pnl", 0.0),
                "unrealized_pnl": metrics.get("unrealized_pnl", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "num_trades": metrics.get("num_trades", 0),
                "win_rate": accounting.calculate_win_rate(),
                "profit_factor": accounting.calculate_profit_factor(),
            },
            "costs": {
                "total_commission": metrics.get("total_commission", 0.0),
                "total_slippage": metrics.get("total_slippage", 0.0),
                "avg_commission_per_trade": accounting.calculate_avg_commission(),
                "avg_slippage_per_trade": accounting.calculate_avg_slippage(),
            },
            "portfolio": {
                "initial_cash": accounting.portfolio.initial_cash,
                "final_equity": summary.get("equity", 0.0),
                "final_cash": summary.get("cash", 0.0),
                "num_positions": summary.get("positions", 0),
            },
            "risk": {
                "max_leverage": metrics.get("max_leverage", 1.0),
                "max_concentration": metrics.get("max_concentration", 0.0),
            },
            "trades": accounting.get_trades_df(),
            "equity_curve": accounting.get_equity_curve_df(),
            "positions": accounting.get_positions_df(),
        }

        return report_data

