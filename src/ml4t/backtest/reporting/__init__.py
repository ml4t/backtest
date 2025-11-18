"""Reporting module for ml4t.backtest."""

# TODO: Update reporting module to work with new Portfolio architecture
# The following imports are temporarily disabled until reporting is updated
# to use the new Portfolio class instead of the removed PortfolioAccounting
#
# from ml4t.backtest.reporting.base import ReportGenerator
# from ml4t.backtest.reporting.html import HTMLReportGenerator
# from ml4t.backtest.reporting.parquet import ParquetReportGenerator

from ml4t.backtest.reporting.reporter import ConsoleReporter, InMemoryReporter, Reporter
from ml4t.backtest.reporting.trade_analysis import (
    analyze_trades,
    avg_hold_time_by_rule,
    feature_correlation,
    pnl_attribution,
    rule_effectiveness,
    win_rate_by_rule,
)
from ml4t.backtest.reporting.trade_schema import (
    ExitReason,
    MLTradeRecord,
    append_trades,
    export_parquet,
    get_schema,
    import_parquet,
    polars_to_trades,
    trades_to_polars,
)
from ml4t.backtest.reporting.visualizations import (
    plot_exit_reasons,
    plot_feature_importance,
    plot_hold_time_distribution,
    plot_mfe_mae_scatter,
    plot_rule_performance,
)

__all__ = [
    "ConsoleReporter",
    "InMemoryReporter",
    "Reporter",
    # "HTMLReportGenerator",  # TODO: Update to use Portfolio
    # "ParquetReportGenerator",  # TODO: Update to use Portfolio
    # "ReportGenerator",  # TODO: Update to use Portfolio
    # Trade schema
    "MLTradeRecord",
    "ExitReason",
    "get_schema",
    "trades_to_polars",
    "polars_to_trades",
    "export_parquet",
    "import_parquet",
    "append_trades",
    # Trade analysis
    "win_rate_by_rule",
    "avg_hold_time_by_rule",
    "pnl_attribution",
    "rule_effectiveness",
    "feature_correlation",
    "analyze_trades",
    # Visualizations
    "plot_rule_performance",
    "plot_hold_time_distribution",
    "plot_feature_importance",
    "plot_exit_reasons",
    "plot_mfe_mae_scatter",
]
