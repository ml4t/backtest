"""Reporting module for ml4t.backtest."""

# TODO: Update reporting module to work with new Portfolio architecture
# The following imports are temporarily disabled until reporting is updated
# to use the new Portfolio class instead of the removed PortfolioAccounting
#
# from ml4t.backtest.reporting.base import ReportGenerator
# from ml4t.backtest.reporting.html import HTMLReportGenerator
# from ml4t.backtest.reporting.parquet import ParquetReportGenerator

from ml4t.backtest.reporting.reporter import ConsoleReporter, InMemoryReporter, Reporter
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
]
