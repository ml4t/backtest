"""Reporting module for ml4t.backtest."""

# TODO: Update reporting module to work with new Portfolio architecture
# The following imports are temporarily disabled until reporting is updated
# to use the new Portfolio class instead of the removed PortfolioAccounting
#
# from ml4t.backtest.reporting.base import ReportGenerator
# from ml4t.backtest.reporting.html import HTMLReportGenerator
# from ml4t.backtest.reporting.parquet import ParquetReportGenerator

from ml4t.backtest.reporting.reporter import ConsoleReporter, InMemoryReporter, Reporter

__all__ = [
    "ConsoleReporter",
    "InMemoryReporter",
    "Reporter",
    # "HTMLReportGenerator",  # TODO: Update to use Portfolio
    # "ParquetReportGenerator",  # TODO: Update to use Portfolio
    # "ReportGenerator",  # TODO: Update to use Portfolio
]
