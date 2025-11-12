"""Reporting module for QEngine."""

# TODO: Update reporting module to work with new Portfolio architecture
# The following imports are temporarily disabled until reporting is updated
# to use the new Portfolio class instead of the removed PortfolioAccounting
#
# from qengine.reporting.base import ReportGenerator
# from qengine.reporting.html import HTMLReportGenerator
# from qengine.reporting.parquet import ParquetReportGenerator

from qengine.reporting.reporter import ConsoleReporter, InMemoryReporter, Reporter

__all__ = [
    "ConsoleReporter",
    "InMemoryReporter",
    "Reporter",
    # "HTMLReportGenerator",  # TODO: Update to use Portfolio
    # "ParquetReportGenerator",  # TODO: Update to use Portfolio
    # "ReportGenerator",  # TODO: Update to use Portfolio
]
