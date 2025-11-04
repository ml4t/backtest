"""Trade comparison and matching utilities."""

import sys
from pathlib import Path

# Handle imports when run as script vs module
try:
    from .matcher import TradeMatch, match_trades
    from .reporter import generate_trade_report, generate_summary_report
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from matcher import TradeMatch, match_trades
    from reporter import generate_trade_report, generate_summary_report

__all__ = ['TradeMatch', 'match_trades', 'generate_trade_report', 'generate_summary_report']
